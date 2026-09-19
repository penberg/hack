//! The shell: an input box at the bottom of the terminal, with the model
//! answering above it. The model runs on a thread of its own so the screen
//! stays responsive while it downloads, loads, and generates.

use std::{
    env,
    error::Error,
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    style::Stylize,
};
use dwim_gpu::{Cpu, Gpu};
use dwim_harness::{self as harness, Harness};
use dwim_models::{Chat, Gguf, Tokenizer};

use crate::{
    fetch::{self, Progress},
    models,
    opts::Device,
    tui::{self, Input, Line, Screen, span},
};

/// How often the screen is redrawn while the model is busy, to animate the
/// spinner.
const TICK: Duration = Duration::from_millis(80);

/// Runs the shell until the user quits, showing the model's thinking if
/// `thinking` is set, or once ctrl+o is pressed.
pub fn run(
    name: &str,
    device: Device,
    context: usize,
    thinking: bool,
) -> Result<(), Box<dyn Error>> {
    let (model, dir) = fetch::locate(name)?;

    let (requests, requests_rx) = mpsc::channel();
    let (replies_tx, replies) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let dir = dir.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            if let Err(e) = work(
                model,
                &dir,
                device,
                context,
                requests_rx,
                &replies_tx,
                &stop,
            ) {
                let _ = replies_tx.send(Reply::Failed(e.to_string()));
            }
        });
    }

    let mut app = App {
        screen: Screen::new()?,
        model: name.to_string(),
        device: None,
        cwd: env::current_dir().unwrap_or_default(),
        input: Input::default(),
        history: Vec::new(),
        recalled: None,
        status: Status::Loading {
            since: Instant::now(),
            done: 0,
            total: 0,
        },
        started: Instant::now(),
        loading: Duration::ZERO,
        stats: harness::Stats::default(),
        reply: String::new(),
        segment: Segment::Text,
        reply_width: 0,
        committed: 0,
        interrupted: false,
        thinking,
        lines: Vec::new(),
        requests,
        replies,
        stop,
    };
    let result = app.run();
    // Leave the conversation on screen, but not the input box, with a blank
    // line between it and whatever the shell prints next.
    app.lines.push(Line::new());
    app.screen.draw(&app.lines, &[], (0, 0))?;
    result
}

/// What the shell asks of the model thread.
enum Request {
    /// A message from the user.
    Message(String),
    /// Compact the conversation now, as the `/compact` command asks.
    Compact,
}

/// What the model thread reports back.
enum Reply {
    /// Part of one of the model's files has been downloaded.
    Downloading(Progress),
    /// Which device the model is being loaded onto.
    Device(String),
    /// How many of the model's tensors are loaded, out of how many.
    Loading {
        done: usize,
        total: usize,
    },
    /// The model is reading the system prompt: how many of its tokens so
    /// far, out of how many.
    Prompting {
        read: usize,
        total: usize,
    },
    /// The model is ready for a message.
    Ready,
    /// Part of the model's thought, before it replies.
    Thought(String),
    Text(String),
    /// A tool is about to run: its name and how it was called.
    Call {
        name: String,
        detail: String,
    },
    /// What the tool returned.
    Output(String),
    /// The reply was stopped short because the context window was full.
    Cut,
    /// The conversation is being compacted: the model's note to go on
    /// from follows, in pieces.
    Compacting,
    Note(String),
    /// The conversation was compacted: from how many tokens, to how many.
    Compacted {
        before: usize,
        after: usize,
    },
    /// Something to tell the user, in place of a reply.
    Notice(String),
    /// The reply is over, and where the conversation's time has gone.
    Done(harness::Stats),
    /// The turn failed, though the model is still there for the next one.
    Error(String),
    /// The model is gone: it could not be fetched or loaded.
    Failed(String),
}

/// Fetches the model if needed, then loads it and answers messages until
/// the UI hangs up.
fn work(
    model: &'static models::Model,
    dir: &Path,
    device: Device,
    context: usize,
    requests: Receiver<Request>,
    replies: &Sender<Reply>,
    stop: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    fetch::fetch(model, dir, |progress| {
        let _ = replies.send(Reply::Downloading(progress));
    })?;
    let gguf = model.open(dir)?;
    match device {
        Device::Cpu => {
            let _ = replies.send(Reply::Device("cpu".to_string()));
            serve(gguf, Cpu, model, context, requests, replies, stop)
        }
        Device::Gpu => {
            let gpu = Gpu::new()?;
            // Drivers append their own name in parentheses; the GPU's is enough.
            let name = gpu
                .name()
                .split(" (")
                .next()
                .unwrap_or(gpu.name())
                .to_string();
            let _ = replies.send(Reply::Device(name));
            serve(gguf, gpu, model, context, requests, replies, stop)
        }
    }
}

/// Loads the model and answers messages until the UI hangs up.
fn serve<D: dwim_gpu::Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    which: &'static models::Model,
    context: usize,
    requests: Receiver<Request>,
    replies: &Sender<Reply>,
    stop: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let sampler = models::sampler(&gguf);
    let model = models::load(gguf, device, context, |done, total| {
        let _ = replies.send(Reply::Loading { done, total });
    })?;
    let mut chat = Chat::new(model, tokenizer, sampler)?;
    let cwd = env::current_dir()?;
    let system = harness::system_prompt(&cwd);
    models::start(&mut chat, which, &system, |read, total| {
        let _ = replies.send(Reply::Prompting { read, total });
    })?;
    let mut harness = Harness::new(chat, &cwd, move |chat| {
        models::start(chat, which, &system, |_, _| {}).map(|_| ())
    })?;
    let _ = replies.send(Reply::Ready);

    for request in requests {
        let on_event = |event: harness::Event| {
            let _ = replies.send(reply(event));
            if stop.load(Ordering::Relaxed) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let result = match request {
            Request::Message(message) => harness.send(&message, on_event),
            Request::Compact => harness.compact(on_event).map(|compacted| {
                if !compacted && !stop.load(Ordering::Relaxed) {
                    let _ = replies.send(Reply::Notice("Nothing to compact yet".to_string()));
                }
            }),
        };
        if let Err(e) = result {
            let _ = replies.send(Reply::Error(e.to_string()));
        }
        let _ = replies.send(Reply::Done(harness.stats()));
    }
    Ok(())
}

/// What the model thread reports for an event of a turn.
fn reply(event: harness::Event) -> Reply {
    match event {
        harness::Event::Thought(text) => Reply::Thought(text.to_string()),
        harness::Event::Text(text) => Reply::Text(text.to_string()),
        harness::Event::Call { name, detail } => Reply::Call {
            name: name.to_string(),
            detail: detail.to_string(),
        },
        harness::Event::Output(output) => Reply::Output(output.to_string()),
        harness::Event::Cut => Reply::Cut,
        harness::Event::Compacting => Reply::Compacting,
        harness::Event::Note(text) => Reply::Note(text.to_string()),
        harness::Event::Compacted { before, after } => Reply::Compacted { before, after },
    }
}

/// What the model is doing.
enum Status {
    /// Downloading one of the model's files, since when.
    Downloading {
        since: Instant,
        progress: Progress,
    },
    /// Loading the weights, since when: how many tensors so far, out of how
    /// many.
    Loading {
        since: Instant,
        done: usize,
        total: usize,
    },
    /// Reading the system prompt, since when: how many of its tokens so
    /// far, out of how many.
    Prompting {
        since: Instant,
        read: usize,
        total: usize,
    },
    Idle,
    /// Reading the prompt, before the first token of the reply.
    Reading(Instant),
    Generating {
        start: Instant,
        tokens: usize,
    },
}

/// What part of the reply is being generated.
#[derive(Clone, Copy, PartialEq)]
enum Segment {
    /// The model's thought, before it replies.
    Thought,
    /// The reply itself.
    Text,
    /// The note the model goes on from after the conversation is compacted.
    Note,
}

struct App {
    screen: Screen,
    model: String,
    /// The device the model runs on, once known.
    device: Option<String>,
    cwd: PathBuf,
    input: Input,
    /// Messages sent so far, and which one is recalled into the input.
    history: Vec<String>,
    recalled: Option<usize>,
    status: Status,
    /// When the shell started, how long loading the model took, and where
    /// the conversation's time has gone, as of the last reply.
    started: Instant,
    loading: Duration,
    stats: harness::Stats,
    /// The part of the reply being generated, which part it is, the width
    /// it is wrapped at, and how many of its lines have been printed.
    reply: String,
    segment: Segment,
    reply_width: usize,
    committed: usize,
    interrupted: bool,
    /// Whether the model's thinking is shown. While it is hidden, the
    /// thought is kept but none of it is printed, so showing it prints
    /// all of it so far.
    thinking: bool,
    /// Lines waiting to be printed above the live region.
    lines: Vec<Line>,
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    stop: Arc<AtomicBool>,
}

impl App {
    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut dirty = true;
        loop {
            if dirty || !matches!(self.status, Status::Idle) {
                self.draw()?;
                dirty = false;
            }
            if event::poll(TICK)? {
                match event::read()? {
                    Event::Key(key) if key.kind != KeyEventKind::Release => {
                        if self.key(key) {
                            return Ok(());
                        }
                    }
                    Event::Paste(text) => self.input.insert(&tui::sanitize(&text)),
                    _ => {}
                }
                dirty = true;
            }
            loop {
                match self.replies.try_recv() {
                    Ok(reply) => self.reply(reply)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err("the model stopped unexpectedly".into());
                    }
                }
                dirty = true;
            }
        }
    }

    /// Handles a key press, returning whether to quit.
    fn key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let busy = !matches!(self.status, Status::Idle);
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if matches!(self.status, Status::Reading(_) | Status::Generating { .. }) {
                    self.interrupt();
                } else if !self.input.is_empty() {
                    self.input.take();
                } else {
                    return true;
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => return true,
            KeyCode::Esc => self.interrupt(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
            {
                self.input.insert("\n")
            }
            KeyCode::Char('j') if ctrl => self.input.insert("\n"),
            KeyCode::Char('o') if ctrl => {
                self.thinking = !self.thinking;
                if self.thinking
                    && self.segment == Segment::Thought
                    && matches!(self.status, Status::Generating { .. })
                {
                    self.commit(false);
                }
            }
            KeyCode::Enter if !busy => self.submit(),
            KeyCode::Enter => {}
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            KeyCode::Char('w') if ctrl => self.input.delete_word(),
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::ALT) => {
                self.input.delete_word()
            }
            KeyCode::Char(c) if !ctrl => self.input.insert(c.encode_utf8(&mut [0; 4])),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Up => self.recall(-1),
            KeyCode::Down => self.recall(1),
            _ => {}
        }
        false
    }

    /// Stops the reply being generated, if any.
    fn interrupt(&mut self) {
        if matches!(self.status, Status::Reading(_) | Status::Generating { .. }) {
            self.stop.store(true, Ordering::Relaxed);
            self.interrupted = true;
        }
    }

    /// Steps through the messages sent so far, like a shell's history.
    fn recall(&mut self, step: isize) {
        if self.input.text().contains('\n') || self.history.is_empty() {
            return;
        }
        let last = self.history.len() - 1;
        self.recalled = match (self.recalled, step) {
            (None, -1) => Some(last),
            (None, _) => None,
            (Some(i), -1) => Some(i.saturating_sub(1)),
            (Some(i), _) if i < last => Some(i + 1),
            (Some(_), _) => None,
        };
        let text = self
            .recalled
            .map(|i| self.history[i].clone())
            .unwrap_or_default();
        self.input.set(text);
    }

    fn submit(&mut self) {
        let message = self.input.text().trim().to_string();
        if message.is_empty() {
            return;
        }
        self.input.take();
        self.recalled = None;
        if self.history.last() != Some(&message) {
            self.history.push(message.clone());
        }

        let (columns, _) = self.screen.size();
        self.lines.push(Line::new());
        for (i, line) in tui::wrap(&message, columns.saturating_sub(2))
            .into_iter()
            .enumerate()
        {
            let prompt = if i == 0 {
                span("› ").bold()
            } else {
                span("  ")
            };
            self.lines.push(vec![prompt, span(line).bold()]);
        }
        self.lines.push(Line::new());

        let request = match message.strip_prefix('/') {
            Some("compact") => Request::Compact,
            Some(command) => {
                self.command(command);
                return;
            }
            None => Request::Message(message),
        };

        self.reply.clear();
        self.segment = Segment::Text;
        self.reply_width = columns.saturating_sub(2);
        self.committed = 0;
        self.interrupted = false;
        self.stop.store(false, Ordering::Relaxed);
        self.status = Status::Reading(Instant::now());
        let _ = self.requests.send(request);
    }

    fn reply(&mut self, reply: Reply) -> Result<(), Box<dyn Error>> {
        match reply {
            Reply::Downloading(progress) => {
                // Time each file on its own, so the speed is of this one.
                let since = match self.status {
                    Status::Downloading {
                        since,
                        progress: last,
                    } if last.file == progress.file => since,
                    _ => Instant::now(),
                };
                self.status = Status::Downloading { since, progress };
            }
            Reply::Device(device) => self.device = Some(device),
            Reply::Loading { done, total } => {
                let since = match self.status {
                    Status::Loading { since, .. } => since,
                    _ => Instant::now(),
                };
                self.status = Status::Loading { since, done, total };
            }
            Reply::Prompting { read, total } => {
                let since = match self.status {
                    Status::Prompting { since, .. } => since,
                    Status::Loading { since, .. } => {
                        self.loading = since.elapsed();
                        Instant::now()
                    }
                    _ => Instant::now(),
                };
                self.status = Status::Prompting { since, read, total };
            }
            Reply::Ready => self.status = Status::Idle,
            Reply::Thought(text) => self.generated(Segment::Thought, &text),
            Reply::Text(text) => self.generated(Segment::Text, &text),
            Reply::Call { name, detail } => {
                // The text so far is finished; the tool's output and the
                // rest of the reply follow it.
                self.finish();
                let detail: Vec<&str> = detail.split_whitespace().collect();
                self.lines.push(vec![
                    span("● "),
                    span(name).bold(),
                    span(format!("({})", detail.join(" "))).dark_grey(),
                ]);
                self.status = Status::Reading(Instant::now());
            }
            Reply::Output(output) => {
                let (columns, _) = self.screen.size();
                let output = tui::sanitize(&output);
                let mut first = true;
                for line in output.lines() {
                    for text in tui::wrap(line, columns.saturating_sub(4)) {
                        let prefix = if first { "  ⎿ " } else { "    " };
                        first = false;
                        self.lines
                            .push(vec![span(format!("{prefix}{text}")).dark_grey()]);
                    }
                }
                self.lines.push(Line::new());
            }
            Reply::Cut => {
                self.finish();
                self.lines.push(vec![
                    span("  ⎿ Stopped short: the context window is full").dark_grey(),
                ]);
                self.lines.push(Line::new());
            }
            Reply::Compacting => {
                self.finish();
                self.lines.push(vec![
                    span("● ").dark_grey(),
                    span("Compacting the conversation…").dark_grey(),
                ]);
                self.status = Status::Reading(Instant::now());
            }
            Reply::Note(text) => self.generated(Segment::Note, &text),
            Reply::Compacted { before, after } => {
                self.finish();
                self.lines.push(vec![
                    span(format!("  ⎿ Compacted from {before} to {after} tokens")).dark_grey(),
                ]);
                self.lines.push(Line::new());
                self.status = Status::Reading(Instant::now());
            }
            Reply::Done(stats) => {
                self.stats = stats;
                self.reply.truncate(self.reply.trim_end().len());
                self.commit(true);
                if self.interrupted {
                    self.lines.push(vec![span("  ⎿ Interrupted").dark_grey()]);
                }
                self.status = Status::Idle;
            }
            Reply::Notice(text) => {
                self.finish();
                self.lines.push(vec![span(format!("● {text}")).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::Error(e) => {
                self.finish();
                self.lines
                    .push(vec![span("● ").red(), span(format!("error: {e}"))]);
                self.lines.push(Line::new());
            }
            Reply::Failed(e) => return Err(e.into()),
        }
        Ok(())
    }

    /// Adds a piece of the reply to the part being generated, or starts
    /// the next part if it is of a different kind.
    fn generated(&mut self, segment: Segment, text: &str) {
        if segment != self.segment {
            self.finish();
            self.segment = segment;
        }
        let text = tui::sanitize(text);
        // The model tends to open with blank lines.
        let text = if self.reply.is_empty() {
            text.trim_start()
        } else {
            &text
        };
        self.reply.push_str(text);
        self.status = match self.status {
            Status::Generating { start, tokens } => Status::Generating {
                start,
                tokens: tokens + 1,
            },
            _ => Status::Generating {
                start: Instant::now(),
                tokens: 1,
            },
        };
        self.commit(false);
    }

    /// Prints the rest of the part of the reply being generated, with a
    /// blank line after it if there was anything to print, and starts over.
    fn finish(&mut self) {
        self.reply.truncate(self.reply.trim_end().len());
        self.commit(true);
        if self.committed > 0 {
            self.lines.push(Line::new());
        }
        self.reply.clear();
        self.committed = 0;
    }

    /// Prints the lines of the reply that are finished: all of them once the
    /// reply is done, and otherwise all but the last, which may still grow.
    /// Blank lines are held back until something follows them.
    fn commit(&mut self, done: bool) {
        if self.segment == Segment::Thought && !self.thinking {
            return;
        }
        let lines = tui::wrap(&self.reply, self.reply_width);
        let end = if done { lines.len() } else { lines.len() - 1 };
        let end = lines[..end]
            .iter()
            .rposition(|line| !line.is_empty())
            .map_or(0, |i| i + 1);
        if end > self.committed {
            let segment = self.segment;
            let committed = self.committed;
            self.lines.extend(
                lines
                    .into_iter()
                    .enumerate()
                    .take(end)
                    .skip(committed)
                    .map(|line| reply_line(segment, line)),
            );
            self.committed = end;
        }
    }

    /// Runs a slash command, printing what it has to say.
    fn command(&mut self, command: &str) {
        let (columns, _) = self.screen.size();
        let lines = match command.trim() {
            "model" => models_listing(&self.model),
            "stats" => {
                let report = self.stats.report(self.loading, self.started.elapsed());
                report
                    .into_iter()
                    .map(|line| vec![span(format!("  {line}"))])
                    .collect()
            }
            other => vec![vec![span(format!("● Unknown command /{other}"))]],
        };
        self.lines
            .extend(lines.iter().map(|line| tui::truncate(line, columns)));
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        let (columns, rows) = self.screen.size();
        let mut live = Vec::new();

        // The line of the reply still being generated, and the status. The
        // message a reply is to, or a tool's output, already ends in a blank
        // line.
        if matches!(self.status, Status::Generating { .. })
            && (self.segment == Segment::Text || self.thinking)
        {
            let lines = tui::wrap(&self.reply, self.reply_width);
            let segment = self.segment;
            live.extend(
                lines
                    .into_iter()
                    .enumerate()
                    .skip(self.committed)
                    .map(|line| reply_line(segment, line)),
            );
        }
        if !matches!(self.status, Status::Reading(_)) {
            live.push(Line::new());
        }
        if let Some(status) = self.status_line() {
            live.push(status);
            live.push(Line::new());
        }

        // The input box, as tall as its text but no taller than the screen,
        // scrolled to show the cursor.
        let (text, (row, column)) = self.input.layout(columns.saturating_sub(6).max(1));
        let height = rows.saturating_sub(live.len() + 3).max(1);
        let first = (row + 1).saturating_sub(height);
        let border = |left: &str, right: &str| {
            vec![
                span(format!(
                    "{left}{}{right}",
                    "─".repeat(columns.saturating_sub(2))
                ))
                .dark_grey(),
            ]
        };
        live.push(border("╭", "╮"));
        let cursor = (live.len() + row - first, 4 + column);
        for (i, line) in text.into_iter().enumerate().skip(first).take(height) {
            let prompt = match (i, &self.status) {
                (0, Status::Idle) => span("› ").bold(),
                (0, _) => span("› ").dark_grey(),
                _ => span("  "),
            };
            let content = if self.input.is_empty() {
                span("Ask anything, or describe what to build").dark_grey()
            } else {
                span(line)
            };
            live.push(tui::boxed(vec![prompt, content], columns));
        }
        live.push(border("╰", "╯"));
        live.push(self.footer());

        let lines = std::mem::take(&mut self.lines);
        self.screen.draw(&lines, &live, cursor)?;
        Ok(())
    }

    /// What the model is doing, with a spinner, while it's busy.
    fn status_line(&self) -> Option<Line> {
        const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let (verb, since, details) = match self.status {
            Status::Idle => return None,
            Status::Downloading { since, progress } => (
                format!("Downloading {}…", self.model),
                since,
                download_details(progress, since.elapsed()),
            ),
            Status::Loading { since, done, total } => (
                match &self.device {
                    Some(device) => format!("Loading {} onto {device}…", self.model),
                    None => format!("Loading {}…", self.model),
                },
                since,
                format!("{done}/{total} tensors · {}s", since.elapsed().as_secs()),
            ),
            Status::Prompting { since, read, total } => (
                "Reading system prompt…".to_string(),
                since,
                format!("{read}/{total} tokens · {}s", since.elapsed().as_secs()),
            ),
            Status::Reading(since) => (
                "Reading…".to_string(),
                since,
                format!("{}s · esc to interrupt", since.elapsed().as_secs()),
            ),
            Status::Generating { start, tokens } => (
                match self.segment {
                    Segment::Thought => "Thinking…".to_string(),
                    Segment::Text => "Generating…".to_string(),
                    Segment::Note => "Compacting…".to_string(),
                },
                start,
                format!(
                    "{tokens} tokens · {:.1} tok/s · {}esc to interrupt",
                    tokens as f32 / start.elapsed().as_secs_f32(),
                    match (self.segment, self.thinking) {
                        (Segment::Thought, false) => "ctrl+o to show · ",
                        (Segment::Thought, true) => "ctrl+o to hide · ",
                        (Segment::Text | Segment::Note, _) => "",
                    }
                ),
            ),
        };
        let frame = SPINNER[(since.elapsed().as_millis() / 150) as usize % SPINNER.len()];
        Some(vec![
            span(format!("{frame} {verb}")).bold(),
            span(format!(" {details}")).dark_grey(),
        ])
    }

    /// The directory the agent works in.
    fn footer(&self) -> Line {
        vec![span(format!("  {}", tilde(&self.cwd))).dark_grey()]
    }
}

/// How a download is going: the file, how much of it has arrived, and at what
/// speed, with the time left once the file's size is known.
fn download_details(progress: Progress, elapsed: Duration) -> String {
    let Progress { file, done, total } = progress;
    let mut details = format!("{file} · {}", fetch::size(done));
    if let Some(total) = total {
        details.push_str(&format!(
            " / {} ({}%)",
            fetch::size(total),
            done * 100 / total.max(1)
        ));
    }
    // Too early to tell the speed, or nothing has arrived yet.
    if elapsed < Duration::from_secs(1) || done == 0 {
        return details;
    }
    let rate = done as f64 / elapsed.as_secs_f64();
    details.push_str(&format!(" · {}/s", fetch::size(rate as u64)));
    if let Some(total) = total
        && done < total
    {
        let left = ((total - done) as f64 / rate).round() as u64;
        let left = match left {
            0..60 => format!("{left}s"),
            60..3600 => format!("{}m {}s", left / 60, left % 60),
            _ => format!("{}h {}m", left / 3600, left % 3600 / 60),
        };
        details.push_str(&format!(" · {left} left"));
    }
    details
}

/// A path, with the home directory as `~`.
fn tilde(path: &Path) -> String {
    match dirs::home_dir().and_then(|home| path.strip_prefix(home).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// The models `dwim` knows, where their weights are kept, and how much of
/// each is there, for the `/model` command. The one running is marked.
fn models_listing(running: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    for (i, model) in models::MODELS.iter().enumerate() {
        let bullet = if i == 0 { "● " } else { "  " };
        let mut line = vec![
            span(bullet),
            span(model.name).bold(),
            span(format!("  {}", model.repo)),
        ];
        if model.name == running {
            line.push(span("  running").dark_grey());
        }
        lines.push(line);
        let location = match model.dir() {
            Some(dir) => {
                let (files, bytes) = fetch::on_disk(model, &dir);
                let status = match files {
                    0 => "not fetched".to_string(),
                    n if n == model.files.len() => fetch::size(bytes),
                    n => format!("{n}/{} files, {}", model.files.len(), fetch::size(bytes)),
                };
                format!("{} · {status}", tilde(&dir))
            }
            None => "no cache directory".to_string(),
        };
        lines.push(vec![span(format!("  {location}")).dark_grey()]);
    }
    lines
}

/// A line of a part of the reply, the first one marked with a bullet. A
/// thought is dimmed, so the reply stands out from it, and so is the note
/// the model goes on from after a compaction.
fn reply_line(segment: Segment, (i, text): (usize, &str)) -> Line {
    match segment {
        Segment::Thought | Segment::Note => {
            let bullet = if i == 0 { span("✻ ") } else { span("  ") };
            vec![bullet.dark_grey(), span(text).dark_grey().italic()]
        }
        Segment::Text => {
            let bullet = if i == 0 { span("● ") } else { span("  ") };
            vec![bullet, span(text)]
        }
    }
}
