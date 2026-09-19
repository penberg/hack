//! Answering one prompt and exiting, without the shell.
//!
//! The model's reply goes to standard output and nothing else does, so that
//! a script can read the reply on its own. Getting ready, what the model is
//! thinking, the commands it runs, and what they printed all go to standard
//! error.

use std::{
    env,
    error::Error,
    io::{self, IsTerminal, Write},
    ops::ControlFlow,
    sync::Arc,
    time::{Duration, Instant},
};

use dwim_gpu::{Cpu, Gpu};
use dwim_harness::{self as harness, Harness};
use dwim_models::{Chat, Gguf, Tokenizer};

use crate::{fetch, models, opts::Device};

/// Answers `prompt` with the model, running the tools it calls, and returns
/// once it replies with text alone; then reports where the time went, if
/// `stats` asks for it.
pub fn once(
    name: &str,
    device: Device,
    context: usize,
    prompt: &str,
    stats: bool,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let (model, dir) = fetch::locate(name)?;
    let mut progress = Progress::new();
    fetch::fetch(model, &dir, |file| {
        let total = file.total.unwrap_or(file.done);
        progress.report(
            format!("downloading {}", file.file),
            megabytes(file.done),
            megabytes(total),
        );
    })?;
    let gguf = model.open(&dir)?;
    let (report, loading) = match device {
        Device::Cpu => answer(gguf, Cpu, model, "the CPU", context, prompt)?,
        Device::Gpu => {
            let gpu = Gpu::new()?;
            // Drivers append their own name in parentheses; the GPU's is enough.
            let device = gpu
                .name()
                .split(" (")
                .next()
                .unwrap_or(gpu.name())
                .to_string();
            answer(gguf, gpu, model, &device, context, prompt)?
        }
    };
    if stats {
        for line in report.report(loading, started.elapsed()) {
            eprintln!("{line}");
        }
    }
    Ok(())
}

/// Loads `name` onto `device`, with room for `context` tokens, and answers
/// `prompt` with it, returning where the time went and how long the
/// loading took.
fn answer<D: dwim_gpu::Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    which: &'static models::Model,
    on: &str,
    context: usize,
    prompt: &str,
) -> Result<(harness::Stats, Duration), Box<dyn Error>> {
    let name = which.name;
    let mut progress = Progress::new();
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let sampler = models::sampler(&gguf);
    let loading = Instant::now();
    let model = models::load(gguf, device, context, |done, total| {
        progress.report(format!("loading {name} on {on}"), done, total);
    })?;
    let loading = loading.elapsed();
    let mut chat = Chat::new(model, tokenizer, sampler)?;
    let cwd = env::current_dir()?;
    let system = harness::system_prompt(&cwd);
    models::start(&mut chat, which, &system, |read, total| {
        progress.report("reading the system prompt".to_string(), read, total);
    })?;

    let mut printer = Printer::default();
    let mut harness = Harness::new(chat, &cwd, move |chat| {
        models::start(chat, which, &system, |_, _| {}).map(|_| ())
    })?;
    harness.send(prompt, |event| {
        printer.print(event);
        ControlFlow::Continue(())
    })?;
    Ok((harness.stats(), loading))
}

/// Bytes as whole megabytes.
fn megabytes(bytes: u64) -> usize {
    (bytes / (1 << 20)) as usize
}

/// How far along getting ready is, on standard error: a line rewritten as it
/// goes when standard error is a terminal, and one line for each stage when
/// it is not.
struct Progress {
    terminal: bool,
    stage: String,
}

impl Progress {
    fn new() -> Self {
        Self {
            terminal: io::stderr().is_terminal(),
            stage: String::new(),
        }
    }

    fn report(&mut self, stage: String, done: usize, total: usize) {
        if self.stage != stage {
            self.stage = stage;
            if !self.terminal {
                eprintln!("{}…", self.stage);
            }
        }
        if self.terminal {
            eprint!("\r\x1b[K{}… {done}/{total}", self.stage);
            if done >= total {
                eprintln!();
            }
            let _ = io::stderr().flush();
        }
    }
}

/// Prints a turn as it happens: the reply to standard output, and the
/// thought behind it and the tools it runs to standard error.
#[derive(Default)]
struct Printer {
    /// Whether anything of the reply has been printed, so that the blank
    /// lines the model opens with are left out.
    replied: bool,
    /// Whether standard error is part way through a line of thought.
    thinking: bool,
}

impl Printer {
    fn print(&mut self, event: harness::Event) {
        match event {
            harness::Event::Thought(text) => {
                let text = if self.thinking {
                    text
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                    self.thinking = true;
                }
            }
            harness::Event::Text(text) => {
                self.end_thought();
                let text = if self.replied {
                    text
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    print!("{text}");
                    let _ = io::stdout().flush();
                    self.replied = true;
                }
            }
            harness::Event::Call { name, detail } => {
                self.end_thought();
                match name {
                    "bash" => eprintln!("$ {detail}"),
                    name => eprintln!("{name}({detail})"),
                }
            }
            harness::Event::Output(output) => eprintln!("{output}"),
            harness::Event::Cut => {
                self.end_thought();
                eprintln!("[stopped short: the context window is full]");
            }
            harness::Event::Compacting => {
                self.end_thought();
                eprintln!("[compacting the conversation]");
            }
            harness::Event::Note(text) => self.print(harness::Event::Thought(text)),
            harness::Event::Compacted { before, after } => {
                self.end_thought();
                eprintln!("[compacted from {before} to {after} tokens]");
            }
        }
    }

    /// Ends the line of thought being printed, if there is one.
    fn end_thought(&mut self) {
        if self.thinking {
            eprintln!();
            self.thinking = false;
        }
    }
}

impl Drop for Printer {
    /// Ends the reply with a newline, as a command's output should end.
    fn drop(&mut self) {
        self.end_thought();
        if self.replied {
            println!();
        }
    }
}
