//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The tools are `bash`, which runs a shell command,
//! and `read`, which reads a file a page at a time; they live in
//! `dwim_tools`. The system prompt pushes the model to use them rather
//! than answer from memory or ask the user for a command. It also tells
//! the model about the project it works in, since a model won't always go
//! looking on its own.

use std::{
    error::Error,
    fs,
    ops::ControlFlow,
    path::Path,
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dwim_models::{Chat, Chunk, LanguageModel, ToolCall};
use dwim_tools::Tools;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 4000;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// How the agent should behave: the start of the system prompt.
const INSTRUCTIONS: &str = r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there and a read tool that reads files, and you may use them at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- To read a file, use read, not cat: it gives you a page of 200 numbered lines and says where the next page starts. Read the next page when you need more, and start from a line to read the middle of a file.
- If a command fails, read the error and try another way. A command's result ends with its exit code, and what it printed to standard error comes after a `[stderr]` line.
- When a command prints more than fits, the result shows the start and the end of its output and names a file that holds all of it, with the line to read it from: use read on that file instead of running the command again.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`."#;

/// How the tools are called, declared as Bonsai's chat template puts it,
/// after their signatures.
const CALLING: &str = r#"If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// The system prompt for an agent working in `dir`: the tools, then how to
/// behave, where it is and what the project looks like, and the project's
/// own instructions from its `AGENTS.md` if it has one.
pub fn system_prompt(dir: &Path) -> String {
    let mut prompt = format!("{}\n\n{INSTRUCTIONS}\n\n{}", tools(), environment(dir));
    if let Ok(instructions) = fs::read_to_string(dir.join("AGENTS.md")) {
        let instructions = truncate(instructions.trim(), MAX_INSTRUCTIONS);
        prompt.push_str(&format!("\n\n# Project instructions\n\nFrom AGENTS.md:\n\n{instructions}"));
    }
    prompt
}

/// The tools, declared as Bonsai's chat template puts them: the start of
/// the system prompt, before what the user's system prompt says.
fn tools() -> String {
    format!(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}\n</tools>\n\n{CALLING}",
        dwim_tools::signatures()
    )
}

/// Where the agent is: the working directory and the files in it, whether
/// it is a git repository, the platform, and the date.
fn environment(dir: &Path) -> String {
    let branch = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    let git = match branch {
        Some(branch) if !branch.is_empty() => format!("yes, on branch {branch}"),
        Some(_) => "yes".to_string(),
        None => "no".to_string(),
    };
    let platform = match std::env::consts::OS {
        "macos" => "macOS",
        "linux" => "Linux",
        os => os,
    };
    let days = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs() / 86400);
    format!(
        "# Environment\n\nWorking directory: {}\nFiles: {}\nGit repository: {git}\nPlatform: {platform}\nDate: {}",
        dir.display(),
        files(dir),
        date(days as i64),
    )
}

/// The files and directories in `dir`, directories with a trailing slash,
/// leaving out hidden ones.
fn files(dir: &Path) -> String {
    let Ok(entries) = fs::read_dir(dir) else {
        return String::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            match entry.file_type().ok()? {
                _ if name.starts_with('.') => None,
                kind if kind.is_dir() => Some(format!("{name}/")),
                _ => Some(name),
            }
        })
        .collect();
    names.sort();
    if names.len() > MAX_FILES {
        names.truncate(MAX_FILES);
        names.push("…".to_string());
    }
    names.join(" ")
}

/// The date `days` after 1970-01-01, as year-month-day.
fn date(days: i64) -> String {
    // Howard Hinnant's civil_from_days: count in 400-year eras of 146097
    // days, from a year that starts in March so that leap days come last.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let day_of_era = z.rem_euclid(146097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Cuts `text` down to at most `max` bytes, on a character boundary, marking
/// the cut with an ellipsis.
fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
    format!("{}…", &text[..end])
}

/// What happens during a turn, as it happens.
pub enum Event<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Text of the reply.
    Text(&'a str),
    /// A tool is about to run: its name and how it was called.
    Call { name: &'a str, detail: &'a str },
    /// What the tool returned, as the model sees it.
    Output(&'a str),
}

/// The loop around a [`Chat`] that runs the tools the model calls and feeds
/// the results back, until the model replies with text alone.
pub struct Harness<M: LanguageModel> {
    chat: Chat<M>,
    tools: Tools,
    calls: usize,
    tool_seconds: f64,
}

/// Where a conversation's time has gone: the model's, by what it was
/// doing, and the tools'.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub model: dwim_models::Stats,
    /// Tool calls run, and the time they took.
    pub calls: usize,
    pub tool_seconds: f64,
}

impl Stats {
    /// The accounting as lines, given how long loading the model took and
    /// how long everything took: what is left over is the time outside the
    /// model and the tools.
    pub fn report(&self, loading: Duration, total: Duration) -> Vec<String> {
        let m = &self.model;
        let row = |name: &str, count: String, seconds: f64, rate: String| format!("{name:<8}{count:>13}{seconds:>8.1} s{rate:>11}");
        let tokens = |name: &str, tally: &dwim_models::Tally| row(name, format!("{} tokens", tally.tokens), tally.seconds, format!("{:.0} tok/s", tally.rate()));
        let accounted = loading.as_secs_f64() + m.cached.seconds + m.prompt.seconds + m.thought.seconds + m.answer.seconds + self.tool_seconds;
        vec![
            row("loading", String::new(), loading.as_secs_f64(), String::new()),
            row("cached", format!("{} tokens", m.cached.tokens), m.cached.seconds, String::new()),
            tokens("prompt", &m.prompt),
            tokens("thought", &m.thought),
            tokens("answer", &m.answer),
            row("drafted", format!("{} tokens", m.drafted), 0.0, format!("{} taken", m.accepted)),
            row("tools", format!("{} calls", self.calls), self.tool_seconds, String::new()),
            row("other", String::new(), (total.as_secs_f64() - accounted).max(0.0), String::new()),
            row("total", String::new(), total.as_secs_f64(), String::new()),
        ]
        .into_iter()
        .map(|line| line.trim_end().to_string())
        .collect()
    }
}

impl<M: LanguageModel> Harness<M> {
    pub fn new(chat: Chat<M>) -> Self {
        Self {
            chat,
            tools: Tools::new(),
            calls: 0,
            tool_seconds: 0.0,
        }
    }

    /// Where the conversation's time has gone so far.
    pub fn stats(&self) -> Stats {
        Stats {
            model: self.chat.stats(),
            calls: self.calls,
            tool_seconds: self.tool_seconds,
        }
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.chat.tokens()
    }

    /// Sends a message from the user, reporting the reply and any tool
    /// calls it makes to `on_event`. The turn ends early if `on_event`
    /// breaks.
    pub fn send(
        &mut self,
        message: &str,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let mut calls = self.chat.send(message, |chunk| on_event(event(chunk)))?;
        // The last call run, as its name and arguments.
        let mut last = None;
        while !calls.is_empty() {
            let mut outputs = Vec::new();
            for call in &calls {
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = dwim_tools::describe(&call);
                        if on_event(Event::Call {
                            name: &call.name,
                            detail: &detail,
                        })
                        .is_break()
                        {
                            return Ok(());
                        }
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if this == last {
                            REPEATED.to_string()
                        } else {
                            last = this;
                            let start = Instant::now();
                            let output = self.tools.run(&call);
                            self.calls += 1;
                            self.tool_seconds += start.elapsed().as_secs_f64();
                            output
                        }
                    }
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                if on_event(Event::Output(&output)).is_break() {
                    return Ok(());
                }
                outputs.push(output);
            }
            calls = self.chat.respond(&outputs, |chunk| on_event(event(chunk)))?;
        }
        Ok(())
    }
}

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text),
        Chunk::Text(text) => Event::Text(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_where_the_time_went() {
        let stats = Stats {
            model: dwim_models::Stats {
                cached: dwim_models::Tally { tokens: 1500, seconds: 0.5 },
                prompt: dwim_models::Tally { tokens: 1200, seconds: 15.0 },
                thought: dwim_models::Tally { tokens: 900, seconds: 30.0 },
                answer: dwim_models::Tally { tokens: 300, seconds: 10.0 },
                drafted: 200,
                accepted: 120,
            },
            calls: 2,
            tool_seconds: 0.5,
        };
        let lines = stats.report(Duration::from_secs_f64(2.5), Duration::from_secs_f64(60.0));
        assert_eq!(lines[0], "loading                   2.5 s");
        assert_eq!(lines[1], "cached    1500 tokens     0.5 s");
        assert_eq!(lines[2], "prompt    1200 tokens    15.0 s   80 tok/s");
        assert_eq!(lines[3], "thought    900 tokens    30.0 s   30 tok/s");
        assert_eq!(lines[4], "answer     300 tokens    10.0 s   30 tok/s");
        assert_eq!(lines[5], "drafted    200 tokens     0.0 s  120 taken");
        assert_eq!(lines[6], "tools         2 calls     0.5 s");
        assert_eq!(lines[7], "other                     1.5 s");
        assert_eq!(lines[8], "total                    60.0 s");
    }

    #[test]
    fn dates() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(11016), "2000-02-29");
        assert_eq!(date(19782), "2024-02-29");
        assert_eq!(date(20711), "2026-09-15");
    }

    #[test]
    fn describes_the_project() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let prompt = system_prompt(dir);
        assert!(prompt.starts_with("# Tools\n"));
        assert!(prompt.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"bash\""));
        assert!(prompt.contains("\n{\"type\": \"function\", \"function\": {\"name\": \"read\""));
        assert!(prompt.contains(CALLING));
        assert!(prompt.contains(INSTRUCTIONS));
        let files = prompt.lines().find_map(|line| line.strip_prefix("Files: ")).unwrap();
        assert!(files.split(' ').any(|file| file == "Cargo.toml"));
        assert!(files.split(' ').any(|file| file == "harness/"));
        assert!(!files.contains(".git"));
        assert!(prompt.contains("From AGENTS.md:\n\n# `dwim`"));
    }
}
