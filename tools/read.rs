//! The `read` tool: a file, a page of numbered lines at a time, with
//! where the next page starts. The page is fixed and the model is told
//! its size, so it knows what it will get before it asks and never has to
//! guess at a range: it reads the next page, or starts from a line.

use std::{
    fs::File,
    io::{self, BufRead, BufReader},
};

use serde_json::Value;

use crate::string;

/// Most lines in a page.
pub const PAGE: usize = 200;

/// Most bytes in a page: a page of long lines ends early, so that one
/// page can't take up much of the context.
pub const BYTES: usize = 16 << 10;

/// Most bytes of a line shown; a longer line is cut there, with a mark.
pub const LINE: usize = 300;

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "read", "description": "Read a file, a page of up to 200 numbered lines (or 16 KB) at a time. The result says how many lines the file has and where the next page starts.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file to read."}, "start": {"type": "integer", "description": "The line to start from, counting from 1. The first line if left out."}}, "required": ["path"]}}}"#;

/// Reads the page of the file in `arguments`, returning what the model
/// should see.
pub fn run(arguments: &Value) -> String {
    let Some(path) = string(arguments, "path") else {
        return "error: read needs a 'path' string".to_string();
    };
    let start = match arguments.get("start") {
        None => 1,
        Some(_) => match start(arguments) {
            Some(start) => start.max(1),
            None => return "error: 'start' is a line number".to_string(),
        },
    };
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return format!("error: cannot read {path}: {e}"),
    };
    if file.metadata().is_ok_and(|metadata| metadata.is_dir()) {
        return format!("error: {path} is a directory; list it with ls");
    }
    match page(BufReader::new(file), path, start) {
        Ok(text) => text,
        Err(e) => format!("error: reading {path}: {e}"),
    }
}

/// The `start` argument as a line number, written as a number or, as the
/// `<function=…>` call format writes every argument, as a string.
pub fn start(arguments: &Value) -> Option<u64> {
    match arguments.get("start")? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// The page of `file` from line `start`: its lines, each after its number
/// and a tab, up to [`PAGE`] of them or [`BYTES`] in all, then a line of
/// where the page sits in the file and where the next page starts.
fn page(mut file: impl BufRead, path: &str, start: u64) -> io::Result<String> {
    let mut text = String::new();
    let mut line = Vec::new();
    // Lines read so far, shown in the page, and cut for length.
    let (mut lines, mut shown, mut cut) = (0, 0, 0);
    loop {
        line.clear();
        if file.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        lines += 1;
        if lines < start {
            continue;
        }
        let mut content = line.strip_suffix(b"\n").unwrap_or(&line);
        content = content.strip_suffix(b"\r").unwrap_or(content);
        let mut content = String::from_utf8_lossy(content).into_owned();
        let long = content.len() > LINE;
        if long {
            let end = (0..=LINE)
                .rev()
                .find(|&i| content.is_char_boundary(i))
                .unwrap_or(0);
            content.truncate(end);
            content.push('…');
        }
        let entry = format!("{lines}\t{content}\n");
        if shown == PAGE || (shown > 0 && text.len() + entry.len() > BYTES) {
            // The file goes on past the page: count the rest.
            lines += count(&mut file)?;
            break;
        }
        text.push_str(&entry);
        shown += 1;
        cut += u64::from(long);
    }
    let end = start + shown as u64 - 1;
    let mut note = if lines == 0 {
        format!("[{path} is empty]")
    } else if shown == 0 {
        format!(
            "[{path} has {lines} {}; nothing from line {start}]",
            plural(lines, "line")
        )
    } else if end < lines {
        format!(
            "[lines {start}–{end} of {lines} in {path}; next: read {path} from line {}]",
            end + 1
        )
    } else {
        format!("[lines {start}–{end} of {lines} in {path}; end of file]")
    };
    if cut > 0 {
        note.insert_str(
            note.len() - 1,
            &format!(
                "; {cut} {} longer than {LINE} bytes cut",
                plural(cut, "line")
            ),
        );
    }
    text.push_str(&note);
    Ok(text)
}

/// Number of lines left in `file`, reading it to the end. A last line
/// without a newline counts.
fn count(file: &mut impl BufRead) -> io::Result<u64> {
    let mut lines = 0;
    let mut last = b'\n';
    loop {
        let buf = file.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        lines += buf.iter().filter(|&&b| b == b'\n').count() as u64;
        last = buf[buf.len() - 1];
        let n = buf.len();
        file.consume(n);
    }
    Ok(lines + u64::from(last != b'\n'))
}

/// `word`, or its plural if `n` is not one.
fn plural(n: u64, word: &str) -> String {
    if n == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    /// A file with `text` in a directory of its own, and its path.
    fn file(name: &str, text: &str) -> (PathBuf, String) {
        static FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-read-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        let text = path.to_str().unwrap().to_string();
        (dir, text)
    }

    fn read(path: &str, start: Option<u64>) -> String {
        let mut arguments = serde_json::json!({ "path": path });
        if let Some(start) = start {
            arguments["start"] = start.into();
        }
        run(&arguments)
    }

    #[test]
    fn pages_through_a_file() {
        let text: String = (1..=450).map(|i| format!("line {i}\n")).collect();
        let (dir, path) = file("long.txt", &text);
        let first = read(&path, None);
        assert!(first.starts_with("1\tline 1\n2\tline 2\n"), "{first}");
        assert!(first.contains("\n200\tline 200\n"), "{first}");
        assert!(!first.contains("line 201\n"));
        assert!(
            first.ends_with(&format!(
                "[lines 1–200 of 450 in {path}; next: read {path} from line 201]"
            )),
            "{first}"
        );
        assert_eq!(first.lines().count(), PAGE + 1);

        let second = read(&path, Some(201));
        assert!(second.starts_with("201\tline 201\n"), "{second}");
        assert!(
            second.ends_with(&format!(
                "[lines 201–400 of 450 in {path}; next: read {path} from line 401]"
            )),
            "{second}"
        );

        let last = read(&path, Some(401));
        assert!(last.starts_with("401\tline 401\n"));
        assert!(last.contains("\n450\tline 450\n"));
        assert!(
            last.ends_with(&format!("[lines 401–450 of 450 in {path}; end of file]")),
            "{last}"
        );

        // Following the cursor covers the file exactly once.
        let mut seen = Vec::new();
        for page in [&first, &second, &last] {
            seen.extend(
                page.lines()
                    .filter(|line| !line.starts_with('['))
                    .map(|line| line.split('\t').next().unwrap().parse::<u64>().unwrap()),
            );
        }
        assert_eq!(seen, (1..=450).collect::<Vec<_>>());

        assert_eq!(
            read(&path, Some(451)),
            format!("[{path} has 450 lines; nothing from line 451]")
        );
        assert_eq!(
            read(&path, Some(0)).lines().next().unwrap(),
            "1\tline 1",
            "a start below 1 means the first line"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_the_edges() {
        let (dir, path) = file("empty.txt", "");
        assert_eq!(read(&path, None), format!("[{path} is empty]"));
        let one = dir.join("one.txt");
        fs::write(&one, "just this").unwrap();
        let one = one.to_str().unwrap();
        assert_eq!(
            read(one, None),
            format!("1\tjust this\n[lines 1–1 of 1 in {one}; end of file]")
        );
        let crlf = dir.join("crlf.txt");
        fs::write(&crlf, "a\r\nb\r\n").unwrap();
        let crlf = crlf.to_str().unwrap();
        assert_eq!(
            read(crlf, None),
            format!("1\ta\n2\tb\n[lines 1–2 of 2 in {crlf}; end of file]")
        );
        let bytes = dir.join("bytes.bin");
        fs::write(
            &bytes,
            [b"x\xc3\xa4\n".as_slice(), &[0xff, 0xfe, b'y']].concat(),
        )
        .unwrap();
        let bytes = bytes.to_str().unwrap();
        assert_eq!(
            read(bytes, None),
            format!("1\tx\u{e4}\n2\t\u{FFFD}\u{FFFD}y\n[lines 1–2 of 2 in {bytes}; end of file]")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cuts_long_lines_between_characters() {
        let long = "ä".repeat(400);
        let (dir, path) = file("wide.txt", &format!("short\n{long}\n"));
        let page = read(&path, None);
        let second = page.lines().nth(1).unwrap();
        assert!(second.starts_with("2\t"));
        assert!(second.ends_with('…'));
        assert!(!second.contains('\u{FFFD}'));
        assert!(second.len() <= "2\t".len() + LINE + '…'.len_utf8());
        assert!(
            page.ends_with(&format!(
                "[lines 1–2 of 2 in {path}; end of file; 1 line longer than {LINE} bytes cut]"
            )),
            "{page}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ends_a_page_of_long_lines_early() {
        // 100 lines of 299 bytes: three pages by bytes, half of one by lines.
        let text: String = (1..=100).map(|i| format!("{i:<299}\n")).collect();
        let (dir, path) = file("wide.txt", &text);
        let mut start = 1;
        let mut seen = Vec::new();
        while start <= 100 {
            let page = read(&path, Some(start));
            let body: usize = page
                .lines()
                .filter(|line| !line.starts_with('['))
                .map(|line| line.len() + 1)
                .sum();
            assert!(body <= BYTES, "{body} bytes of lines");
            let shown: Vec<u64> = page
                .lines()
                .filter(|line| !line.starts_with('['))
                .map(|line| line.split('\t').next().unwrap().parse().unwrap())
                .collect();
            assert!(!shown.is_empty());
            let end = *shown.last().unwrap();
            if end < 100 {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–{end} of 100 in {path}; next: read {path} from line {}]",
                        end + 1
                    )),
                    "{page}"
                );
            } else {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–100 of 100 in {path}; end of file]"
                    )),
                    "{page}"
                );
            }
            seen.extend(shown);
            start = end + 1;
        }
        assert_eq!(seen, (1..=100).collect::<Vec<_>>());
        assert!(seen.len() > 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(
            read("/nonexistent/file", None).starts_with("error: cannot read /nonexistent/file: ")
        );
        assert!(read("/", None).starts_with("error: / is a directory"));
        assert_eq!(
            run(&serde_json::json!({ "path": "Cargo.toml", "start": "abc" })),
            "error: 'start' is a line number"
        );
        assert!(
            run(&serde_json::json!({ "path": "Cargo.toml", "start": "1" }))
                .starts_with("1\t[package]")
        );
    }
}
