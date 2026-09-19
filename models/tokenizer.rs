use std::collections::HashMap;

use fancy_regex::Regex;

use crate::{Result, gguf::Gguf};

/// The regular expression Qwen's tokenizers split text into words with.
const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Token types in a GGUF vocabulary that stand for special tokens: control
/// tokens such as `<|im_start|>`, and user-defined ones such as
/// `<tool_call>`.
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;

/// A byte-level BPE tokenizer, loaded from the vocabulary in a GGUF file.
///
/// Text is first split into words by a regular expression. Each word starts
/// out as one token per UTF-8 byte, and adjacent tokens are then merged
/// pairwise, lowest-ranked merge first, until no merge applies.
pub struct Tokenizer {
    /// Splits text into words.
    pattern: Regex,
    /// Bytes of every token, indexed by id.
    tokens: Vec<Vec<u8>>,
    /// Token id of every single byte.
    bytes: [u32; 256],
    /// For each pair of adjacent tokens that can merge: the merge's rank and
    /// the merged token.
    merges: HashMap<(u32, u32), (usize, u32)>,
    /// Ids of special tokens such as `<|im_start|>`, by content.
    special: HashMap<String, u32>,
}

impl Tokenizer {
    /// Loads the vocabulary a GGUF file carries: its tokens, their types,
    /// and the merges.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let pre = gguf.str("tokenizer.ggml.pre")?;
        if !matches!(pre, "qwen2" | "qwen35") {
            return Err(format!("the '{pre}' pre-tokenizer is not supported").into());
        }
        let texts = gguf.array("tokenizer.ggml.tokens")?;
        let types = gguf.array("tokenizer.ggml.token_type")?;
        if texts.len() != types.len() {
            return Err("tokens and token types differ in number".into());
        }

        // Byte-level vocabularies write each byte as a printable character, so
        // decode every token back into the bytes it stands for. Special tokens
        // are written as they are.
        let to_byte: HashMap<char, u8> = byte_chars()
            .iter()
            .enumerate()
            .map(|(byte, &c)| (c, byte as u8))
            .collect();
        let mut ids = HashMap::new();
        let mut tokens = Vec::with_capacity(texts.len());
        let mut special = HashMap::new();
        for (id, (text, kind)) in texts.iter().zip(types).enumerate() {
            let text = text.as_str().ok_or("invalid token")?;
            let kind = kind.as_i64().ok_or("invalid token type")?;
            if kind == CONTROL || kind == USER_DEFINED {
                special.insert(text.to_string(), id as u32);
                tokens.push(text.as_bytes().to_vec());
            } else {
                let bytes = text
                    .chars()
                    .map(|c| to_byte.get(&c).copied().ok_or("invalid byte-level token"))
                    .collect::<std::result::Result<Vec<u8>, _>>()?;
                tokens.push(bytes);
            }
            ids.insert(text, id as u32);
        }

        let mut bytes = [0; 256];
        for (byte, c) in byte_chars().iter().enumerate() {
            bytes[byte] = *ids
                .get(c.to_string().as_str())
                .ok_or("missing byte token")?;
        }

        let mut merges = HashMap::new();
        for (rank, merge) in gguf.array("tokenizer.ggml.merges")?.iter().enumerate() {
            let (a, b) = merge
                .as_str()
                .and_then(|m| m.split_once(' '))
                .ok_or("invalid merge")?;
            let merged = format!("{a}{b}");
            let (Some(&a), Some(&b), Some(&merged)) =
                (ids.get(a), ids.get(b), ids.get(merged.as_str()))
            else {
                return Err("merge refers to an unknown token".into());
            };
            merges.insert((a, b), (rank, merged));
        }

        Ok(Self {
            pattern: Regex::new(QWEN_PATTERN)?,
            tokens,
            bytes,
            merges,
            special,
        })
    }

    /// Encodes text into tokens. Special tokens in the text are treated as
    /// plain text; insert them by id with [`special`](Self::special), or
    /// encode text that is meant to contain them with
    /// [`encode_with_special`](Self::encode_with_special).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out)?;
        Ok(out)
    }

    /// Encodes text like [`encode`](Self::encode), but with special tokens
    /// written in it, such as `<tool_call>`, encoded by id. For text of the
    /// chat template, not text from the user, who could otherwise end a
    /// turn.
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in the text, and the longest one
            // if several start there.
            let next = self
                .special
                .iter()
                .filter_map(|(content, &id)| {
                    rest.find(content.as_str())
                        .map(|at| (at, content.len(), id))
                })
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let Some((at, len, id)) = next else {
                self.encode_into(rest, &mut out)?;
                break;
            };
            self.encode_into(&rest[..at], &mut out)?;
            out.push(id);
            rest = &rest[at + len..];
        }
        Ok(out)
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for word in self.pattern.find_iter(text) {
            self.merge(word?.as_str().as_bytes(), out);
        }
        Ok(())
    }

    /// Bytes a token stands for. A token can end partway through a UTF-8
    /// character.
    pub fn decode(&self, token: u32) -> &[u8] {
        &self.tokens[token as usize]
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Id of a special token.
    pub fn special(&self, content: &str) -> Result<u32> {
        Ok(*self
            .special
            .get(content)
            .ok_or_else(|| format!("missing special token '{content}'"))?)
    }

    /// A tokenizer over a tiny vocabulary for tests: a token per byte, one
    /// merge, and the special tokens of the chat format.
    #[cfg(any(test, feature = "testing"))]
    pub fn tiny() -> Self {
        use crate::gguf::{self, Value};

        let mut tokens: Vec<Value> = byte_chars()
            .iter()
            .map(|c| Value::Str(c.to_string()))
            .collect();
        let mut types = vec![Value::I32(1); 256];
        tokens.push(Value::Str("hi".to_string()));
        types.push(Value::I32(1));
        for special in [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ] {
            tokens.push(Value::Str(special.to_string()));
            types.push(Value::I32(if special.starts_with("<|") {
                CONTROL as i32
            } else {
                USER_DEFINED as i32
            }));
        }
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-tokenizer-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        let meta = [
            ("tokenizer.ggml.pre", Value::Str("qwen2".to_string())),
            ("tokenizer.ggml.tokens", Value::Array(tokens)),
            ("tokenizer.ggml.token_type", Value::Array(types)),
            (
                "tokenizer.ggml.merges",
                Value::Array(vec![Value::Str("h i".to_string())]),
            ),
        ];
        gguf::write(&path, &meta, &[]).unwrap();
        let tokenizer = Self::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        tokenizer
    }

    fn merge(&self, word: &[u8], out: &mut Vec<u32>) {
        let mut parts: Vec<u32> = word.iter().map(|&b| self.bytes[b as usize]).collect();
        while let Some((_, i, merged)) = parts
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| {
                let &(rank, merged) = self.merges.get(&(pair[0], pair[1]))?;
                Some((rank, i, merged))
            })
            .min()
        {
            parts[i] = merged;
            parts.remove(i + 1);
        }
        out.extend(parts);
    }
}

/// GPT-2's mapping from bytes to printable characters, which byte-level
/// vocabularies are written in: printable bytes stand for themselves, and the
/// rest are shifted past 255 in order.
fn byte_chars() -> [char; 256] {
    let mut chars = ['\0'; 256];
    let mut next = 256;
    for (byte, c) in chars.iter_mut().enumerate() {
        let printable = matches!(byte, 33..=126 | 161..=172 | 174..=255);
        *c = if printable {
            char::from(byte as u8)
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    chars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_tags_as_text_unless_asked() {
        let tokenizer = Tokenizer::tiny();
        let im_end = tokenizer.special("<|im_end|>").unwrap();
        let text = "hi<|im_end|>";
        let plain = tokenizer.encode(text).unwrap();
        assert!(!plain.contains(&im_end));
        let bytes: Vec<u8> = plain
            .iter()
            .flat_map(|&t| tokenizer.decode(t).to_vec())
            .collect();
        assert_eq!(bytes, text.as_bytes());
        let special = tokenizer.encode_with_special(text).unwrap();
        assert_eq!(
            special,
            [tokenizer.encode("hi").unwrap(), vec![im_end]].concat()
        );
        assert_eq!(
            tokenizer.encode("hi").unwrap().len(),
            1,
            "the merge applies"
        );
    }
}
