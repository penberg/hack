//! A model and a tokenizer for tests, here and in the crates around this
//! one: the tokenizer over a tiny vocabulary, and a model that says what
//! it is told to.

use std::collections::VecDeque;

use crate::{LanguageModel, Tokenizer};

/// A model that says what it is told to: it records every token fed to it,
/// and each time the chat template opens a reply for it, it says the next
/// of its scripts, then ends the reply. Once its scripts run out, it ends
/// every reply at once.
pub struct Scripted {
    /// Every token fed so far, in order.
    pub fed: Vec<u32>,
    scripts: VecDeque<Vec<u32>>,
    /// What is left of the reply being said.
    saying: VecDeque<u32>,
    vocab: usize,
    think: u32,
    think_end: u32,
    newline: u32,
    end: u32,
    max_len: usize,
}

impl Scripted {
    /// A model over `tokenizer`'s vocabulary that says `scripts` in turn,
    /// each given as text with the chat format's tags in it, and has room
    /// for `max_len` tokens.
    pub fn new(tokenizer: &Tokenizer, scripts: &[&str], max_len: usize) -> Self {
        Self {
            fed: Vec::new(),
            scripts: scripts
                .iter()
                .map(|script| tokenizer.encode_with_special(script).unwrap())
                .collect(),
            saying: VecDeque::new(),
            vocab: tokenizer.vocab_size(),
            think: tokenizer.special("<think>").unwrap(),
            think_end: tokenizer.special("</think>").unwrap(),
            newline: tokenizer.encode("\n").unwrap()[0],
            end: tokenizer.special("<|im_end|>").unwrap(),
            max_len,
        }
    }
}

impl LanguageModel for Scripted {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert_eq!(
            pos,
            self.fed.len(),
            "the conversation must only append tokens"
        );
        assert!(
            pos + tokens.len() <= self.max_len,
            "tokens past the end of the context"
        );
        self.fed.extend(tokens);
        // The reply's prompt ends with `<think>\n`, or with an empty
        // thought when the reply is to be made without one; a sampled token
        // comes alone.
        let opened = tokens.len() > 1 && tokens.ends_with(&[self.think, self.newline]);
        let no_thought =
            tokens.len() > 1 && tokens.ends_with(&[self.think_end, self.newline, self.newline]);
        if opened || no_thought {
            self.saying = self.scripts.pop_front().unwrap_or_default().into();
        } else if tokens.len() != 1 {
            self.saying.clear();
        }
        let next = self.saying.pop_front().unwrap_or(self.end);
        let mut logits = vec![0.0; self.vocab];
        logits[next as usize] = 1.0;
        logits
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}
