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
    /// What it says when asked for a plain answer, if not the next script.
    answer: Option<Vec<u32>>,
    /// What is left of the reply being said.
    saying: VecDeque<u32>,
    /// Whether the last batch ended with the template's opening of a
    /// reply, and with the thought closed, so the next sampled token
    /// starts one.
    pending: Option<bool>,
    vocab: usize,
    /// How the chat template opens a reply, with the thought open or with
    /// an empty one when the reply is to be made without thinking.
    opening: Vec<u32>,
    unthinking: Vec<u32>,
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
            answer: None,
            saying: VecDeque::new(),
            pending: None,
            vocab: tokenizer.vocab_size(),
            opening: tokenizer
                .encode_with_special("<|im_start|>assistant\n<think>\n")
                .unwrap(),
            unthinking: tokenizer
                .encode_with_special("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                .unwrap(),
            end: tokenizer.special("<|im_end|>").unwrap(),
            max_len,
        }
    }

    /// Has the model say `text` whenever it is asked for a plain answer,
    /// rather than its next script.
    pub fn answers(mut self, tokenizer: &Tokenizer, text: &str) -> Self {
        self.answer = Some(tokenizer.encode(text).unwrap());
        self
    }

    /// What the next reply says: the answer, if there is one and the reply
    /// is to be made without thinking, and otherwise the next script,
    /// which it takes.
    fn script(&mut self, unthinking: bool) -> Vec<u32> {
        match (&self.answer, unthinking) {
            (Some(answer), true) => answer.clone(),
            _ => self.scripts.pop_front().unwrap_or_default(),
        }
    }

    /// The first token of what the next reply says, leaving it in place.
    fn peek(&self, unthinking: bool) -> Option<u32> {
        match (&self.answer, unthinking) {
            (Some(answer), true) => answer.first().copied(),
            _ => self.scripts.front()?.first().copied(),
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
        // A reply's prompt ends with the template's opening of one, and
        // the reply is then sampled a token at a time. The script is taken
        // up only once the sampling starts: a prompt read in batches may
        // end a batch at an opening it goes on past.
        if tokens.len() > 1 {
            self.saying.clear();
            self.pending = if self.fed.ends_with(&self.unthinking) {
                Some(true)
            } else if self.fed.ends_with(&self.opening) {
                Some(false)
            } else {
                None
            };
        } else if let Some(unthinking) = self.pending.take() {
            self.saying = self.script(unthinking).into();
            // The batch's logits already said the first token.
            self.saying.pop_front();
        }
        let next = match self.pending {
            Some(unthinking) => self.peek(unthinking).unwrap_or(self.end),
            None => self.saying.pop_front().unwrap_or(self.end),
        };
        let mut logits = vec![0.0; self.vocab];
        logits[next as usize] = 1.0;
        logits
    }

    fn reset(&mut self) {
        self.fed.clear();
        self.saying.clear();
        self.pending = None;
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}
