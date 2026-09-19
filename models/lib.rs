//! Language models: Bonsai, its tokenizer and weights, and a chat around
//! it.

pub mod bonsai;
mod chat;
pub mod gguf;
mod sampler;
mod tokenizer;

pub use chat::{Chat, Chunk, Stats, Tally, ToolCall};
pub use gguf::Gguf;
pub use dwim_gpu::{Device, Tensor, ternary};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A model that predicts the next token, keeping the sequence so far in
/// its own state.
/// Most tokens a draft verified at once has.
pub const VERIFY: usize = 8;

pub trait LanguageModel {
    /// The model's state after the first `len` positions, as bytes for
    /// [`restore`](Self::restore) to take up from, if the model can give
    /// it.
    fn save(&self, _len: usize) -> Option<Vec<u8>> {
        None
    }

    /// Puts the model in a state [`save`](Self::save) gave, and returns
    /// how many positions it holds.
    fn restore(&mut self, _state: &[u8]) -> Result<usize> {
        Err("this model cannot restore a state".into())
    }

    /// Runs a draft of up to [`VERIFY`] tokens, the first at position
    /// `pos`, and returns the logits after each of them in turn, without
    /// committing the model's state to them: [`commit`](Self::commit) then
    /// says how many of them stand. A model that cannot returns nothing.
    fn verify(&mut self, _tokens: &[u32], _pos: usize) -> Option<Vec<f32>> {
        None
    }

    /// Commits the state to the first `count` tokens of the last draft
    /// verified, as if they had been run through [`forward`](Self::forward).
    fn commit(&mut self, _count: usize) {}

    /// Runs `tokens`, the first at position `pos`, through the model, and
    /// returns the logits for the token that follows the last of them.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32>;

    /// Longest sequence the state has room for.
    fn max_len(&self) -> usize;
}

impl<M: LanguageModel + ?Sized> LanguageModel for Box<M> {
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        (**self).save(len)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        (**self).restore(state)
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Option<Vec<f32>> {
        (**self).verify(tokens, pos)
    }

    fn commit(&mut self, count: usize) {
        (**self).commit(count)
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).forward(tokens, pos)
    }

    fn max_len(&self) -> usize {
        (**self).max_len()
    }
}

/// The cosines and sines of the angles rotary position embeddings rotate
/// by, laid out as [`Device::rope`] expects, for positions up to `max_len`
/// and `rot_dim` rotated elements per head. Each pair of elements rotates at
/// its own frequency.
pub fn rope_table(max_len: usize, rot_dim: usize, theta: f32) -> Vec<f32> {
    let half = rot_dim / 2;
    let mut table = Vec::with_capacity(max_len * rot_dim);
    for pos in 0..max_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / rot_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            table.extend([cos, sin]);
        }
    }
    table
}
