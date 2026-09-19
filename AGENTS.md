# `dwim`

A coding agent harness in Rust.

## Layout

* `cli/`: the `dwim` command: arguments, fetching the model, and the terminal UI.
* `models/`: the language model: `gguf` reads the file the weights ship in, `tokenizer` builds the byte-level BPE tokenizer from the vocabulary in it, `bonsai` is the transformer as operations on a device, and `chat` is the conversation around it.
* `harness/`: the agent around the model: runs the tools it calls and feeds the results back until it replies with text alone. The system prompt declares the tools in the form the model's chat template uses: their JSON signatures, and calls as `<function=…>` blocks; it holds only what is the same from one run to the next (plus `AGENTS.md`), so the model's state after it can be saved and reused, and the working directory, its files, the branch, and the date go ahead of the first message. Tool results are encoded as plain text, and only the chat's own framing as special tokens, so a command's output cannot end a turn. The harness keeps a transcript of the turns and compacts the conversation before a message or a result would take the context past four fifths of the window: it has the model write a note to go on from, starts over from the system prompt (from its saved state), and reads back where it is, the note, the user's earlier messages, and the most recent exchanges without their thoughts. It reserves room past every reply for that note, so a reply that runs out of room is stopped short and goes on after a compaction; `/compact` in the shell does it by hand.
* `tools/`: the tools, one file each. `bash` runs a shell command: its result shows standard output and standard error apart, each cut to a preview of its start and end when long, and always ends with the exit code; a stream that outgrew its preview is kept whole (up to 4 MiB) in a private directory under the temporary directory, `dwim-<pid>-<n>/`, which the result names along with the line to `read` it from. The directory goes when the tool is dropped, and a new one removes those left by processes that are no longer running. `read` gives the model a file a page of up to 200 numbered lines, or 16 KB, at a time, each line cut at 300 bytes, and says where the next page starts, so it never has to guess a range.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference, `vulkan` with hand-written WGSL compute kernels compiled to SPIR-V at build time, and `metal` with hand-written Metal Shading Language kernels compiled when the device opens. `ternary` is the 1.75-bit weight format. `Gpu` is Metal on Apple platforms and Vulkan elsewhere. The Vulkan ternary matmul comes in three kernels picked by the batch: for a single token, eight rows per workgroup with the trits of each byte looked up in a 4 KB table of half floats; eight tokens at a time for batches of up to 31, from the activations packed as half floats but with the products summed in single precision; and for batches of 32 tokens or more a tiled product over the activations packed as half floats, which unpacks each block once into workgroup memory for 64 rows and 64 tokens; its attention splits positions into chunks of 128 across workgroups and merges them, reading each key/value head once for the query heads that share it; and it submits command buffers every 128 kernels from a ring so the GPU runs while the CPU records.
* `scripts/`: `swebench.py`, which runs the agent over SWE-bench Lite instances and writes the patches it makes as predictions; and `release`, which sets the version, commits it, and tags it for `.github/workflows/release.yml` to build.
* `man/`: `dwim.1`, the manual page: the options, the shell's commands and keys, and the files it reads.
* `third_party/`: reference implementations studied for design, not built.

## Model

The one model is `bonsai-2-27b`: PrismML's Ternary Bonsai 2 27B, Qwen3.8-27B with its matrices made ternary (`{-1, 0, 1}` with an f16 scale per 128 weights, 5.95 GB as shipped in the `PTQ1_0` GGUF packing). It is a hybrid: 64 layers of which every fourth is full attention (24 query heads, 4 key/value heads, 256 wide, the first 64 elements rotated by RoPE) and the rest Gated DeltaNet linear attention (16 key heads and 48 value heads of 128, a causal convolution of 4 taps, and a 128×128 recurrent state per value head). The matrices are stored in a rotated basis: activations are rotated by a blockwise (1024) Walsh-Hadamard transform with fixed signs before every ternary matmul, and the embedding table's rows are rotated back after lookup. The file stores the linear-attention value heads in llama.cpp's tiled head order; `bonsai.rs` puts them back into the checkpoint's grouped order when loading. The whole model runs on the device; only the embedding lookup is on the CPU.

The model thinks by default (its template opens `<think>` for it), calls tools as `<tool_call><function=bash>…`, and samples at temperature 1.0, top-k 20, top-p 0.95 as the file recommends. Its own chat template puts the tools at the top of the system prompt, which the harness follows.

## Building

`cargo build` needs nothing beyond Rust: the Vulkan kernels are compiled by `naga` in `gpu/build.rs`, and Metal compiles its own at runtime. Running on the GPU needs Metal on macOS, and a Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. The weights are fetched into `dwim/models/` under the user's cache directory (`~/Library/Caches` on macOS, `~/.cache` elsewhere) on first use (a resumable 6 GB download) and need about 6 GB of device memory plus 64 KB per token of context for the key/value caches.

## Verifying

* `cargo test -p dwim-gpu` checks every Vulkan and Metal kernel against the CPU reference, and skips a backend whose GPU is not available.
* `cargo test -p dwim-models` checks the GGUF reader and runs a tiny random model on the CPU and the GPU and compares their logits. `cargo test -p dwim-models real_model -- --ignored --nocapture` completes a prompt with the real model if it is in the cache: it must say Paris, as PrismML's llama.cpp does, and it reports prompt and decode speeds.
* Exercise `./target/debug/dwim` in a terminal (or tmux) and ask the model something before calling a change done.
* `CONTRIBUTING.md` has the longer version of all this, and how to run SWE-bench Lite.

## Coding Style

* Never use `pub(crate)` in source code. A definition is either `pub` or not.
* Use one `use` per crate with nested paths (`use std::{path::Path, sync::Arc};`, not a `use` per item), grouped as `std`, external crates, then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
