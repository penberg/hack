//! Bonsai 2 27B: PrismML's ternary compression of Qwen3.8-27B, a hybrid
//! transformer in which every fourth layer is full attention and the rest
//! are Gated DeltaNet linear attention, written as a sequence of operations
//! on a [`Device`].
//!
//! The matrices are ternary in a rotated basis: the activations a matrix
//! multiplies are first rotated by a blockwise Hadamard transform with fixed
//! signs, whose inverse was folded into the weights, and the embedding table
//! holds rotated rows that are rotated back after lookup. The small
//! recurrent-state tensors of linear attention and the norms are kept in
//! higher precision and take unrotated activations.
//!
//! The file stores the value heads of linear attention in the order
//! llama.cpp's kernels want, in which value head `h` shares key head
//! `h % n_k`; they are put back into the checkpoint's order, in which value
//! head `h` shares key head `h / (n_v / n_k)`, as the loader reads them. The
//! output projection was rotated in the checkpoint's order and is left as
//! it is.

use std::sync::Arc;

use dwim_gpu::{CONV_KERNEL, HADAMARD_BLOCK};

use crate::{Device, Gguf, LanguageModel, Result, Tensor, VERIFY, rope_table, ternary};

/// Most tokens a forward pass runs through the model at once. Running a
/// batch of tokens together reads each weight once for the whole batch,
/// rather than once per token; the activations are allocated for this many.
/// A batch is one submission to the device, which a driver may abandon as
/// hung if it runs for seconds, so batches are kept short.
pub const BATCH: usize = 64;

/// The architecture, read from the GGUF metadata.
#[derive(Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// Full attention: heads, key/value heads, their width, and how much of
    /// each head rotary position embeddings rotate.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f32,
    /// Every this many layers, the last is full attention.
    pub full_attention_interval: usize,
    /// Linear attention: key heads, value heads, and the width of both.
    pub k_heads: usize,
    pub v_heads: usize,
    pub state_dim: usize,
    pub eps: f32,
    pub vocab: usize,
    /// Positions the model was trained for.
    pub context_length: usize,
}

impl Config {
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let arch = gguf.str("general.architecture")?;
        if arch != "qwen35" {
            return Err(format!("the '{arch}' architecture is not supported").into());
        }
        let key = |name: &str| gguf.u32(&format!("{arch}.{name}"));
        let config = Self {
            hidden: key("embedding_length")? as usize,
            intermediate: key("feed_forward_length")? as usize,
            layers: key("block_count")? as usize,
            heads: key("attention.head_count")? as usize,
            kv_heads: key("attention.head_count_kv")? as usize,
            head_dim: key("attention.key_length")? as usize,
            rot_dim: key("rope.dimension_count")? as usize,
            rope_theta: gguf.f32(&format!("{arch}.rope.freq_base"))?,
            full_attention_interval: key("full_attention_interval")? as usize,
            k_heads: key("ssm.group_count")? as usize,
            v_heads: key("ssm.time_step_rank")? as usize,
            state_dim: key("ssm.state_size")? as usize,
            eps: gguf.f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
            vocab: gguf.info("token_embd.weight")?.dims[1],
            context_length: key("context_length")? as usize,
        };
        if key("ssm.conv_kernel")? as usize != CONV_KERNEL {
            return Err("the convolution kernel is not four wide".into());
        }
        if key("ssm.inner_size")? as usize != config.v_heads * config.state_dim {
            return Err("the linear attention sizes disagree".into());
        }
        if !config.v_heads.is_multiple_of(config.k_heads) {
            return Err("value heads are not a multiple of key heads".into());
        }
        for (key, want) in [
            ("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard"),
            ("prism.hadamard.axis", "input-last-dimension"),
            ("prism.hadamard.sign_mode", "explicit"),
        ] {
            let got = gguf.str(key)?;
            if got != want {
                return Err(format!("{key} is '{got}', not '{want}'").into());
            }
        }
        if gguf.u32("prism.hadamard.version")? != 1 || gguf.u32("prism.hadamard.block_size")? as usize != HADAMARD_BLOCK {
            return Err("the weights are rotated in a way the model does not implement".into());
        }
        if gguf.get("prism.hadamard.gdn_v_grouped").and_then(|v| v.as_bool()) != Some(true) {
            return Err("the linear attention output projection is not in the checkpoint's head order".into());
        }
        Ok(config)
    }

    /// Whether the layer is full attention rather than linear.
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Width of the linear attention's queries and keys per token, and of
    /// its values.
    /// Activations of a linear layer's convolution state: the tokens before
    /// the batch, a row of channels each.
    fn conv_width(&self) -> usize {
        (CONV_KERNEL - 1) * (2 * self.k_dim() + self.v_dim())
    }

    /// Activations of a linear layer's recurrent state: a matrix per value
    /// head.
    fn ssm_width(&self) -> usize {
        self.v_heads * self.state_dim * self.state_dim
    }

    fn k_dim(&self) -> usize {
        self.k_heads * self.state_dim
    }

    fn v_dim(&self) -> usize {
        self.v_heads * self.state_dim
    }
}

/// The model, with its weights on a device, and the state of a sequence.
pub struct Model<D: Device> {
    pub config: Config,
    pub device: D,
    /// The file, kept mapped for the embedding table, which is looked up
    /// on the CPU: it is most of a third of a gigabyte, and a token needs
    /// one row of it.
    gguf: Arc<Gguf>,
    layers: Vec<Layer<D>>,
    norm: D::Buffer,
    lm_head: D::Weight,
    /// The signs of the rotation, by the width of the activations they
    /// apply to.
    signs: Vec<(usize, D::Buffer)>,
    state: State<D>,
}

struct Layer<D: Device> {
    attn_norm: D::Buffer,
    mixer: Mixer<D>,
    mlp_norm: D::Buffer,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

/// What mixes the tokens in a layer.
enum Mixer<D: Device> {
    Attention {
        q: D::Weight,
        /// The output gate, projected alongside the query.
        gate: D::Weight,
        k: D::Weight,
        v: D::Weight,
        o: D::Weight,
        q_norm: D::Buffer,
        k_norm: D::Buffer,
        /// Which key/value cache is the layer's.
        cache: usize,
    },
    Linear {
        qkv: D::Weight,
        /// The output gate.
        z: D::Weight,
        /// The decay and forget projections, `alpha` then `beta`, in one
        /// bf16 matrix.
        ab: D::Weight,
        /// The convolution taps, four per channel.
        conv: D::Buffer,
        /// `a` for each value head, then `dt_bias` for each.
        decay: D::Buffer,
        norm: D::Buffer,
        out: D::Weight,
        /// Which recurrent state is the layer's.
        slot: usize,
    },
}

impl<D: Device> Model<D> {
    /// Loads the model from its GGUF file, uploading its weights to
    /// `device`, with state for sequences of up to `max_len` tokens, and
    /// reporting how many of its tensors are loaded, out of how many, as it
    /// goes.
    pub fn load(gguf: Arc<Gguf>, device: D, max_len: usize, on_progress: impl FnMut(usize, usize)) -> Result<Self> {
        let config = Config::load(&gguf)?;
        let c = &config;
        let attention_layers = (0..c.layers).filter(|&i| c.is_attention(i)).count();
        let mut l = Loader {
            device: &device,
            gguf: &gguf,
            done: 0,
            total: attention_layers * 11 + (c.layers - attention_layers) * 12 + 2,
            on_progress,
            rotated: Vec::new(),
        };
        let g = &*gguf;

        let mut layers = Vec::with_capacity(c.layers);
        let mut caches = 0;
        let mut slots = 0;
        for i in 0..c.layers {
            let p = format!("blk.{i}");
            let attn_norm = l.buffer(g.f32s(&format!("{p}.attn_norm.weight"))?);
            let mixer = if c.is_attention(i) {
                // The query projection puts each head's gate after its
                // query: the rows are split into two matrices.
                let name = format!("{p}.attn_q.weight");
                let (q, gate) = match g.ternary(&name)? {
                    Tensor::Ternary { shape, data } => {
                        let head = c.head_dim * ternary::row_bytes(shape[1]);
                        let (mut q, mut gate) = (Vec::with_capacity(data.len() / 2), Vec::with_capacity(data.len() / 2));
                        for h in data.chunks_exact(2 * head) {
                            q.extend_from_slice(&h[..head]);
                            gate.extend_from_slice(&h[head..]);
                        }
                        let shape = vec![shape[0] / 2, shape[1]];
                        (
                            Tensor::Ternary {
                                shape: shape.clone(),
                                data: q,
                            },
                            Tensor::Ternary { shape, data: gate },
                        )
                    }
                    _ => unreachable!(),
                };
                let q = l.rotated(&name, q);
                let gate = l.device.upload(gate);
                caches += 1;
                Mixer::Attention {
                    q,
                    gate,
                    k: l.weight(&format!("{p}.attn_k.weight"))?,
                    v: l.weight(&format!("{p}.attn_v.weight"))?,
                    o: l.weight(&format!("{p}.attn_output.weight"))?,
                    q_norm: l.buffer(g.f32s(&format!("{p}.attn_q_norm.weight"))?),
                    k_norm: l.buffer(g.f32s(&format!("{p}.attn_k_norm.weight"))?),
                    cache: caches - 1,
                }
            } else {
                // The value heads, and everything per value head, are put
                // back into the checkpoint's order.
                let name = format!("{p}.attn_qkv.weight");
                let qkv = match g.ternary(&name)? {
                    Tensor::Ternary { shape, mut data } => {
                        let row = ternary::row_bytes(shape[1]);
                        let v = 2 * c.k_dim() * row;
                        let untiled = untile(&data[v..], c.k_heads, c.v_heads, c.state_dim * row);
                        data[v..].copy_from_slice(&untiled);
                        Tensor::Ternary { shape, data }
                    }
                    _ => unreachable!(),
                };
                let qkv = l.rotated(&name, qkv);
                let name = format!("{p}.attn_gate.weight");
                let z = match g.ternary(&name)? {
                    Tensor::Ternary { shape, data } => Tensor::Ternary {
                        data: untile(&data, c.k_heads, c.v_heads, c.state_dim * ternary::row_bytes(shape[1])),
                        shape,
                    },
                    _ => unreachable!(),
                };
                let z = l.rotated(&name, z);
                let ab = match (g.bf16(&format!("{p}.ssm_alpha.weight"))?, g.bf16(&format!("{p}.ssm_beta.weight"))?) {
                    (Tensor::Bf16 { shape, data: alpha }, Tensor::Bf16 { data: beta, .. }) => {
                        let mut data = untile(&alpha, c.k_heads, c.v_heads, c.hidden);
                        data.extend(untile(&beta, c.k_heads, c.v_heads, c.hidden));
                        Tensor::Bf16 {
                            shape: vec![2 * shape[0], shape[1]],
                            data,
                        }
                    }
                    _ => unreachable!(),
                };
                let ab = l.device.upload(ab);
                l.tick();
                let mut conv = g.f32s(&format!("{p}.ssm_conv1d.weight"))?;
                let v = 2 * c.k_dim() * CONV_KERNEL;
                let untiled = untile(&conv[v..], c.k_heads, c.v_heads, c.state_dim * CONV_KERNEL);
                conv[v..].copy_from_slice(&untiled);
                let conv = l.buffer(conv);
                let mut decay = untile(&g.f32s(&format!("{p}.ssm_a"))?, c.k_heads, c.v_heads, 1);
                decay.extend(untile(&g.f32s(&format!("{p}.ssm_dt.bias"))?, c.k_heads, c.v_heads, 1));
                let decay = l.buffer(decay);
                let norm = l.buffer(g.f32s(&format!("{p}.ssm_norm.weight"))?);
                slots += 1;
                Mixer::Linear {
                    qkv,
                    z,
                    ab,
                    conv,
                    decay,
                    norm,
                    out: l.weight(&format!("{p}.ssm_out.weight"))?,
                    slot: slots - 1,
                }
            };
            let mlp_norm = l.buffer(g.f32s(&format!("{p}.post_attention_norm.weight"))?);
            layers.push(Layer {
                attn_norm,
                mixer,
                mlp_norm,
                gate: l.weight(&format!("{p}.ffn_gate.weight"))?,
                up: l.weight(&format!("{p}.ffn_up.weight"))?,
                down: l.weight(&format!("{p}.ffn_down.weight"))?,
            });
        }
        let norm = l.buffer(g.f32s("output_norm.weight")?);
        let lm_head = l.weight("output.weight")?;

        // The file says which weights were rotated: they must be exactly the
        // ones the forward pass rotates activations for.
        let mut listed: Vec<String> = g
            .array("prism.hadamard.weight_names")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("invalid weight name"))
            .collect::<std::result::Result<_, _>>()?;
        listed.sort();
        l.rotated.sort();
        if listed != l.rotated {
            return Err("the rotated weights are not the ones the model expects".into());
        }
        let inverse: Vec<&str> = g.array("prism.hadamard.inverse_weight_names")?.iter().filter_map(|v| v.as_str()).collect();
        if inverse != ["token_embd.weight"] {
            return Err("the embedding table is not rotated as the model expects".into());
        }
        let widths = g.array("prism.hadamard.sign_widths")?;
        let values = g.array("prism.hadamard.sign_values")?;
        let mut signs = Vec::new();
        let mut at = 0;
        for width in widths {
            let width = width.as_i64().ok_or("invalid sign width")? as usize;
            let values: Vec<f32> = values.get(at..at + width).ok_or("too few rotation signs")?.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
            if !values.iter().all(|&s| s == 1.0 || s == -1.0) {
                return Err("rotation signs must be 1 or -1".into());
            }
            let mut buf = device.alloc(width);
            device.write(&mut buf, &values);
            signs.push((width, buf));
            at += width;
        }
        for width in [c.hidden, c.v_dim(), c.heads * c.head_dim, c.intermediate] {
            if !signs.iter().any(|(w, _)| *w == width) {
                return Err(format!("no rotation signs for activations {width} wide").into());
            }
        }

        let state = State::new(&config, &device, max_len, caches, slots);
        Ok(Self {
            config,
            device,
            gguf,
            layers,
            norm,
            lm_head,
            signs,
            state,
        })
    }

    /// Runs a batch through the model. Verifying a draft, the linear layers'
    /// recurrence runs on a scratch copy of its state and their projections
    /// are kept, for `commit` to run the accepted tokens through the real
    /// state, and the logits of every token come back rather than the
    /// last's.
    fn forward_batch(&mut self, tokens: &[u32], pos: usize, verify: bool) -> Vec<f32> {
        let c = &self.config;
        let d = &self.device;
        let s = &mut self.state;
        let n = tokens.len();
        let eps = c.eps;
        let kv_dim = c.kv_heads * c.head_dim;
        s.batch(d, n);

        // Embeddings are looked up on the CPU and rotated back on the device.
        let table = self.gguf.bytes("token_embd.weight").expect("embeddings");
        let row = ternary::row_bytes(c.hidden);
        let mut x = vec![0.0; n * c.hidden];
        for (x, &token) in x.chunks_exact_mut(c.hidden).zip(tokens) {
            ternary::dequantize_row(&table[token as usize * row..][..row], x);
        }
        d.write(&mut s.x, &x);
        // The signs of the rotation for activations of each width.
        let signs = |width: usize| &self.signs.iter().find(|(w, _)| *w == width).expect("checked at load").1;
        d.hadamard(&mut s.x, signs(c.hidden), true);

        for layer in &self.layers {
            // The token mixer, with its result added back into the
            // residual stream. It sees the normalized activations rotated,
            // and the recurrent-state projections see them as they are.
            d.norm_rotate(&mut s.xh, &s.x, &layer.attn_norm, signs(c.hidden), eps);
            match &layer.mixer {
                Mixer::Attention { q, gate, k, v, o, q_norm, k_norm, cache } => {
                    let q_dim = c.heads * c.head_dim;
                    d.resize(&mut s.q, n * q_dim);
                    d.resize(&mut s.gate, n * q_dim);
                    d.resize(&mut s.k, n * kv_dim);
                    d.resize(&mut s.v, n * kv_dim);
                    d.resize(&mut s.att, n * q_dim);
                    d.matmul(&mut s.q, q, &s.xh);
                    d.matmul(&mut s.gate, gate, &s.xh);
                    d.matmul(&mut s.k, k, &s.xh);
                    d.matmul(&mut s.v, v, &s.xh);
                    d.rmsnorm(&mut s.q, q_norm, eps);
                    d.rmsnorm(&mut s.k, k_norm, eps);
                    d.rope(&mut s.q, &s.rope, pos, c.heads, c.head_dim, c.rot_dim);
                    d.rope(&mut s.k, &s.rope, pos, c.kv_heads, c.head_dim, c.rot_dim);
                    d.store(&mut s.k_cache[*cache], pos * kv_dim, &s.k);
                    d.store(&mut s.v_cache[*cache], pos * kv_dim, &s.v);
                    d.attention(&mut s.att, &s.q, &s.k_cache[*cache], &s.v_cache[*cache], pos, c.heads, c.head_dim, c.kv_heads);
                    d.sigmoid_mul(&mut s.att, &s.gate);
                    d.hadamard(&mut s.att, signs(q_dim), false);
                    d.matmul(&mut s.xb, o, &s.att);
                }
                Mixer::Linear { qkv, z, ab, conv, decay, norm, out, slot } => {
                    let (k_dim, v_dim) = (c.k_dim(), c.v_dim());
                    d.resize(&mut s.q, n * k_dim);
                    d.resize(&mut s.k, n * k_dim);
                    d.resize(&mut s.v, n * v_dim);
                    d.resize(&mut s.att, n * v_dim);
                    d.matmul(&mut s.qkv, qkv, &s.xh);
                    d.matmul(&mut s.z, z, &s.xh);
                    d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden);
                    d.rmsnorm(&mut s.xb, &layer.attn_norm, eps);
                    d.matmul(&mut s.ab, ab, &s.xb);
                    if verify {
                        d.copy(&mut s.stash_qkv[*slot], 0, &s.qkv, 0, n * (2 * k_dim + v_dim));
                        d.copy(&mut s.stash_ab[*slot], 0, &s.ab, 0, n * 2 * c.v_heads);
                    }
                    // The convolution reads the tokens before the batch from
                    // one state buffer and leaves the ones after it in the
                    // other.
                    let [a, b] = &mut s.conv_state[*slot];
                    let (state, state_out) = if s.parity { (&*a, b) } else { (&*b, a) };
                    d.conv(&mut s.q, &mut s.k, &mut s.v, state_out, &s.qkv, state, conv);
                    d.l2norm(&mut s.q, c.state_dim, eps);
                    d.l2norm(&mut s.k, c.state_dim, eps);
                    let ssm = if verify {
                        d.copy(&mut s.scratch_state, 0, &s.ssm_state[*slot], 0, c.ssm_width());
                        &mut s.scratch_state
                    } else {
                        &mut s.ssm_state[*slot]
                    };
                    d.delta_net(&mut s.att, &s.q, &s.k, &s.v, &s.ab, decay, ssm, c.k_heads, c.v_heads, c.state_dim);
                    d.rmsnorm(&mut s.att, norm, eps);
                    d.silu_mul(&mut s.z, &s.att);
                    d.hadamard(&mut s.z, signs(v_dim), false);
                    d.matmul(&mut s.xb, out, &s.z);
                }
            }
            d.add(&mut s.x, &s.xb);

            // Feed-forward network, likewise added back.
            d.norm_rotate(&mut s.xh, &s.x, &layer.mlp_norm, signs(c.hidden), eps);
            d.matmul(&mut s.up, &layer.gate, &s.xh);
            d.matmul(&mut s.gate_ffn, &layer.up, &s.xh);
            d.silu_mul(&mut s.up, &s.gate_ffn);
            d.hadamard(&mut s.up, signs(c.intermediate), false);
            d.matmul(&mut s.xb, &layer.down, &s.up);
            d.add(&mut s.x, &s.xb);
        }
        s.parity = !s.parity;

        if verify {
            // Every token's logits.
            s.verified = n;
            d.norm_rotate(&mut s.xh, &s.x, &self.norm, signs(c.hidden), eps);
            d.resize(&mut s.verify_logits, n * c.vocab);
            d.matmul(&mut s.verify_logits, &self.lm_head, &s.xh);
            return d.read(&s.verify_logits);
        }
        // Only the last token's logits are wanted.
        d.resize(&mut s.xb, c.hidden);
        d.resize(&mut s.xh, c.hidden);
        d.copy(&mut s.xb, 0, &s.x, (n - 1) * c.hidden, c.hidden);
        d.norm_rotate(&mut s.xh, &s.xb, &self.norm, signs(c.hidden), eps);
        d.matmul(&mut s.logits, &self.lm_head, &s.xh);
        d.read(&s.logits)
    }
}

/// Uploads the weights, counting them off.
struct Loader<'a, D: Device, F: FnMut(usize, usize)> {
    device: &'a D,
    gguf: &'a Gguf,
    done: usize,
    total: usize,
    on_progress: F,
    /// Every weight that takes rotated activations.
    rotated: Vec<String>,
}

impl<D: Device, F: FnMut(usize, usize)> Loader<'_, D, F> {
    fn tick(&mut self) {
        self.done += 1;
        (self.on_progress)(self.done, self.total);
    }

    /// Uploads a ternary matrix that takes rotated activations.
    fn weight(&mut self, name: &str) -> Result<D::Weight> {
        let tensor = self.gguf.ternary(name)?;
        Ok(self.rotated(name, tensor))
    }

    /// Uploads a matrix, read from the named tensor, that takes rotated
    /// activations.
    fn rotated(&mut self, name: &str, tensor: Tensor) -> D::Weight {
        self.rotated.push(name.to_string());
        let w = self.device.upload(tensor);
        self.tick();
        w
    }

    fn buffer(&mut self, values: Vec<f32>) -> D::Buffer {
        let mut buf = self.device.alloc(values.len());
        self.device.write(&mut buf, &values);
        self.tick();
        buf
    }
}

/// Reorders `n_v` heads of `per_head` elements each from the tiled order
/// llama.cpp keeps the value heads in, where head `j` shares key head
/// `j % n_k`, into the checkpoint's grouped order, where head `h` shares
/// key head `h / (n_v / n_k)`.
fn untile<T: Clone>(data: &[T], n_k: usize, n_v: usize, per_head: usize) -> Vec<T> {
    assert_eq!(data.len(), n_v * per_head);
    let rep = n_v / n_k;
    let mut out = Vec::with_capacity(data.len());
    for h in 0..n_v {
        let j = (h % rep) * n_k + h / rep;
        out.extend_from_slice(&data[j * per_head..][..per_head]);
    }
    out
}

/// The first bytes of a saved state, and its layout's version.
const STATE_MAGIC: &[u8; 8] = b"dwimstat";
const STATE_VERSION: u32 = 1;

impl<D: Device> LanguageModel for Model<D> {
    /// The state after `len` positions: a header of the layout's version and
    /// the model's dimensions, then each attention layer's keys and values
    /// for the positions, then each linear layer's convolution state and
    /// recurrent state.
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        let (c, d, s) = (&self.config, &self.device, &self.state);
        let kv_dim = c.kv_heads * c.head_dim;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        for value in [len, s.k_cache.len(), s.ssm_state.len(), kv_dim, c.conv_width(), c.ssm_width()] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for cache in s.k_cache.iter().chain(&s.v_cache) {
            out.extend(d.read_cache(cache, len * kv_dim).iter().flat_map(|v| v.to_le_bytes()));
        }
        for slot in 0..s.ssm_state.len() {
            // The convolution reads the batch before from one buffer of the
            // pair or the other, by the parity.
            let conv = &s.conv_state[slot][if s.parity { 0 } else { 1 }];
            for buf in [conv, &s.ssm_state[slot]] {
                out.extend(d.read(buf).iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        Some(out)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        let kv_dim = c.kv_heads * c.head_dim;
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u8]> {
            let bytes = state.get(at..at + n).ok_or("the saved state is cut short")?;
            at += n;
            Ok(bytes)
        };
        if take(8)? != STATE_MAGIC {
            return Err("not a saved state".into());
        }
        if u32::from_le_bytes(take(4)?.try_into()?) != STATE_VERSION {
            return Err("a saved state of another version".into());
        }
        let mut header = [0; 6];
        for value in &mut header {
            *value = u64::from_le_bytes(take(8)?.try_into()?) as usize;
        }
        let len = header[0];
        if header[1..] != [s.k_cache.len(), s.ssm_state.len(), kv_dim, c.conv_width(), c.ssm_width()] {
            return Err("a saved state of another model".into());
        }
        if len > s.max_len {
            return Err(format!("a saved state of {len} tokens does not fit in the context").into());
        }
        for cache in s.k_cache.iter_mut().chain(&mut s.v_cache) {
            let halves: Vec<u16> = take(len * kv_dim * 2)?.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            d.write_cache(cache, &halves);
        }
        for slot in 0..s.ssm_state.len() {
            let [conv, _] = &mut s.conv_state[slot];
            for (buf, width) in [(conv, c.conv_width()), (&mut s.ssm_state[slot], c.ssm_width())] {
                let floats: Vec<f32> = take(width * 4)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                d.write(buf, &floats);
            }
        }
        s.parity = true;
        Ok(len)
    }

    /// Adds the tokens to the model's state. Tokens run through the model
    /// in batches of up to [`BATCH`].
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(pos + tokens.len() <= self.state.max_len, "tokens past the end of the cache");
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            logits = self.forward_batch(batch, pos + i * BATCH, false);
        }
        logits
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Option<Vec<f32>> {
        assert!(!tokens.is_empty() && tokens.len() <= VERIFY, "a draft is one to {VERIFY} tokens");
        assert!(pos + tokens.len() <= self.state.max_len, "tokens past the end of the cache");
        Some(self.forward_batch(tokens, pos, true))
    }

    /// The attention layers' caches hold every token of the draft by
    /// position and the ones past `count` are never read; the linear
    /// layers' convolution and recurrent states, which the verify left as
    /// they were, take the first `count` tokens' projections now.
    fn commit(&mut self, count: usize) {
        let c = &self.config;
        let d = &self.device;
        let s = &mut self.state;
        assert!(count <= s.verified, "committing more tokens than were verified");
        if count == 0 {
            return;
        }
        let (k_dim, v_dim) = (c.k_dim(), c.v_dim());
        let eps = c.eps;
        d.resize(&mut s.q, count * k_dim);
        d.resize(&mut s.k, count * k_dim);
        d.resize(&mut s.v, count * v_dim);
        d.resize(&mut s.att, count * v_dim);
        for layer in &self.layers {
            let Mixer::Linear { conv, decay, slot, .. } = &layer.mixer else {
                continue;
            };
            d.resize(&mut s.stash_qkv[*slot], count * (2 * k_dim + v_dim));
            d.resize(&mut s.stash_ab[*slot], count * 2 * c.v_heads);
            // The parity has turned since the verify: its input buffer is
            // the other one, and the verify's output buffer takes the
            // state after the accepted tokens.
            let [a, b] = &mut s.conv_state[*slot];
            let (state, state_out) = if s.parity { (&*b, a) } else { (&*a, b) };
            d.conv(&mut s.q, &mut s.k, &mut s.v, state_out, &s.stash_qkv[*slot], state, conv);
            d.l2norm(&mut s.q, c.state_dim, eps);
            d.l2norm(&mut s.k, c.state_dim, eps);
            d.delta_net(&mut s.att, &s.q, &s.k, &s.v, &s.stash_ab[*slot], decay, &mut s.ssm_state[*slot], c.k_heads, c.v_heads, c.state_dim);
            d.resize(&mut s.stash_qkv[*slot], VERIFY * (2 * k_dim + v_dim));
            d.resize(&mut s.stash_ab[*slot], VERIFY * 2 * c.v_heads);
        }
        s.verified = 0;
    }

    fn max_len(&self) -> usize {
        self.state.max_len
    }
}

/// Buffers the forward pass computes in, with room for a batch of
/// [`BATCH`] tokens; the key/value caches of the attention layers, holding
/// every position seen so far in half precision; the convolution and
/// recurrent states of the linear attention layers; and the rotary position
/// embedding table for every position there is room for.
struct State<D: Device> {
    /// Activations per token of each per-token buffer, in field order.
    widths: [usize; 11],
    x: D::Buffer,
    xb: D::Buffer,
    xh: D::Buffer,
    q: D::Buffer,
    gate: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    qkv: D::Buffer,
    z: D::Buffer,
    ab: D::Buffer,
    /// The feed-forward network's two projections, `intermediate` wide.
    up: D::Buffer,
    gate_ffn: D::Buffer,
    intermediate: usize,
    logits: D::Buffer,
    k_cache: Vec<D::Cache>,
    v_cache: Vec<D::Cache>,
    /// Two per linear layer, the batch's inputs coming from one and its
    /// outputs going to the other, swapping every batch.
    conv_state: Vec<[D::Buffer; 2]>,
    parity: bool,
    ssm_state: Vec<D::Buffer>,
    /// For verifying a draft: each linear layer's projections and gates
    /// for its tokens, a recurrent state to run on in place of the layer's
    /// own, and room for every token's logits; and how many tokens the last
    /// verify ran.
    stash_qkv: Vec<D::Buffer>,
    stash_ab: Vec<D::Buffer>,
    scratch_state: D::Buffer,
    verify_logits: D::Buffer,
    verified: usize,
    rope: D::Buffer,
    max_len: usize,
}

impl<D: Device> State<D> {
    /// Allocates state for sequences of up to `max_len` tokens.
    fn new(c: &Config, d: &D, max_len: usize, caches: usize, slots: usize) -> Self {
        let q_dim = c.heads * c.head_dim;
        let kv_dim = c.kv_heads * c.head_dim;
        let (k_dim, v_dim) = (c.k_dim(), c.v_dim());
        let widths = [
            c.hidden,
            c.hidden,
            c.hidden,
            q_dim.max(k_dim),
            q_dim,
            kv_dim.max(k_dim),
            kv_dim.max(v_dim),
            q_dim.max(v_dim),
            2 * k_dim + v_dim,
            v_dim,
            2 * c.v_heads,
        ];
        let [x, xb, xh, q, gate, k, v, att, qkv, z, ab] = widths.map(|width| d.alloc(BATCH * width));
        let table = rope_table(max_len, c.rot_dim, c.rope_theta);
        let mut rope = d.alloc(table.len());
        d.write(&mut rope, &table);
        let conv_width = (CONV_KERNEL - 1) * (2 * k_dim + v_dim);
        Self {
            widths,
            x,
            xb,
            xh,
            q,
            gate,
            k,
            v,
            att,
            qkv,
            z,
            ab,
            up: d.alloc(BATCH * c.intermediate),
            gate_ffn: d.alloc(BATCH * c.intermediate),
            intermediate: c.intermediate,
            logits: d.alloc(c.vocab),
            stash_qkv: (0..slots).map(|_| d.alloc(VERIFY * (2 * k_dim + v_dim))).collect(),
            stash_ab: (0..slots).map(|_| d.alloc(VERIFY * 2 * c.v_heads)).collect(),
            scratch_state: d.alloc(c.ssm_width()),
            verify_logits: d.alloc(VERIFY * c.vocab),
            verified: 0,
            k_cache: (0..caches).map(|_| d.alloc_cache(max_len * kv_dim)).collect(),
            v_cache: (0..caches).map(|_| d.alloc_cache(max_len * kv_dim)).collect(),
            conv_state: (0..slots).map(|_| [d.alloc(conv_width), d.alloc(conv_width)]).collect(),
            parity: false,
            ssm_state: (0..slots).map(|_| d.alloc(c.v_heads * c.state_dim * c.state_dim)).collect(),
            rope,
            max_len,
        }
    }

    /// Sizes the buffers for a batch of `n` tokens.
    fn batch(&mut self, device: &D, n: usize) {
        assert!(n <= BATCH, "a batch of {n} tokens is more than {BATCH}");
        let buffers = [
            &mut self.x,
            &mut self.xb,
            &mut self.xh,
            &mut self.q,
            &mut self.gate,
            &mut self.k,
            &mut self.v,
            &mut self.att,
            &mut self.qkv,
            &mut self.z,
            &mut self.ab,
        ];
        for (buffer, width) in buffers.into_iter().zip(self.widths) {
            device.resize(buffer, n * width);
        }
        device.resize(&mut self.up, n * self.intermediate);
        device.resize(&mut self.gate_ffn, n * self.intermediate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{self, Value};
    use dwim_gpu::{Cpu, Gpu};
    use std::path::Path;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn floats(&mut self, n: usize, scale: f32, offset: f32) -> Vec<u8> {
            (0..n).flat_map(|_| (self.next() * scale + offset).to_le_bytes()).collect()
        }

        fn bf16(&mut self, n: usize, scale: f32) -> Vec<u8> {
            (0..n).flat_map(|_| (((self.next() * scale).to_bits() >> 16) as u16).to_le_bytes()).collect()
        }

        fn ternary(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<u8> {
            let mut data = vec![0; rows * ternary::row_bytes(cols)];
            for row in data.chunks_exact_mut(ternary::row_bytes(cols)) {
                let values: Vec<f32> = (0..cols).map(|_| self.next() * scale).collect();
                ternary::quantize_row(&values, row);
            }
            data
        }
    }

    /// Writes a tiny random model in the file format, four layers of which
    /// the last is full attention.
    fn synthetic(path: &Path) {
        let (hidden, inter, layers, heads, kv_heads, head_dim, rot_dim) = (1024, 1024, 4, 4, 2, 256, 64);
        let (k_heads, v_heads, state_dim, vocab) = (2, 8, 128, 64);
        let (k_dim, v_dim) = (k_heads * state_dim, v_heads * state_dim);
        let mut rng = Rng(42);
        let mut signs = Vec::new();
        for _ in 0..hidden {
            signs.push(Value::I32(if rng.next() < 0.0 { -1 } else { 1 }));
        }
        let mut names = Vec::new();
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        // Weights are small so that the activations stay sane through the
        // layers; ternary scales are their largest magnitude.
        let mut tern = |name: &str, rows: usize, cols: usize, names: &mut Vec<Value>| {
            names.push(Value::Str(name.to_string()));
            let scale = 1.0 / (cols as f32).sqrt();
            tensors.push((name.to_string(), vec![cols, rows], gguf::PTQ1_0, rng.ternary(rows, cols, scale)));
        };
        tern("output.weight", vocab, hidden, &mut names);
        for i in 0..layers {
            let p = format!("blk.{i}");
            if (i + 1) % 4 == 0 {
                tern(&format!("{p}.attn_q.weight"), heads * head_dim * 2, hidden, &mut names);
                tern(&format!("{p}.attn_k.weight"), kv_heads * head_dim, hidden, &mut names);
                tern(&format!("{p}.attn_v.weight"), kv_heads * head_dim, hidden, &mut names);
                tern(&format!("{p}.attn_output.weight"), hidden, heads * head_dim, &mut names);
            } else {
                tern(&format!("{p}.attn_qkv.weight"), 2 * k_dim + v_dim, hidden, &mut names);
                tern(&format!("{p}.attn_gate.weight"), v_dim, hidden, &mut names);
                tern(&format!("{p}.ssm_out.weight"), hidden, v_dim, &mut names);
            }
            tern(&format!("{p}.ffn_gate.weight"), inter, hidden, &mut names);
            tern(&format!("{p}.ffn_up.weight"), inter, hidden, &mut names);
            tern(&format!("{p}.ffn_down.weight"), hidden, inter, &mut names);
        }
        let mut rng = Rng(7);
        tensors.push(("token_embd.weight".into(), vec![hidden, vocab], gguf::PTQ1_0, rng.ternary(vocab, hidden, 0.5)));
        tensors.push(("output_norm.weight".into(), vec![hidden], gguf::F32, rng.floats(hidden, 0.1, 1.0)));
        for i in 0..layers {
            let p = format!("blk.{i}");
            tensors.push((format!("{p}.attn_norm.weight"), vec![hidden], gguf::F32, rng.floats(hidden, 0.1, 1.0)));
            tensors.push((format!("{p}.post_attention_norm.weight"), vec![hidden], gguf::F32, rng.floats(hidden, 0.1, 1.0)));
            if (i + 1) % 4 == 0 {
                tensors.push((format!("{p}.attn_q_norm.weight"), vec![head_dim], gguf::F32, rng.floats(head_dim, 0.1, 1.0)));
                tensors.push((format!("{p}.attn_k_norm.weight"), vec![head_dim], gguf::F32, rng.floats(head_dim, 0.1, 1.0)));
            } else {
                tensors.push((format!("{p}.ssm_alpha.weight"), vec![hidden, v_heads], gguf::BF16, rng.bf16(v_heads * hidden, 0.05)));
                tensors.push((format!("{p}.ssm_beta.weight"), vec![hidden, v_heads], gguf::BF16, rng.bf16(v_heads * hidden, 0.05)));
                let channels = 2 * k_dim + v_dim;
                tensors.push((format!("{p}.ssm_conv1d.weight"), vec![CONV_KERNEL, channels], gguf::F32, rng.floats(channels * CONV_KERNEL, 0.5, 0.0)));
                tensors.push((format!("{p}.ssm_a"), vec![v_heads], gguf::F32, rng.floats(v_heads, 0.5, -0.6)));
                tensors.push((format!("{p}.ssm_dt.bias"), vec![v_heads], gguf::F32, rng.floats(v_heads, 2.0, 0.0)));
                tensors.push((format!("{p}.ssm_norm.weight"), vec![state_dim], gguf::F32, rng.floats(state_dim, 0.1, 1.0)));
            }
        }
        let n = |v: usize| Value::U32(v as u32);
        let meta = [
            ("general.architecture", Value::Str("qwen35".into())),
            ("qwen35.block_count", n(layers)),
            ("qwen35.context_length", n(256)),
            ("qwen35.embedding_length", n(hidden)),
            ("qwen35.feed_forward_length", n(inter)),
            ("qwen35.attention.head_count", n(heads)),
            ("qwen35.attention.head_count_kv", n(kv_heads)),
            ("qwen35.attention.key_length", n(head_dim)),
            ("qwen35.rope.dimension_count", n(rot_dim)),
            ("qwen35.rope.freq_base", Value::F32(10000.0)),
            ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("qwen35.ssm.conv_kernel", n(CONV_KERNEL)),
            ("qwen35.ssm.state_size", n(state_dim)),
            ("qwen35.ssm.group_count", n(k_heads)),
            ("qwen35.ssm.time_step_rank", n(v_heads)),
            ("qwen35.ssm.inner_size", n(v_dim)),
            ("qwen35.full_attention_interval", n(4)),
            ("prism.hadamard.version", n(1)),
            ("prism.hadamard.block_size", n(HADAMARD_BLOCK)),
            ("prism.hadamard.transform", Value::Str("normalized-sylvester-walsh-hadamard".into())),
            ("prism.hadamard.axis", Value::Str("input-last-dimension".into())),
            ("prism.hadamard.sign_mode", Value::Str("explicit".into())),
            ("prism.hadamard.weight_names", Value::Array(names)),
            ("prism.hadamard.sign_widths", Value::Array(vec![n(hidden)])),
            ("prism.hadamard.sign_values", Value::Array(signs)),
            ("prism.hadamard.inverse_weight_names", Value::Array(vec![Value::Str("token_embd.weight".into())])),
            ("prism.hadamard.gdn_v_grouped", Value::Bool(true)),
        ];
        gguf::write(path, &meta, &tensors).unwrap();
    }

    fn run<D: Device>(path: &Path, device: D) -> [Vec<f32>; 3] {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let mut model = Model::load(gguf, device, 64, |_, _| {}).unwrap();
        // A batch of several tokens, then one more, then a batch of many,
        // each carrying the state.
        let first = model.forward(&[3, 17, 42, 7, 9], 0);
        let second = model.forward(&[11], 5);
        let many: Vec<u32> = (0..40).map(|i| (i * 5 % 43) as u32).collect();
        let third = model.forward(&many, 6);
        [first, second, third]
    }

    #[test]
    fn restores_a_saved_state() {
        let dir = std::env::temp_dir().join(format!("dwim-bonsai-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        // A conversation of two batches, and one that takes up from the
        // first's saved state: the same logits, since the state holds the
        // caches exactly.
        let mut whole = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        whole.forward(&[3, 17, 42, 7, 9], 0);
        let state = whole.save(5).unwrap();
        let want = whole.forward(&[11, 2], 5);
        let mut resumed = Model::load(gguf, Cpu, 64, |_, _| {}).unwrap();
        assert_eq!(resumed.restore(&state).unwrap(), 5);
        assert_eq!(resumed.forward(&[11, 2], 5), want);
        assert!(resumed.restore(&state[..100]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verifies_a_draft_and_commits_part_of_it() {
        let dir = std::env::temp_dir().join(format!("dwim-bonsai-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        // One model runs a draft of four tokens after a prompt and commits
        // two; another runs the prompt and those two tokens plainly. The
        // draft's logits are the plain ones token by token, and both
        // models agree on the token after.
        let mut drafted = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        let mut plain = Model::load(gguf, Cpu, 64, |_, _| {}).unwrap();
        let prompt = [3, 17, 42, 7, 9];
        drafted.forward(&prompt, 0);
        plain.forward(&prompt, 0);
        let draft = [11, 2, 30, 5];
        let logits = drafted.verify(&draft, 5).unwrap();
        let vocab = logits.len() / draft.len();
        for (i, &token) in draft.iter().enumerate() {
            let want = plain.forward(&[token], 5 + i);
            for (a, b) in logits[i * vocab..][..vocab].iter().zip(&want) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "token {i}: {a} vs {b}");
            }
        }
        drafted.commit(2);
        let mut plain = Model::load(Arc::new(Gguf::open(&path).unwrap()), Cpu, 64, |_, _| {}).unwrap();
        plain.forward(&prompt, 0);
        plain.forward(&draft[..2], 5);
        let (a, b) = (drafted.forward(&[8], 7), plain.forward(&[8], 7));
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "{a} vs {b}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gpu_agrees_with_cpu() {
        let dir = std::env::temp_dir().join(format!("dwim-bonsai-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let cpu = run(&path, Cpu);
        assert!(cpu[0].iter().all(|v| v.is_finite()));
        assert!(cpu[0].iter().any(|&v| v != 0.0));
        assert_ne!(cpu[0], cpu[1]);
        if let Ok(gpu) = Gpu::new() {
            let gpu = run(&path, gpu);
            // A batch of tokens goes through the GPU's half-precision
            // matmuls, whose difference the state carries forward.
            for ((cpu, gpu), tolerance) in cpu.iter().zip(&gpu).zip([1e-2, 1e-2, 1e-2]) {
                for (a, b) in cpu.iter().zip(gpu) {
                    assert!((a - b).abs() <= tolerance * (1.0 + a.abs()), "{a} vs {b}");
                }
            }
        } else {
            eprintln!("skipping the GPU comparison: no GPU");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Greedy completion of a prompt by the real model, if its file is in
    /// the cache: run with `--ignored --nocapture` and compare with what
    /// PrismML's llama.cpp says.
    #[test]
    #[ignore]
    fn real_model_completes_a_prompt() {
        let path = dirs::cache_dir().unwrap().join("dwim/models/bonsai-2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf");
        if !path.exists() {
            eprintln!("skipping: no model at {}", path.display());
            return;
        }
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        let tokenizer = crate::Tokenizer::from_gguf(&gguf).unwrap();
        let device = Gpu::new().unwrap();
        let start = std::time::Instant::now();
        let mut model = Model::load(gguf, device, 256, |_, _| {}).unwrap();
        eprintln!("loaded in {:.1}s", start.elapsed().as_secs_f32());
        let mut tokens = tokenizer.encode("The capital of France is").unwrap();
        // The prompt first, then the tokens one at a time, each timed on
        // its own.
        let decode = 16;
        let mut text = Vec::new();
        let start = std::time::Instant::now();
        let mut logits = model.forward(&tokens, 0);
        eprintln!("prompt: {:.1} tok/s", tokens.len() as f32 / start.elapsed().as_secs_f32());
        let start = std::time::Instant::now();
        for _ in 0..decode {
            let next = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
            text.extend_from_slice(tokenizer.decode(next));
            tokens.push(next);
            logits = model.forward(&tokens[tokens.len() - 1..], tokens.len() - 1);
        }
        eprintln!("decode: {:.1} tok/s over {decode} tokens", decode as f32 / start.elapsed().as_secs_f32());
        let text = String::from_utf8_lossy(&text);
        eprintln!("completion: {text:?}");
        assert!(text.starts_with(" Paris."), "{text:?}");
        // A long prompt, carrying on from the completion, for the speed of
        // the batch kernels at their size.
        let passage = "The quick brown fox jumps over the lazy dog while the river runs to the sea. ".repeat(20);
        let mut long = tokenizer.encode(&passage).unwrap();
        long.truncate(model.max_len() - tokens.len());
        let start = std::time::Instant::now();
        model.forward(&long, tokens.len());
        eprintln!("long prompt: {:.1} tok/s over {} tokens", long.len() as f32 / start.elapsed().as_secs_f32(), long.len());
    }

    #[test]
    fn untiling_puts_value_heads_back_by_key_head() {
        // Tiled: head j shares key head j % 2; grouped: head h shares h / 3.
        let tiled: Vec<u32> = vec![0, 1, 0, 1, 0, 1];
        let grouped = untile(&tiled, 2, 6, 1);
        assert_eq!(grouped, [0, 0, 0, 1, 1, 1]);
        let tiled: Vec<u32> = (0..6).collect();
        assert_eq!(untile(&tiled, 2, 6, 1), [0, 2, 4, 1, 3, 5]);
    }
}
