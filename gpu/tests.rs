//! Checks of a GPU device's operations against the CPU reference.

use crate::{CONV_KERNEL, Cpu, Device, HADAMARD_BLOCK, Tensor, ternary};

/// Defines a test for each check, run on the device `$open` returns, and
/// skipped if it returns an error.
macro_rules! check_against_cpu {
    ($open:expr) => {
        check_against_cpu!(
            $open;
            write_and_read_round_trip,
            alloc_is_zeroed_and_copy_moves_ranges,
            matmul_matches_cpu,
            ternary_matmul_matches_cpu,
            rmsnorm_matches_cpu,
            l2norm_matches_cpu,
            rope_matches_cpu,
            attention_matches_cpu,
            store_rounds_like_cpu,
            cache_round_trips,
            elementwise_match_cpu,
            hadamard_matches_cpu,
            norm_rotate_matches_cpu,
            conv_matches_cpu,
            delta_net_matches_cpu,
            resize_keeps_capacity
        );
    };
    ($open:expr; $($check:ident),*) => {
        $(
            #[test]
            fn $check() {
                match $open {
                    Ok(gpu) => crate::tests::$check(&gpu),
                    Err(e) => eprintln!("skipping: {e}"),
                }
            }
        )*
    };
}

/// Deterministic pseudo-random values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    fn bf16(&mut self, shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Tensor::Bf16 {
            shape: shape.to_vec(),
            data: (0..n).map(|_| (self.next().to_bits() >> 16) as u16).collect(),
        }
    }

    fn ternary(&mut self, shape: &[usize]) -> Tensor {
        let (rows, cols) = (shape[0], shape[1]);
        let mut data = vec![0; rows * ternary::row_bytes(cols)];
        for row in data.chunks_exact_mut(ternary::row_bytes(cols)) {
            ternary::quantize_row(&self.floats(cols), row);
        }
        Tensor::Ternary {
            shape: shape.to_vec(),
            data,
        }
    }
}

fn clone(t: &Tensor) -> Tensor {
    match t {
        Tensor::Bf16 { shape, data } => Tensor::Bf16 {
            shape: shape.clone(),
            data: data.clone(),
        },
        Tensor::Ternary { shape, data } => Tensor::Ternary {
            shape: shape.clone(),
            data: data.clone(),
        },
    }
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())), "element {i}: {a} vs {b}");
    }
}


fn buffer<D: Device>(gpu: &D, data: &[f32]) -> D::Buffer {
    let mut buf = gpu.alloc(data.len());
    gpu.write(&mut buf, data);
    buf
}

/// A cache holding `data`, stored in two parts split at `split`, as the
/// prompt and then the tokens after it are.
fn cache<D: Device>(gpu: &D, data: &[f32], split: usize) -> D::Cache {
    let mut cache = gpu.alloc_cache(data.len());
    for (offset, part) in [(0, &data[..split]), (split, &data[split..])] {
        if !part.is_empty() {
            gpu.store(&mut cache, offset, &buffer(gpu, part));
        }
    }
    cache
}

pub fn write_and_read_round_trip<D: Device>(gpu: &D) {
    let data = Rng(1).floats(100_003);
    let buf = buffer(gpu, &data);
    assert_eq!(gpu.read(&buf), data);
}

pub fn alloc_is_zeroed_and_copy_moves_ranges<D: Device>(gpu: &D) {
    let zero = gpu.alloc(1000);
    assert!(gpu.read(&zero).iter().all(|&v| v == 0.0));
    let data = Rng(2).floats(50);
    let src = buffer(gpu, &data);
    let mut dst = gpu.alloc(100);
    gpu.copy(&mut dst, 30, &src, 10, 20);
    let mut want = vec![0.0; 100];
    Cpu.copy(&mut want, 30, &data, 10, 20);
    assert_eq!(gpu.read(&dst), want);
}

pub fn matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(4);
    for (rows, cols, n) in [(1, 8, 1), (200, 192, 1), (77, 1032, 3), (70_000, 8, 2)] {
        let w = rng.bf16(&[rows, cols]);
        let x = rng.floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &clone(&w), &x);
        let weight = gpu.upload(w);
        let x = buffer(gpu, &x);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &x);
        close(&gpu.read(&out), &want, 1e-4);
    }
}

pub fn ternary_matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(14);
    // Single tokens, small batches, and batches of half a tile of tokens or
    // more at the edges of the tiles: a whole one, partial row and token
    // tiles, and the model's width. A device may multiply a batch in half
    // precision, as the Vulkan batch kernels do, so a batch is held to an
    // error of a thousandth of the outputs' typical magnitude in root mean
    // square and a hundredth at worst, which is a tenth of what eight-bit
    // activations would cost; a single token to the usual.
    for (rows, cols, n) in [(1, 128, 1), (200, 5120, 1), (77, 1024, 3), (70_000, 128, 2), (200, 5120, 8), (64, 128, 16), (100, 256, 47), (1030, 1152, 33), (300, 1152, 130), (2500, 5120, 64)] {
        let w = rng.ternary(&[rows, cols]);
        let x = rng.floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &clone(&w), &x);
        let weight = gpu.upload(w);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &buffer(gpu, &x));
        let got = gpu.read(&out);
        if n >= 2 {
            let typical = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();
            let (mut worst, mut sum) = (0.0f64, 0.0f64);
            for (g, w) in got.iter().zip(&want) {
                let e = (g - w).abs() as f64;
                worst = worst.max(e);
                sum += e * e;
            }
            let rms = (sum / got.len() as f64).sqrt();
            assert!(rms <= 1e-3 * typical && worst <= 1e-2 * typical, "{rows}x{cols} n={n}: rms {rms:.2e}, worst {worst:.2e}, typical {typical:.2e}");
        } else {
            close(&got, &want, 1e-4);
        }
    }
}

/// Prints how far the tile kernel is from the CPU on a batch: the largest
/// and root-mean-square error against the CPU on the activations as given,
/// and on them rounded to half floats first.
pub fn tile_error<D: Device>(gpu: &D) {
    let mut rng = Rng(14);
    for (rows, cols, n) in [(100, 256, 47), (1030, 1152, 33), (2500, 5120, 64)] {
        let w = rng.ternary(&[rows, cols]);
        let x = rng.floats(n * cols);
        let halved: Vec<f32> = x.iter().map(|&v| crate::from_f16(crate::to_f16(v))).collect();
        let mut want = vec![0.0; n * rows];
        let mut want_half = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &clone(&w), &x);
        Cpu.matmul(&mut want_half, &clone(&w), &halved);
        let weight = gpu.upload(w);
        let x = buffer(gpu, &x);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &x);
        let got = gpu.read(&out);
        for (name, want) in [("f32", &want), ("f16", &want_half)] {
            let (mut worst, mut sum) = (0.0f32, 0.0f64);
            for (g, w) in got.iter().zip(want) {
                let e = (g - w).abs();
                worst = worst.max(e);
                sum += (e * e) as f64;
            }
            let rms = (sum / got.len() as f64).sqrt();
            let typical = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();
            eprintln!("{rows}x{cols} n={n} vs {name}: worst {worst:.2e} rms {rms:.2e} typical {typical:.2e}");
        }
    }
}

/// Times the ternary matmul on the model's 17408x5120 for batches of
/// tokens, and prints each batch's rate.
pub fn ternary_matmul_speed<D: Device>(gpu: &D) {
    let mut rng = Rng(21);
    let (rows, cols) = (17408, 5120);
    let weight = gpu.upload(rng.ternary(&[rows, cols]));
    for n in [1, 4, 8, 16, 32, 64, 96, 128, 256] {
        let x = buffer(gpu, &rng.floats(n * cols));
        let mut out = gpu.alloc(n * rows);
        // Warm up until the clocks are up.
        for _ in 0..(1600 / n).max(40) {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let runs = (800 / n).max(20);
        let start = std::time::Instant::now();
        for _ in 0..runs {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let flops = 2.0 * (rows * cols * n) as f64 / each;
        eprintln!("{rows}x{cols} n={n:<3} {:8.3} ms  {:6.3} ms/token  {:6.2} TFLOP/s", each * 1e3, each * 1e3 / n as f64, flops / 1e12);
    }
    // The model's other shapes, for a single token: the ones with fewer
    // rows take the other kernels.
    for (rows, cols) in [(5120, 17408), (6144, 5120), (10240, 5120), (1024, 5120)] {
        let weight = gpu.upload(rng.ternary(&[rows, cols]));
        let x = buffer(gpu, &rng.floats(cols));
        let mut out = gpu.alloc(rows);
        for _ in 0..2000 {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let runs = 2000;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let bytes = rows * ternary::row_bytes(cols);
        eprintln!("{rows}x{cols} n=1   {:8.3} ms  {:6.0} GB/s", each * 1e3, bytes as f64 / each / 1e9);
    }
}

/// Times a run of small elementwise dispatches, for the cost of a dispatch
/// itself: adds of a token's width of activations, back to back.
pub fn dispatch_overhead<D: Device>(gpu: &D) {
    let mut rng = Rng(22);
    let mut x = buffer(gpu, &rng.floats(5120));
    let y = buffer(gpu, &rng.floats(5120));
    for _ in 0..2000 {
        gpu.add(&mut x, &y);
    }
    gpu.read(&x);
    let runs = 4000;
    let start = std::time::Instant::now();
    for _ in 0..runs {
        gpu.add(&mut x, &y);
    }
    gpu.read(&x);
    eprintln!("add of 5120: {:.2} us a dispatch", start.elapsed().as_secs_f64() * 1e6 / runs as f64);
}

/// Times every kernel at the model's shapes for a batch of `n` tokens, the
/// ternary matmuls other than the largest included, and prints each one's
/// cost and, times its calls a batch, its share of a token.
pub fn kernel_speed<D: Device>(gpu: &D, n: usize) {
    let mut rng = Rng(23);
    let (hidden, inter, heads, head_dim, rot_dim, kv_heads) = (5120, 17408, 24, 256, 64, 4);
    let (k_heads, v_heads, state_dim) = (16, 48, 128);
    let (k_dim, v_dim) = (k_heads * state_dim, v_heads * state_dim);
    let (q_dim, kv_dim) = (heads * head_dim, kv_heads * head_dim);
    let channels = 2 * k_dim + v_dim;
    let context = 1024;
    let sync = gpu.alloc(1);
    let mut total = 0.0;
    let mut time = |name: &str, calls: usize, op: &mut dyn FnMut()| {
        for _ in 0..(200 / n).max(20) {
            op();
        }
        gpu.read(&sync);
        let runs = (500 / n).max(20);
        let start = std::time::Instant::now();
        for _ in 0..runs {
            op();
        }
        gpu.read(&sync);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let per_token = each * calls as f64 / n as f64;
        total += per_token;
        eprintln!("{name:<28} {:7.1} us  x{calls:<3} {:6.2} ms/token", each * 1e6, per_token * 1e3);
    };
    let (mut h, mut i, mut qd, mut vd, mut kd) = (buffer(gpu, &rng.floats(n * hidden)), buffer(gpu, &rng.floats(n * inter)), buffer(gpu, &rng.floats(n * q_dim)), buffer(gpu, &rng.floats(n * v_dim)), buffer(gpu, &rng.floats(n * kv_dim)));
    let (h2, i2, vd2) = (buffer(gpu, &rng.floats(n * hidden)), buffer(gpu, &rng.floats(n * inter)), buffer(gpu, &rng.floats(n * v_dim)));
    let (signs_h, signs_i, signs_v) = (buffer(gpu, &rng.floats(hidden)), buffer(gpu, &rng.floats(inter)), buffer(gpu, &rng.floats(v_dim)));
    let (norm_h, norm_head, norm_state) = (buffer(gpu, &rng.floats(hidden)), buffer(gpu, &rng.floats(head_dim)), buffer(gpu, &rng.floats(state_dim)));
    let mut out_h = gpu.alloc(n * hidden);
    time("norm_rotate 5120", 129, &mut || gpu.norm_rotate(&mut out_h, &h, &norm_h, &signs_h, 1e-6));
    time("hadamard 6144", 64, &mut || gpu.hadamard(&mut vd, &signs_v, false));
    time("hadamard 17408", 64, &mut || gpu.hadamard(&mut i, &signs_i, false));
    time("rmsnorm 5120", 48, &mut || gpu.rmsnorm(&mut h, &norm_h, 1e-6));
    time("rmsnorm 6144 by 256", 16, &mut || gpu.rmsnorm(&mut qd, &norm_head, 1e-6));
    time("rmsnorm 1024 by 256", 16, &mut || gpu.rmsnorm(&mut kd, &norm_head, 1e-6));
    time("rmsnorm 6144 by 128", 48, &mut || gpu.rmsnorm(&mut vd, &norm_state, 1e-6));
    time("add 5120", 128, &mut || gpu.add(&mut h, &h2));
    time("copy 5120", 48, &mut || gpu.copy(&mut h, 0, &h2, 0, n * hidden));
    time("silu_mul 17408", 64, &mut || gpu.silu_mul(&mut i, &i2));
    time("silu_mul 6144", 48, &mut || gpu.silu_mul(&mut vd, &vd2));
    time("sigmoid_mul 6144", 16, &mut || gpu.sigmoid_mul(&mut qd, &vd2));
    let mut kq = buffer(gpu, &rng.floats(n * k_dim));
    time("l2norm 2048 by 128", 96, &mut || gpu.l2norm(&mut kq, state_dim, 1e-6));
    let table = buffer(gpu, &rng.floats((context + n) * rot_dim));
    time("rope 24 heads", 16, &mut || gpu.rope(&mut qd, &table, 500, heads, head_dim, rot_dim));
    time("rope 4 heads", 16, &mut || gpu.rope(&mut kd, &table, 500, kv_heads, head_dim, rot_dim));
    let mut k_cache = cache(gpu, &rng.floats((context + n) * kv_dim), 0);
    let v_cache = cache(gpu, &rng.floats((context + n) * kv_dim), 0);
    time("store 1024", 32, &mut || gpu.store(&mut k_cache, 500 * kv_dim, &kd));
    let mut att = gpu.alloc(n * q_dim);
    time("attention at 1024", 16, &mut || gpu.attention(&mut att, &qd, &k_cache, &v_cache, context - 1, heads, head_dim, kv_heads));
    let long = 4 * context;
    let k_long = cache(gpu, &rng.floats((long + n) * kv_dim), 0);
    let v_long = cache(gpu, &rng.floats((long + n) * kv_dim), 0);
    time("attention at 4096", 16, &mut || gpu.attention(&mut att, &qd, &k_long, &v_long, long - 1, heads, head_dim, kv_heads));
    let (mut cq, mut ck, mut cv) = (gpu.alloc(n * k_dim), gpu.alloc(n * k_dim), gpu.alloc(n * v_dim));
    let (xc, cstate, cweight) = (buffer(gpu, &rng.floats(n * channels)), buffer(gpu, &rng.floats((CONV_KERNEL - 1) * channels)), buffer(gpu, &rng.floats(channels * CONV_KERNEL)));
    let mut cstate_out = gpu.alloc((CONV_KERNEL - 1) * channels);
    time("conv 10240", 48, &mut || gpu.conv(&mut cq, &mut ck, &mut cv, &mut cstate_out, &xc, &cstate, &cweight));
    let gates = buffer(gpu, &rng.floats(n * 2 * v_heads));
    let decay: Vec<f32> = (0..2 * v_heads).map(|i| if i < v_heads { -rng.next().abs() } else { 4.0 * rng.next() }).collect();
    let decay = buffer(gpu, &decay);
    let mut dstate = buffer(gpu, &rng.floats(v_heads * state_dim * state_dim));
    let mut dout = gpu.alloc(n * v_dim);
    time("delta_net 48 heads", 48, &mut || gpu.delta_net(&mut dout, &cq, &ck, &cv, &gates, &decay, &mut dstate, k_heads, v_heads, state_dim));
    let ab = gpu.upload(rng.bf16(&[2 * v_heads, hidden]));
    let mut gout = gpu.alloc(n * 2 * v_heads);
    time("matmul bf16 96x5120", 48, &mut || gpu.matmul(&mut gout, &ab, &h));
    for (rows, cols, calls) in [(5120, 17408, 64), (6144, 5120, 80), (10240, 5120, 48), (1024, 5120, 32)] {
        let weight = gpu.upload(rng.ternary(&[rows, cols]));
        let x = buffer(gpu, &rng.floats(n * cols));
        let mut out = gpu.alloc(n * rows);
        time(&format!("matmul {rows}x{cols}"), calls, &mut || gpu.matmul(&mut out, &weight, &x));
    }
    eprintln!("{:<28} {:22} {:6.2} ms/token", "total", "", total * 1e3);
}

pub fn rmsnorm_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(5);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let w = rng.floats(dim);
        let x = rng.floats(dim * rows);
        let mut want = x.clone();
        Cpu.rmsnorm(&mut want, &w, 1e-6);
        let weight = buffer(gpu, &w);
        let mut buf = buffer(gpu, &x);
        gpu.rmsnorm(&mut buf, &weight, 1e-6);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn l2norm_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(15);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let x = rng.floats(dim * rows);
        let mut want = x.clone();
        Cpu.l2norm(&mut want, dim, 1e-6);
        let mut buf = buffer(gpu, &x);
        gpu.l2norm(&mut buf, dim, 1e-6);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn rope_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(6);
    for (n_heads, head_dim, rot_dim, n, pos) in [(4, 16, 16, 3, 7), (3, 256, 64, 2, 5)] {
        let table = rng.floats((pos + n) * rot_dim);
        let x = rng.floats(n * n_heads * head_dim);
        let mut want = x.clone();
        Cpu.rope(&mut want, &table, pos, n_heads, head_dim, rot_dim);
        let table = buffer(gpu, &table);
        let mut buf = buffer(gpu, &x);
        gpu.rope(&mut buf, &table, pos, n_heads, head_dim, rot_dim);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn attention_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(7);
    // The last attends over more positions than one dispatch has scores
    // for, all its tokens at once.
    let cases = [
        (4, 8, 2, 3, 5),
        (16, 128, 8, 2, 300),
        (2, 128, 1, 1, 0),
        (24, 256, 4, 3, 1000),
        (3, 24, 1, 2, 9),
        (16, 128, 4, 64, 32704),
    ];
    for (n_heads, head_dim, n_kv_heads, n, pos) in cases {
        let kv_dim = n_kv_heads * head_dim;
        let q = rng.floats(n * n_heads * head_dim);
        let k_cache = rng.floats((pos + n) * kv_dim);
        let v_cache = rng.floats((pos + n) * kv_dim);
        let mut want = vec![0.0; q.len()];
        let (cpu_k, cpu_v) = (cache(&Cpu, &k_cache, pos * kv_dim), cache(&Cpu, &v_cache, pos * kv_dim));
        Cpu.attention(&mut want, &q, &cpu_k, &cpu_v, pos, n_heads, head_dim, n_kv_heads);
        let q = buffer(gpu, &q);
        let (k, v) = (cache(gpu, &k_cache, pos * kv_dim), cache(gpu, &v_cache, pos * kv_dim));
        let mut out = gpu.alloc(want.len());
        gpu.attention(&mut out, &q, &k, &v, pos, n_heads, head_dim, n_kv_heads);
        // Within a couple of half-precision ulps: a driver whose `store`
        // truncates rather than rounds (Mesa's `pack2x16float` does) puts
        // slightly different keys and values in the cache, which
        // `store_rounds_like_cpu` reports on its own.
        close(&gpu.read(&out), &want, 3e-3);
    }
}

pub fn store_rounds_like_cpu<D: Device>(gpu: &D) {
    // Attention over a single position weighs its value by exactly one, so
    // it reads back what the value cache holds.
    let head_dim = 128;
    let mut v = Rng(9).floats(head_dim);
    let edges = [
        1e6,
        -1e6,
        65504.0,
        65520.0,
        1.0 + 2.0f32.powi(-11),
        1.0 + 3.0 * 2.0f32.powi(-11),
        0.1,
        1e-9,
        -3e-7,
        2.0f32.powi(-14) * 0.99999,
        -0.0,
    ];
    v[..edges.len()].copy_from_slice(&edges);
    let q = vec![0.0; head_dim];
    let mut want = vec![0.0; head_dim];
    let cpu_v = cache(&Cpu, &v, 0);
    Cpu.attention(&mut want, &q, &cpu_v, &cpu_v, 0, 1, head_dim, 1);
    let gpu_v = cache(gpu, &v, 0);
    let mut out = gpu.alloc(head_dim);
    gpu.attention(&mut out, &buffer(gpu, &q), &gpu_v, &gpu_v, 0, 1, head_dim, 1);
    assert_eq!(gpu.read(&out), want);
}

pub fn cache_round_trips<D: Device>(gpu: &D) {
    let data = Rng(10).floats(1002);
    let mut c = cache(gpu, &data, 300);
    let want: Vec<u16> = data.iter().map(|&v| crate::to_f16(v)).collect();
    assert_eq!(gpu.read_cache(&c, 1002), want);
    assert_eq!(gpu.read_cache(&c, 7), want[..7]);
    let bits: Vec<u16> = (0..500).map(|i| i as u16 * 3).collect();
    gpu.write_cache(&mut c, &bits);
    let mut expect = want.clone();
    expect[..500].copy_from_slice(&bits);
    assert_eq!(gpu.read_cache(&c, 1002), expect);
}

pub fn elementwise_match_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(8);
    let a = rng.floats(3001);
    let b = rng.floats(3001);
    let mut want = a.clone();
    Cpu.silu_mul(&mut want, &b);
    let mut gate = buffer(gpu, &a);
    let up = buffer(gpu, &b);
    gpu.silu_mul(&mut gate, &up);
    close(&gpu.read(&gate), &want, 1e-6);

    let mut want = a.clone();
    Cpu.sigmoid_mul(&mut want, &b);
    let mut x = buffer(gpu, &a);
    gpu.sigmoid_mul(&mut x, &up);
    close(&gpu.read(&x), &want, 1e-6);

    let mut want = a.clone();
    Cpu.add(&mut want, &b);
    let mut x = buffer(gpu, &a);
    gpu.add(&mut x, &up);
    close(&gpu.read(&x), &want, 1e-6);
}

pub fn hadamard_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(10);
    for (width, rows) in [(HADAMARD_BLOCK, 1), (5 * HADAMARD_BLOCK, 3)] {
        let signs: Vec<f32> = (0..width).map(|_| if rng.next() < 0.0 { -1.0 } else { 1.0 }).collect();
        let x = rng.floats(width * rows);
        for inverse in [false, true] {
            let mut want = x.clone();
            Cpu.hadamard(&mut want, &signs, inverse);
            let signs = buffer(gpu, &signs);
            let mut buf = buffer(gpu, &x);
            gpu.hadamard(&mut buf, &signs, inverse);
            close(&gpu.read(&buf), &want, 1e-5);
        }
        // The inverse undoes the rotation.
        let mut back = x.clone();
        Cpu.hadamard(&mut back, &signs, false);
        Cpu.hadamard(&mut back, &signs, true);
        close(&back, &x, 1e-5);
    }
}

pub fn norm_rotate_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(13);
    for (width, rows) in [(HADAMARD_BLOCK, 1), (5 * HADAMARD_BLOCK, 3)] {
        let signs: Vec<f32> = (0..width).map(|_| if rng.next() < 0.0 { -1.0 } else { 1.0 }).collect();
        let weight: Vec<f32> = (0..width).map(|_| 1.0 + 0.2 * rng.next()).collect();
        let x = rng.floats(width * rows);
        let mut want = vec![0.0; x.len()];
        Cpu.norm_rotate(&mut want, &x, &weight, &signs, 1e-6);
        let mut out = gpu.alloc(x.len());
        gpu.norm_rotate(&mut out, &buffer(gpu, &x), &buffer(gpu, &weight), &buffer(gpu, &signs), 1e-6);
        close(&gpu.read(&out), &want, 1e-5);
    }
}

pub fn conv_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(11);
    let (q_dim, k_dim, v_dim) = (16, 16, 48);
    let channels = q_dim + k_dim + v_dim;
    for n in [1, 2, 7] {
        let x = rng.floats(n * channels);
        let state = rng.floats((CONV_KERNEL - 1) * channels);
        let weight = rng.floats(channels * CONV_KERNEL);
        let (mut want_q, mut want_k, mut want_v) = (vec![0.0; n * q_dim], vec![0.0; n * k_dim], vec![0.0; n * v_dim]);
        let mut want_state = vec![0.0; state.len()];
        Cpu.conv(&mut want_q, &mut want_k, &mut want_v, &mut want_state, &x, &state, &weight);
        let (mut q, mut k, mut v) = (gpu.alloc(n * q_dim), gpu.alloc(n * k_dim), gpu.alloc(n * v_dim));
        let mut state_out = gpu.alloc(state.len());
        gpu.conv(&mut q, &mut k, &mut v, &mut state_out, &buffer(gpu, &x), &buffer(gpu, &state), &buffer(gpu, &weight));
        close(&gpu.read(&q), &want_q, 1e-5);
        close(&gpu.read(&k), &want_k, 1e-5);
        close(&gpu.read(&v), &want_v, 1e-5);
        close(&gpu.read(&state_out), &want_state, 1e-6);
    }
}

pub fn delta_net_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(12);
    let (n_k_heads, n_v_heads, head_dim) = (2, 6, 128);
    for n in [1, 5] {
        let q = rng.floats(n * n_k_heads * head_dim);
        let k = rng.floats(n * n_k_heads * head_dim);
        let v = rng.floats(n * n_v_heads * head_dim);
        let gates = rng.floats(n * 2 * n_v_heads);
        let decay: Vec<f32> = (0..2 * n_v_heads).map(|i| if i < n_v_heads { -rng.next().abs() } else { 4.0 * rng.next() }).collect();
        let state = rng.floats(n_v_heads * head_dim * head_dim);
        let mut want = vec![0.0; v.len()];
        let mut want_state = state.clone();
        Cpu.delta_net(&mut want, &q, &k, &v, &gates, &decay, &mut want_state, n_k_heads, n_v_heads, head_dim);
        let mut out = gpu.alloc(v.len());
        let mut gpu_state = buffer(gpu, &state);
        gpu.delta_net(
            &mut out,
            &buffer(gpu, &q),
            &buffer(gpu, &k),
            &buffer(gpu, &v),
            &buffer(gpu, &gates),
            &buffer(gpu, &decay),
            &mut gpu_state,
            n_k_heads,
            n_v_heads,
            head_dim,
        );
        close(&gpu.read(&out), &want, 1e-4);
        close(&gpu.read(&gpu_state), &want_state, 1e-4);
    }
}

pub fn resize_keeps_capacity<D: Device>(gpu: &D) {
    let mut buf = gpu.alloc(64);
    gpu.resize(&mut buf, 16);
    assert_eq!(gpu.read(&buf).len(), 16);
    gpu.resize(&mut buf, 64);
    assert_eq!(gpu.read(&buf).len(), 64);
}
