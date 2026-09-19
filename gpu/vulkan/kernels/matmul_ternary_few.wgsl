// out[t][r] = w[r] · x[t] for ternary weights and a few tokens, up to
// eight: as `matmul_ternary.wgsl`, one workgroup per eight rows in eight
// groups of eight threads, each group taking every eighth block and each
// thread a word of it, looked up byte by byte in the table of trits, but
// with the tokens' activations already half floats packed two tokens to a
// word by `pack_halves.wgsl`, so that a trit multiplies into two tokens
// with one packed half-precision multiply-add. A block's products are
// summed in half precision and added to the totals, scaled, in single
// precision. The weights are read once for all the tokens, which is what
// makes verifying a draft of tokens cheaper than decoding them one by one.

enable f16;

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid, for matrices with more rows
    // than one dimension of the grid allows.
    stride: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
// The activations as `pack_halves.wgsl` lays them out for up to eight
// tokens: element k's four pairs of tokens as one vector.
@group(0) @binding(2) var<storage, read> x: array<vec4<u32>>;

// Threads and rows per workgroup, and the most tokens.
const THREADS: u32 = 64u;
const ROWS: u32 = 8u;
const TOKENS: u32 = 8u;

// Words per block: 28 bytes.
const WORDS: u32 = 7u;

// Each subgroup's sums of the rows and tokens, for the workgroup to add up.
var<workgroup> partial: array<f32, 4u * ROWS * TOKENS>;

// The trits of each byte value, as `matmul_ternary.wgsl` builds them.
var<workgroup> table: array<vec4<u32>, 256>;

fn build(lid: u32) {
    for (var v = lid; v < 256u; v += THREADS) {
        var q = v;
        var t: array<f32, 5>;
        for (var n = 0u; n < 5u; n++) {
            t[n] = f32((q * 3u) >> 8u) - 1.0;
            q = (q * 3u) & 0xffu;
        }
        table[v] = vec4(pack2x16float(vec2(t[0], t[1])), pack2x16float(vec2(t[2], t[3])), pack2x16float(vec2(t[4], 0.0)), 0u);
    }
}

// The element of a block that trit `n` of byte `i` of word `j` belongs to.
fn element(j: u32, i: u32, n: u32) -> u32 {
    if j < 4u {
        return 4u * j + i + 16u * n;
    } else if j < 6u {
        return 80u + 4u * (j - 4u) + i + 8u * n;
    }
    return 120u + i + 2u * n;
}

// The four pairs of tokens of an element, as halves.
fn pairs(v: vec4<u32>) -> array<vec2<f16>, 4> {
    return array(vec2<f16>(unpack2x16float(v.x)), vec2<f16>(unpack2x16float(v.y)), vec2<f16>(unpack2x16float(v.z)), vec2<f16>(unpack2x16float(v.w)));
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let r0 = (wg.y * p.stride + wg.x) * ROWS;
    let blocks = p.cols / 128u;
    let groups = THREADS / 8u;
    let group = lid / 8u;
    let j = lid % 8u;
    build(lid);
    workgroupBarrier();
    // The totals over the blocks, a row's tokens each, in single precision.
    var acc: array<array<f32, TOKENS>, ROWS>;
    for (var b = group; b < blocks; b += groups) {
        // The activations of the word's elements: byte i's trit n goes with
        // element(j, i, n), whose four pairs of tokens are one vector.
        var xs: array<array<array<vec2<f16>, 4>, 4>, 5>;
        for (var n = 0u; n < 5u; n++) {
            for (var i = 0u; i < 4u; i++) {
                xs[n][i] = pairs(x[b * 128u + element(min(j, 6u), i, n)]);
            }
        }
        // The words of all eight rows first, so that their loads are in
        // flight together. A row past the matrix reads the last row
        // instead, in bounds, and its sum goes nowhere. The scale is in
        // the last word, which the group's seventh lane has.
        var words: array<u32, ROWS>;
        var scales: array<f32, ROWS>;
        for (var r = 0u; r < ROWS; r++) {
            let wbase = (b * p.rows + min(r0 + r, p.rows - 1u)) * WORDS;
            words[r] = w[wbase + min(j, 6u)];
        }
        for (var r = 0u; r < ROWS; r++) {
            scales[r] = unpack2x16float(subgroupShuffle(words[r], (sinv & ~7u) + 6u)).y;
        }
        for (var r = 0u; r < ROWS; r++) {
            let word = words[r];
            // The block's products for the row, a pair of tokens to a sum.
            var sum: array<vec2<f16>, 4>;
            if j < 6u {
                for (var i = 0u; i < 4u; i++) {
                    let e = table[(word >> (8u * i)) & 0xffu];
                    let t01 = vec2<f16>(unpack2x16float(e.x));
                    let t23 = vec2<f16>(unpack2x16float(e.y));
                    let t4 = vec2<f16>(unpack2x16float(e.z)).x;
                    let trits = array(t01.x, t01.y, t23.x, t23.y, t4);
                    for (var n = 0u; n < 5u; n++) {
                        let t = vec2<f16>(trits[n]);
                        for (var q = 0u; q < 4u; q++) {
                            sum[q] = fma(t, xs[n][i][q], sum[q]);
                        }
                    }
                }
            } else if j == 6u {
                // The low two bytes of the last word hold four trits each.
                for (var i = 0u; i < 2u; i++) {
                    let e = table[(word >> (8u * i)) & 0xffu];
                    let t01 = vec2<f16>(unpack2x16float(e.x));
                    let t23 = vec2<f16>(unpack2x16float(e.y));
                    let trits = array(t01.x, t01.y, t23.x, t23.y);
                    for (var n = 0u; n < 4u; n++) {
                        let t = vec2<f16>(trits[n]);
                        for (var q = 0u; q < 4u; q++) {
                            sum[q] = fma(t, xs[n][i][q], sum[q]);
                        }
                    }
                }
            }
            for (var q = 0u; q < 4u; q++) {
                acc[r][2u * q] += scales[r] * f32(sum[q].x);
                acc[r][2u * q + 1u] += scales[r] * f32(sum[q].y);
            }
        }
    }
    // The groups covered every block between them: each subgroup sums its
    // lanes, and the workgroup the subgroups.
    for (var r = 0u; r < ROWS; r++) {
        for (var t = 0u; t < TOKENS; t++) {
            let s = subgroupAdd(acc[r][t]);
            if sinv == 0u {
                partial[(sid * ROWS + r) * TOKENS + t] = s;
            }
        }
    }
    workgroupBarrier();
    if lid < ROWS * TOKENS {
        let r = lid / TOKENS;
        let t = lid % TOKENS;
        if r0 + r < p.rows && t < p.n {
            var total = 0.0;
            for (var i = 0u; i < nsg; i++) {
                total += partial[(i * ROWS + r) * TOKENS + t];
            }
            out[t * p.rows + r0 + r] = total;
        }
    }
}
