/// Picks the next token from a model's logits: from the `top_k` most likely
/// tokens, then the smallest set of those whose probability reaches `top_p`,
/// at `temperature`. A temperature of zero always picks the most likely token.
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: Rng(seed.max(1)),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let candidates = self.candidates(logits);
        let r = self.rng.next();
        pick(&candidates, r)
    }

    /// Takes a drafted token or not: the token itself with the probability
    /// the sampler would have picked it, and otherwise one picked from the
    /// rest by their probabilities, which leaves the tokens taken
    /// distributed as `sample` would have picked them. Nothing if the draft
    /// was the only candidate and was not taken, which cannot happen.
    pub fn accept(&mut self, logits: &[f32], draft: u32) -> Option<u32> {
        let candidates = self.candidates(logits);
        let p = candidates.iter().find(|&&(token, _)| token == draft).map_or(0.0, |&(_, p)| p);
        if p >= 1.0 || self.rng.next() < p {
            return Some(draft);
        }
        let rest: Vec<(u32, f32)> = candidates.into_iter().filter(|&(token, _)| token != draft).collect();
        let total: f32 = rest.iter().map(|&(_, p)| p).sum();
        if total <= 0.0 {
            return None;
        }
        let r = self.rng.next() * total;
        Some(pick(&rest, r))
    }

    /// The tokens the sampler may pick, with their probabilities, which add
    /// up to one: the top k by logit, in one pass over the logits (a
    /// vocabulary is a quarter of a million entries, and k is twenty), at
    /// the temperature, cut to the most likely up to a total of top p. At
    /// temperature zero, the top one alone.
    fn candidates(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let k = self.top_k.clamp(1, logits.len());
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (token, &logit) in logits.iter().enumerate() {
            if candidates.len() == k && logit <= candidates[k - 1].1 {
                continue;
            }
            let at = candidates.partition_point(|&(_, l)| l >= logit);
            candidates.insert(at, (token as u32, logit));
            candidates.truncate(k);
        }
        if self.temperature == 0.0 {
            return vec![(candidates[0].0, 1.0)];
        }
        let max = candidates[0].1;
        for (_, logit) in &mut candidates {
            *logit = ((*logit - max) / self.temperature).exp();
        }
        let sum: f32 = candidates.iter().map(|&(_, p)| p).sum();
        let mut kept = 0.0;
        let mut n = 0;
        while n < candidates.len() && kept < self.top_p {
            kept += candidates[n].1 / sum;
            n += 1;
        }
        candidates.truncate(n);
        for (_, p) in &mut candidates {
            *p /= sum * kept;
        }
        candidates
    }
}

/// xorshift64*, uniform in [0, 1).
/// The candidate at `r` in [0, 1) along their probabilities.
fn pick(candidates: &[(u32, f32)], r: f32) -> u32 {
    let mut acc = 0.0;
    for &(token, p) in candidates {
        acc += p;
        if r < acc {
            return token;
        }
    }
    candidates[candidates.len() - 1].0
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (bits >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_from_the_top_k_in_order() {
        let mut logits = vec![0.0; 1000];
        logits[7] = 5.0;
        logits[300] = 4.0;
        logits[999] = 3.0;
        let mut greedy = Sampler::new(0.0, 20, 0.95, 1);
        assert_eq!(greedy.sample(&logits), 7);
        // At a low temperature the top token is all but certain, and with
        // top_p of nearly nothing it is the only one kept.
        let mut sampler = Sampler::new(1.0, 3, 0.01, 1);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits), 7);
        }
        // With k of two, the third never comes up.
        let mut sampler = Sampler::new(100.0, 2, 1.0, 1);
        let picks: std::collections::HashSet<u32> = (0..200).map(|_| sampler.sample(&logits)).collect();
        assert!(picks.contains(&7) && picks.contains(&300) && !picks.contains(&999));
    }
}
