// QuPU kernel — quantum state-vector simulator compiled to WebAssembly.
//
// State vector layout: interleaved f64 pairs [re0, im0, re1, im1, ...] of
// length 2 * 2^n_qubits. The vector lives on the JS side (typed-array view
// over a SharedArrayBuffer); each Web Worker passes its own tile of the
// vector into the kernel, so multiple workers can apply the same gate to
// disjoint partitions in parallel — mimicking how a GPU dispatches a
// kernel across thread blocks.
//
// Partitioning rule: for a 1-qubit gate on `target`, slices must be aligned
// to (and multiples of) 2^(target+1) amplitudes. This guarantees that
// amplitude pairs (i, i ^ (1<<target)) never straddle a slice boundary,
// so workers never touch each other's memory.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct QKernel {
    n_qubits: u32,
}

#[wasm_bindgen]
impl QKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(n_qubits: u32) -> Self {
        Self { n_qubits }
    }

    #[wasm_bindgen(getter)]
    pub fn n_qubits(&self) -> u32 { self.n_qubits }

    #[wasm_bindgen(getter)]
    pub fn n_amplitudes(&self) -> u32 { 1u32 << self.n_qubits }

    /// Reset slice to |0...0>: this is only meaningful when called on the
    /// full state. We accept a slice + a flag for the global index 0.
    pub fn init_zero(&self, slice: &mut [f64], contains_zero: bool) {
        for v in slice.iter_mut() { *v = 0.0; }
        if contains_zero {
            slice[0] = 1.0;
        }
    }

    /// Apply a 1-qubit gate to a tile of the state vector.
    ///
    /// `slice` is the interleaved (re, im) amplitudes for absolute indices
    /// [base_index, base_index + slice.len()/2). Both `base_index` and the
    /// number of amplitudes must be a multiple of 2^(target+1).
    ///
    /// `matrix` = [a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im] for
    /// gate [[a,b],[c,d]].
    pub fn apply_1q(
        &self,
        slice: &mut [f64],
        target: u32,
        base_index: u32,
        matrix: &[f64],
    ) {
        let tmask = 1usize << target;
        let n_amps = slice.len() / 2;
        let base = base_index as usize;
        let (a_re, a_im) = (matrix[0], matrix[1]);
        let (b_re, b_im) = (matrix[2], matrix[3]);
        let (c_re, c_im) = (matrix[4], matrix[5]);
        let (d_re, d_im) = (matrix[6], matrix[7]);

        let mut i = 0usize;
        while i < n_amps {
            if ((base + i) & tmask) == 0 {
                let i0 = i * 2;
                let i1 = (i + tmask) * 2;
                let x0r = slice[i0];     let x0i = slice[i0 + 1];
                let x1r = slice[i1];     let x1i = slice[i1 + 1];

                let n0r = a_re * x0r - a_im * x0i + b_re * x1r - b_im * x1i;
                let n0i = a_re * x0i + a_im * x0r + b_re * x1i + b_im * x1r;
                let n1r = c_re * x0r - c_im * x0i + d_re * x1r - d_im * x1i;
                let n1i = c_re * x0i + c_im * x0r + d_re * x1i + d_im * x1r;

                slice[i0]     = n0r; slice[i0 + 1] = n0i;
                slice[i1]     = n1r; slice[i1 + 1] = n1i;
            }
            i += 1;
        }
    }

    /// Controlled 1-qubit gate. Only updates pairs where the `ctrl` bit is 1.
    /// Same alignment rule as `apply_1q` (based on `target`); `ctrl` does not
    /// affect alignment because both members of each pair share the same
    /// ctrl-bit value.
    pub fn apply_c1q(
        &self,
        slice: &mut [f64],
        ctrl: u32,
        target: u32,
        base_index: u32,
        matrix: &[f64],
    ) {
        let tmask = 1usize << target;
        let cmask = 1usize << ctrl;
        let n_amps = slice.len() / 2;
        let base = base_index as usize;
        let (a_re, a_im) = (matrix[0], matrix[1]);
        let (b_re, b_im) = (matrix[2], matrix[3]);
        let (c_re, c_im) = (matrix[4], matrix[5]);
        let (d_re, d_im) = (matrix[6], matrix[7]);

        let mut i = 0usize;
        while i < n_amps {
            let g = base + i;
            if (g & tmask) == 0 && (g & cmask) != 0 {
                let i0 = i * 2;
                let i1 = (i + tmask) * 2;
                let x0r = slice[i0];     let x0i = slice[i0 + 1];
                let x1r = slice[i1];     let x1i = slice[i1 + 1];

                let n0r = a_re * x0r - a_im * x0i + b_re * x1r - b_im * x1i;
                let n0i = a_re * x0i + a_im * x0r + b_re * x1i + b_im * x1r;
                let n1r = c_re * x0r - c_im * x0i + d_re * x1r - d_im * x1i;
                let n1i = c_re * x0i + c_im * x0r + d_re * x1i + d_im * x1r;

                slice[i0]     = n0r; slice[i0 + 1] = n0i;
                slice[i1]     = n1r; slice[i1 + 1] = n1i;
            }
            i += 1;
        }
    }

    /// Compute |amp|^2 for each amplitude in slice into `probs`.
    pub fn probabilities(&self, slice: &[f64], probs: &mut [f64]) {
        let n_amps = slice.len() / 2;
        for i in 0..n_amps {
            let r = slice[2 * i];
            let im = slice[2 * i + 1];
            probs[i] = r * r + im * im;
        }
    }

    /// Norm squared sum (used to validate unitarity / detect drift).
    pub fn norm_sq(&self, slice: &[f64]) -> f64 {
        let n_amps = slice.len() / 2;
        let mut s = 0.0_f64;
        for i in 0..n_amps {
            let r = slice[2 * i];
            let im = slice[2 * i + 1];
            s += r * r + im * im;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT2_INV: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn h_matrix() -> [f64; 8] {
        [ROOT2_INV, 0.0,  ROOT2_INV, 0.0,
         ROOT2_INV, 0.0, -ROOT2_INV, 0.0]
    }
    fn x_matrix() -> [f64; 8] {
        [0.0, 0.0,  1.0, 0.0,
         1.0, 0.0,  0.0, 0.0]
    }

    fn fresh(n: u32) -> (QKernel, Vec<f64>) {
        let k = QKernel::new(n);
        let n_amps = (1usize) << n;
        let mut state = vec![0.0; n_amps * 2];
        k.init_zero(&mut state, true);
        (k, state)
    }

    #[test]
    fn init_is_pure_zero() {
        let (k, state) = fresh(3);
        assert!((state[0] - 1.0).abs() < 1e-12);
        assert!((k.norm_sq(&state) - 1.0).abs() < 1e-12);
        for (i, v) in state.iter().enumerate() {
            if i == 0 { continue; }
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn h_on_q0_gives_plus_state() {
        let (k, mut state) = fresh(2);
        let h = h_matrix();
        k.apply_1q(&mut state, 0, 0, &h);
        // expect (|00> + |10>)/sqrt(2)... wait, H on q0 of |00> →
        //   (|0>+|1>)/√2 ⊗ |0> = (|00> + |01>)/√2 in little-endian indexing
        //   where amp index = (q_{n-1} q_{n-2} ... q_0) binary.
        // With our convention bit i of index is qubit i, so index 1 has q0=1.
        // |00>=idx0, q0=1 → idx1. So result: amp[0] = amp[1] = 1/√2, rest 0.
        assert!((state[0] - ROOT2_INV).abs() < 1e-12);
        assert!((state[2] - ROOT2_INV).abs() < 1e-12);
        assert!((k.norm_sq(&state) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bell_state_via_h_then_cnot() {
        let (k, mut state) = fresh(2);
        let h = h_matrix();
        let x = x_matrix();
        k.apply_1q(&mut state, 0, 0, &h);
        k.apply_c1q(&mut state, 0, 1, 0, &x);
        // Bell state: (|00> + |11>)/√2.
        // |00>=idx0, |11>=idx3 (q0=1 bit0=1, q1=1 bit1=1 → 0b11=3)
        let mut probs = vec![0.0; 4];
        k.probabilities(&state, &mut probs);
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!(probs[1].abs() < 1e-12);
        assert!(probs[2].abs() < 1e-12);
        assert!((probs[3] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ghz_state_three_qubits() {
        let (k, mut state) = fresh(3);
        let h = h_matrix();
        let x = x_matrix();
        k.apply_1q(&mut state, 0, 0, &h);
        k.apply_c1q(&mut state, 0, 1, 0, &x);
        k.apply_c1q(&mut state, 0, 2, 0, &x);
        let mut probs = vec![0.0; 8];
        k.probabilities(&state, &mut probs);
        // expect 0.5 at index 0 (|000>) and 0.5 at index 7 (|111>)
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!((probs[7] - 0.5).abs() < 1e-12);
        for i in 1..7 { assert!(probs[i].abs() < 1e-12); }
    }

    #[test]
    fn tiled_h_matches_unsharded() {
        // Apply H to q1 on full state vs same gate on 2 tiles aligned to 2^(1+1)=4.
        // For n=3 (8 amps), tile size = 4 amps, two tiles: [0,4) and [4,8).
        let (k, mut state_full) = fresh(3);
        let (k2, mut state_tiled) = fresh(3);
        let h = h_matrix();
        k.apply_1q(&mut state_full, 1, 0, &h);

        // tile 0: indices 0..4
        let (left, right) = state_tiled.split_at_mut(8); // 4 amps * 2 floats
        k2.apply_1q(left, 1, 0, &h);
        k2.apply_1q(right, 1, 4, &h);

        for i in 0..state_full.len() {
            assert!((state_full[i] - state_tiled[i]).abs() < 1e-12,
                    "mismatch at {}: {} vs {}", i, state_full[i], state_tiled[i]);
        }
    }
}
