//! Gate types, gate matrices, and preset circuits.
//!
//! A `Op` is a single instruction in a quantum program. Gate matrices are
//! returned as `[a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im]` row-major.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    H, X, Y, Z, S, T,
    RX, RY, RZ,
    CNOT, CZ, SWAP,
}

impl GateKind {
    pub fn label(self) -> &'static str {
        match self {
            GateKind::H => "H", GateKind::X => "X", GateKind::Y => "Y", GateKind::Z => "Z",
            GateKind::S => "S", GateKind::T => "T",
            GateKind::RX => "RX", GateKind::RY => "RY", GateKind::RZ => "RZ",
            GateKind::CNOT => "CNOT", GateKind::CZ => "CZ", GateKind::SWAP => "SWAP",
        }
    }
    pub fn is_2q(self) -> bool {
        matches!(self, GateKind::CNOT | GateKind::CZ | GateKind::SWAP)
    }
    pub fn is_rot(self) -> bool {
        matches!(self, GateKind::RX | GateKind::RY | GateKind::RZ)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Op {
    pub gate: GateKind,
    pub target: u32,
    pub ctrl: u32,   // unused for 1q
    pub theta: f64,  // unused for non-rot
}

impl Op {
    pub fn one(g: GateKind, target: u32) -> Self {
        Self { gate: g, target, ctrl: 0, theta: 0.0 }
    }
    pub fn rot(g: GateKind, target: u32, theta: f64) -> Self {
        Self { gate: g, target, ctrl: 0, theta }
    }
    pub fn two(g: GateKind, ctrl: u32, target: u32) -> Self {
        Self { gate: g, target, ctrl, theta: 0.0 }
    }
}

/// Return the row-major 2x2 complex matrix for a 1-qubit gate (or for the
/// "target half" of a 2-qubit gate). For SWAP returns identity — SWAP is
/// expanded into three CNOTs upstream.
pub fn matrix(op: Op) -> [f64; 8] {
    match op.gate {
        GateKind::H => [FRAC_1_SQRT_2, 0.0,  FRAC_1_SQRT_2, 0.0,
                        FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2, 0.0],
        GateKind::X | GateKind::CNOT => [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        GateKind::Y => [0.0, 0.0,  0.0, -1.0,  0.0, 1.0,  0.0, 0.0],
        GateKind::Z | GateKind::CZ => [1.0, 0.0,  0.0, 0.0,  0.0, 0.0, -1.0, 0.0],
        GateKind::S => [1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 1.0],
        GateKind::T => {
            let (c, s) = ((PI / 4.0).cos(), (PI / 4.0).sin());
            [1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  c, s]
        }
        GateKind::RX => {
            let (c, s) = ((op.theta / 2.0).cos(), (op.theta / 2.0).sin());
            [c, 0.0,  0.0, -s,  0.0, -s,  c, 0.0]
        }
        GateKind::RY => {
            let (c, s) = ((op.theta / 2.0).cos(), (op.theta / 2.0).sin());
            [c, 0.0,  -s, 0.0,  s, 0.0,  c, 0.0]
        }
        GateKind::RZ => {
            let (ch, sh) = ((op.theta / 2.0).cos(), (op.theta / 2.0).sin());
            [ch, -sh,  0.0, 0.0,  0.0, 0.0,  ch, sh]
        }
        GateKind::SWAP => [1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0],
    }
}

/// Expand a `SWAP` into three CNOTs; pass other ops through unchanged.
pub fn expand(ops: &[Op]) -> Vec<Op> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        if op.gate == GateKind::SWAP {
            out.push(Op::two(GateKind::CNOT, op.ctrl, op.target));
            out.push(Op::two(GateKind::CNOT, op.target, op.ctrl));
            out.push(Op::two(GateKind::CNOT, op.ctrl, op.target));
        } else {
            out.push(*op);
        }
    }
    out
}

// --- presets ---

pub fn preset_bell(_n: u32) -> Vec<Op> {
    vec![Op::one(GateKind::H, 0), Op::two(GateKind::CNOT, 0, 1)]
}

pub fn preset_ghz(n: u32) -> Vec<Op> {
    let mut ops = vec![Op::one(GateKind::H, 0)];
    for i in 1..n { ops.push(Op::two(GateKind::CNOT, 0, i)); }
    ops
}

pub fn preset_qft(n: u32) -> Vec<Op> {
    let mut ops = Vec::new();
    for i in 0..n {
        ops.push(Op::one(GateKind::H, i));
        for j in (i + 1)..n {
            let theta = PI / (1u64 << (j - i)) as f64;
            // CRZ(theta) decomposed as RZ(theta/2)·CNOT·RZ(-theta/2)·CNOT
            ops.push(Op::rot(GateKind::RZ, j, theta / 2.0));
            ops.push(Op::two(GateKind::CNOT, i, j));
            ops.push(Op::rot(GateKind::RZ, j, -theta / 2.0));
            ops.push(Op::two(GateKind::CNOT, i, j));
        }
    }
    for i in 0..(n / 2) {
        ops.push(Op::two(GateKind::SWAP, i, n - 1 - i));
    }
    ops
}

pub fn preset_grover(n: u32) -> Vec<Op> {
    let mut ops = Vec::new();
    // Uniform superposition.
    for i in 0..n { ops.push(Op::one(GateKind::H, i)); }
    // Oracle marks |11...1>.
    push_multi_cz(&mut ops, n);
    // Diffuser
    for i in 0..n { ops.push(Op::one(GateKind::H, i)); }
    for i in 0..n { ops.push(Op::one(GateKind::X, i)); }
    push_multi_cz(&mut ops, n);
    for i in 0..n { ops.push(Op::one(GateKind::X, i)); }
    for i in 0..n { ops.push(Op::one(GateKind::H, i)); }
    ops
}

fn push_multi_cz(ops: &mut Vec<Op>, n: u32) {
    if n < 2 { return; }
    if n == 2 {
        ops.push(Op::two(GateKind::CZ, 0, 1));
        return;
    }
    // Rough proxy for larger n — a CZ ladder. (For exact multi-CZ we'd want
    // a Toffoli decomposition; this still produces visible structure.)
    for i in 0..(n - 1) {
        ops.push(Op::two(GateKind::CZ, i, n - 1));
    }
}

pub fn preset_scramble(n: u32) -> Vec<Op> {
    let mut ops = Vec::new();
    let onesie = [GateKind::H, GateKind::X, GateKind::Y, GateKind::S, GateKind::T];
    for layer in 0..3 {
        for i in 0..n {
            let g = onesie[((i + layer) as usize) % onesie.len()];
            ops.push(Op::one(g, i));
        }
        let mut i = 0;
        while i + 1 < n { ops.push(Op::two(GateKind::CNOT, i, i + 1)); i += 2; }
        let mut i = 1;
        while i + 1 < n { ops.push(Op::two(GateKind::CNOT, i, i + 1)); i += 2; }
    }
    ops
}

pub const PRESETS: &[(&str, fn(u32) -> Vec<Op>)] = &[
    ("Bell",     preset_bell),
    ("GHZ",      preset_ghz),
    ("QFT",      preset_qft),
    ("Grover",   preset_grover),
    ("Scramble", preset_scramble),
];
