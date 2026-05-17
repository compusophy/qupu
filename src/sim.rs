//! The QuPU simulator.
//!
//! There is no separate "backend" here. The fields that the renderer reads
//! (state vector, per-core busy timers, scheduler queue) are the same fields
//! the gate kernels write — the visualization *is* the simulation.
//!
//! Scheduling model: each gate is split into K tiles aligned to 2^(target+1)
//! amplitudes. Tiles are pushed onto a queue; idle cores pop one per tick
//! and "occupy" themselves for `1 / clock_hz` seconds of wall time before
//! the kernel actually mutates the state and the core returns to idle.
//! Because the alignment rule means tiles are independent, completion order
//! does not affect correctness.

use crate::circuit::{expand, matrix, Op};

/// One simulated quantum compute core.
#[derive(Debug, Clone)]
pub struct Core {
    pub id: u32,
    /// Wall-time seconds until this core finishes its current tile (0 = idle).
    pub busy_for: f32,
    /// Wall-time since this core last finished a tile (used for fade-out).
    pub idle_for: f32,
    /// Total seconds this core spent computing across all gates.
    pub total_busy: f64,
    /// Most recent tile this core was assigned (for hover/labeling).
    pub last_tile: Option<AssignedTile>,
}

#[derive(Debug, Clone, Copy)]
pub struct AssignedTile {
    pub op: Op,
    pub base_index: u32,
    pub n_amps: u32,
}

/// A tile that has not yet been picked up by a core.
#[derive(Debug, Clone, Copy)]
pub struct PendingTile {
    pub op: Op,
    pub base_index: u32,
    pub n_amps: u32,
}

pub struct Simulator {
    pub n_qubits: u32,
    pub n_cores: u32,
    pub state: Vec<f64>,
    pub cores: Vec<Core>,
    pub circuit: Vec<Op>,
    /// Index into `circuit` of the next op to schedule.
    pub pc: usize,
    /// Whether to keep scheduling new ops automatically.
    pub running: bool,
    pub queue: Vec<PendingTile>,
    /// Tiles per second a single core processes (controls animation pace).
    pub clock_hz: f32,
    pub gates_completed: u64,
    pub tiles_completed: u64,
    pub wall_seconds: f64,
}

impl Simulator {
    pub fn new(n_qubits: u32, n_cores: u32) -> Self {
        let n_amps = 1usize << n_qubits;
        let mut state = vec![0.0; n_amps * 2];
        state[0] = 1.0;
        let n_cores = highest_pow2_at_most(n_cores).max(1).min(n_amps as u32);
        let cores = (0..n_cores).map(|id| Core {
            id, busy_for: 0.0, idle_for: 1e9, total_busy: 0.0, last_tile: None,
        }).collect();
        Self {
            n_qubits, n_cores, state, cores,
            circuit: Vec::new(),
            pc: 0, running: false,
            queue: Vec::new(),
            clock_hz: 8.0,
            gates_completed: 0,
            tiles_completed: 0,
            wall_seconds: 0.0,
        }
    }

    pub fn n_amps(&self) -> u32 { 1u32 << self.n_qubits }

    pub fn reset_state(&mut self) {
        for v in self.state.iter_mut() { *v = 0.0; }
        self.state[0] = 1.0;
    }

    pub fn clear(&mut self) {
        self.reset_state();
        self.circuit.clear();
        self.pc = 0;
        self.running = false;
        self.queue.clear();
        for c in &mut self.cores {
            c.busy_for = 0.0;
            c.idle_for = 1e9;
            c.total_busy = 0.0;
            c.last_tile = None;
        }
        self.gates_completed = 0;
        self.tiles_completed = 0;
        self.wall_seconds = 0.0;
    }

    pub fn norm_sq(&self) -> f64 {
        self.state.chunks_exact(2).map(|c| c[0]*c[0] + c[1]*c[1]).sum()
    }

    pub fn set_circuit(&mut self, ops: Vec<Op>) {
        self.circuit = expand(&ops);
        self.pc = 0;
        self.reset_state();
        self.gates_completed = 0;
        self.tiles_completed = 0;
        self.wall_seconds = 0.0;
    }

    pub fn append(&mut self, op: Op) {
        for o in expand(&[op]) { self.circuit.push(o); }
    }

    pub fn start_running(&mut self) {
        self.running = true;
    }

    pub fn stop_running(&mut self) {
        self.running = false;
    }

    pub fn step_once(&mut self) {
        if self.pc < self.circuit.len() && self.queue.is_empty() && self.all_idle() {
            self.schedule_next();
        }
    }

    pub fn all_idle(&self) -> bool {
        self.cores.iter().all(|c| c.busy_for <= 0.0)
    }

    fn schedule_next(&mut self) {
        if self.pc >= self.circuit.len() { return; }
        let op = self.circuit[self.pc];
        self.pc += 1;
        // tile partitioning
        let target = op.target;
        let align: u32 = 1u32 << (target + 1);
        let max_tiles = self.n_amps() / align; // power of two
        let n_tiles = self.n_cores.min(max_tiles).max(1);
        let tile_amps = self.n_amps() / n_tiles;
        for t in 0..n_tiles {
            self.queue.push(PendingTile {
                op,
                base_index: t * tile_amps,
                n_amps: tile_amps,
            });
        }
    }

    /// Advance the wall clock by `dt` seconds. Drives core completion,
    /// dispatch from the queue, and (if `running`) auto-schedule of the
    /// next gate when the prior one fully drains.
    pub fn tick(&mut self, dt: f32) {
        self.wall_seconds += dt as f64;

        // 1. Advance busy cores; if any complete, apply their tile to state.
        let mut completed: Vec<(usize, AssignedTile)> = Vec::new();
        for (i, core) in self.cores.iter_mut().enumerate() {
            if core.busy_for > 0.0 {
                core.busy_for -= dt;
                core.total_busy += dt as f64;
                if core.busy_for <= 0.0 {
                    core.busy_for = 0.0;
                    core.idle_for = 0.0;
                    if let Some(tile) = core.last_tile {
                        completed.push((i, tile));
                    }
                }
            } else {
                core.idle_for += dt;
            }
        }
        for (_, tile) in completed {
            self.apply_tile(tile);
            self.tiles_completed += 1;
            if self.queue.is_empty() && self.all_idle() {
                self.gates_completed += 1;
            }
        }

        // 2. Idle cores pick up pending tiles (in order).
        let per_tile = (1.0 / self.clock_hz).max(0.001);
        for core in &mut self.cores {
            if core.busy_for <= 0.0 {
                if let Some(p) = self.queue.first().copied() {
                    self.queue.remove(0);
                    core.busy_for = per_tile;
                    core.idle_for = 0.0;
                    core.last_tile = Some(AssignedTile {
                        op: p.op, base_index: p.base_index, n_amps: p.n_amps,
                    });
                }
            }
        }

        // 3. Auto-advance to the next gate when current drains.
        if self.running && self.queue.is_empty() && self.all_idle()
            && self.pc < self.circuit.len()
        {
            self.schedule_next();
        }
    }

    fn apply_tile(&mut self, tile: AssignedTile) {
        let op = tile.op;
        let base = tile.base_index as usize;
        let len = tile.n_amps as usize;
        let slice = &mut self.state[base * 2 .. (base + len) * 2];
        if op.gate.is_2q() {
            apply_c1q(slice, op.ctrl, op.target, tile.base_index, &matrix(op));
        } else {
            apply_1q(slice, op.target, tile.base_index, &matrix(op));
        }
    }
}

fn highest_pow2_at_most(n: u32) -> u32 {
    if n == 0 { return 1; }
    let mut p = 1;
    while p * 2 <= n { p *= 2; }
    p
}

// --- Kernels (operate on a tile, in-place) ---

pub fn apply_1q(slice: &mut [f64], target: u32, base_index: u32, m: &[f64; 8]) {
    let tmask = 1usize << target;
    let n_amps = slice.len() / 2;
    let base = base_index as usize;
    let (ar, ai, br, bi) = (m[0], m[1], m[2], m[3]);
    let (cr, ci, dr, di) = (m[4], m[5], m[6], m[7]);
    let mut i = 0usize;
    while i < n_amps {
        if ((base + i) & tmask) == 0 {
            let i0 = i * 2;
            let i1 = (i + tmask) * 2;
            let (x0r, x0i) = (slice[i0], slice[i0 + 1]);
            let (x1r, x1i) = (slice[i1], slice[i1 + 1]);
            slice[i0]     = ar*x0r - ai*x0i + br*x1r - bi*x1i;
            slice[i0 + 1] = ar*x0i + ai*x0r + br*x1i + bi*x1r;
            slice[i1]     = cr*x0r - ci*x0i + dr*x1r - di*x1i;
            slice[i1 + 1] = cr*x0i + ci*x0r + dr*x1i + di*x1r;
        }
        i += 1;
    }
}

pub fn apply_c1q(slice: &mut [f64], ctrl: u32, target: u32, base_index: u32, m: &[f64; 8]) {
    let tmask = 1usize << target;
    let cmask = 1usize << ctrl;
    let n_amps = slice.len() / 2;
    let base = base_index as usize;
    let (ar, ai, br, bi) = (m[0], m[1], m[2], m[3]);
    let (cr, ci, dr, di) = (m[4], m[5], m[6], m[7]);
    let mut i = 0usize;
    while i < n_amps {
        let g = base + i;
        if (g & tmask) == 0 && (g & cmask) != 0 {
            let i0 = i * 2;
            let i1 = (i + tmask) * 2;
            let (x0r, x0i) = (slice[i0], slice[i0 + 1]);
            let (x1r, x1i) = (slice[i1], slice[i1 + 1]);
            slice[i0]     = ar*x0r - ai*x0i + br*x1r - bi*x1i;
            slice[i0 + 1] = ar*x0i + ai*x0r + br*x1i + bi*x1r;
            slice[i1]     = cr*x0r - ci*x0i + dr*x1r - di*x1i;
            slice[i1 + 1] = cr*x0i + ci*x0r + dr*x1i + di*x1r;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{GateKind, Op};

    #[test]
    fn bell_state() {
        let mut s = Simulator::new(2, 2);
        s.set_circuit(vec![
            Op::one(GateKind::H, 0),
            Op::two(GateKind::CNOT, 0, 1),
        ]);
        s.start_running();
        // tick generously until all done
        for _ in 0..10_000 {
            s.tick(0.01);
            if !s.running { break; }
            if s.pc >= s.circuit.len() && s.queue.is_empty() && s.all_idle() { break; }
        }
        let p = |i: usize| s.state[2*i]*s.state[2*i] + s.state[2*i+1]*s.state[2*i+1];
        assert!((p(0) - 0.5).abs() < 1e-12);
        assert!(p(1).abs() < 1e-12);
        assert!(p(2).abs() < 1e-12);
        assert!((p(3) - 0.5).abs() < 1e-12);
        assert!((s.norm_sq() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tiled_h_matches_single() {
        let m_h = matrix(Op::one(GateKind::H, 1));
        let n_amps = 8;
        let mut full = vec![0.0; n_amps * 2];
        full[0] = 1.0;
        let mut tiled = full.clone();
        apply_1q(&mut full, 1, 0, &m_h);
        let (left, right) = tiled.split_at_mut(8);
        apply_1q(left, 1, 0, &m_h);
        apply_1q(right, 1, 4, &m_h);
        for (a, b) in full.iter().zip(tiled.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
