//! GPU path for the core's symbol-level simulation (wgpu: Vulkan, Metal,
//! DX12 natively; WebGPU in the browser).
//!
//! One dispatch computes every symbol in an interval in full: it generates
//! the workload in the shader, forms each row's dot product from the
//! effective weights, adds receiver noise, quantises through the ADC,
//! corrects the weight mapping, and accumulates squared error against the
//! ideal product. The CPU keeps the clock, thermal state and ledgers.
//!
//! `cpu_twin` replays the same arithmetic in f64 with the same hash RNG, so
//! the kernel is checked against a reference rather than trusted.

use crate::core::{input_phases, Core, INPUT_NOISE};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Symbols per workgroup; must match `SB` in the shader.
const SB: u32 = 4;
/// Largest dispatch, in symbols.
pub const MAX_SYMBOLS: u32 = 1 << 22;
const MAX_DIM: usize = 64;
const GROUPS_X: u32 = 32768;

const SHADER: &str = r#"
struct Params {
  rows: u32, channels: u32, symbols: u32, seed: u32,
  s_lo: u32, noise_on: u32, groups_x: u32, _pad0: u32,
  levels: f32, sigma: f32, mid: f32, half: f32,
  inv_n: f32, input_noise: f32, _pad1: f32, _pad2: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
// Effective weights, then logical weights (rows × channels each).
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> chan: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>;

const SB: u32 = 4u;
const WG: u32 = 64u;
var<workgroup> xs: array<f32, 256>;
var<workgroup> red: array<f32, 64>;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn unif(a: u32, b: u32, c: u32) -> f32 {
  return f32(pcg(a ^ pcg(b ^ pcg(c))) >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let g = wid.y * P.groups_x + wid.x;
  let first = g * SB;
  let n = P.channels;
  for (var i = lid; i < SB * n; i += WG) {
    let k = first + i / n;
    let j = i % n;
    var x = 0.0;
    if (k < P.symbols) {
      let c = chan[j];
      let kf = f32(k);
      let u = unif(P.seed ^ 0x9e3779b9u, P.s_lo + k, j);
      x = clamp(0.5 + 0.30 * sin(c.x + c.y * kf) + 0.12 * sin(c.z + c.w * kf)
                + P.input_noise * (2.0 * u - 1.0), 0.0, 1.0);
    }
    xs[i] = x;
  }
  workgroupBarrier();
  var err = 0.0;
  let lm1 = P.levels - 1.0;
  for (var p = lid; p < SB * P.rows; p += WG) {
    let kk = p / P.rows;
    let r = p % P.rows;
    let k = first + kk;
    if (k < P.symbols) {
      var acc = 0.0;
      var ideal = 0.0;
      var sx = 0.0;
      let log0 = P.rows * n;
      for (var j = 0u; j < n; j++) {
        let x = xs[kk * n + j];
        acc += w[r * n + j] * x;
        ideal += w[log0 + r * n + j] * x;
        sx += x;
      }
      var meas = acc * P.inv_n;
      if (P.noise_on != 0u) {
        let u1 = max(unif(P.seed, P.s_lo + k, 1000u + 2u * r), 1e-7);
        let u2 = unif(P.seed, P.s_lo + k, 1001u + 2u * r);
        meas += P.sigma * sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
      }
      let q = round((clamp(meas, -1.0, 1.0) + 1.0) * 0.5 * lm1) / lm1 * 2.0 - 1.0;
      let y = (q - P.mid * sx * P.inv_n) / P.half;
      let e = y - ideal * P.inv_n;
      err += e * e;
      if (k + 1u == P.symbols) { y_out[r] = y; }
    }
  }
  red[lid] = err;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partial[g] = red[0]; }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    channels: u32,
    symbols: u32,
    seed: u32,
    s_lo: u32,
    noise_on: u32,
    groups_x: u32,
    _pad0: u32,
    levels: f32,
    sigma: f32,
    mid: f32,
    half: f32,
    inv_n: f32,
    input_noise: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Everything one dispatch needs, captured from the core.
#[derive(Clone)]
pub struct Batch {
    pub rows: usize,
    pub channels: usize,
    pub s0: u64,
    pub symbols: u32,
    pub seed: u32,
    pub noise: bool,
    pub levels: f64,
    pub sigma: f64,
    pub mid: f64,
    pub half: f64,
    pub w_eff: Vec<f64>,
    pub w_log: Vec<f64>,
    pub chan: Vec<[f64; 4]>,
}

impl Batch {
    pub fn from_core(core: &Core, s0: u64, symbols: u32, noise: bool) -> Self {
        let (mid, half) = core.weight_scale();
        Batch {
            rows: core.cfg.rows,
            channels: core.cfg.channels,
            s0,
            symbols: symbols.min(MAX_SYMBOLS),
            seed: (core.cfg.seed as u32) ^ 0xa511_e9b3,
            noise,
            levels: (1u64 << core.cfg.adc_bits) as f64,
            sigma: core.receiver_sigma(),
            mid,
            half,
            w_eff: core.w_eff.clone(),
            w_log: core.logical_w.clone(),
            chan: (0..core.cfg.channels)
                .map(|j| input_phases(core.cfg.baud, j, s0))
                .collect(),
        }
    }

    fn groups(&self) -> u32 {
        self.symbols.div_ceil(SB)
    }
}

/// Result of a dispatch.
#[derive(Debug, Clone)]
pub struct Stats {
    pub symbols: u64,
    pub rows: usize,
    pub err_sq: f64,
    pub y_last: Vec<f64>,
}

impl Stats {
    pub fn outputs(&self) -> f64 {
        self.symbols as f64 * self.rows as f64
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    w: wgpu::Buffer,
    chan: wgpu::Buffer,
    partial: wgpu::Buffer,
    y_out: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
    in_flight: Option<(Batch, Arc<AtomicBool>)>,
    pub adapter: String,
}

impl Gpu {
    pub async fn new() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        // Browsers hide adapter names from pages; say which API it is instead.
        let adapter_name = if info.name.is_empty() {
            format!("{:?}", info.backend)
        } else {
            format!("{} ({:?})", info.name, info.backend)
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("studio"),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("core"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("core"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let w = storage(
            "w",
            (2 * MAX_DIM * MAX_DIM * 4) as u64,
            wgpu::BufferUsages::empty(),
        );
        let chan = storage("chan", (MAX_DIM * 16) as u64, wgpu::BufferUsages::empty());
        let partial_bytes = (MAX_SYMBOLS / SB) as u64 * 4;
        let partial = storage("partial", partial_bytes, wgpu::BufferUsages::COPY_SRC);
        let y_out = storage("y_out", (MAX_DIM * 4) as u64, wgpu::BufferUsages::COPY_SRC);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: partial_bytes + (MAX_DIM * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("core"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y_out.as_entire_binding(),
                },
            ],
        });
        Ok(Gpu {
            device,
            queue,
            pipeline,
            params,
            w,
            chan,
            partial,
            y_out,
            readback,
            bind,
            in_flight: None,
            adapter: adapter_name,
        })
    }

    pub fn busy(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Encode and submit one batch; the readback is mapped asynchronously.
    pub fn submit(&mut self, b: Batch) {
        assert!(b.rows <= MAX_DIM && b.channels <= MAX_DIM && self.in_flight.is_none());
        let groups = b.groups();
        let params = Params {
            rows: b.rows as u32,
            channels: b.channels as u32,
            symbols: b.symbols,
            seed: b.seed,
            s_lo: b.s0 as u32,
            noise_on: b.noise as u32,
            groups_x: GROUPS_X,
            _pad0: 0,
            levels: b.levels as f32,
            sigma: b.sigma as f32,
            mid: b.mid as f32,
            half: b.half as f32,
            inv_n: 1.0 / b.channels as f32,
            input_noise: INPUT_NOISE as f32,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let chan: Vec<f32> = b.chan.iter().flat_map(|c| c.map(|x| x as f32)).collect();
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut w = f32s(&b.w_eff);
        w.extend(f32s(&b.w_log));
        self.queue
            .write_buffer(&self.w, 0, bytemuck::cast_slice(&w));
        self.queue
            .write_buffer(&self.chan, 0, bytemuck::cast_slice(&chan));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(groups.min(GROUPS_X), groups.div_ceil(GROUPS_X), 1);
        }
        let partial_bytes = groups as u64 * 4;
        enc.copy_buffer_to_buffer(&self.partial, 0, &self.readback, 0, partial_bytes);
        let y_off = (MAX_SYMBOLS / SB) as u64 * 4;
        enc.copy_buffer_to_buffer(&self.y_out, 0, &self.readback, y_off, (b.rows * 4) as u64);
        self.queue.submit([enc.finish()]);
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
        self.in_flight = Some((b, done));
    }

    /// Collect a finished batch without blocking (browser frame loop).
    pub fn try_collect(&mut self) -> Option<Stats> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let ready = self
            .in_flight
            .as_ref()
            .is_some_and(|(_, d)| d.load(Ordering::Acquire));
        if !ready {
            return None;
        }
        let (b, _) = self.in_flight.take()?;
        let groups = b.groups() as usize;
        let y_off = (MAX_SYMBOLS / SB) as usize * 4;
        let stats = {
            let data = self.readback.get_mapped_range(..).ok()?;
            let partial: &[f32] = bytemuck::cast_slice(&data[..groups * 4]);
            let y: &[f32] = bytemuck::cast_slice(&data[y_off..y_off + b.rows * 4]);
            Stats {
                symbols: b.symbols as u64,
                rows: b.rows,
                err_sq: partial.iter().map(|&v| v as f64).sum(),
                y_last: y.iter().map(|&v| v as f64).collect(),
            }
        };
        self.readback.unmap();
        Some(stats)
    }

    /// Submit and wait (native tools and tests).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&mut self, b: Batch) -> Stats {
        self.submit(b);
        loop {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(s) = self.try_collect() {
                return s;
            }
        }
    }
}

use bytemuck::Zeroable;

fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn unif(a: u32, b: u32, c: u32) -> f64 {
    (pcg(a ^ pcg(b ^ pcg(c))) >> 8) as f64 / 16_777_216.0
}

/// The kernel's arithmetic in f64, symbol by symbol.
pub fn cpu_twin(b: &Batch) -> Stats {
    let (m, n) = (b.rows, b.channels);
    let lm1 = b.levels - 1.0;
    let mut err_sq = 0.0;
    let mut y_last = vec![0.0; m];
    let mut x = vec![0.0; n];
    for k in 0..b.symbols {
        let s_lo = (b.s0 as u32).wrapping_add(k);
        for (j, xj) in x.iter_mut().enumerate() {
            let c = b.chan[j];
            let u = unif(b.seed ^ 0x9e37_79b9, s_lo, j as u32);
            let kf = k as f64;
            *xj = (0.5
                + 0.30 * (c[0] + c[1] * kf).sin()
                + 0.12 * (c[2] + c[3] * kf).sin()
                + INPUT_NOISE * (2.0 * u - 1.0))
                .clamp(0.0, 1.0);
        }
        let sx: f64 = x.iter().sum();
        for (r, y_slot) in y_last.iter_mut().enumerate() {
            let row = r * n..(r + 1) * n;
            let acc: f64 = b.w_eff[row.clone()]
                .iter()
                .zip(&x)
                .map(|(w, x)| w * x)
                .sum();
            let ideal: f64 = b.w_log[row].iter().zip(&x).map(|(w, x)| w * x).sum();
            let mut meas = acc / n as f64;
            if b.noise {
                let u1 = unif(b.seed, s_lo, 1000 + 2 * r as u32).max(1e-7);
                let u2 = unif(b.seed, s_lo, 1001 + 2 * r as u32);
                meas += b.sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            }
            let q = ((meas.clamp(-1.0, 1.0) + 1.0) * 0.5 * lm1).round_ties_even() / lm1 * 2.0 - 1.0;
            let y = (q - b.mid * sx / n as f64) / b.half;
            err_sq += (y - ideal / n as f64).powi(2);
            if k + 1 == b.symbols {
                *y_slot = y;
            }
        }
    }
    Stats {
        symbols: b.symbols as u64,
        rows: m,
        err_sq,
        y_last,
    }
}
