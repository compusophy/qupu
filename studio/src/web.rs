//! Browser view. Each frame advances the core's femtosecond clock by
//! (wall time × playback) and draws what the core holds: light on every
//! bus, ring resonances and dropped power, heater warmth, detector
//! outputs. The renderer reads `Core` fields directly; there is no copy of
//! the simulation to keep in sync.
//!
//! Drawing is one WebGL2 instanced pass: every waveguide segment, ring and
//! glow is a quad whose fragment shader computes distance to the shape and
//! emits a core + halo, composited with screen blending. Labels and plots
//! are Canvas 2D.

use crate::core::{Core, CoreConfig, Pattern, FS_PER_S};
use crate::fit::{fit_device, DeviceFit};
use crate::measure::Measurement;
use crate::process::{ProcessModel, Sample};
use crate::ring::{add_drop, Ring};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d as Ctx2d, Document, HtmlCanvasElement, HtmlElement, HtmlInputElement,
    HtmlSelectElement, MouseEvent, WebGl2RenderingContext as GL, WebGlBuffer, WebGlProgram,
    WebGlShader, WebGlUniformLocation, WebGlVertexArrayObject,
};

/// Hardware time between automatic weight reloads.
const RELOAD_FS: u64 = 90_000_000_000; // 90 µs
/// Symbols computed in full per frame; the rest are counted in the ledgers.
const MAX_FULL_SYMBOLS: usize = 1024;

const VS: &str = r#"#version 300 es
layout(location=0) in vec2 corner;
layout(location=1) in vec4 seg;
layout(location=2) in vec4 col;
layout(location=3) in vec2 kw;
uniform vec2 u_scale;
uniform vec2 u_offset;
uniform float u_px;
out vec2 v_local;
out vec4 v_col;
flat out float v_kind;
out float v_w;
out float v_len;
out float v_r;
void main() {
  float kind = kw.x;
  float w = max(kw.y, 0.9 * u_px);
  float glow = w * 7.0 + 3.0 * u_px;
  vec2 pos;
  v_len = 0.0;
  v_r = 0.0;
  if (kind < 0.5) {
    vec2 d = seg.zw - seg.xy;
    float L = length(d);
    vec2 t = L > 0.0 ? d / L : vec2(1.0, 0.0);
    vec2 n = vec2(-t.y, t.x);
    float along = mix(-glow, L + glow, corner.x * 0.5 + 0.5);
    float across = corner.y * glow;
    pos = seg.xy + t * along + n * across;
    v_local = vec2(along, across);
    v_len = L;
  } else {
    float R = kind < 1.5 ? seg.z : 0.0;
    float ext = R + glow;
    v_local = corner * ext;
    pos = seg.xy + v_local;
    v_r = R;
  }
  v_col = col;
  v_kind = kind;
  v_w = w;
  gl_Position = vec4(pos * u_scale + u_offset, 0.0, 1.0);
}
"#;

const FS: &str = r#"#version 300 es
precision highp float;
in vec2 v_local;
in vec4 v_col;
flat in float v_kind;
in float v_w;
in float v_len;
in float v_r;
out vec4 o;
void main() {
  float d;
  if (v_kind < 0.5) {
    float x = clamp(v_local.x, 0.0, v_len);
    d = length(vec2(v_local.x - x, v_local.y));
  } else if (v_kind < 1.5) {
    d = abs(length(v_local) - v_r);
  } else {
    d = length(v_local);
  }
  float core = exp(-(d * d) / (v_w * v_w));
  float halo = 0.32 * exp(-d / (v_w * 2.6));
  float a = clamp((core + halo) * v_col.a, 0.0, 4.0);
  o = vec4(v_col.rgb * a, 1.0);
}
"#;

type Rgb = [f32; 3];

const OXIDE: Rgb = [0.20, 0.25, 0.40];
const HEAT: Rgb = [1.0, 0.52, 0.18];
const POS: Rgb = [0.39, 0.89, 0.91];
const NEG: Rgb = [0.93, 0.45, 0.84];
const WHITE: Rgb = [0.92, 0.94, 1.0];

/// Short wavelengths violet, long ones red: the comb as a spectrum.
fn spectral(f: f64) -> Rgb {
    const STOPS: [(f64, Rgb); 5] = [
        (0.00, [0.50, 0.38, 1.00]),
        (0.27, [0.22, 0.62, 1.00]),
        (0.52, [0.25, 0.98, 0.78]),
        (0.76, [1.00, 0.84, 0.32]),
        (1.00, [1.00, 0.36, 0.38]),
    ];
    let f = f.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let ((f0, c0), (f1, c1)) = (w[0], w[1]);
        if f <= f1 {
            let t = ((f - f0) / (f1 - f0)) as f32;
            return [
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
            ];
        }
    }
    STOPS[4].1
}

fn css(c: Rgb, a: f64) -> String {
    format!(
        "rgba({},{},{},{a})",
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8
    )
}

/// Instance stream: 10 floats per shape (seg.xyzw, col.rgba, kind, width).
#[derive(Default)]
struct Batch(Vec<f32>);

impl Batch {
    fn push(&mut self, a: [f64; 4], c: Rgb, alpha: f64, kind: f32, w: f64) {
        if alpha <= 0.002 {
            return;
        }
        self.0.extend_from_slice(&[
            a[0] as f32,
            a[1] as f32,
            a[2] as f32,
            a[3] as f32,
            c[0],
            c[1],
            c[2],
            alpha as f32,
            kind,
            w as f32,
        ]);
    }
    fn seg(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: Rgb, alpha: f64, w: f64) {
        self.push([x0, y0, x1, y1], c, alpha, 0.0, w);
    }
    fn ring(&mut self, x: f64, y: f64, r: f64, c: Rgb, alpha: f64, w: f64) {
        self.push([x, y, r, 0.0], c, alpha, 1.0, w);
    }
    fn dot(&mut self, x: f64, y: f64, c: Rgb, alpha: f64, w: f64) {
        self.push([x, y, 0.0, 0.0], c, alpha, 2.0, w);
    }
}

struct Renderer {
    gl: GL,
    prog: WebGlProgram,
    vao: WebGlVertexArrayObject,
    inst: WebGlBuffer,
    u_scale: WebGlUniformLocation,
    u_offset: WebGlUniformLocation,
    u_px: WebGlUniformLocation,
}

fn shader(gl: &GL, kind: u32, src: &str) -> Result<WebGlShader, String> {
    let s = gl.create_shader(kind).ok_or("no shader")?;
    gl.shader_source(&s, src);
    gl.compile_shader(&s);
    if gl
        .get_shader_parameter(&s, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(s)
    } else {
        Err(gl.get_shader_info_log(&s).unwrap_or_default())
    }
}

impl Renderer {
    fn new(canvas: &HtmlCanvasElement) -> Result<Self, String> {
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(&opts, &"antialias".into(), &true.into()).ok();
        js_sys::Reflect::set(&opts, &"alpha".into(), &false.into()).ok();
        let gl: GL = canvas
            .get_context_with_context_options("webgl2", &opts)
            .map_err(|_| "WebGL2 unavailable")?
            .ok_or("WebGL2 unavailable")?
            .dyn_into()
            .map_err(|_| "not a WebGL2 context")?;
        let prog = gl.create_program().ok_or("no program")?;
        gl.attach_shader(&prog, &shader(&gl, GL::VERTEX_SHADER, VS)?);
        gl.attach_shader(&prog, &shader(&gl, GL::FRAGMENT_SHADER, FS)?);
        gl.link_program(&prog);
        if !gl
            .get_program_parameter(&prog, GL::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            return Err(gl.get_program_info_log(&prog).unwrap_or_default());
        }
        let vao = gl.create_vertex_array().ok_or("no vao")?;
        gl.bind_vertex_array(Some(&vao));
        let quad = gl.create_buffer().ok_or("no buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&quad));
        let corners: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        // SAFETY: the view is consumed by buffer_data before any allocation.
        unsafe {
            gl.buffer_data_with_array_buffer_view(
                GL::ARRAY_BUFFER,
                &js_sys::Float32Array::view(&corners),
                GL::STATIC_DRAW,
            );
        }
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 2, GL::FLOAT, false, 8, 0);
        let inst = gl.create_buffer().ok_or("no buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inst));
        for (loc, size, off) in [(1u32, 4, 0), (2, 4, 16), (3, 2, 32)] {
            gl.enable_vertex_attrib_array(loc);
            gl.vertex_attrib_pointer_with_i32(loc, size, GL::FLOAT, false, 40, off);
            gl.vertex_attrib_divisor(loc, 1);
        }
        gl.bind_vertex_array(None);
        let u = |n: &str| {
            gl.get_uniform_location(&prog, n)
                .ok_or(format!("no uniform {n}"))
        };
        Ok(Renderer {
            u_scale: u("u_scale")?,
            u_offset: u("u_offset")?,
            u_px: u("u_px")?,
            gl,
            prog,
            vao,
            inst,
        })
    }

    fn draw(&self, batch: &Batch, view: &View) {
        let gl = &self.gl;
        gl.viewport(0, 0, view.w as i32, view.h as i32);
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::ONE, GL::ONE_MINUS_SRC_COLOR);
        gl.use_program(Some(&self.prog));
        gl.uniform2f(
            Some(&self.u_scale),
            (2.0 * view.s / view.w) as f32,
            (-2.0 * view.s / view.h) as f32,
        );
        gl.uniform2f(
            Some(&self.u_offset),
            (2.0 * view.ox / view.w - 1.0) as f32,
            (1.0 - 2.0 * view.oy / view.h) as f32,
        );
        gl.uniform1f(Some(&self.u_px), (1.0 / view.s) as f32);
        gl.bind_vertex_array(Some(&self.vao));
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.inst));
        // SAFETY: as above, no allocation between view creation and upload.
        unsafe {
            gl.buffer_data_with_array_buffer_view(
                GL::ARRAY_BUFFER,
                &js_sys::Float32Array::view(&batch.0),
                GL::DYNAMIC_DRAW,
            );
        }
        gl.draw_arrays_instanced(GL::TRIANGLE_STRIP, 0, 4, (batch.0.len() / 10) as i32);
        gl.bind_vertex_array(None);
    }
}

/// µm → device pixels.
#[derive(Clone, Copy, Default)]
struct View {
    s: f64,
    ox: f64,
    oy: f64,
    w: f64,
    h: f64,
}

impl View {
    fn fit(width_um: f64, height_um: f64, w: f64, h: f64, pad: f64) -> Self {
        let s = ((w - 2.0 * pad) / width_um)
            .min((h - 2.0 * pad) / height_um)
            .max(0.01);
        View {
            s,
            ox: (w - width_um * s) / 2.0,
            oy: (h - height_um * s) / 2.0,
            w,
            h,
        }
    }
    fn px(&self, x: f64, y: f64) -> (f64, f64) {
        (x * self.s + self.ox, y * self.s + self.oy)
    }
    fn um(&self, px: f64, py: f64) -> (f64, f64) {
        ((px - self.ox) / self.s, (py - self.oy) / self.s)
    }
}

struct Ui {
    doc: Document,
}

impl Ui {
    fn el<T: JsCast>(&self, id: &str) -> T {
        self.doc
            .get_element_by_id(id)
            .unwrap_or_else(|| panic!("#{id} missing"))
            .dyn_into()
            .unwrap_or_else(|_| panic!("#{id} type"))
    }
    fn text(&self, id: &str, s: &str) {
        if let Some(e) = self.doc.get_element_by_id(id) {
            e.set_text_content(Some(s));
        }
    }
    fn canvas2d(&self, id: &str) -> (HtmlCanvasElement, Ctx2d) {
        let c: HtmlCanvasElement = self.el(id);
        let ctx: Ctx2d = c.get_context("2d").unwrap().unwrap().dyn_into().unwrap();
        (c, ctx)
    }
}

struct App {
    ui: Ui,
    core: Core,
    model: ProcessModel,
    devices: Vec<DeviceFit>,
    measurements: Vec<Measurement>,
    fit_ms: f64,
    device: usize,
    spec_dirty: bool,
    speed_log10: f64,
    paused: bool,
    auto: bool,
    pattern_idx: usize,
    next_reload_fs: u64,
    fs_carry: f64,
    wall_s: f64,
    last_ms: Option<f64>,
    frame: u64,
    cost_ms: f64,
    settled_phase: bool,
    settled_fs: u64,
    hover: Option<(f64, f64, f64, f64)>,
    renderer: Renderer,
    gl_canvas: HtmlCanvasElement,
    hud_canvas: HtmlCanvasElement,
    hud: Ctx2d,
    spec_canvas: HtmlCanvasElement,
    spec: Ctx2d,
    row_canvas: HtmlCanvasElement,
    row: Ctx2d,
    batch: Batch,
    view: View,
    dpr: f64,
}

fn ring_for(model: &ProcessModel, size: usize) -> (Ring, &'static str) {
    if size <= 16 {
        (
            model.predict(5.0, 80.0).expect("R 5 learned"),
            "R 5 µm, gap 80 nm",
        )
    } else {
        (
            model.predict(3.0, 70.0).expect("R 3 learned"),
            "R 3 µm, gap 70 nm",
        )
    }
}

fn fmt_time(s: f64) -> String {
    let a = s.abs();
    if a < 1e-12 {
        format!("{:.0} fs", s * 1e15)
    } else if a < 1e-9 {
        format!("{:.1} ps", s * 1e12)
    } else if a < 1e-6 {
        format!("{:.2} ns", s * 1e9)
    } else if a < 1e-3 {
        format!("{:.2} µs", s * 1e6)
    } else if a < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.2} s", s)
    }
}

fn si(v: f64) -> String {
    let (d, u) = [(1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]
        .into_iter()
        .find(|(d, _)| v >= *d)
        .unwrap_or((1.0, ""));
    format!("{:.2} {u}", v / d)
}

impl App {
    fn new(doc: Document, perf_now: impl Fn() -> f64) -> Result<Self, String> {
        let ui = Ui { doc };
        let t0 = perf_now();
        let measurements = Measurement::bundled();
        let devices: Vec<DeviceFit> = measurements.iter().map(fit_device).collect();
        let fit_ms = perf_now() - t0;
        let samples: Vec<Sample> = devices.iter().map(Sample::of).collect();
        let model = ProcessModel::learn(&samples);
        let (ring, _) = ring_for(&model, 16);
        let core = Core::new(CoreConfig::new(ring, 16, 16));

        let gl_canvas: HtmlCanvasElement = ui.el("gl");
        let renderer = Renderer::new(&gl_canvas)?;
        let hud_canvas: HtmlCanvasElement = ui.el("hud");
        let hud: Ctx2d = hud_canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into()
            .unwrap();
        let (spec_canvas, spec) = ui.canvas2d("spec");
        let (row_canvas, row) = ui.canvas2d("rowspec");
        let device = devices.iter().position(|d| d.name == "r5g80").unwrap_or(0);
        let mut app = App {
            ui,
            core,
            model,
            devices,
            measurements,
            fit_ms,
            device,
            spec_dirty: true,
            speed_log10: -5.3,
            paused: false,
            auto: true,
            pattern_idx: 4,
            next_reload_fs: RELOAD_FS,
            fs_carry: 0.0,
            wall_s: 0.0,
            last_ms: None,
            frame: 0,
            cost_ms: 0.0,
            settled_phase: false,
            settled_fs: 0,
            hover: None,
            renderer,
            gl_canvas,
            hud_canvas,
            hud,
            spec_canvas,
            spec,
            row_canvas,
            row,
            batch: Batch::default(),
            view: View::default(),
            dpr: 1.0,
        };
        app.build_controls();
        Ok(app)
    }

    fn build_controls(&mut self) {
        let doc = &self.ui.doc;
        let sel: HtmlSelectElement = self.ui.el("device");
        sel.set_inner_html("");
        for (k, d) in self.devices.iter().enumerate() {
            let o = doc.create_element("option").unwrap();
            o.set_attribute("value", &k.to_string()).ok();
            o.set_text_content(Some(&format!(
                "{} · R {:.0} µm · gap {:.0} nm",
                d.name, d.radius_um, d.gap_nm
            )));
            if k == self.device {
                o.set_attribute("selected", "").ok();
            }
            sel.append_child(&o).ok();
        }
        let holder: HtmlElement = self.ui.el("patterns");
        holder.set_inner_html("");
        for (k, p) in Pattern::ALL.iter().enumerate() {
            let b = doc.create_element("button").unwrap();
            b.set_attribute("type", "button").ok();
            b.set_attribute("id", &format!("pat-{k}")).ok();
            b.set_attribute(
                "aria-pressed",
                if k == self.pattern_idx {
                    "true"
                } else {
                    "false"
                },
            )
            .ok();
            b.set_text_content(Some(p.name()));
            holder.append_child(&b).ok();
        }
        let ledger: HtmlElement = self.ui.el("ledger");
        let mut html = String::new();
        for (key, label) in [
            ("dac", "DAC"),
            ("adc", "ADC"),
            ("tia", "TIA"),
            ("laser", "laser"),
            ("heat", "heaters"),
        ] {
            let cls = if key == "heat" { " class=\"heat\"" } else { "" };
            html += &format!("<span class=\"k\">{label}</span><span class=\"bar\"><i id=\"bar-{key}\"{cls}></i></span><span class=\"v\" id=\"val-{key}\">–</span>");
        }
        ledger.set_inner_html(&html);
        self.update_speed_label();
    }

    fn update_speed_label(&self) {
        let hw = 10f64.powf(self.speed_log10);
        self.ui
            .text("speed-val", &format!("1 s = {}", fmt_time(hw)));
    }

    fn set_pattern(&mut self, k: usize) {
        self.pattern_idx = k % Pattern::ALL.len();
        self.core.program(Pattern::ALL[self.pattern_idx]);
        self.next_reload_fs = self.core.t_fs + RELOAD_FS;
        self.settled_phase = false;
        for i in 0..Pattern::ALL.len() {
            if let Some(b) = self.ui.doc.get_element_by_id(&format!("pat-{i}")) {
                b.set_attribute(
                    "aria-pressed",
                    if i == self.pattern_idx {
                        "true"
                    } else {
                        "false"
                    },
                )
                .ok();
            }
        }
    }

    fn rebuild(&mut self, size: usize) {
        let (ring, _) = ring_for(&self.model, size);
        let cal = self.core.calibrate;
        let t = self.core.t_fs;
        let mut core = Core::new(CoreConfig::new(ring, size, size));
        core.calibrate = cal;
        core.t_fs = t;
        self.core = core;
        self.set_pattern(self.pattern_idx);
    }

    fn frame(&mut self, now_ms: f64, perf_now: &dyn Fn() -> f64) {
        let dt = self
            .last_ms
            .map(|l| ((now_ms - l) / 1000.0).clamp(0.0, 0.1))
            .unwrap_or(0.0);
        self.last_ms = Some(now_ms);
        self.frame += 1;
        if !self.paused {
            self.wall_s += dt;
            self.fs_carry += dt * 10f64.powf(self.speed_log10) * FS_PER_S;
            let fs = self.fs_carry.floor();
            self.fs_carry -= fs;
            let t0 = perf_now();
            self.core.step(fs as u64, MAX_FULL_SYMBOLS);
            if self.auto && self.core.t_fs >= self.next_reload_fs {
                self.set_pattern(self.pattern_idx + 1);
            }
            if !self.settled_phase && self.core.settled() >= 0.99 {
                self.settled_phase = true;
                self.core.ledger.err_sq = 0.0;
                self.core.ledger.err_n = 0.0;
            }
            if self.settled_phase {
                self.settled_fs += fs as u64;
            }
            self.cost_ms = 0.9 * self.cost_ms + 0.1 * (perf_now() - t0);
        }
        self.resize();
        self.draw_chip();
        self.renderer.draw(&self.batch, &self.view);
        self.draw_hud();
        if self.frame % 4 == 0 {
            self.draw_row();
        }
        if self.spec_dirty {
            self.draw_silicon();
            self.spec_dirty = false;
        }
        if self.frame % 6 == 0 {
            self.update_stats();
        }
    }

    fn resize(&mut self) {
        let dpr = web_sys::window().unwrap().device_pixel_ratio().max(1.0);
        self.dpr = dpr;
        for c in [
            &self.gl_canvas,
            &self.hud_canvas,
            &self.spec_canvas,
            &self.row_canvas,
        ] {
            let (w, h) = (
                (c.client_width() as f64 * dpr) as u32,
                (c.client_height() as f64 * dpr) as u32,
            );
            if c.width() != w || c.height() != h {
                c.set_width(w.max(1));
                c.set_height(h.max(1));
                if c == &self.spec_canvas {
                    self.spec_dirty = true;
                }
            }
        }
        let l = &self.core.layout;
        self.view = View::fit(
            l.width,
            l.height,
            self.gl_canvas.width() as f64,
            self.gl_canvas.height() as f64,
            18.0 * dpr,
        );
    }

    /// Wavefronts are visible only when a frame is shorter than a symbol.
    fn slow_motion(&self) -> bool {
        10f64.powf(self.speed_log10) / 60.0 < 1.0 / self.core.cfg.baud
    }

    fn draw_chip(&mut self) {
        let slow = self.slow_motion();
        let hover = self.hover;
        let core = &self.core;
        let l = &core.layout;
        let (m, n) = (core.cfg.rows, core.cfg.channels);
        let b = &mut self.batch;
        b.0.clear();
        let mean_ok = core.x_mean.iter().any(|v| *v > 0.0);
        let x_at = |s: f64| inputs_at(core, slow, mean_ok, s);
        let colors: Vec<Rgb> = (0..n)
            .map(|j| spectral(j as f64 / (n.max(2) - 1) as f64))
            .collect();
        let mix = |p: &dyn Fn(usize) -> f64| -> (Rgb, f64) {
            let mut c = [0.0f32; 3];
            let mut tot = 0.0;
            for j in 0..n {
                let w = p(j).max(0.0);
                tot += w;
                for k in 0..3 {
                    c[k] += colors[j][k] * w as f32;
                }
            }
            if tot > 0.0 {
                for v in &mut c {
                    *v /= tot as f32;
                }
            }
            (c, tot / n as f64)
        };
        let wg = 0.45; // waveguide width, µm

        // Chip edge.
        let (w, h) = (l.width, l.height);
        for (x0, y0, x1, y1) in [
            (0.0, 0.0, w, 0.0),
            (w, 0.0, w, h),
            (w, h, 0.0, h),
            (0.0, h, 0.0, 0.0),
        ] {
            b.seg(x0, y0, x1, y1, OXIDE, 0.18, 0.5);
        }

        // Comb source: a small spectrum bar.
        let (sx, sy) = l.source;
        let span = (n as f64 * 1.4).min(l.height * 0.3);
        for j in 0..n {
            let y = sy - span / 2.0 + span * j as f64 / (n.max(2) - 1) as f64;
            b.seg(sx - 7.0, y, sx - 1.0, y, colors[j], 0.9, 0.35);
        }
        b.dot(sx, sy, WHITE, 0.9, 1.6);

        // Input bus, carved by each modulator.
        let mut xs = vec![sx];
        xs.extend(&l.mod_x);
        xs.push(l.split_x0);
        for k in 0..xs.len() - 1 {
            let x = x_at(0.5 * (xs[k] + xs[k + 1]) - sx);
            let (c, p) = mix(&|j| if j + 1 <= k { x[j] } else { 1.0 });
            b.seg(
                xs[k],
                l.bus_y,
                xs[k + 1],
                l.bus_y,
                c,
                0.25 + 0.9 * p.sqrt(),
                wg,
            );
        }
        for j in 0..n {
            let x = x_at(l.mod_x[j] - sx);
            let cy = l.bus_y - l.mod_r - 1.2;
            b.ring(
                l.mod_x[j],
                cy,
                l.mod_r,
                colors[j],
                0.15 + 0.9 * (1.0 - x[j]),
                0.35,
            );
        }

        // Broadcast tree.
        let x_tree = x_at(l.split_x0 - sx);
        let (tc, tp) = mix(&|j| x_tree[j]);
        for i in 0..m {
            let y = l.row_y[i];
            let alpha = 0.10 + 0.55 * tp.sqrt();
            b.seg(
                l.split_x0,
                l.bus_y,
                l.split_x0 + 14.0,
                l.bus_y,
                tc,
                alpha,
                wg,
            );
            b.seg(
                l.split_x0 + 14.0,
                l.bus_y,
                l.split_x1 - 14.0,
                y,
                tc,
                alpha,
                wg,
            );
            b.seg(l.split_x1 - 14.0, y, l.row_x0, y, tc, alpha, wg);
        }

        // Rows: through bus, rings, drop bus, detectors, outputs.
        let max_heat = core.shift_nm.iter().cloned().fold(1e-9, f64::max) / core.cfg.nm_per_mw;
        let lam_lo = core.channel_nm[0] - 0.5 * core.cfg.spacing_nm();
        let lam_hi = core.channel_nm[n - 1] + 0.5 * core.cfg.spacing_nm();
        let hover_row = hover.and_then(|(x, y, _, _)| row_at(core, x, y));
        for i in 0..m {
            let y = l.row_y[i];
            let dy = l.drop_y[i];
            let s0 = l.row_start_s[i];
            if hover_row == Some(i) {
                b.seg(
                    l.row_x0,
                    0.5 * (y + dy),
                    l.det_x,
                    0.5 * (y + dy),
                    [0.45, 0.55, 1.0],
                    0.07,
                    l.ring_r * 1.4,
                );
            }
            let frac = |j: usize| {
                let k = i * n + j;
                core.ring_drop[k] / core.ring_in[k].max(1e-9)
            };
            let mut pts = vec![l.row_x0];
            pts.extend(&l.ring_x);
            pts.push(l.row_x1);
            for k in 0..pts.len() - 1 {
                let x = x_at(s0 + 0.5 * (pts[k] + pts[k + 1]) - l.row_x0);
                let (c, p) = mix(&|j| {
                    if j + 1 <= k {
                        x[j] * (1.0 - frac(j))
                    } else {
                        x[j]
                    }
                });
                b.seg(pts[k], y, pts[k + 1], y, c, 0.10 + 0.8 * p.sqrt(), wg);
                let (dc, dp) = mix(&|j| if j + 1 <= k { x[j] * frac(j) } else { 0.0 });
                if k >= 1 {
                    b.seg(pts[k], dy, pts[k + 1], dy, dc, 0.06 + 1.6 * dp.sqrt(), wg);
                }
            }
            b.seg(l.row_x1, dy, l.det_x, dy, OXIDE, 0.25, wg);
            for j in 0..n {
                let k = i * n + j;
                let x = x_at(s0 + l.ring_x[j] - l.row_x0);
                let cy = y - l.ring_r - 1.5;
                let res = core.ring_res_nm(k);
                let hue = spectral((res - lam_lo) / (lam_hi - lam_lo));
                let heat = core.ring_heater_mw(k) / max_heat;
                b.dot(l.ring_x[j], cy, HEAT, 0.42 * heat, l.ring_r * 0.75);
                b.ring(
                    l.ring_x[j],
                    cy,
                    l.ring_r,
                    hue,
                    0.10 + 2.4 * x[j] * core.ring_drop[k],
                    0.32,
                );
            }
            // Balanced detector pair and the row's output.
            let x = x_at(s0 + l.det_x - l.row_x0);
            let thr: f64 = (0..n).map(|j| x[j] * (1.0 - frac(j))).sum::<f64>() / n as f64;
            let drp: f64 = (0..n).map(|j| x[j] * frac(j)).sum::<f64>() / n as f64;
            b.seg(
                l.det_x,
                y - 1.6,
                l.det_x,
                y + 1.6,
                WHITE,
                0.2 + 1.5 * thr,
                1.1,
            );
            b.seg(
                l.det_x,
                dy - 1.6,
                l.det_x,
                dy + 1.6,
                WHITE,
                0.2 + 3.0 * drp,
                1.1,
            );
            let oy = 0.5 * (y + dy);
            let x0 = l.det_x + 16.0;
            let scale = 55.0;
            let v = core.y[i].clamp(-1.0, 1.0);
            b.seg(
                x0,
                oy,
                x0 + v.abs() * scale,
                oy,
                if v >= 0.0 { POS } else { NEG },
                0.85,
                1.4,
            );
            let t = core.y_ideal[i].clamp(-1.0, 1.0);
            b.seg(
                x0 + t.abs() * scale,
                oy - 2.6,
                x0 + t.abs() * scale,
                oy + 2.6,
                WHITE,
                0.55,
                0.25,
            );
        }
    }

    fn draw_hud(&self) {
        let c = &self.hud;
        let (w, h) = (
            self.hud_canvas.width() as f64,
            self.hud_canvas.height() as f64,
        );
        c.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0).ok();
        c.clear_rect(0.0, 0.0, w, h);
        let d = self.dpr;
        let core = &self.core;
        let l = &core.layout;
        let v = &self.view;
        let fs = (10.5 * d).round();
        c.set_font(&format!(
            "500 {fs}px 'IBM Plex Mono', ui-monospace, monospace"
        ));
        c.set_text_baseline("bottom");
        c.set_fill_style_str("rgba(125,135,157,0.95)");
        let (n, m) = (core.cfg.channels, core.cfg.rows);
        let top = v.px(0.0, l.row_y[0] - 4.0 * l.ring_r - 6.0).1.max(14.0 * d);
        let label = |x_um: f64, y_px: f64, s: &str| {
            let (px, _) = v.px(x_um, 0.0);
            c.fill_text(s, px, y_px).ok();
        };
        if self.slow_motion() {
            c.set_fill_style_str("rgba(185,198,255,0.95)");
            c.fill_text(
                "slow motion: each symbol's wavefront crosses the chip in a few ps",
                10.0 * d,
                h - 10.0 * d,
            )
            .ok();
        }
        if n > 38 {
            c.set_fill_style_str("rgba(255,161,73,0.95)");
            c.fill_text(
                "64 channels exceed one free spectral range: rings alias onto other channels",
                10.0 * d,
                h - 10.0 * d,
            )
            .ok();
        }
        // Scale bar: the chip is drawn to scale.
        let (bx, by) = v.px(l.source.0, l.height - 8.0);
        c.set_fill_style_str("rgba(125,135,157,0.9)");
        c.fill_rect(bx, by, 100.0 * v.s, 1.0 * d);
        c.fill_text("100 µm", bx, by - 3.0 * d).ok();
        if self.hud_canvas.client_width() < 640 {
            return; // labels collide at phone scale; the panels carry the numbers
        }
        let (_, bus_px) = v.px(0.0, l.bus_y);
        label(l.source.0 - 6.0, bus_px - 14.0 * d, "comb");
        label(
            l.mod_x[0] - 4.0,
            bus_px - 2.6 * l.mod_r * v.s - 4.0 * d,
            &format!("{n} modulators · 10 GBd"),
        );
        label(l.split_x0, top, &format!("broadcast 1→{m}"));
        label(
            l.row_x0 + 8.0,
            top,
            &format!(
                "weight bank {m}×{n} · {} · Q {:.0}",
                ring_for(&self.model, n).1,
                core.cfg.ring.q_loaded()
            ),
        );
        label(l.det_x - 6.0, top, "PD → ADC → y");
        c.set_text_baseline("top");
        let lo = core.channel_nm[0];
        let hi = core.channel_nm[n - 1];
        label(
            l.mod_x[0] - 4.0,
            bus_px + 8.0 * d,
            &format!("{lo:.2} … {hi:.2} nm · 100 GHz grid"),
        );
    }

    fn draw_row(&self) {
        let core = &self.core;
        let c = &self.row;
        let (w, h) = (
            self.row_canvas.width() as f64,
            self.row_canvas.height() as f64,
        );
        let d = self.dpr;
        c.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0).ok();
        c.clear_rect(0.0, 0.0, w, h);
        let row = self
            .hover
            .and_then(|(x, y, _, _)| row_at(core, x, y))
            .unwrap_or(0);
        self.ui.text("row-title", &format!("Row {row} transfer"));
        let n = core.cfg.channels;
        let sp = core.cfg.spacing_nm();
        let (lo, hi) = (
            core.channel_nm[0] - 0.7 * sp,
            core.channel_nm[n - 1] + 0.7 * sp,
        );
        let (pl, pr, pt, pb) = (6.0 * d, 6.0 * d, 8.0 * d, 18.0 * d);
        let xpx = |lam: f64| pl + (lam - lo) / (hi - lo) * (w - pl - pr);
        let ypx = |v: f64| pt + (1.0 - v.clamp(0.0, 1.0)) * (h - pt - pb);
        for j in 0..n {
            let col = spectral(j as f64 / (n.max(2) - 1) as f64);
            c.set_stroke_style_str(&css(col, 0.55));
            c.set_line_width(1.0 * d);
            c.begin_path();
            c.move_to(xpx(core.channel_nm[j]), ypx(1.0));
            c.line_to(xpx(core.channel_nm[j]), ypx(0.0));
            c.stroke();
        }
        let samples = 420;
        let mut thr = Vec::with_capacity(samples);
        let mut drp = Vec::with_capacity(samples);
        for s in 0..samples {
            let lam = lo + (hi - lo) * s as f64 / (samples - 1) as f64;
            let mut p = 1.0;
            let mut dsum = 0.0;
            for k in 0..n {
                let ring = Ring {
                    lambda_res_nm: core.ring_res_nm(row * n + k),
                    ..core.cfg.ring
                };
                let (t, dd) = add_drop(ring.r1, ring.r2, ring.a, ring.phase(lam));
                dsum += p * dd * p;
                p *= t;
            }
            thr.push((lam, p));
            drp.push((lam, dsum));
        }
        let line = |pts: &[(f64, f64)], color: &str, width: f64, fill: Option<&str>| {
            c.begin_path();
            c.move_to(xpx(pts[0].0), ypx(pts[0].1));
            for &(x, v) in &pts[1..] {
                c.line_to(xpx(x), ypx(v));
            }
            if let Some(f) = fill {
                c.line_to(xpx(pts[pts.len() - 1].0), ypx(0.0));
                c.line_to(xpx(pts[0].0), ypx(0.0));
                c.set_fill_style_str(f);
                c.fill();
            } else {
                c.set_stroke_style_str(color);
                c.set_line_width(width * d);
                c.stroke();
            }
        };
        line(&drp, "", 0.0, Some("rgba(99,227,232,0.16)"));
        line(&drp, "rgba(99,227,232,0.95)", 1.4, None);
        line(&thr, "rgba(217,222,234,0.55)", 1.0, None);
        c.set_font(&format!(
            "{}px 'IBM Plex Mono', monospace",
            (10.0 * d).round()
        ));
        c.set_fill_style_str("rgba(125,135,157,1)");
        c.set_text_baseline("bottom");
        c.set_text_align("left");
        c.fill_text(&format!("{lo:.1} nm"), pl, h - 2.0 * d).ok();
        c.set_text_align("right");
        c.fill_text(&format!("{hi:.1} nm"), w - pr, h - 2.0 * d)
            .ok();
        c.set_text_align("center");
        c.fill_text("drop · through (linear)", w / 2.0, h - 2.0 * d)
            .ok();
        c.set_text_align("left");
    }

    fn draw_silicon(&self) {
        let c = &self.spec;
        let (w, h) = (
            self.spec_canvas.width() as f64,
            self.spec_canvas.height() as f64,
        );
        let d = self.dpr;
        c.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0).ok();
        c.clear_rect(0.0, 0.0, w, h);
        let dev = &self.devices[self.device];
        let m = &self.measurements[self.device];
        let rf = dev.nearest(1550.0);
        let ring = rf.ring;
        let half = (6.0 * ring.fwhm_nm()).max(0.5).min(0.45 * ring.fsr_nm());
        let (lo, hi) = (ring.lambda_res_nm - half, ring.lambda_res_nm + half);
        let idx: Vec<usize> = (0..m.wl_nm.len())
            .filter(|&k| m.wl_nm[k] >= lo && m.wl_nm[k] <= hi)
            .collect();
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;
        for &k in &idx {
            ymin = ymin.min(m.through_dbm[k]).min(m.drop_dbm[k]);
            ymax = ymax.max(m.through_dbm[k]).max(m.drop_dbm[k]);
        }
        let (ymin, ymax) = ((ymin / 10.0).floor() * 10.0, (ymax / 10.0).ceil() * 10.0);
        let (pl, pr, pt, pb) = (30.0 * d, 6.0 * d, 6.0 * d, 18.0 * d);
        let xpx = |lam: f64| pl + (lam - lo) / (hi - lo) * (w - pl - pr);
        let ypx = |db: f64| pt + (ymax - db) / (ymax - ymin) * (h - pt - pb);
        c.set_font(&format!(
            "{}px 'IBM Plex Mono', monospace",
            (10.0 * d).round()
        ));
        c.set_text_baseline("middle");
        c.set_text_align("right");
        let mut g = ymin;
        while g <= ymax + 1e-9 {
            c.set_stroke_style_str("rgba(26,33,51,1)");
            c.set_line_width(1.0);
            c.begin_path();
            c.move_to(pl, ypx(g));
            c.line_to(w - pr, ypx(g));
            c.stroke();
            c.set_fill_style_str("rgba(125,135,157,1)");
            c.fill_text(&format!("{g:.0}"), pl - 4.0 * d, ypx(g)).ok();
            g += 10.0;
        }
        let step = (idx.len() / 500).max(1);
        for (vals, color) in [
            (&m.through_dbm, "rgba(217,222,234,0.5)"),
            (&m.drop_dbm, "rgba(99,227,232,0.6)"),
        ] {
            c.set_fill_style_str(color);
            for &k in idx.iter().step_by(step) {
                c.fill_rect(
                    xpx(m.wl_nm[k]) - 0.8 * d,
                    ypx(vals[k]) - 0.8 * d,
                    1.6 * d,
                    1.6 * d,
                );
            }
        }
        for (port, color) in [(0, "rgba(255,255,255,0.95)"), (1, "rgba(99,227,232,1)")] {
            c.set_stroke_style_str(color);
            c.set_line_width(1.3 * d);
            c.begin_path();
            for s in 0..=300 {
                let lam = lo + (hi - lo) * s as f64 / 300.0;
                let (t, dd) = ring.through_drop(lam);
                let tilt = rf.tilt_db_per_nm * (lam - ring.lambda_res_nm);
                let v = if port == 0 {
                    rf.through_offset_db + tilt + 10.0 * t.log10()
                } else {
                    rf.drop_offset_db + tilt + 10.0 * dd.log10()
                };
                let (x, y) = (xpx(lam), ypx(v.clamp(ymin, ymax)));
                if s == 0 {
                    c.move_to(x, y)
                } else {
                    c.line_to(x, y)
                }
            }
            c.stroke();
        }
        c.set_text_baseline("bottom");
        c.set_text_align("left");
        c.set_fill_style_str("rgba(125,135,157,1)");
        c.fill_text(&format!("{lo:.2} nm"), pl, h - 2.0 * d).ok();
        c.set_text_align("right");
        c.fill_text(&format!("{hi:.2} nm"), w - pr, h - 2.0 * d)
            .ok();
        c.set_text_align("left");
        self.ui
            .text("si-title", &format!("Measured silicon · {}", dev.name));
        self.ui.text("si-note", &format!(
            "Resonance at {:.2} nm: Q {:.0}, κ² {:.4}, loss {:.1} dB/cm, fit residual {:.2} dB. All {} resonances on 8 chips fitted in this tab in {:.0} ms; group index {:.3}.",
            ring.lambda_res_nm, ring.q_loaded(), ring.kappa2(), ring.loss_db_per_cm(), rf.worst_rms_db(),
            self.devices.iter().map(|d| d.resonances.len()).sum::<usize>(), self.fit_ms, self.model.n_g));
    }

    fn update_stats(&self) {
        let core = &self.core;
        let ui = &self.ui;
        ui.text("hw", &fmt_time(core.t_fs as f64 / FS_PER_S));
        ui.text("wall", &format!("{:.1} s", self.wall_s));
        ui.text("cost", &format!("{:.1} ms/frame", self.cost_ms));
        let settled = core.settled();
        ui.text(
            "enob",
            &if self.settled_phase && core.ledger.err_n > 0.0 {
                format!("{:.1} bits", core.ledger.enob())
            } else {
                "–".into()
            },
        );
        ui.text("werr", &format!("{:.2} %", 100.0 * core.weight_error()));
        ui.text("settled", &format!("{:.0} %", 100.0 * settled));
        let peak = (core.cfg.rows * core.cfg.channels) as f64 * core.cfg.baud;
        ui.text("peak", &si(peak));
        ui.text("macs", &si(core.ledger.macs));
        ui.text("heat", &format!("{:.2} W", core.heater_mw() / 1e3));
        let util = self.settled_fs as f64 / (core.t_fs.max(1) as f64);
        let chip: HtmlElement = ui.el("state");
        if self.settled_phase {
            chip.set_class_name("chip live");
            chip.set_text_content(Some(&format!(
                "computing · {} · utilisation {:.0} %",
                core.pattern.name(),
                100.0 * util
            )));
        } else {
            chip.set_class_name("chip warn");
            chip.set_text_content(Some(&format!(
                "reloading weights · {} · {:.0} % settled",
                core.pattern.name(),
                100.0 * settled
            )));
        }
        let led = &core.ledger;
        let macs = led.macs.max(1.0);
        let parts = [
            ("dac", led.dac_j),
            ("adc", led.adc_j),
            ("tia", led.tia_j),
            ("laser", led.laser_j),
            ("heat", led.heater_j),
        ];
        let total = led.total_j().max(1e-30);
        for (k, j) in parts {
            if let Some(e) = ui.doc.get_element_by_id(&format!("bar-{k}")) {
                let e: HtmlElement = e.dyn_into().unwrap();
                e.style()
                    .set_property("width", &format!("{:.1}%", 100.0 * j / total))
                    .ok();
            }
            ui.text(&format!("val-{k}"), &format!("{:.0} fJ", j / macs * 1e15));
        }
        ui.text("pj", &format!("{:.0} fJ/MAC", led.pj_per_mac() * 1e3));
    }

    fn tooltip(&self) {
        let tip: HtmlElement = self.ui.el("tip");
        let Some((x, y, cx, cy)) = self.hover else {
            tip.set_hidden(true);
            return;
        };
        let core = &self.core;
        let l = &core.layout;
        let n = core.cfg.channels;
        let hit = row_at(core, x, y).and_then(|i| {
            let cyr = l.row_y[i] - l.ring_r - 1.5;
            l.ring_x
                .iter()
                .position(|rx| {
                    (rx - x).abs() <= l.ring_r + 4.0 && (cyr - y).abs() <= l.ring_r + 4.0
                })
                .map(|j| (i, j))
        });
        let Some((i, j)) = hit else {
            tip.set_hidden(true);
            return;
        };
        let k = i * n + j;
        let (mid, half) = core.weight_scale();
        tip.set_text_content(Some(&format!(
            "row {i} · channel {j}\nchannel   {:.3} nm\nring at   {:.3} nm\nweight    {:+.2} set · {:+.2} actual\nheater    {:.2} mW",
            core.channel_nm[j], core.ring_res_nm(k), core.logical_w[k], (core.w_eff[k] - mid) / half, core.ring_heater_mw(k))));
        let style = tip.style();
        style.set_property("left", &format!("{}px", cx + 14.0)).ok();
        style.set_property("top", &format!("{}px", cy + 14.0)).ok();
        tip.set_hidden(false);
    }
}

/// Inputs present at path length `s` from the source: the symbol whose
/// wavefront is there in slow motion, otherwise the frame's exposure mean.
fn inputs_at(core: &Core, slow: bool, mean_ok: bool, s: f64) -> &[f64] {
    if slow {
        core.x_at(s)
    } else if mean_ok {
        &core.x_mean
    } else {
        &core.x
    }
}

/// Row under a chip coordinate, if any.
fn row_at(core: &Core, x: f64, y: f64) -> Option<usize> {
    let l = &core.layout;
    if x < l.row_x0 - 10.0 || x > l.det_x + 80.0 {
        return None;
    }
    (0..core.cfg.rows).find(|&i| y >= l.drop_y[i] - l.ring_r && y <= l.row_y[i] + l.ring_r)
}

#[wasm_bindgen(start)]
pub fn boot() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let doc = window.document().ok_or("no document")?;
    let perf = window.performance().ok_or("no performance")?;
    let now = {
        let p = perf.clone();
        move || p.now()
    };
    let app = Rc::new(RefCell::new(
        App::new(doc.clone(), now.clone()).map_err(|e| JsValue::from_str(&e))?,
    ));
    app.borrow_mut().set_pattern(4);

    let on = |id: &str, ev: &str, f: Box<dyn FnMut(web_sys::Event)>| {
        let cb = Closure::wrap(f);
        doc.get_element_by_id(id)
            .unwrap()
            .add_event_listener_with_callback(ev, cb.as_ref().unchecked_ref())
            .ok();
        cb.forget();
    };
    {
        let a = app.clone();
        on(
            "speed",
            "input",
            Box::new(move |e| {
                let v: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                a.speed_log10 = v.value().parse().unwrap_or(-5.3);
                a.update_speed_label();
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "device",
            "change",
            Box::new(move |e| {
                let v: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                a.device = v.value().parse().unwrap_or(0);
                a.spec_dirty = true;
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "size",
            "change",
            Box::new(move |e| {
                let v: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
                a.borrow_mut().rebuild(v.value().parse().unwrap_or(16));
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "auto",
            "change",
            Box::new(move |e| {
                let v: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                a.auto = v.checked();
                a.next_reload_fs = a.core.t_fs + RELOAD_FS;
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "cal",
            "change",
            Box::new(move |e| {
                let v: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                a.core.calibrate = v.checked();
                let k = a.pattern_idx;
                a.set_pattern(k);
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "pause",
            "click",
            Box::new(move |e| {
                let b: HtmlElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                a.paused = !a.paused;
                b.set_text_content(Some(if a.paused { "Resume" } else { "Pause" }));
                b.set_attribute("aria-pressed", if a.paused { "true" } else { "false" })
                    .ok();
            }),
        );
    }
    for k in 0..Pattern::ALL.len() {
        let a = app.clone();
        on(
            &format!("pat-{k}"),
            "click",
            Box::new(move |_| a.borrow_mut().set_pattern(k)),
        );
    }
    {
        let a = app.clone();
        on(
            "gl",
            "mousemove",
            Box::new(move |e| {
                let e: MouseEvent = e.dyn_into().unwrap();
                let mut a = a.borrow_mut();
                let (px, py) = (e.offset_x() as f64 * a.dpr, e.offset_y() as f64 * a.dpr);
                let (x, y) = a.view.um(px, py);
                a.hover = Some((x, y, e.offset_x() as f64, e.offset_y() as f64));
                a.tooltip();
            }),
        );
        let a = app.clone();
        on(
            "gl",
            "mouseleave",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                a.hover = None;
                a.tooltip();
            }),
        );
    }

    // Animation loop.
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let win = window.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |t: f64| {
        app.borrow_mut().frame(t, &now);
        win.request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .ok();
    }) as Box<dyn FnMut(f64)>));
    window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;
    Ok(())
}
