//! The glass-engine demo: the tabletop rig of docs/photonic/TABLETOP.md
//! rendered as the object it would be, and run by the simulator.
//!
//! Every symbol, `freespace::Engine` pushes the input image through the
//! modelled optics. The input LCD shows its copies, the weight LCD shows
//! the masks the engine solved, and the camera's readout is the inset. The
//! scene is one raymarched pass (`glass/scene.frag`): the panels sample
//! the LCD textures, and the beams in the haze carry the same images blurred by
//! the cones they spread through. Evolve runs `tabletop::RigClimb` between
//! frames and moves the bench to each new best design.

use crate::core::Rng;
use crate::freespace::{Engine, Layouts, Readout};
use crate::geometry::Lattice;
use crate::tabletop::{
    field_mm, lcd_map, rig_enob, Input, Rig, RigClimb, Workload, F_STOPS, IMAGE_CIRCLE_MM, N, SIDE,
};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d as Ctx2d, Document, HtmlCanvasElement, HtmlElement, HtmlSelectElement,
    PointerEvent, WebGl2RenderingContext as GL, WebGlFramebuffer, WebGlProgram, WebGlShader,
    WebGlTexture, WebGlUniformLocation, WebGlVertexArrayObject, WheelEvent,
};

const VS: &str = include_str!("glass/fullscreen.vert");
const FS_COMPOSE: &str = include_str!("glass/compose.frag");
const FS_SCENE: &str = include_str!("glass/scene.frag");
const FS_DOWN: &str = include_str!("glass/down.frag");
const FS_UP: &str = include_str!("glass/up.frag");
const FS_FINAL: &str = include_str!("glass/final.frag");

type FrameLoop = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// The rig's LCDs refresh at 50 Hz; the demo runs in real time.
const BAUD: f64 = 50.0;
const AX: f64 = 82.0;
const BLOOM_LEVELS: usize = 5;
const SCALES: [f64; 5] = [0.5, 0.625, 0.75, 0.875, 1.0];

struct Prog {
    p: WebGlProgram,
    u: RefCell<HashMap<&'static str, Option<WebGlUniformLocation>>>,
}

impl Prog {
    fn new(gl: &GL, fs: &str) -> Result<Prog, String> {
        let compile = |kind: u32, src: &str| -> Result<WebGlShader, String> {
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
        };
        let p = gl.create_program().ok_or("no program")?;
        gl.attach_shader(&p, &compile(GL::VERTEX_SHADER, VS)?);
        gl.attach_shader(&p, &compile(GL::FRAGMENT_SHADER, fs)?);
        gl.link_program(&p);
        if !gl
            .get_program_parameter(&p, GL::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            return Err(gl.get_program_info_log(&p).unwrap_or_default());
        }
        Ok(Prog {
            p,
            u: RefCell::new(HashMap::new()),
        })
    }

    fn loc(&self, gl: &GL, name: &'static str) -> Option<WebGlUniformLocation> {
        self.u
            .borrow_mut()
            .entry(name)
            .or_insert_with(|| gl.get_uniform_location(&self.p, name))
            .clone()
    }
    fn f1(&self, gl: &GL, n: &'static str, a: f64) {
        gl.uniform1f(self.loc(gl, n).as_ref(), a as f32);
    }
    fn f2(&self, gl: &GL, n: &'static str, a: f64, b: f64) {
        gl.uniform2f(self.loc(gl, n).as_ref(), a as f32, b as f32);
    }
    fn f3(&self, gl: &GL, n: &'static str, v: [f64; 3]) {
        gl.uniform3f(
            self.loc(gl, n).as_ref(),
            v[0] as f32,
            v[1] as f32,
            v[2] as f32,
        );
    }
    fn f4(&self, gl: &GL, n: &'static str, v: [f64; 4]) {
        gl.uniform4f(
            self.loc(gl, n).as_ref(),
            v[0] as f32,
            v[1] as f32,
            v[2] as f32,
            v[3] as f32,
        );
    }
    fn i1(&self, gl: &GL, n: &'static str, a: i32) {
        gl.uniform1i(self.loc(gl, n).as_ref(), a);
    }
}

struct Target {
    tex: WebGlTexture,
    fb: WebGlFramebuffer,
    w: i32,
    h: i32,
}

#[derive(Clone, Copy)]
enum Fmt {
    R8,
    Hdr,
    Rgba8,
}

fn texture(
    gl: &GL,
    w: i32,
    h: i32,
    fmt: Fmt,
    mips: bool,
    data: Option<&[u8]>,
) -> Result<WebGlTexture, String> {
    let t = gl.create_texture().ok_or("no texture")?;
    gl.bind_texture(GL::TEXTURE_2D, Some(&t));
    let (internal, format, ty) = match fmt {
        Fmt::R8 => (GL::R8, GL::RED, GL::UNSIGNED_BYTE),
        Fmt::Hdr => (GL::RGBA16F, GL::RGBA, GL::HALF_FLOAT),
        Fmt::Rgba8 => (GL::RGBA8, GL::RGBA, GL::UNSIGNED_BYTE),
    };
    gl.pixel_storei(GL::UNPACK_ALIGNMENT, 1);
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        GL::TEXTURE_2D,
        0,
        internal as i32,
        w,
        h,
        0,
        format,
        ty,
        data,
    )
    .map_err(|_| "texImage2D failed")?;
    let min = if mips {
        GL::LINEAR_MIPMAP_LINEAR
    } else {
        GL::LINEAR
    };
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, min as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::LINEAR as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
    if mips {
        gl.generate_mipmap(GL::TEXTURE_2D);
    }
    Ok(t)
}

fn target(gl: &GL, w: i32, h: i32, fmt: Fmt, mips: bool) -> Result<Target, String> {
    let tex = texture(gl, w, h, fmt, mips, None)?;
    let fb = gl.create_framebuffer().ok_or("no framebuffer")?;
    gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&fb));
    gl.framebuffer_texture_2d(
        GL::FRAMEBUFFER,
        GL::COLOR_ATTACHMENT0,
        GL::TEXTURE_2D,
        Some(&tex),
        0,
    );
    let ok = gl.check_framebuffer_status(GL::FRAMEBUFFER) == GL::FRAMEBUFFER_COMPLETE;
    gl.bind_framebuffer(GL::FRAMEBUFFER, None);
    if ok {
        Ok(Target { tex, fb, w, h })
    } else {
        Err(format!("framebuffer incomplete ({w}×{h})"))
    }
}

struct Gfx {
    gl: GL,
    vao: WebGlVertexArrayObject,
    compose: Prog,
    scene: Prog,
    down: Prog,
    up: Prog,
    fin: Prog,
    hdr: bool,
    max_tex: i32,
    map: Option<(WebGlTexture, i32)>,
    vals_a: WebGlTexture,
    vals_b: Option<(WebGlTexture, i32)>,
    lcd: Option<(Target, Target, f64)>,
    frame_t: Option<(Target, Vec<Target>)>,
}

impl Gfx {
    fn new(canvas: &HtmlCanvasElement) -> Result<Gfx, String> {
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(&opts, &"antialias".into(), &false.into()).ok();
        js_sys::Reflect::set(&opts, &"alpha".into(), &false.into()).ok();
        js_sys::Reflect::set(&opts, &"preserveDrawingBuffer".into(), &true.into()).ok();
        let gl: GL = canvas
            .get_context_with_context_options("webgl2", &opts)
            .map_err(|_| "WebGL2 unavailable")?
            .ok_or("WebGL2 unavailable")?
            .dyn_into()
            .map_err(|_| "not a WebGL2 context")?;
        let hdr = gl
            .get_extension("EXT_color_buffer_float")
            .ok()
            .flatten()
            .is_some();
        let max_tex = gl
            .get_parameter(GL::MAX_TEXTURE_SIZE)
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(2048.0) as i32;
        let vao = gl.create_vertex_array().ok_or("no vao")?;
        let vals_a = texture(&gl, N as i32, 1, Fmt::R8, false, Some(&[0u8; N]))?;
        Ok(Gfx {
            compose: Prog::new(&gl, FS_COMPOSE)?,
            scene: Prog::new(&gl, FS_SCENE)?,
            down: Prog::new(&gl, FS_DOWN)?,
            up: Prog::new(&gl, FS_UP)?,
            fin: Prog::new(&gl, FS_FINAL)?,
            hdr,
            max_tex,
            map: None,
            vals_a,
            vals_b: None,
            lcd: None,
            frame_t: None,
            vao,
            gl,
        })
    }

    fn nearest(&self, t: &WebGlTexture) {
        let gl = &self.gl;
        gl.bind_texture(GL::TEXTURE_2D, Some(t));
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
    }

    /// New weight layout: the LCD map, the masks, LCD texture size.
    fn set_layout(&mut self, rgba: &[u8], size: usize, masks: &[Vec<f64>]) -> Result<(), String> {
        let gl = self.gl.clone();
        let size = size as i32;
        let map = texture(&gl, size, size, Fmt::Rgba8, false, Some(rgba))?;
        self.nearest(&map);
        if let Some((old, _)) = self.map.replace((map, size)) {
            gl.delete_texture(Some(&old));
        }
        let tiles = masks.len() as i32;
        let bytes: Vec<u8> = masks
            .iter()
            .flat_map(|m| m.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect();
        let vb = texture(&gl, N as i32, tiles, Fmt::R8, false, Some(&bytes))?;
        self.nearest(&vb);
        if let Some((old, _)) = self.vals_b.replace((vb, tiles)) {
            gl.delete_texture(Some(&old));
        }
        let cap = 2048.min(self.max_tex);
        let k = (cap / size).max(1);
        let res = (size * k).min(self.max_tex);
        let need = self.lcd.as_ref().is_none_or(|l| l.0.w != res);
        if need {
            if let Some((a, b, _)) = self.lcd.take() {
                for t in [a, b] {
                    gl.delete_texture(Some(&t.tex));
                    gl.delete_framebuffer(Some(&t.fb));
                }
            }
            self.lcd = Some((
                target(&gl, res, res, Fmt::R8, true)?,
                target(&gl, res, res, Fmt::R8, true)?,
                k as f64,
            ));
        } else if let Some(l) = self.lcd.as_mut() {
            l.2 = k as f64;
        }
        Ok(())
    }

    fn set_inputs(&self, x: &[f64]) {
        let gl = &self.gl;
        let bytes: Vec<u8> = x
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.vals_a));
        gl.pixel_storei(GL::UNPACK_ALIGNMENT, 1);
        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            0,
            0,
            0,
            N as i32,
            1,
            GL::RED,
            GL::UNSIGNED_BYTE,
            Some(&bytes),
        )
        .ok();
    }

    /// Draw one LCD's active area into its texture and rebuild the mips.
    fn compose_lcd(&self, weights: bool) {
        let (Some((map, size)), Some((a, b, k)), Some((vb, _))) =
            (&self.map, &self.lcd, &self.vals_b)
        else {
            return;
        };
        let gl = &self.gl;
        let t = if weights { b } else { a };
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&t.fb));
        gl.viewport(0, 0, t.w, t.h);
        gl.use_program(Some(&self.compose.p));
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(map));
        gl.active_texture(GL::TEXTURE1);
        gl.bind_texture(
            GL::TEXTURE_2D,
            Some(if weights { vb } else { &self.vals_a }),
        );
        self.compose.i1(gl, "u_map", 0);
        self.compose.i1(gl, "u_vals", 1);
        self.compose.f1(gl, "u_size", *size as f64);
        self.compose.f1(gl, "u_k", *k);
        self.compose.i1(gl, "u_weights", weights as i32);
        gl.bind_vertex_array(Some(&self.vao));
        gl.draw_arrays(GL::TRIANGLES, 0, 3);
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&t.tex));
        gl.generate_mipmap(GL::TEXTURE_2D);
    }

    fn ensure_frame(&mut self, w: i32, h: i32) -> Result<(), String> {
        if self
            .frame_t
            .as_ref()
            .is_some_and(|f| f.0.w == w && f.0.h == h)
        {
            return Ok(());
        }
        let gl = self.gl.clone();
        if let Some((s, b)) = self.frame_t.take() {
            for t in std::iter::once(s).chain(b) {
                gl.delete_texture(Some(&t.tex));
                gl.delete_framebuffer(Some(&t.fb));
            }
        }
        let fmt = if self.hdr { Fmt::Hdr } else { Fmt::Rgba8 };
        let scene = target(&gl, w, h, fmt, false)?;
        let mut bloom = Vec::new();
        let (mut bw, mut bh) = (w, h);
        for _ in 0..BLOOM_LEVELS {
            bw = (bw / 2).max(1);
            bh = (bh / 2).max(1);
            bloom.push(target(&gl, bw, bh, fmt, false)?);
        }
        self.frame_t = Some((scene, bloom));
        Ok(())
    }
}

/// Animated bench geometry: x of LCD A, lens 1 flange, lens 2 flange,
/// LCD B, camera front; lens 2 lift; where the beam leaves the relay;
/// f-number; aperture-ring turn.
type Pose = [f64; 9];

fn pose(rig: &Rig) -> Pose {
    let f = rig.fnum();
    let turn = (rig.stop as f64 - 3.0) * 0.16;
    if rig.symmetric {
        [-80.0, -34.5, 43.5, 89.0, 189.0, 0.0, 50.5, f, turn]
    } else {
        // One lens at 2f–2f: 100 mm from each panel's principal plane.
        [-80.0, 0.0, 43.5, 120.0, 220.0, 260.0, 37.0, f, turn]
    }
}

struct View {
    yaw: f64,
    pitch: f64,
    dist: f64,
    target: [f64; 3],
    goal: (f64, f64, f64, [f64; 3]),
    aperture: f64,
    drag: Option<(f64, f64)>,
    idle_s: f64,
}

impl View {
    fn preset(k: usize, xb: f64) -> (f64, f64, f64, [f64; 3], f64) {
        match k {
            1 => (0.78, 0.24, 270.0, [5.0, AX - 4.0, 0.0], 3.2),
            2 => (1.43, 0.11, 78.0, [xb, AX, 0.0], 2.6),
            _ => (0.24, 0.3, 500.0, [55.0, 64.0, 0.0], 2.2),
        }
    }
    fn eye(&self) -> [f64; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [
            self.target[0] + self.dist * cp * self.yaw.sin(),
            self.target[1] + self.dist * sp,
            self.target[2] + self.dist * cp * self.yaw.cos(),
        ]
    }
}

struct Ui {
    doc: Document,
}

impl Ui {
    fn text(&self, id: &str, s: &str) {
        if let Some(e) = self.doc.get_element_by_id(id) {
            e.set_text_content(Some(s));
        }
    }
    fn pressed(&self, id: &str, on: bool) {
        if let Some(e) = self.doc.get_element_by_id(id) {
            e.set_attribute("aria-pressed", if on { "true" } else { "false" })
                .ok();
        }
    }
}

struct App {
    ui: Ui,
    gfx: Gfx,
    canvas: HtmlCanvasElement,
    io: Ctx2d,
    io_canvas: HtmlCanvasElement,
    layouts: Layouts,
    rig: Rig,
    engine: Engine,
    workload: Workload,
    input: Input,
    rng: Rng,
    t_sim: f64,
    sym_acc: f64,
    symbols: u64,
    prev: Vec<f64>,
    x: Vec<f64>,
    last: Option<Readout>,
    errs: VecDeque<f64>,
    pred: f64,
    field: f64,
    half_mm: f64,
    lit: f64,
    climb: Option<RigClimb>,
    evolving: bool,
    pending: Option<Rig>,
    last_apply_s: f64,
    last_eval_s: f64,
    best_at: usize,
    clock_s: f64,
    pose_now: Pose,
    view: View,
    scale_idx: usize,
    dt_ema: f64,
    last_ms: Option<f64>,
    frame: u64,
    still: bool,
    /// The viewer asked for reduced motion: no idle orbit.
    calm: bool,
    dirty_a: bool,
    dirty_b: bool,
    hud_s: f64,
    perf: web_sys::Performance,
}

impl App {
    fn new(
        doc: Document,
        perf: web_sys::Performance,
        params: &HashMap<String, String>,
    ) -> Result<App, String> {
        let canvas: HtmlCanvasElement = doc
            .get_element_by_id("gl")
            .ok_or("#gl missing")?
            .dyn_into()
            .map_err(|_| "#gl")?;
        let io_canvas: HtmlCanvasElement = doc
            .get_element_by_id("io")
            .ok_or("#io missing")?
            .dyn_into()
            .map_err(|_| "#io")?;
        let io: Ctx2d = io_canvas
            .get_context("2d")
            .ok()
            .flatten()
            .ok_or("no 2d")?
            .dyn_into()
            .map_err(|_| "2d")?;
        let gfx = Gfx::new(&canvas)?;
        let mut layouts = Layouts::default();
        let rig = if params.get("rig").map(String::as_str) == Some("naive") {
            Rig::naive()
        } else {
            Rig::as_built()
        };
        let pick = |k: &str, n: usize| {
            params
                .get(k)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
                .min(n - 1)
        };
        let workload = Workload::ALL[pick("workload", 4)];
        let input = Input::ALL[pick("input", 4)];
        let engine = Engine::staged(&rig.design(), &mut layouts, workload.weights(), 1);
        let view_k = pick("view", 3);
        let p = pose(&rig);
        let (yaw, pitch, dist, tgt, ap) = View::preset(view_k, p[3]);
        let t0: f64 = params.get("t").and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let mut app = App {
            ui: Ui { doc },
            gfx,
            canvas,
            io,
            io_canvas,
            pred: rig_enob(&rig, &mut layouts),
            field: field_mm(&rig.design(), &mut layouts),
            layouts,
            rig,
            engine,
            workload,
            input,
            rng: Rng::new(7),
            t_sim: t0,
            sym_acc: 0.0,
            symbols: 0,
            prev: input.image(t0),
            x: input.image(t0),
            last: None,
            errs: VecDeque::new(),
            half_mm: 1.0,
            lit: 0.3,
            climb: None,
            evolving: false,
            pending: None,
            last_apply_s: 0.0,
            last_eval_s: 0.0,
            best_at: 0,
            clock_s: 0.0,
            pose_now: p,
            view: View {
                yaw,
                pitch,
                dist,
                target: tgt,
                goal: (yaw, pitch, dist, tgt),
                aperture: ap,
                drag: None,
                idle_s: 0.0,
            },
            scale_idx: if params.contains_key("still") { 4 } else { 2 },
            dt_ema: 16.0,
            last_ms: None,
            frame: 0,
            still: params.contains_key("still"),
            calm: web_sys::window()
                .and_then(|w| {
                    w.match_media("(prefers-reduced-motion: reduce)")
                        .ok()
                        .flatten()
                })
                .is_some_and(|m| m.matches()),
            dirty_a: true,
            dirty_b: true,
            hud_s: 0.0,
            perf,
        };
        if let Some(steps) = params.get("evolve").and_then(|v| v.parse::<usize>().ok()) {
            // Jump ahead: climb from the naive bench, show where it got to.
            let mut c = RigClimb::new(Rig::naive(), &mut app.layouts, 5);
            for _ in 0..steps {
                c.step(&mut app.layouts);
            }
            app.rig = c.best.0;
            app.engine = Engine::staged(
                &app.rig.design(),
                &mut app.layouts,
                app.workload.weights(),
                1,
            );
            app.pred = c.best.1;
            app.field = field_mm(&app.rig.design(), &mut app.layouts);
            app.pose_now = pose(&app.rig);
        }
        app.upload_layout()?;
        app.step_symbol();
        app.sync_controls();
        Ok(app)
    }

    fn upload_layout(&mut self) -> Result<(), String> {
        let map = lcd_map(&self.engine, self.rig.cell as usize);
        self.half_mm = 0.5 * map.side_mm;
        let lit = map.rgba.chunks(4).filter(|p| p[2] > 0).count() as f64;
        self.lit = lit / (map.size * map.size) as f64 * 0.64;
        self.gfx
            .set_layout(&map.rgba, map.size, &self.engine.masks)?;
        self.dirty_a = true;
        self.dirty_b = true;
        Ok(())
    }

    /// Rebuild the engine for the current rig and workload.
    fn rebuild(&mut self) {
        self.engine = Engine::staged(
            &self.rig.design(),
            &mut self.layouts,
            self.workload.weights(),
            1 + self.symbols,
        );
        self.errs.clear();
        if let Err(e) = self.upload_layout() {
            self.ui.text("state", &format!("GPU: {e}"));
        }
        self.sync_controls();
    }

    fn set_rig(&mut self, rig: Rig) {
        self.rig = rig;
        self.pred = rig_enob(&rig, &mut self.layouts);
        self.field = field_mm(&rig.design(), &mut self.layouts);
        self.rebuild();
    }

    fn step_symbol(&mut self) {
        self.t_sim += 1.0 / BAUD;
        self.x = self.input.image(self.t_sim);
        let r = self.engine.run(&self.x, &self.prev, &mut self.rng);
        let mse =
            r.y.iter()
                .zip(&r.ideal)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / r.y.len() as f64;
        self.errs.push_back(mse);
        if self.errs.len() > 50 {
            self.errs.pop_front();
        }
        self.prev = self.x.clone();
        self.symbols += 1;
        self.gfx.set_inputs(&self.x);
        self.dirty_a = true;
        self.last = Some(r);
        self.draw_io();
    }

    fn live_bits(&self) -> f64 {
        let mse = self.errs.iter().sum::<f64>() / self.errs.len().max(1) as f64;
        (self.engine.output_span() / (12f64.sqrt() * mse.sqrt().max(1e-12))).log2()
    }

    fn draw_io(&self) {
        let Some(r) = &self.last else { return };
        let ctx = &self.io;
        let w = self.io_canvas.width() as f64;
        let cell = (w / (3.0 * SIDE as f64 + 4.0)).floor();
        let gap = 2.0 * cell;
        ctx.set_fill_style_str("#05070c");
        ctx.fill_rect(0.0, 0.0, w, self.io_canvas.height() as f64);
        let spectrum = self.workload == Workload::Spectrum;
        let (lo, hi) = r
            .ideal
            .iter()
            .fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let amax = r.ideal.iter().fold(1e-12f64, |a, v| a.max(v.abs()));
        let shade = |v: f64| -> f64 {
            if spectrum {
                (v.abs() / amax).sqrt().clamp(0.0, 1.0)
            } else {
                ((v - lo) / (hi - lo).max(1e-12)).clamp(0.0, 1.0)
            }
        };
        for (k, img) in [&self.x, &r.y, &r.ideal].iter().enumerate() {
            let ox = k as f64 * (SIDE as f64 * cell + gap);
            for j in 0..N {
                let v = if k == 0 { img[j] } else { shade(img[j]) };
                let (g, rb) = if k == 0 { (v, v) } else { (v, v * 0.35) };
                ctx.set_fill_style_str(&format!(
                    "rgb({},{},{})",
                    (rb * 255.0) as u8,
                    (g * 255.0) as u8,
                    (rb * 255.0 * 0.9) as u8
                ));
                let (cx, cy) = ((j % SIDE) as f64, (j / SIDE) as f64);
                ctx.fill_rect(ox + cx * cell, cy * cell, cell - 0.5, cell - 0.5);
            }
        }
    }

    fn sync_controls(&self) {
        for (k, w) in Workload::ALL.iter().enumerate() {
            self.ui.pressed(&format!("wl-{k}"), *w == self.workload);
        }
        for (k, i) in Input::ALL.iter().enumerate() {
            self.ui.pressed(&format!("in-{k}"), *i == self.input);
        }
        self.ui.pressed("evolve", self.evolving);
        if !self.evolving {
            self.ui.text(
                "state",
                if self.rig == Rig::as_built() {
                    "as costed in TABLETOP.md"
                } else if self.rig == Rig::naive() {
                    "a naive bench: one fast lens, quasi lattice"
                } else {
                    "a custom bench"
                },
            );
        }
        self.ui.pressed("relay", self.rig.symmetric);
        self.ui.text(
            "relay",
            if self.rig.symmetric {
                "Two lenses"
            } else {
                "One lens"
            },
        );
        if let Some(s) = self
            .ui
            .doc
            .get_element_by_id("fstop")
            .and_then(|e| e.dyn_into::<HtmlSelectElement>().ok())
        {
            s.set_value(&self.rig.stop.to_string());
        }
        if let Some(s) = self
            .ui
            .doc
            .get_element_by_id("lattice")
            .and_then(|e| e.dyn_into::<HtmlSelectElement>().ok())
        {
            s.set_value(
                &Lattice::ALL
                    .iter()
                    .position(|l| *l == self.rig.lattice)
                    .unwrap_or(0)
                    .to_string(),
            );
        }
    }

    fn evolve_toggle(&mut self) {
        if self.evolving {
            self.evolving = false;
            self.sync_controls();
            self.ui.text("state", "evolution paused");
        } else {
            self.set_rig(Rig::naive());
            self.climb = Some(RigClimb::new(
                Rig::naive(),
                &mut self.layouts,
                5 + self.symbols,
            ));
            self.evolving = true;
            self.best_at = 1;
            self.sync_controls();
            self.ui.text("state", "evolving from a naive bench");
        }
    }

    fn evolve_tick(&mut self) {
        if !self.evolving {
            return;
        }
        let Some(c) = self.climb.as_mut() else { return };
        // About eight new designs a second, so the bench can be watched
        // rearranging itself; revisits of known designs are free.
        if self.clock_s - self.last_eval_s >= 0.12 {
            self.last_eval_s = self.clock_s;
            let (e0, t0) = (c.evals, self.perf.now());
            while c.evals == e0 && self.perf.now() - t0 < 7.0 {
                if c.step(&mut self.layouts) {
                    self.pending = Some(c.best.0);
                    self.best_at = c.evals;
                }
            }
        }
        let (evals, best) = (c.evals, c.best);
        if let Some(r) = self.pending {
            if self.clock_s - self.last_apply_s > 0.9 {
                self.pending = None;
                self.last_apply_s = self.clock_s;
                self.rig = r;
                self.pred = best.1;
                self.field = field_mm(&r.design(), &mut self.layouts);
                self.rebuild();
            }
        }
        self.ui.text(
            "evolve-val",
            &format!("{evals} designs · best {:.2} bits", best.1),
        );
        if evals - self.best_at > 45 && self.pending.is_none() {
            self.evolving = false;
            self.sync_controls();
            self.ui
                .text("state", &format!("converged after {evals} designs"));
        }
    }

    fn resize(&mut self) -> (i32, i32) {
        let win = web_sys::window().unwrap();
        let dpr = win.device_pixel_ratio().min(1.5);
        let (cw, ch) = (
            self.canvas.client_width() as f64,
            self.canvas.client_height() as f64,
        );
        let (w, h) = (
            (cw * dpr).round().max(1.0) as u32,
            (ch * dpr).round().max(1.0) as u32,
        );
        if self.canvas.width() != w || self.canvas.height() != h {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
        }
        let iw = (self.io_canvas.client_width() as f64 * dpr).round() as u32;
        let ih = (self.io_canvas.client_height() as f64 * dpr).round() as u32;
        if iw > 0 && (self.io_canvas.width() != iw || self.io_canvas.height() != ih) {
            self.io_canvas.set_width(iw);
            self.io_canvas.set_height(ih);
            self.draw_io();
        }
        (w as i32, h as i32)
    }

    fn frame(&mut self, now_ms: f64) {
        if self.still && self.frame > 3 {
            // Stills (screenshots): a few frames to settle, then hold.
            return;
        }
        let dt = self
            .last_ms
            .map_or(0.016, |l| ((now_ms - l) / 1000.0).clamp(0.0, 0.1));
        self.last_ms = Some(now_ms);
        self.clock_s += dt;
        self.frame += 1;
        if !self.still {
            self.dt_ema = 0.95 * self.dt_ema + 0.05 * dt * 1000.0;
            if self.frame.is_multiple_of(30) {
                if self.dt_ema > 26.0 && self.scale_idx > 0 {
                    self.scale_idx -= 1;
                } else if self.dt_ema < 15.0 && self.scale_idx + 1 < SCALES.len() {
                    self.scale_idx += 1;
                }
            }
            self.sym_acc += dt * BAUD;
            let mut n = 0;
            while self.sym_acc >= 1.0 && n < 3 {
                self.sym_acc -= 1.0;
                self.step_symbol();
                n += 1;
            }
            self.sym_acc = self.sym_acc.min(1.0);
            self.evolve_tick();
        }

        // Ease the bench and the camera toward their goals.
        let goal = pose(&self.rig);
        let k = if self.still {
            1.0
        } else {
            1.0 - (-3.0 * dt).exp()
        };
        for (c, g) in self.pose_now.iter_mut().zip(goal) {
            *c += (g - *c) * k;
        }
        let v = &mut self.view;
        if v.drag.is_none() {
            v.idle_s += dt;
        }
        let kv = if self.still {
            1.0
        } else {
            1.0 - (-2.5 * dt).exp()
        };
        if v.drag.is_none() {
            v.yaw += (v.goal.0 - v.yaw) * kv;
            v.pitch += (v.goal.1 - v.pitch) * kv;
            v.dist += (v.goal.2 - v.dist) * kv;
        }
        for i in 0..3 {
            v.target[i] += (v.goal.3[i] - v.target[i]) * kv;
        }
        if v.idle_s > 8.0 && !self.still && !self.calm {
            v.goal.0 += 0.035 * dt * (0.21 * self.clock_s).sin().signum();
        }

        let (w, h) = self.resize();
        self.render(w, h);
        if self.clock_s - self.hud_s > 0.12 || self.still {
            self.hud_s = self.clock_s;
            self.hud();
        }
    }

    fn render(&mut self, w: i32, h: i32) {
        let s = SCALES[self.scale_idx];
        let (sw, sh) = (((w as f64) * s) as i32, ((h as f64) * s) as i32);
        if let Err(e) = self.gfx.ensure_frame(sw.max(1), sh.max(1)) {
            self.ui.text("state", &format!("GPU: {e}"));
            return;
        }
        if self.dirty_b {
            self.gfx.compose_lcd(true);
            self.dirty_b = false;
        }
        if self.dirty_a {
            self.gfx.compose_lcd(false);
            self.dirty_a = false;
        }
        let gfx = &self.gfx;
        let gl = &gfx.gl;
        let (Some((scene_t, bloom)), Some((lcd_a, lcd_b, _))) = (&gfx.frame_t, &gfx.lcd) else {
            return;
        };
        gl.bind_vertex_array(Some(&gfx.vao));
        gl.disable(GL::BLEND);

        // Scene.
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&scene_t.fb));
        gl.viewport(0, 0, scene_t.w, scene_t.h);
        let sp = &gfx.scene;
        gl.use_program(Some(&sp.p));
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&lcd_a.tex));
        gl.active_texture(GL::TEXTURE1);
        gl.bind_texture(GL::TEXTURE_2D, Some(&lcd_b.tex));
        sp.i1(gl, "u_texA", 0);
        sp.i1(gl, "u_texB", 1);
        sp.f2(gl, "u_res", scene_t.w as f64, scene_t.h as f64);
        sp.f3(gl, "u_eye", self.view.eye());
        sp.f3(gl, "u_target", self.view.target);
        // Keep the bench's width in frame on portrait screens.
        let aspect = scene_t.w as f64 / scene_t.h as f64;
        sp.f1(
            gl,
            "u_fov",
            0.52f64.max(2.0 * (0.36f64.tan() / aspect).atan()).min(1.25),
        );
        sp.f1(gl, "u_time", self.clock_s);
        sp.f1(gl, "u_frame", self.frame as f64);
        let p = &self.pose_now;
        sp.f4(gl, "u_pos", [p[0], p[1], p[2], p[3]]);
        sp.f4(gl, "u_pos2", [p[4], p[5], p[6], p[7]]);
        let mean_x = self.x.iter().sum::<f64>() / N as f64;
        let mean_mask =
            self.engine.masks.iter().flatten().sum::<f64>() / (self.engine.masks.len() * N) as f64;
        sp.f4(
            gl,
            "u_lcd",
            [
                self.half_mm,
                mean_x * self.lit,
                mean_x * mean_mask * self.lit,
                p[8],
            ],
        );
        sp.f1(gl, "u_tex", lcd_a.w as f64);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // Bloom down and up.
        gl.use_program(Some(&gfx.down.p));
        gfx.down.i1(gl, "u_src", 0);
        gl.active_texture(GL::TEXTURE0);
        for (i, t) in bloom.iter().enumerate() {
            gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&t.fb));
            gl.viewport(0, 0, t.w, t.h);
            gl.bind_texture(
                GL::TEXTURE_2D,
                Some(if i == 0 {
                    &scene_t.tex
                } else {
                    &bloom[i - 1].tex
                }),
            );
            gfx.down.f2(gl, "u_dst", t.w as f64, t.h as f64);
            gfx.down.i1(gl, "u_first", (i == 0) as i32);
            gl.draw_arrays(GL::TRIANGLES, 0, 3);
        }
        gl.use_program(Some(&gfx.up.p));
        gfx.up.i1(gl, "u_src", 0);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::ONE, GL::ONE);
        for i in (0..bloom.len() - 1).rev() {
            let t = &bloom[i];
            gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&t.fb));
            gl.viewport(0, 0, t.w, t.h);
            gl.bind_texture(GL::TEXTURE_2D, Some(&bloom[i + 1].tex));
            gfx.up.f2(gl, "u_dst", t.w as f64, t.h as f64);
            gfx.up.f1(gl, "u_weight", 0.85);
            gl.draw_arrays(GL::TRIANGLES, 0, 3);
        }
        gl.disable(GL::BLEND);

        // Lens and film, to the canvas.
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);
        gl.viewport(0, 0, w, h);
        let fp = &gfx.fin;
        gl.use_program(Some(&fp.p));
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&scene_t.tex));
        gl.active_texture(GL::TEXTURE1);
        gl.bind_texture(GL::TEXTURE_2D, Some(&bloom[0].tex));
        fp.i1(gl, "u_scene", 0);
        fp.i1(gl, "u_bloom", 1);
        fp.f2(gl, "u_res", w as f64, h as f64);
        fp.f2(gl, "u_src", scene_t.w as f64, scene_t.h as f64);
        fp.f1(gl, "u_focus", self.view.dist);
        fp.f1(
            gl,
            "u_aperture",
            self.view.aperture * scene_t.h as f64 / 900.0,
        );
        fp.f1(gl, "u_exposure", 1.15);
        fp.f1(gl, "u_frame", self.frame as f64);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);
    }

    fn hud(&self) {
        let bits = self.live_bits();
        self.ui.text("bits", &format!("{:.2}", bits.max(0.0)));
        self.ui.text("pred", &format!("{:.2}", self.pred));
        self.ui
            .text("fstop-val", &format!("f/{}", F_STOPS[self.rig.stop]));
        self.ui.text("lattice-val", self.rig.lattice.short());
        self.ui.text(
            "exposure",
            &format!("{:.0} %", self.engine.exposure * 100.0),
        );
        self.ui.text("sym", &format!("{}", self.symbols));
        self.ui.text("symc", &format!("{}", self.symbols));
        self.ui.text(
            "outs",
            &format!(
                "{} · {} tiles",
                self.engine.design.m_out,
                self.engine.tiles()
            ),
        );
        self.ui.text(
            "field",
            &if self.field > IMAGE_CIRCLE_MM {
                format!("{:.0} mm · vignettes", self.field)
            } else {
                format!("{:.0} / {IMAGE_CIRCLE_MM:.0} mm", self.field)
            },
        );
        let macs = (N * N) as f64 * self.symbols as f64;
        self.ui.text(
            "macs",
            &match macs {
                m if m >= 1e9 => format!("{:.2} G", m / 1e9),
                m if m >= 1e6 => format!("{:.1} M", m / 1e6),
                m => format!("{:.0} k", m / 1e3),
            },
        );
        let quality = if bits >= 5.0 {
            "clean"
        } else if bits >= 3.0 {
            "noisy"
        } else {
            "lost in noise"
        };
        self.ui.text("quality", quality);
    }
}

fn params() -> HashMap<String, String> {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    search
        .trim_start_matches('?')
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            let mut it = kv.splitn(2, '=');
            (
                it.next().unwrap_or("").to_string(),
                it.next().unwrap_or("1").to_string(),
            )
        })
        .collect()
}

#[wasm_bindgen]
pub fn start_glass() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let doc = window.document().ok_or("no document")?;
    let perf = window.performance().ok_or("no performance")?;
    let params = params();
    let app = Rc::new(RefCell::new(
        App::new(doc.clone(), perf, &params).map_err(|e| JsValue::from_str(&e))?,
    ));

    let on = |id: &str, ev: &str, f: Box<dyn FnMut(web_sys::Event)>| {
        let cb = Closure::wrap(f);
        if let Some(el) = doc.get_element_by_id(id) {
            el.add_event_listener_with_callback(ev, cb.as_ref().unchecked_ref())
                .ok();
        }
        cb.forget();
    };
    for k in 0..Workload::ALL.len() {
        let a = app.clone();
        on(
            &format!("wl-{k}"),
            "click",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                a.workload = Workload::ALL[k];
                a.rebuild();
            }),
        );
    }
    for k in 0..Input::ALL.len() {
        let a = app.clone();
        on(
            &format!("in-{k}"),
            "click",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                a.input = Input::ALL[k];
                a.sync_controls();
            }),
        );
    }
    for k in 0..3 {
        let a = app.clone();
        on(
            &format!("view-{k}"),
            "click",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                let xb = pose(&a.rig)[3];
                let (yaw, pitch, dist, tgt, ap) = View::preset(k, xb);
                a.view.goal = (yaw, pitch, dist, tgt);
                a.view.aperture = ap;
                a.view.idle_s = 0.0;
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "evolve",
            "click",
            Box::new(move |_| a.borrow_mut().evolve_toggle()),
        );
        let a = app.clone();
        on(
            "asbuilt",
            "click",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                a.evolving = false;
                a.set_rig(Rig::as_built());
            }),
        );
        let a = app.clone();
        on(
            "relay",
            "click",
            Box::new(move |_| {
                let mut a = a.borrow_mut();
                let mut r = a.rig;
                r.symmetric = !r.symmetric;
                a.evolving = false;
                a.set_rig(r);
            }),
        );
        let a = app.clone();
        on(
            "fstop",
            "change",
            Box::new(move |e| {
                let v: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                let mut r = a.rig;
                r.stop = v.value().parse().unwrap_or(2);
                a.evolving = false;
                a.set_rig(r);
            }),
        );
        let a = app.clone();
        on(
            "lattice",
            "change",
            Box::new(move |e| {
                let v: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
                let mut a = a.borrow_mut();
                let mut r = a.rig;
                r.lattice = Lattice::ALL[v.value().parse::<usize>().unwrap_or(0).min(3)];
                r.cell = if r.lattice == Lattice::Square {
                    r.cell
                } else {
                    2
                };
                a.evolving = false;
                a.set_rig(r);
            }),
        );
    }
    {
        let a = app.clone();
        on(
            "gl",
            "pointerdown",
            Box::new(move |e| {
                let e: PointerEvent = e.dyn_into().unwrap();
                if let Some(t) = e.target().and_then(|t| t.dyn_into::<HtmlElement>().ok()) {
                    t.set_pointer_capture(e.pointer_id()).ok();
                }
                let mut a = a.borrow_mut();
                a.view.drag = Some((e.client_x() as f64, e.client_y() as f64));
                a.view.idle_s = 0.0;
            }),
        );
        let a = app.clone();
        on(
            "gl",
            "pointermove",
            Box::new(move |e| {
                let e: PointerEvent = e.dyn_into().unwrap();
                let mut a = a.borrow_mut();
                if let Some((x0, y0)) = a.view.drag {
                    let (x, y) = (e.client_x() as f64, e.client_y() as f64);
                    let v = &mut a.view;
                    v.yaw -= (x - x0) * 0.005;
                    v.pitch = (v.pitch + (y - y0) * 0.004).clamp(0.02, 1.35);
                    v.goal.0 = v.yaw;
                    v.goal.1 = v.pitch;
                    v.drag = Some((x, y));
                }
            }),
        );
        for ev in ["pointerup", "pointercancel"] {
            let a = app.clone();
            on("gl", ev, Box::new(move |_| a.borrow_mut().view.drag = None));
        }
        let a = app.clone();
        on(
            "gl",
            "wheel",
            Box::new(move |e| {
                let e: WheelEvent = e.dyn_into().unwrap();
                let mut a = a.borrow_mut();
                let v = &mut a.view;
                v.goal.2 = (v.goal.2 * (e.delta_y() * 0.0012).exp()).clamp(45.0, 1400.0);
                v.idle_s = 0.0;
            }),
        );
    }

    let f: FrameLoop = Rc::new(RefCell::new(None));
    let g = f.clone();
    let win = window.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |t: f64| {
        app.borrow_mut().frame(t);
        win.request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .ok();
    }) as Box<dyn FnMut(f64)>));
    window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;
    Ok(())
}
