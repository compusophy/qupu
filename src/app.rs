//! egui application — the UI is rendered directly from the simulator's own
//! fields. There is no parallel model; what you see on screen is what the
//! simulator owns.

use eframe::egui;
use egui::{
    Align, Align2, CentralPanel, Color32, CornerRadius, FontFamily, FontId, Frame, Panel, Pos2,
    Rect, Sense, Stroke, StrokeKind, TextStyle, Ui, Vec2,
};

use crate::circuit::{GateKind, Op, PRESETS};
use crate::sim::Simulator;

// ----- palette -----
const BG_0:        Color32 = Color32::from_rgb(6, 8, 16);
const BG_1:        Color32 = Color32::from_rgb(10, 15, 26);
const BG_2:        Color32 = Color32::from_rgb(14, 20, 34);
const BG_3:        Color32 = Color32::from_rgb(19, 26, 44);
const LINE:        Color32 = Color32::from_rgb(30, 38, 64);
const LINE_BRIGHT: Color32 = Color32::from_rgb(44, 57, 97);
const TEXT:        Color32 = Color32::from_rgb(214, 227, 244);
const TEXT_DIM:    Color32 = Color32::from_rgb(126, 138, 166);
const TEXT_FAINT:  Color32 = Color32::from_rgb(66, 82, 107);
const CYAN:        Color32 = Color32::from_rgb(0, 255, 209);
const MAGENTA:     Color32 = Color32::from_rgb(255, 45, 173);
const AMBER:       Color32 = Color32::from_rgb(255, 200, 61);

pub struct QupuApp {
    sim: Simulator,
    target_q: u32,
    ctrl_q: u32,
    theta: f64,
    requested_qubits: u32,
    requested_cores: u32,
    speed_hz: f32,
    last_norm_warn: bool,
    theme_applied: bool,
}

impl QupuApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let n_qubits = 6u32;
        let n_cores = 8u32;
        let mut sim = Simulator::new(n_qubits, n_cores);
        sim.clock_hz = 8.0;
        Self {
            sim,
            target_q: 1,
            ctrl_q: 0,
            theta: std::f64::consts::FRAC_PI_2,
            requested_qubits: n_qubits,
            requested_cores: n_cores,
            speed_hz: 8.0,
            last_norm_warn: false,
            theme_applied: false,
        }
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if self.theme_applied { return; }
        self.theme_applied = true;

        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(TEXT);
        visuals.panel_fill = BG_1;
        visuals.window_fill = BG_2;
        visuals.extreme_bg_color = BG_0;
        visuals.faint_bg_color = BG_2;
        visuals.widgets.noninteractive.bg_fill = BG_2;
        visuals.widgets.inactive.bg_fill = BG_3;
        visuals.widgets.inactive.weak_bg_fill = BG_3;
        visuals.widgets.hovered.bg_fill = BG_3;
        visuals.widgets.active.bg_fill = BG_3;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, CYAN);
        visuals.widgets.active.fg_stroke = Stroke::new(1.0, CYAN);
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, LINE_BRIGHT);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, CYAN);
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, CYAN);
        visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(0, 255, 209, 40);
        visuals.selection.stroke = Stroke::new(1.0, CYAN);
        visuals.window_stroke = Stroke::new(1.0, LINE);
        visuals.window_corner_radius = CornerRadius::same(4);
        visuals.menu_corner_radius = CornerRadius::same(4);
        ctx.set_visuals(visuals);

        let mut style = (*ctx.style()).clone();
        style.text_styles.insert(TextStyle::Body, FontId::new(13.0, FontFamily::Monospace));
        style.text_styles.insert(TextStyle::Button, FontId::new(13.0, FontFamily::Monospace));
        style.text_styles.insert(TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace));
        style.text_styles.insert(TextStyle::Heading, FontId::new(14.0, FontFamily::Monospace));
        style.text_styles.insert(TextStyle::Small, FontId::new(11.0, FontFamily::Monospace));
        style.spacing.item_spacing = Vec2::new(6.0, 6.0);
        style.spacing.button_padding = Vec2::new(8.0, 4.0);
        ctx.set_style(style);
    }
}

impl eframe::App for QupuApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let [r, g, b, _] = BG_0.to_array();
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.apply_theme(ui.ctx());

        // Drive the simulation forward by the frame delta.
        self.sim.clock_hz = self.speed_hz;
        let dt = ui.ctx().input(|i| i.stable_dt).min(0.05);
        self.sim.tick(dt);
        ui.ctx().request_repaint();

        let norm = self.sim.norm_sq();
        self.last_norm_warn = (norm - 1.0).abs() > 1e-6;

        // ----- top bar -----
        Panel::top("top")
            .frame(Frame::default().fill(BG_1).inner_margin(egui::Margin::symmetric(20, 12)))
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    paint_brand_dot(ui);
                    ui.label(rich_strong("QuPU"));
                    ui.label(rich_dim("virtual quantum processing unit"));
                    ui.add_space(20.0);
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        let (txt, col) = if self.last_norm_warn {
                            ("// drift detected", AMBER)
                        } else {
                            ("// coherent", CYAN)
                        };
                        ui.colored_label(col, txt);
                        ui.label(rich_faint("  ·  "));
                        ui.label(rich_dim(&format!(
                            "{} qubits · {} amps · {} cores · pure rust → wasm",
                            self.sim.n_qubits, self.sim.n_amps(), self.sim.n_cores,
                        )));
                    });
                });
            });

        // ----- left side panel -----
        Panel::left("left")
            .resizable(false)
            .exact_size(310.0)
            .frame(Frame::default().fill(BG_1).inner_margin(12))
            .show_inside(ui, |ui| {
                self.draw_die_panel(ui);
                ui.add_space(12.0);
                self.draw_telemetry_panel(ui);
                ui.add_space(12.0);
                self.draw_config_panel(ui);
            });

        // ----- bottom panel -----
        Panel::bottom("bottom")
            .resizable(true)
            .default_size(220.0)
            .frame(Frame::default().fill(BG_1).inner_margin(12))
            .show_inside(ui, |ui| {
                self.draw_compose_panel(ui);
                ui.add_space(8.0);
                self.draw_circuit_panel(ui);
            });

        // ----- central panel: state vector -----
        CentralPanel::default()
            .frame(Frame::default().fill(BG_0).inner_margin(0))
            .show_inside(ui, |ui| {
                self.draw_state_panel(ui);
            });
    }
}

// ---------- panels ----------

impl QupuApp {
    fn draw_die_panel(&mut self, ui: &mut Ui) {
        section_header(ui, "die", &format!("{} cores", self.sim.n_cores));
        let cols = (self.sim.n_cores as f32).sqrt().ceil() as u32;
        let cell_w = ((ui.available_width() - (cols as f32 - 1.0) * 6.0) / cols as f32).max(40.0);
        let cell_h = 46.0;
        let mut idx = 0u32;
        let n = self.sim.n_cores;
        let painter = ui.painter().clone();
        for _row in 0..((n + cols - 1) / cols) {
            ui.horizontal(|ui| {
                for _col in 0..cols {
                    if idx >= n { break; }
                    let (rect, _resp) = ui.allocate_exact_size(
                        Vec2::new(cell_w, cell_h), Sense::hover());
                    self.paint_core(&painter, rect, idx);
                    idx += 1;
                }
            });
        }
        ui.add_space(4.0);
        ui.label(rich_faint(
            "Each core is one Rust scheduler slot; tile work animates in real time."
        ));
    }

    fn paint_core(&self, painter: &egui::Painter, rect: Rect, idx: u32) {
        let core = &self.sim.cores[idx as usize];
        let intensity = if core.busy_for > 0.0 {
            1.0
        } else {
            (1.0 - (core.idle_for / 0.4).clamp(0.0, 1.0)).max(0.0)
        };

        painter.rect(
            rect, CornerRadius::same(3),
            BG_3, Stroke::new(1.0, LINE_BRIGHT), StrokeKind::Inside,
        );
        let glow_rect = Rect::from_min_size(
            rect.left_top() + Vec2::new(rect.width() * 0.5, 0.0),
            Vec2::new(rect.width() * 0.5, rect.height() * 0.6),
        );
        painter.rect_filled(
            glow_rect, CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(0, 255, 209, (12.0 + intensity * 30.0) as u8),
        );

        painter.text(
            rect.left_top() + Vec2::new(8.0, 7.0), Align2::LEFT_TOP,
            format!("C{:X}", idx), FontId::monospace(11.0), CYAN,
        );

        let meter_rect = Rect::from_min_size(
            rect.left_top() + Vec2::new(8.0, 22.0),
            Vec2::new(rect.width() - 16.0, 4.0),
        );
        painter.rect_filled(meter_rect, CornerRadius::same(1),
            Color32::from_rgba_unmultiplied(255, 255, 255, 12));
        let fill_w = (meter_rect.width() * (0.05 + intensity * 0.95)).max(1.0);
        let fill_rect = Rect::from_min_size(meter_rect.min, Vec2::new(fill_w, meter_rect.height()));
        let fill_col = lerp_color(CYAN, MAGENTA, intensity * 0.5);
        painter.rect_filled(fill_rect, CornerRadius::same(1), fill_col);

        let label = if core.busy_for > 0.0 {
            if let Some(t) = core.last_tile {
                format!("{} t={}", t.op.gate.label(), t.op.target)
            } else { "computing".into() }
        } else if intensity > 0.05 { "settling".into() } else { "idle".into() };
        let lab_col = if intensity > 0.05 { CYAN } else { TEXT_FAINT };
        painter.text(
            rect.left_top() + Vec2::new(8.0, 30.0), Align2::LEFT_TOP,
            label, FontId::monospace(10.0), lab_col,
        );
    }

    fn draw_telemetry_panel(&self, ui: &mut Ui) {
        section_header(ui, "telemetry", "live");
        egui::Grid::new("telem").num_columns(2).spacing([12.0, 4.0]).striped(false)
            .show(ui, |ui| {
                row(ui, "qubits", &self.sim.n_qubits.to_string(), TEXT);
                row(ui, "amplitudes", &self.sim.n_amps().to_string(), TEXT);
                row(ui, "gates done", &self.sim.gates_completed.to_string(), TEXT);
                row(ui, "tiles done", &self.sim.tiles_completed.to_string(), TEXT);
                row(ui, "wall time", &format!("{:.2} s", self.sim.wall_seconds), AMBER);
                let core_seconds: f64 = self.sim.cores.iter().map(|c| c.total_busy).sum();
                let busy_frac = if self.sim.wall_seconds > 0.0 {
                    core_seconds / (self.sim.wall_seconds * self.sim.n_cores as f64)
                } else { 0.0 };
                row(ui, "core utilization",
                    &format!("{:.0} %", busy_frac * 100.0),
                    MAGENTA);
                row(ui, "norm² (Σ)",
                    &format!("{:.8}", self.sim.norm_sq()),
                    if self.last_norm_warn { AMBER } else { CYAN });
            });
    }

    fn draw_config_panel(&mut self, ui: &mut Ui) {
        section_header(ui, "config", "");
        ui.horizontal(|ui| {
            ui.label(rich_dim("qubits"));
            ui.add(egui::Slider::new(&mut self.requested_qubits, 2..=8).show_value(true));
        });
        ui.horizontal(|ui| {
            ui.label(rich_dim("cores  "));
            ui.add(egui::Slider::new(&mut self.requested_cores, 1..=16).show_value(true));
        });
        ui.horizontal(|ui| {
            ui.label(rich_dim("clock   "));
            ui.add(egui::Slider::new(&mut self.speed_hz, 1.0..=60.0)
                .suffix(" Hz").show_value(true));
        });
        if self.requested_qubits != self.sim.n_qubits || self.requested_cores != self.sim.n_cores {
            if ui.button("apply (resets state)").clicked() {
                let q = self.requested_qubits;
                let c = self.requested_cores;
                self.sim = Simulator::new(q, c);
                self.sim.clock_hz = self.speed_hz;
                self.requested_qubits = q;
                self.requested_cores = self.sim.n_cores;
                if self.target_q >= q { self.target_q = q - 1; }
                if self.ctrl_q >= q { self.ctrl_q = 0; }
            }
        }
        ui.add_space(4.0);
        ui.label(rich_faint(
            "Tile alignment caps parallelism on the top qubit at 1 core — same constraint distributed quantum simulators face.",
        ));
    }

    // ---------- state vector grid ----------

    fn draw_state_panel(&self, ui: &mut Ui) {
        let avail = ui.available_rect_before_wrap();
        let painter = ui.painter().clone();

        let header_h = 32.0;
        let header_rect = Rect::from_min_size(avail.min, Vec2::new(avail.width(), header_h));
        painter.rect_filled(header_rect, CornerRadius::ZERO, BG_1);
        painter.line_segment(
            [header_rect.left_bottom(), header_rect.right_bottom()],
            Stroke::new(1.0, LINE),
        );
        painter.text(
            header_rect.left_top() + Vec2::new(16.0, 9.0), Align2::LEFT_TOP,
            "STATE VECTOR", FontId::monospace(11.0), TEXT,
        );
        painter.text(
            header_rect.right_top() + Vec2::new(-16.0, 9.0), Align2::RIGHT_TOP,
            "size = |amplitude|   ·   hue = phase",
            FontId::monospace(11.0), TEXT_DIM,
        );

        let body = Rect::from_min_max(
            avail.left_top() + Vec2::new(12.0, header_h + 12.0),
            avail.right_bottom() - Vec2::new(12.0, 12.0),
        );

        let n = self.sim.n_amps() as usize;
        let cols = (n as f32).sqrt().ceil() as usize;
        let rows = (n + cols - 1) / cols;
        let cell_w = body.width()  / cols as f32;
        let cell_h = body.height() / rows as f32;
        let cell = cell_w.min(cell_h);
        let grid_w = cell * cols as f32;
        let grid_h = cell * rows as f32;
        let origin = body.center() - Vec2::new(grid_w * 0.5, grid_h * 0.5);

        let max_disc_r = cell * 0.42;
        for i in 0..n {
            let c = (i % cols) as f32;
            let r = (i / cols) as f32;
            let center = origin + Vec2::new(c * cell + cell * 0.5, r * cell + cell * 0.5);
            let re = self.sim.state[2 * i];
            let im = self.sim.state[2 * i + 1];
            let mag = (re * re + im * im).sqrt() as f32;
            let phase = im.atan2(re) as f32;

            let cell_rect = Rect::from_center_size(center, Vec2::splat(cell - 2.0));
            painter.rect(
                cell_rect, CornerRadius::same(2),
                Color32::from_rgba_unmultiplied(255, 255, 255, 4),
                Stroke::new(0.6, LINE),
                StrokeKind::Inside,
            );

            let radius = (max_disc_r * mag).max(1.5);
            let hue = (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let col = hsv(hue, 0.95, (0.30 + mag * 0.65).clamp(0.0, 1.0));
            let glow_alpha = ((mag * 0.6 + 0.05) * 255.0) as u8;
            painter.circle_filled(
                center, radius * 1.6,
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), (glow_alpha / 3).max(8)),
            );
            painter.circle_filled(center, radius, col);

            if cell > 28.0 {
                let label = format!("|{:0width$b}⟩", i, width = self.sim.n_qubits as usize);
                painter.text(
                    center + Vec2::new(0.0, cell * 0.34),
                    Align2::CENTER_CENTER, label,
                    FontId::monospace((cell * 0.18).clamp(8.5, 11.0)),
                    TEXT_FAINT,
                );
            }
        }
    }

    // ---------- compose toolbar ----------

    fn draw_compose_panel(&mut self, ui: &mut Ui) {
        section_header(ui, "compose", &format!("{} ops in circuit", self.sim.circuit.len()));
        ui.horizontal_wrapped(|ui| {
            ui.label(rich_faint("1q "));
            for g in [GateKind::H, GateKind::X, GateKind::Y, GateKind::Z, GateKind::S, GateKind::T] {
                if ui.button(g.label()).clicked() { self.append_one(g); }
            }
            ui.label(rich_faint("  rot "));
            for g in [GateKind::RX, GateKind::RY, GateKind::RZ] {
                if ui.button(g.label()).clicked() { self.append_rot(g); }
            }
            ui.label(rich_faint("  2q "));
            for g in [GateKind::CNOT, GateKind::CZ, GateKind::SWAP] {
                if ui.button(g.label()).clicked() { self.append_two(g); }
            }
            ui.separator();
            ui.label(rich_dim("target"));
            ui.add(egui::DragValue::new(&mut self.target_q).range(0..=(self.sim.n_qubits - 1)));
            ui.label(rich_dim("ctrl"));
            ui.add(egui::DragValue::new(&mut self.ctrl_q).range(0..=(self.sim.n_qubits - 1)));
            ui.label(rich_dim("θ"));
            ui.add(egui::DragValue::new(&mut self.theta).speed(0.05).max_decimals(4));

            ui.separator();
            let run_text = if self.sim.running { "⏸ pause" } else { "▶ run" };
            if ui.button(run_text).clicked() {
                if self.sim.running { self.sim.stop_running(); } else { self.sim.start_running(); }
            }
            if ui.button("▷ step").clicked() { self.sim.step_once(); }
            if ui.button("↺ reset").clicked() {
                self.sim.reset_state();
                self.sim.pc = 0;
                self.sim.gates_completed = 0;
                self.sim.tiles_completed = 0;
                self.sim.wall_seconds = 0.0;
                for c in &mut self.sim.cores {
                    c.busy_for = 0.0; c.idle_for = 1e9;
                    c.total_busy = 0.0; c.last_tile = None;
                }
                self.sim.queue.clear();
            }
            if ui.button("⌫ clear").clicked() { self.sim.clear(); }
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(rich_faint("presets"));
            for (name, build) in PRESETS {
                if magenta_button(ui, name).clicked() {
                    let ops = build(self.sim.n_qubits);
                    self.sim.set_circuit(ops);
                    self.sim.start_running();
                }
            }
        });
    }

    fn append_one(&mut self, g: GateKind) {
        self.sim.append(Op::one(g, self.target_q));
    }
    fn append_rot(&mut self, g: GateKind) {
        self.sim.append(Op::rot(g, self.target_q, self.theta));
    }
    fn append_two(&mut self, g: GateKind) {
        if self.ctrl_q == self.target_q { return; }
        self.sim.append(Op::two(g, self.ctrl_q, self.target_q));
    }

    // ---------- circuit diagram ----------

    fn draw_circuit_panel(&self, ui: &mut Ui) {
        section_header(ui, "circuit", &format!("pc = {}", self.sim.pc));
        let lane_h = 22.0;
        let col_w = 30.0;
        let pad_x = 44.0;
        let pad_y = 8.0;
        let total_h = pad_y * 2.0 + lane_h * self.sim.n_qubits as f32;
        let total_w = pad_x + col_w * (self.sim.circuit.len().max(1) as f32 + 1.0);

        egui::ScrollArea::horizontal()
            .id_salt("circuit-scroll")
            .max_height(total_h + 4.0)
            .show(ui, |ui| {
                let (rect, _) = ui.allocate_exact_size(
                    Vec2::new(total_w, total_h), Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, CornerRadius::ZERO, BG_0);

                for q in 0..self.sim.n_qubits {
                    let y = rect.min.y + pad_y + q as f32 * lane_h + lane_h * 0.5;
                    painter.text(
                        Pos2::new(rect.min.x + 6.0, y - 7.0), Align2::LEFT_TOP,
                        format!("q{}", q), FontId::monospace(11.0), CYAN,
                    );
                    painter.line_segment(
                        [Pos2::new(rect.min.x + 32.0, y), Pos2::new(rect.max.x - 6.0, y)],
                        Stroke::new(1.0, LINE_BRIGHT),
                    );
                }

                let executing = self.executing_op_index();

                for (idx, op) in self.sim.circuit.iter().enumerate() {
                    let active = Some(idx) == executing;
                    let stroke_col = if active { CYAN } else { TEXT_DIM };
                    let fill_col = if active {
                        Color32::from_rgba_unmultiplied(0, 255, 209, 40)
                    } else { BG_3 };
                    let x = rect.min.x + pad_x + col_w * idx as f32 + col_w * 0.5;
                    let yt = rect.min.y + pad_y + op.target as f32 * lane_h + lane_h * 0.5;

                    if op.gate.is_2q() {
                        let yc = rect.min.y + pad_y + op.ctrl as f32 * lane_h + lane_h * 0.5;
                        painter.line_segment(
                            [Pos2::new(x, yc.min(yt)), Pos2::new(x, yc.max(yt))],
                            Stroke::new(1.4, stroke_col),
                        );
                        painter.circle_filled(Pos2::new(x, yc), 3.5, stroke_col);
                        match op.gate {
                            GateKind::CZ => {
                                painter.circle(
                                    Pos2::new(x, yt), 8.5, fill_col,
                                    Stroke::new(1.0, stroke_col),
                                );
                                painter.text(
                                    Pos2::new(x, yt), Align2::CENTER_CENTER,
                                    "Z", FontId::monospace(10.0), TEXT,
                                );
                            }
                            _ => {
                                painter.circle(
                                    Pos2::new(x, yt), 8.5, fill_col,
                                    Stroke::new(1.0, stroke_col),
                                );
                                painter.line_segment(
                                    [Pos2::new(x - 5.5, yt), Pos2::new(x + 5.5, yt)],
                                    Stroke::new(1.2, stroke_col),
                                );
                                painter.line_segment(
                                    [Pos2::new(x, yt - 5.5), Pos2::new(x, yt + 5.5)],
                                    Stroke::new(1.2, stroke_col),
                                );
                            }
                        }
                    } else {
                        let r = Rect::from_center_size(Pos2::new(x, yt), Vec2::new(18.0, 16.0));
                        painter.rect(
                            r, CornerRadius::same(3), fill_col,
                            Stroke::new(1.0, stroke_col), StrokeKind::Inside,
                        );
                        painter.text(
                            r.center(), Align2::CENTER_CENTER,
                            op.gate.label(), FontId::monospace(9.5), TEXT,
                        );
                        if op.gate.is_rot() {
                            painter.text(
                                Pos2::new(x, yt + 12.5), Align2::CENTER_CENTER,
                                format!("{:.2}", op.theta), FontId::monospace(8.0), TEXT_DIM,
                            );
                        }
                    }
                }
            });
    }

    fn executing_op_index(&self) -> Option<usize> {
        if self.sim.pc == 0 { return None; }
        if self.sim.queue.is_empty() && self.sim.all_idle() { return None; }
        Some(self.sim.pc - 1)
    }
}

// ---------- small helpers ----------

fn section_header(ui: &mut Ui, title: &str, meta: &str) {
    let avail = ui.available_width();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title.to_uppercase())
            .color(TEXT).monospace().size(11.0)
            .extra_letter_spacing(2.5));
        if !meta.is_empty() {
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                ui.label(egui::RichText::new(meta).color(TEXT_DIM).monospace().size(10.5));
            });
        }
    });
    let (rect, _) = ui.allocate_exact_size(Vec2::new(avail, 1.0), Sense::hover());
    ui.painter().line_segment(
        [rect.left_center(), rect.right_center()],
        Stroke::new(1.0, LINE),
    );
    ui.add_space(6.0);
}

fn row(ui: &mut Ui, k: &str, v: &str, col: Color32) {
    ui.label(rich_dim(k));
    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
        ui.label(egui::RichText::new(v).color(col).monospace());
    });
    ui.end_row();
}

fn rich_strong(s: &str) -> egui::RichText {
    egui::RichText::new(s).color(TEXT).strong().size(15.0)
}
fn rich_dim(s: &str) -> egui::RichText {
    egui::RichText::new(s).color(TEXT_DIM).monospace()
}
fn rich_faint(s: &str) -> egui::RichText {
    egui::RichText::new(s).color(TEXT_FAINT).monospace().size(11.0)
}

fn paint_brand_dot(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), Sense::hover());
    let center = rect.center();
    let t = ui.input(|i| i.time);
    let pulse = 0.6 + 0.4 * ((t as f32 * 2.0).sin() * 0.5 + 0.5);
    let glow = (pulse * 30.0) as u8;
    ui.painter().circle_filled(center, 7.0, Color32::from_rgba_unmultiplied(0, 255, 209, glow));
    ui.painter().circle_filled(center, 4.0, CYAN);
}

fn magenta_button(ui: &mut Ui, label: &str) -> egui::Response {
    let btn = egui::Button::new(egui::RichText::new(label).color(MAGENTA).monospace())
        .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 45, 173, 110)))
        .fill(BG_3);
    ui.add(btn)
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 {
        (x as f32 + (y as f32 - x as f32) * t) as u8
    };
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
