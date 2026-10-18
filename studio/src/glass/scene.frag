#version 300 es
// The tabletop rig, raymarched. Units are millimetres; x runs along the
// rail from the backlight toward the camera, y is up, the optical axis sits
// AX above the breadboard. What the LCDs show and what travels down the
// beams comes from the simulator: u_texA is the input panel's active area,
// u_texB the weight panel's, both mip-mapped so a beam can carry the image
// blurred by exactly the cone it has spread through.
precision highp float;
precision highp sampler2D;

uniform vec2 u_res;
uniform vec3 u_eye;
uniform vec3 u_target;
uniform float u_fov;
uniform float u_time;
uniform float u_frame;
uniform vec4 u_pos;   // x of: LCD A, lens 1 flange, lens 2 flange, LCD B
uniform vec4 u_pos2;  // x of camera lens front, lens 2 lift (mm), x where the beam leaves the relay, f-number
uniform vec4 u_lcd;   // active half-side (mm), mean input level, mean transmitted level, aperture ring turn (rad)
uniform float u_tex;  // texels per side of u_texA / u_texB
uniform sampler2D u_texA;
uniform sampler2D u_texB;

out vec4 o;

const float AX = 82.0;
const vec3 GREEN = vec3(0.09, 1.0, 0.2);
const float PI = 3.14159265;
const vec3 KEY_DIR = vec3(-0.33, 0.86, 0.39);
const vec3 KEY_COL = vec3(1.0, 0.95, 0.88) * 1.6;
const vec3 RIM_DIR = vec3(0.62, 0.42, -0.66);
const vec3 RIM_COL = vec3(0.7, 0.82, 1.0) * 1.1;
const float E_PANEL = 7.0;   // radiance of a fully open input pixel

float xA, xL1, xL2, xB, xC, lift, xF, fnum, hA;

// ---------------------------------------------------------------- helpers
float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float noise3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdRBox(vec3 p, vec3 b, float r) { vec3 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
float sdBox2(vec2 p, vec2 b) { vec2 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0); }
float sdCap(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
// A ring (or solid, r0 = 0) of revolution: u along the axis, r radial.
float lathe(vec2 ur, float u0, float u1, float r0, float r1) {
  return sdBox2(ur - vec2(0.5 * (u0 + u1), 0.5 * (r0 + r1)), vec2(0.5 * (u1 - u0), 0.5 * (r1 - r0)));
}
vec2 U(vec2 a, vec2 b) { return a.x < b.x ? a : b; }
float rAx(vec3 p) { return length(vec2(p.y - AX, p.z)); }

// ------------------------------------------------------------- materials
// 1 breadboard  2 rail  3 printed holder  4 anodized black  5 knurled ring
// 6 chrome  7 lens glass  8 input LCD  9 weight LCD  10 steel  11 PCB
// 12 chip  13 flex  14 LED panel  15 camera body  16 cable  17 aperture ring
// 18 bayonet mount

// A 50 mm prime; u from the mount flange toward the front element.
vec2 lens50(vec3 p, float flange, float dir) {
  float u = (p.x - flange) * dir;
  float r = rAx(p);
  float bound = max(max(-9.0 - u, u - 39.0), r - 30.0);
  if (bound > 1.0) return vec2(bound, 0.0);
  vec2 ur = vec2(u, r);
  vec2 d = vec2(lathe(ur, -7.0, 0.0, 13.5, 20.5), 18.0);
  d = U(d, vec2(lathe(ur, -5.6, -4.8, 0.0, 13.6), 7.0));
  d = U(d, vec2(lathe(ur, 0.0, 7.0, 0.0, 28.4) - 0.2, 17.0));
  d = U(d, vec2(lathe(ur, 7.0, 8.6, 0.0, 27.2), 6.0));
  d = U(d, vec2(lathe(ur, 8.6, 14.0, 0.0, 26.4), 4.0));
  d = U(d, vec2(lathe(ur, 14.0, 31.0, 0.0, 29.4) - 0.3, 5.0));
  d = U(d, vec2(lathe(ur, 31.0, 37.0, 16.4, 27.6) - 0.2, 4.0));
  float cap = max(length(vec2(u + 4.5, r)) - 38.0, max(r - 16.5, 30.0 - u));
  d = U(d, vec2(cap, 7.0));
  return d;
}

// Printed clamp + post + rail carrier around a lens at axial position xc.
vec2 lensHolder(vec3 p, float xc) {
  vec3 q = p - vec3(xc, 0.0, 0.0);
  float b = sdBox(q - vec3(0.0, 55.0, 0.0), vec3(14.0, 40.0, 36.0));
  if (b > 1.0) return vec2(b, 0.0);
  float ring = lathe(vec2(q.x, rAx(p)), -2.6, 2.6, 26.4, 31.5) - 0.4;
  float stem = sdRBox(q - vec3(0.0, 0.5 * (20.0 + AX - 29.0), 0.0), vec3(2.6, 0.5 * (AX - 49.0), 7.0), 0.8);
  float block = sdRBox(q - vec3(0.0, 27.0, 0.0), vec3(12.0, 7.0, 16.0), 1.6);
  return vec2(min(min(ring, stem), block), 3.0);
}

// An LCD with bezel, flex, driver board and printed holder. side = -1 puts
// the electronics behind (toward the backlight), +1 toward the camera.
vec2 panel(vec3 p, float x0, float side, float id) {
  vec3 q = p - vec3(x0, AX, 0.0);
  float b = sdBox(q - vec3(0.0, -10.0, 8.0), vec3(16.0, 52.0, 76.0));
  if (b > 1.0) return vec2(b, 0.0);
  vec2 d = vec2(sdBox(q, vec3(0.9, 36.4, 63.4)), id);
  float frame = max(sdBox(q, vec3(1.5, 38.0, 65.0)), -sdBox(q, vec3(2.0, 36.5, 63.5)));
  d = U(d, vec2(frame - 0.1, 10.0));
  d = U(d, vec2(sdBox(q - vec3(side * 4.2, -37.4, 48.0), vec3(4.2, 0.14, 8.0)), 13.0));
  d = U(d, vec2(sdBox(q - vec3(side * 8.4, -49.0, 56.0), vec3(0.8, 11.0, 19.0)), 11.0));
  d = U(d, vec2(sdRBox(q - vec3(side * 9.6, -47.0, 50.0), vec3(0.6, 4.5, 5.0), 0.3), 12.0));
  d = U(d, vec2(sdRBox(q - vec3(side * 9.4, -54.0, 63.0), vec3(0.45, 2.6, 2.6), 0.2), 12.0));
  d = U(d, vec2(sdBox(q - vec3(side * 10.6, -41.0, 69.0), vec3(1.8, 2.6, 5.5)), 10.0));
  float holder = sdRBox(q - vec3(0.0, -48.5, 0.0), vec3(8.0, 14.0, 30.0), 1.5);
  holder = max(holder, -sdBox(q - vec3(0.0, -36.0, 0.0), vec3(2.2, 2.5, 31.0)));
  d = U(d, vec2(holder, 3.0));
  return d;
}

vec2 camera(vec3 p) {
  vec3 q = p - vec3(xC, AX, 0.0);
  float b = sdBox(q - vec3(36.0, -20.0, 0.0), vec3(38.0, 44.0, 22.0));
  if (b > 1.0) return vec2(b, 0.0);
  vec2 ur = vec2(q.x, rAx(p));
  vec2 d = vec2(lathe(ur, 0.0, 4.0, 10.6, 15.0) - 0.2, 4.0);
  d = U(d, vec2(max(length(vec2(q.x - 14.0, ur.y)) - 13.0, max(ur.y - 10.7, 1.4 - q.x)), 7.0));
  d = U(d, vec2(lathe(ur, 4.0, 14.0, 0.0, 16.0) - 0.3, 5.0));
  d = U(d, vec2(lathe(ur, 14.0, 15.4, 0.0, 14.6), 6.0));
  d = U(d, vec2(lathe(ur, 15.4, 24.0, 0.0, 15.4) - 0.2, 17.0));
  d = U(d, vec2(lathe(ur, 24.0, 34.0, 0.0, 13.4), 4.0));
  d = U(d, vec2(sdRBox(q - vec3(51.0, 0.0, 0.0), vec3(17.0, 17.0, 17.0), 2.4), 15.0));
  d = U(d, vec2(sdRBox(q - vec3(51.0, -0.5 * (17.0 + AX - 20.0), 0.0), vec3(12.0, 0.5 * (AX - 37.0), 12.0), 1.5), 3.0));
  return d;
}

vec2 cable(vec3 p) {
  vec3 a = vec3(xC + 68.0, AX - 6.0, 0.0), b = vec3(xC + 86.0, AX - 8.0, 4.0);
  vec3 c = vec3(xC + 100.0, AX - 40.0, 14.0), e = vec3(xC + 108.0, 2.4, 34.0), f = vec3(xC + 150.0, 2.4, 95.0);
  float d = min(min(sdCap(p, a, b, 2.3), sdCap(p, b, c, 2.3)), min(sdCap(p, c, e, 2.3), sdCap(p, e, f, 2.3)));
  return vec2(d, 16.0);
}

vec2 map(vec3 p) {
  vec2 d = vec2(sdBox(p - vec3(60.0, -6.35, 0.0), vec3(330.0, 6.35, 220.0)), 1.0);
  float rail = sdBox(p - vec3(70.0, 10.0, 0.0), vec3(240.0, 10.0, 10.0));
  rail = max(rail, -sdBox(p - vec3(70.0, 20.0, 0.0), vec3(241.0, 3.2, 3.0)));
  rail = max(rail, -sdBox(vec3(p.x - 70.0, p.y - 10.0, abs(p.z) - 10.0), vec3(241.0, 3.0, 3.2)));
  d = U(d, vec2(rail, 2.0));
  // Backlight, input LCD and their holder.
  d = U(d, vec2(sdRBox(p - vec3(xA - 7.6, AX, 0.0), vec3(3.2, 38.0, 38.0), 1.0), 14.0));
  d = U(d, vec2(sdRBox(p - vec3(xA - 7.6, 32.0, 0.0), vec3(3.0, 12.0, 20.0), 1.0), 3.0));
  d = U(d, panel(p, xA, -1.0, 8.0));
  // The relay.
  d = U(d, lens50(p, xL1, 1.0));
  d = U(d, lensHolder(p, xL1 + 11.3));
  vec3 p2 = p - vec3(0.0, lift, lift * 0.25);
  d = U(d, lens50(p2, xL2, -1.0));
  d = U(d, lensHolder(p2, xL2 - 11.3));
  d = U(d, vec2(lathe(vec2(p2.x, rAx(p2)), xL1 + 35.5, xL2 - 35.5, 26.8, 28.9) - 0.2, 6.0));
  // Weight LCD, camera.
  d = U(d, panel(p, xB, 1.0, 9.0));
  d = U(d, camera(p));
  d = U(d, cable(p));
  return d;
}

vec3 normalAt(vec3 p, float t) {
  float e = 0.0006 * t + 0.01;
  vec2 k = vec2(1.0, -1.0);
  return normalize(k.xyy * map(p + k.xyy * e).x + k.yyx * map(p + k.yyx * e).x +
                   k.yxy * map(p + k.yxy * e).x + k.xxx * map(p + k.xxx * e).x);
}

vec2 march(vec3 ro, vec3 rd) {
  float t = 1.0;
  for (int i = 0; i < 220; i++) {
    vec2 h = map(ro + rd * t);
    if (h.x < 0.00025 * t + 0.003) return vec2(t, h.y);
    t += h.x * 0.92;
    if (t > 4000.0) break;
  }
  return vec2(t, -1.0);
}

float softShadow(vec3 ro, vec3 rd) {
  float res = 1.0, t = 0.6;
  for (int i = 0; i < 32; i++) {
    float h = map(ro + rd * t).x;
    res = min(res, 9.0 * h / t);
    t += clamp(h, 0.4, 24.0);
    if (res < 0.002 || t > 500.0) break;
  }
  return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
  float occ = 0.0, s = 1.0;
  for (int i = 0; i < 5; i++) {
    float h = 0.4 + 2.6 * float(i);
    occ += (h - map(p + n * h).x) * s;
    s *= 0.72;
  }
  return clamp(1.0 - 0.09 * occ, 0.0, 1.0);
}

// The studio around the bench: a dark room, an overhead softbox and a cool
// strip light, blurred by roughness.
vec3 env(vec3 d, float rough) {
  float w = 0.05 + rough * 0.8;
  vec3 c = vec3(0.010, 0.011, 0.014) * (0.55 + 0.45 * clamp(d.y, 0.0, 1.0));
  float a1 = acos(clamp(dot(d, normalize(vec3(-0.35, 0.85, 0.45))), -1.0, 1.0));
  c += vec3(1.0, 0.96, 0.9) * 4.0 * smoothstep(0.34 + w, 0.34 - 0.6 * w, a1) / (1.0 + rough * 5.0);
  vec3 s2 = normalize(vec3(0.75, 0.3, -0.6));
  vec3 t2 = normalize(cross(s2, vec3(0.0, 1.0, 0.0)));
  float along = dot(d, vec3(0.0, 1.0, 0.0)), across = dot(d, t2);
  float strip = smoothstep(0.12 + w, 0.12 - 0.5 * w, abs(across)) * smoothstep(0.65 + w, 0.65 - w, abs(along - 0.3)) * step(0.0, dot(d, s2));
  c += vec3(0.72, 0.84, 1.0) * 2.2 * strip / (1.0 + rough * 5.0);
  return c;
}

// Active-area texture coordinates on a panel plane (z, y relative to the axis).
vec2 panelUV(vec2 q) { return vec2(0.5 + q.x / (2.0 * hA), 0.5 - q.y / (2.0 * hA)); }
bool inActive(vec2 q) { return abs(q.x) < hA && abs(q.y) < hA; }
float lodFor(float footprintMM) { return log2(max(footprintMM / (2.0 * hA / u_tex), 1.0)); }
float tapA(vec2 uv, float lod) { return textureLod(u_texA, uv, lod).r; }
float tapB(vec2 uv, float lod) { return textureLod(u_texB, uv, lod).r; }

float pupil() { return 0.5 * 50.0 / fnum * 0.9; }
float coneHalfAngle() { return atan(0.5 / fnum); }

struct Mat { vec3 alb; float rough; float metal; float f0; vec3 emit; };

Mat material(float id, vec3 p, inout vec3 n, vec3 rd, float t) {
  Mat m = Mat(vec3(0.03), 0.5, 0.0, 0.04, vec3(0.0));
  float pixMM = t * 2.0 * tan(0.5 * u_fov) / u_res.y;
  if (id < 1.5) { // black anodized breadboard, M6 holes on 25 mm
    m.alb = vec3(0.028);
    m.rough = 0.38 + 0.18 * noise3(p * 0.08);
    if (p.y > -0.2) {
      vec2 c = mod(p.xz + 12.5, 25.0) - 12.5;
      float r = length(c);
      if (r < 3.1) { m.alb = vec3(0.0); m.rough = 1.0; m.f0 = 0.0; }
      else if (r < 3.7) { n = normalize(n + vec3(c.x, 0.0, c.y) / r * 0.55); m.rough = 0.25; }
      m.rough += 0.25 * smoothstep(0.62, 0.7, noise3(vec3(p.x * 0.02, 0.0, p.z * 0.6)));
    }
  } else if (id < 2.5) { // brushed aluminium extrusion
    m.alb = vec3(0.86, 0.87, 0.89);
    m.metal = 1.0;
    m.rough = 0.26 + 0.12 * noise3(vec3(p.x * 0.05, p.y * 9.0, p.z * 9.0));
  } else if (id < 3.5) { // matte black PLA with 0.2 mm layer lines
    m.alb = vec3(0.022);
    m.rough = 0.62;
    float s = sin(p.y * 2.0 * PI / 0.2);
    n = normalize(n + vec3(0.0, 0.18 * s, 0.0) * (1.0 - abs(n.y)) * smoothstep(1.2, 0.2, pixMM));
  } else if (id < 4.5) {
    m.alb = vec3(0.016);
    m.rough = 0.3;
  } else if (id < 5.5) { // knurled focus ring: axial ribs
    vec2 yz = vec2(p.y - AX, p.z);
    float th = atan(yz.x, yz.y);
    vec3 tang = normalize(vec3(0.0, yz.y, -yz.x));
    n = normalize(n + tang * 0.55 * sin(th * 140.0) * smoothstep(0.9, 0.2, pixMM) * step(0.5, length(n.yz)));
    m.alb = vec3(0.012);
    m.rough = 0.58;
  } else if (id < 6.5) {
    m.alb = vec3(0.95);
    m.metal = 1.0;
    m.rough = 0.07;
  } else if (id < 7.5) { // coated glass: dark, with purple-green reflections and the image inside
    m.alb = vec3(0.0);
    m.rough = 0.02;
    m.f0 = 0.05;
    m.emit = GREEN * 0.06 * u_lcd.y * E_PANEL * smoothstep(0.0, 0.9, abs(dot(n, rd)));
  } else if (id < 8.5) { // input LCD
    m.alb = vec3(0.004);
    m.rough = 0.22;
    vec2 q = vec2(p.z, p.y - AX);
    if (n.x > 0.5) {
      // Shown rotated 180°: the 1:1 relay turns it upright on the weight LCD.
      float v = inActive(q) ? tapA(vec2(1.0) - panelUV(q), lodFor(pixMM / max(abs(rd.x), 0.15))) : 0.002;
      // Alignment marks a real rig would draw: a frame round the active area
      // and crosshairs near the corners.
      float lw = max(0.3, 0.7 * pixMM);
      float fr = abs(max(abs(q.x), abs(q.y)) - (hA + 2.5));
      float mark = smoothstep(lw, 0.0, fr);
      for (int k = 0; k < 4; k++) {
        vec2 c = vec2(k % 2 == 0 ? -52.0 : 52.0, k < 2 ? -28.0 : 28.0);
        vec2 dq = abs(q - c);
        mark = max(mark, smoothstep(lw, 0.0, min(dq.x, dq.y)) * step(max(dq.x, dq.y), 4.0));
      }
      m.emit = GREEN * E_PANEL * max(v, 0.35 * mark);
    } else if (n.x < -0.5) {
      m.emit = GREEN * 0.3;
    }
  } else if (id < 9.5) { // weight LCD: the relayed input through the mask
    m.alb = vec3(0.004);
    m.rough = 0.22;
    vec2 q = vec2(p.z, p.y - AX);
    if (abs(n.x) > 0.5 && inActive(q)) {
      vec2 uv = panelUV(q);
      float lod = lodFor(pixMM / max(abs(rd.x), 0.15));
      float a = tapA(uv, lod);
      if (n.x > 0.0) {
        // Transmitted light leaves in the relay's cone; the panel's haze
        // scatters a little everywhere.
        float off = acos(clamp(-rd.x, -1.0, 1.0));
        float lobe = 0.05 + 1.2 * exp(-pow(off / (coneHalfAngle() + 0.1), 2.0));
        m.emit = GREEN * E_PANEL * 0.8 * a * tapB(uv, lod) * lobe;
      } else {
        m.emit = GREEN * E_PANEL * 0.12 * a;
      }
    }
  } else if (id < 10.5) {
    m.alb = vec3(0.58, 0.58, 0.6);
    m.metal = 1.0;
    m.rough = 0.36;
  } else if (id < 11.5) { // green solder mask with traces
    float tr = smoothstep(0.35, 0.0, abs(fract(p.y * 0.9 + 0.2 * floor(p.z * 0.4)) - 0.5) - 0.1);
    m.alb = mix(vec3(0.012, 0.075, 0.03), vec3(0.03, 0.14, 0.05), tr * step(0.6, noise3(p * 0.7)));
    m.rough = 0.32;
  } else if (id < 12.5) {
    m.alb = vec3(0.018);
    m.rough = 0.45;
  } else if (id < 13.5) { // polyimide flex
    m.alb = vec3(0.6, 0.26, 0.035);
    m.rough = 0.22;
    m.f0 = 0.05;
  } else if (id < 14.5) { // LED light pad, diffuser toward the panel
    m.alb = vec3(0.75);
    m.rough = 0.6;
    if (n.x > 0.5) m.emit = GREEN * 3.0;
  } else if (id < 15.5) {
    m.alb = vec3(0.017);
    m.rough = 0.28;
    vec3 q = p - vec3(xC + 51.0, AX, 0.0);
    vec2 s = abs(q.zy) - 12.0;
    if (abs(q.x) < 16.9 && length(s) < 1.3) { m.alb = vec3(0.5); m.metal = 1.0; m.rough = 0.3; }
  } else if (id < 16.5) {
    m.alb = vec3(0.01);
    m.rough = 0.18;
  } else if (id < 17.5) { // aperture ring: f-stop ticks, index mark
    m.alb = vec3(0.016);
    m.rough = 0.38;
    vec2 yz = vec2(p.y - AX, p.z);
    float th = atan(yz.x, sign(p.x - xL1 - 20.0) * yz.y);
    float r = length(yz);
    if (r > 27.8) {
      float local = th - 0.5 * PI + u_lcd.w;
      for (int k = 0; k < 7; k++) {
        if (abs(local - (float(k) - 3.0) * 0.16) < 0.011) m.alb = k == 3 ? vec3(0.9, 0.42, 0.08) : vec3(0.8);
      }
    }
  } else { // bayonet
    m.alb = vec3(0.8);
    m.metal = 1.0;
    m.rough = 0.16;
  }
  return m;
}

// Light the panels throw on everything nearby.
vec3 glow(vec3 p, vec3 n) {
  vec3 c = vec3(0.0);
  vec3 la = vec3(xA + 1.0, AX, 0.0) - p;
  float da = length(la);
  la /= da;
  float areaA = 4.0 * hA * hA;
  c += GREEN * E_PANEL * u_lcd.y * areaA * max(-la.x, 0.0) * max(dot(n, la), 0.0) / (da * da + areaA) * 0.35;
  vec3 lb = vec3(xB + 1.0, AX, 0.0) - p;
  float db = length(lb);
  lb /= db;
  c += GREEN * E_PANEL * u_lcd.z * areaA * pow(max(-lb.x, 0.0), 6.0) * max(dot(n, lb), 0.0) / (db * db + areaA) * 0.6;
  return c;
}

vec3 shade(vec3 p, vec3 n, vec3 rd, Mat m, float t) {
  vec3 v = -rd;
  float ao = ambientOcclusion(p, n);
  vec3 F0 = mix(vec3(m.f0), m.alb, m.metal);
  vec3 diff = m.alb * (1.0 - m.metal);
  float a = max(m.rough * m.rough, 0.002);
  float NoV = max(dot(n, v), 1e-3);
  vec3 col = vec3(0.0);
  for (int i = 0; i < 2; i++) {
    vec3 L = normalize(i == 0 ? KEY_DIR : RIM_DIR);
    vec3 lc = i == 0 ? KEY_COL : RIM_COL;
    float NoL = max(dot(n, L), 0.0);
    if (NoL <= 0.0) continue;
    float sh = i == 0 ? softShadow(p + n * 0.25, L) : 1.0;
    vec3 h = normalize(L + v);
    float NoH = max(dot(n, h), 0.0);
    float dd = NoH * NoH * (a * a - 1.0) + 1.0;
    float D = a * a / (PI * dd * dd);
    float k = a * 0.5;
    float G = NoL / (NoL * (1.0 - k) + k) * NoV / (NoV * (1.0 - k) + k);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    col += (diff / PI + D * G * F / (4.0 * NoL * NoV + 1e-4)) * NoL * lc * sh * PI;
  }
  vec3 R = reflect(rd, n);
  vec3 Fe = F0 + (max(vec3(1.0 - m.rough), F0) - F0) * pow(1.0 - NoV, 5.0);
  if (m.f0 > 0.045 && m.metal < 0.5 && m.alb.r < 0.01) {
    // Anti-reflection coatings: the residual reflection is tinted.
    Fe *= mix(vec3(0.7, 0.35, 1.0), vec3(0.35, 1.0, 0.55), smoothstep(0.2, 0.9, NoV)) * 1.6;
  }
  float specOcc = clamp(pow(NoV + ao, exp2(-16.0 * m.rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
  col += env(R, m.rough) * Fe * specOcc;
  col += diff * (env(n, 1.0) * 1.6 + vec3(0.004)) * ao;
  col += diff * glow(p, n) * (0.5 + 0.5 * ao);
  col += m.emit;
  return col;
}

// ------------------------------------------------------------------ beams
// Light in the haze along the bench. Each segment is the union of rays
// from a square (the active area) to a disc (a pupil): its cross-section is
// the square shrunk by (1 − s) grown by the disc scaled by s, and what it
// carries is the panel image blurred by the disc it has spread through.
float coneSeg(vec2 q, float s, float rho, out vec2 uvP, out float blur) {
  float a = hA * (1.0 - s), rr = rho * s;
  float sd = sdBox2(q, vec2(a)) - rr;
  uvP = panelUV(q / max(1.0 - s, 0.05));
  blur = rr / max(1.0 - s, 0.05);
  float area = 4.0 * a * a + 8.0 * a * rr + PI * rr * rr;
  // Light only leaves the panel's glass, however big the layout is.
  float onPanel = smoothstep(1.0, -1.0, sdBox2(q / max(1.0 - s, 0.05), vec2(63.4, 36.4)) - blur);
  return smoothstep(0.6, -0.6, sd) * 4.0 * hA * hA / max(area, 1.0) * onPanel;
}

vec3 beams(vec3 ro, vec3 rd, float tHit) {
  vec3 c = vec3(0.5 * (xA + xC), AX, 0.0), h = vec3(0.5 * (xC - xA) + 1.0, 50.0, 50.0);
  vec3 inv = 1.0 / rd;
  vec3 t0 = (c - h - ro) * inv, t1 = (c + h - ro) * inv;
  vec3 tn = min(t0, t1), tf = max(t0, t1);
  float ta = max(max(tn.x, tn.y), max(tn.z, 0.0)), tb = min(min(tf.x, tf.y), min(tf.z, tHit));
  if (ta >= tb) return vec3(0.0);
  const int N = 56;
  float dt = (tb - ta) / float(N);
  float jit = hash1(gl_FragCoord.xy + fract(u_frame * 0.1347) * 61.0);
  float cosT = -rd.x;
  float g = 0.55;
  float phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosT, 1.5) * 0.55;
  float rho = pupil();
  float xR1 = xL1 - 7.0;
  float th = tan(coneHalfAngle());
  float acc = 0.0;
  for (int i = 0; i < N; i++) {
    vec3 p = ro + rd * (ta + (float(i) + jit) * dt);
    vec2 q = vec2(p.z, p.y - AX);
    float x = p.x;
    float dens = 0.0;
    vec2 uvP;
    float blur;
    if (x > xA + 1.0 && x < xR1) {
      float s = (x - xA - 1.0) / (xR1 - xA - 1.0);
      float cov = coneSeg(q, s, rho, uvP, blur);
      if (cov > 0.001) dens += cov * tapA(vec2(1.0) - uvP, lodFor(2.0 * blur));
      // Light the lens does not accept: a soft glow off the panel.
      dens += 0.025 * u_lcd.y * exp(-(x - xA) / 14.0) * smoothstep(40.0, 10.0, sdBox2(q, vec2(hA)));
    } else if (x > xF && x < xB - 1.0) {
      float s = (xB - 1.0 - x) / (xB - 1.0 - xF);
      float cov = coneSeg(q, s, rho, uvP, blur);
      if (cov > 0.001) dens += cov * tapA(uvP, lodFor(2.0 * blur));
    } else if (x > xB + 1.0 && x < xC) {
      float d = x - xB - 1.0, rr = d * th;
      float sd = sdBox2(q, vec2(hA)) - rr;
      if (sd < 0.6) {
        vec2 uv = panelUV(q);
        float lod = lodFor(2.0 * rr);
        float area = 4.0 * hA * hA + 8.0 * hA * rr + PI * rr * rr;
        // After the mask the light is the computation itself; it is shown 3×
        // brighter than its share so the dimmer beam still reads.
        dens += 2.4 * smoothstep(0.6, -0.6, sd) * 4.0 * hA * hA / area * tapA(uv, lod) * tapB(uv, lod);
      }
    }
    // Dust in the beam; no haze right at the eye (the camera sits outside it).
    if (dens > 0.0) dens *= 1.0 + 6.0 * pow(noise3(p * 0.9 + vec3(0.0, u_time * 0.6, 0.0)), 18.0);
    acc += dens * smoothstep(15.0, 55.0, ta + (float(i) + jit) * dt);
  }
  return GREEN * E_PANEL * acc * dt * 0.016 * phase * (8.0 / fnum + 0.5);
}

void main() {
  xA = u_pos.x; xL1 = u_pos.y; xL2 = u_pos.z; xB = u_pos.w;
  xC = u_pos2.x; lift = u_pos2.y; xF = u_pos2.z; fnum = u_pos2.w;
  hA = u_lcd.x;
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;
  vec3 fw = normalize(u_target - u_eye);
  vec3 rt = normalize(cross(fw, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(rt, fw);
  vec3 rd = normalize(fw + (uv.x * rt + uv.y * up) * 2.0 * tan(0.5 * u_fov));
  vec2 hit = march(u_eye, rd);
  vec3 col;
  float depth;
  if (hit.y < 0.0) {
    col = env(rd, 0.6) * 0.15;
    depth = 6000.0;
  } else {
    vec3 p = u_eye + rd * hit.x;
    vec3 n = normalAt(p, hit.x);
    Mat m = material(hit.y, p, n, rd, hit.x);
    col = shade(p, n, rd, m, hit.x);
    depth = hit.x;
  }
  col += beams(u_eye, rd, hit.y < 0.0 ? 6000.0 : hit.x);
  o = vec4(col, depth);
}
