#version 300 es
// Lens and film: depth of field (a golden-angle gather sized by circle of
// confusion), bloom, exposure, ACES filmic curve, vignette and grain.
precision highp float;
uniform sampler2D u_scene;  // rgb HDR, a = distance along the ray (mm)
uniform sampler2D u_bloom;
uniform vec2 u_res;         // canvas size
uniform vec2 u_src;         // scene size
uniform float u_focus;      // mm
uniform float u_aperture;   // circle of confusion scale, scene pixels
uniform float u_exposure;
uniform float u_frame;
out vec4 o;

float coc(float z) { return clamp(u_aperture * abs(z - u_focus) / max(z, 1.0), 0.0, 9.0); }

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec2 uv = gl_FragCoord.xy / u_res;
  vec4 c0 = texture(u_scene, uv);
  float r0 = coc(c0.a);
  vec3 acc = c0.rgb;
  float wsum = 1.0;
  if (r0 > 0.6) {
    vec2 px = 1.0 / u_src;
    for (int i = 0; i < 24; i++) {
      float fi = float(i) + 0.5;
      float r = sqrt(fi / 24.0) * r0;
      float a = fi * 2.39996;
      vec4 s = texture(u_scene, uv + r * vec2(cos(a), sin(a)) * px);
      // A sharper sample in front only contributes if its own blur reaches here.
      float w = s.a < c0.a ? smoothstep(r - 1.0, r, coc(s.a)) : 1.0;
      acc += s.rgb * w;
      wsum += w;
    }
  }
  vec3 col = acc / wsum + texture(u_bloom, uv).rgb * 0.055;
  col = aces(col * u_exposure);
  vec2 q = uv - 0.5;
  col *= mix(1.0, smoothstep(0.95, 0.15, length(q * vec2(u_res.x / u_res.y, 1.0))), 0.45);
  col = pow(col, vec3(1.0 / 2.2));
  col += (hash(gl_FragCoord.xy + fract(u_frame * 0.618) * 100.0) - 0.5) * 0.018;
  o = vec4(col, 1.0);
}
