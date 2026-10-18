#version 300 es
// Bloom: 2× downsample with a 4-tap bilinear box; the first level weights
// taps by 1/(1+luma) so single hot pixels don't flicker into big halos.
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_dst;   // destination size
uniform int u_first;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / u_dst;
  vec2 t = 0.5 / u_dst;
  vec3 a = texture(u_src, uv + vec2(-t.x, -t.y)).rgb;
  vec3 b = texture(u_src, uv + vec2(t.x, -t.y)).rgb;
  vec3 c = texture(u_src, uv + vec2(-t.x, t.y)).rgb;
  vec3 d = texture(u_src, uv + vec2(t.x, t.y)).rgb;
  if (u_first == 1) {
    vec4 w = 1.0 / (1.0 + vec4(dot(a, vec3(0.3, 0.6, 0.1)), dot(b, vec3(0.3, 0.6, 0.1)),
                               dot(c, vec3(0.3, 0.6, 0.1)), dot(d, vec3(0.3, 0.6, 0.1))));
    o = vec4((a * w.x + b * w.y + c * w.z + d * w.w) / (w.x + w.y + w.z + w.w), 1.0);
  } else {
    o = vec4(0.25 * (a + b + c + d), 1.0);
  }
}
