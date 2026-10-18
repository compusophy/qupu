#version 300 es
// Draws an LCD's active area at u_k texels per LCD pixel: each pixel shows
// the value of the weight cell the map assigns to it, through an aperture
// of 80 % (the black matrix between pixels), box-filtered per texel.
precision highp float;
precision highp sampler2D;
uniform sampler2D u_map;   // RGBA8: cell j, tile t mod 256, 1 + t / 256, 255
uniform sampler2D u_vals;  // R8: x[j] in row 0 (inputs) or mask[t][j] (weights)
uniform float u_size;      // LCD pixels per side
uniform float u_k;         // texels per LCD pixel
uniform int u_weights;
out vec4 o;
void main() {
  vec2 tc = gl_FragCoord.xy;
  vec2 f0 = (tc - 0.5) / u_k, f1 = (tc + 0.5) / u_k;
  vec2 pc = floor(tc / u_k);
  vec2 cov2 = clamp(min(f1, pc + 0.9) - max(f0, pc + 0.1), 0.0, 1.0) / (f1 - f0);
  float cov = cov2.x * cov2.y;
  ivec2 ip = ivec2(clamp(pc, vec2(0.0), vec2(u_size - 1.0)));
  vec4 m = texelFetch(u_map, ip, 0);
  float val = 0.002; // LCD black level
  if (m.b > 0.5 / 255.0) {
    int j = int(m.r * 255.0 + 0.5);
    int t = int(m.g * 255.0 + 0.5) + 256 * (int(m.b * 255.0 + 0.5) - 1);
    val = max(texelFetch(u_vals, u_weights == 1 ? ivec2(j, t) : ivec2(j, 0), 0).r, 0.002);
  }
  o = vec4(val * cov, 0.0, 0.0, 1.0);
}
