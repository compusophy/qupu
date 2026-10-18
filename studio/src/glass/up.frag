#version 300 es
// Bloom: 3×3 tent upsample of the smaller level, added onto this one.
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_dst;
uniform float u_weight;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / u_dst;
  vec2 t = 1.0 / u_dst;
  vec3 s = texture(u_src, uv).rgb * 4.0;
  s += (texture(u_src, uv + vec2(t.x, 0.0)).rgb + texture(u_src, uv - vec2(t.x, 0.0)).rgb
      + texture(u_src, uv + vec2(0.0, t.y)).rgb + texture(u_src, uv - vec2(0.0, t.y)).rgb) * 2.0;
  s += texture(u_src, uv + t).rgb + texture(u_src, uv - t).rgb
     + texture(u_src, uv + vec2(t.x, -t.y)).rgb + texture(u_src, uv + vec2(-t.x, t.y)).rgb;
  o = vec4(s / 16.0 * u_weight, 1.0);
}
