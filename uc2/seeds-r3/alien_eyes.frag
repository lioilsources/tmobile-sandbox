#include <flutter/runtime_effect.glsl>

uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform vec2 uFaceCenter;
uniform float uFaceScale;

out vec4 fragColor;

// face coordinates: eyes ~ (+-0.30, -0.15), nose ~ (0, 0.05), mouth ~ (0, 0.35)
vec2 faceQ(vec2 uv) { vec2 q = (uv - uFaceCenter) / uFaceScale; q.x *= uResolution.x / uResolution.y; return q; }
float ell(vec2 q, vec2 c, vec2 r) { return 1.0 - smoothstep(0.6, 1.0, length((q - c) / r)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) { vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec2 uv = FlutterFragCoord().xy / uResolution; vec2 q = faceQ(uv);
  vec3 col = texture(uTexture, uv).rgb; float l = lum(col);
  float face = ell(q, vec2(0.0, 0.0), vec2(0.65, 0.92));
  col = mix(col, vec3(0.45, 0.75, 0.40) * (0.35 + l), face * 0.75);
  float side = q.x < 0.0 ? -1.0 : 1.0;
  vec2 d = q - vec2(0.30 * side, -0.15);
  float ang = 0.45 * side;
  d = vec2(d.x * cos(ang) - d.y * sin(ang), d.x * sin(ang) + d.y * cos(ang));
  float eye = 1.0 - smoothstep(0.85, 1.0, length(d / vec2(0.22, 0.11)));
  col = mix(col, vec3(0.01, 0.02, 0.02), eye);
  col += eye * exp(-dot(d - vec2(-0.05, -0.03), d - vec2(-0.05, -0.03)) * 900.0) * 0.8;
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
