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
  vec3 col = texture(uTexture, uv).rgb;
  float side = q.x < 0.0 ? -1.0 : 1.0;
  vec2 d = q - vec2(0.30 * side, -0.15);
  float eye = 1.0 - smoothstep(0.8, 1.0, length(d / vec2(0.12, 0.07)));
  float slit = 1.0 - smoothstep(0.012, 0.022, abs(d.x));
  col = mix(col, mix(vec3(0.75, 0.70, 0.10), vec3(0.02), slit), eye * 0.9);
  float nose = ell(q, vec2(0.0, 0.12), vec2(0.07, 0.05));
  col = mix(col, vec3(0.85, 0.45, 0.55), nose);
  float wh = 0.0;
  for (int i = 0; i < 3; i++) {
    float y = 0.20 + float(i) * 0.05;
    float yy = y + (abs(q.x) - 0.12) * (float(i) - 1.0) * 0.25;
    wh = max(wh, (1.0 - smoothstep(0.003, 0.008, abs(q.y - yy))) * step(0.12, abs(q.x)) * step(abs(q.x), 0.62));
  }
  col = mix(col, vec3(0.05), wh * 0.9);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
