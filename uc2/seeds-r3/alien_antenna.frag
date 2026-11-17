#include <flutter/runtime_effect.glsl>

uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float uTime;
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
  float face = ell(q, vec2(0.0, -0.05), vec2(0.66, 0.95));
  col = mix(col, vec3(0.62, 0.66, 0.70) * (0.3 + l), face * 0.7);
  float ant = 0.0;
  for (int i = 0; i < 2; i++) {
    float s = i == 0 ? -1.0 : 1.0;
    vec2 a = vec2(0.20 * s, -0.85); vec2 b = vec2(0.45 * s + 0.03 * sin(uTime * 2.0), -1.45);
    vec2 ab = b - a; float t = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
    ant = max(ant, 1.0 - smoothstep(0.015, 0.03, length(q - a - ab * t)));
    ant = max(ant, 1.0 - smoothstep(0.05, 0.07, length(q - b)));
  }
  col = mix(col, vec3(0.35, 0.85, 0.35), ant);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
