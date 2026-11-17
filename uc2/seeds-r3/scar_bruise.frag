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
  float n = noise(q * 9.0) * 0.6 + noise(q * 23.0) * 0.4;
  float r = length((q - vec2(0.30, -0.12)) / vec2(0.24, 0.18));
  float core = smoothstep(1.1, 0.3, r + (n - 0.5) * 0.5);
  float edge = smoothstep(1.4, 0.8, r + (n - 0.5) * 0.5) - core;
  col = mix(col, col * vec3(0.45, 0.25, 0.55), core * 0.75);
  col = mix(col, col * vec3(1.05, 0.95, 0.55), max(edge, 0.0) * 0.6);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
