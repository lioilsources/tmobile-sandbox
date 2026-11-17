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
  vec2 d = q - vec2(0.0, 0.20);
  float w = exp(-dot(d / vec2(0.28, 0.24), d / vec2(0.28, 0.24)));
  vec2 src = q - d * 0.35 * w;
  src.x /= uResolution.x / uResolution.y;
  vec3 col = texture(uTexture, clamp(uFaceCenter + src * uFaceScale, 0.0, 1.0)).rgb;
  float face = ell(q, vec2(0.0, 0.0), vec2(0.70, 0.95));
  float fur = noise(vec2(q.x * 30.0, q.y * 90.0)) * 0.6 + noise(q * 140.0) * 0.4;
  col = mix(col, col * vec3(0.85, 0.65, 0.45) * (0.7 + 0.6 * fur), face * 0.6);
  col = mix(col, vec3(0.08, 0.05, 0.05), ell(q, vec2(0.0, 0.14), vec2(0.09, 0.06)));
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
