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
  float zone = ell(q, vec2(0.0, 0.10), vec2(0.55, 0.22)) * (1.0 - ell(q, vec2(0.0, 0.35), vec2(0.20, 0.08)));
  vec2 g = q * 55.0; vec2 id = floor(g); vec2 f = fract(g) - 0.5;
  float h = hash(id);
  float dot1 = (1.0 - smoothstep(0.10, 0.25, length(f - (vec2(hash(id + 3.1), hash(id + 7.7)) - 0.5) * 0.4))) * step(0.72, h);
  col = mix(col, col * vec3(0.62, 0.42, 0.30), dot1 * zone * 0.7);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
