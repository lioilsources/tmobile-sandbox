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
  vec3 c0 = texture(uTexture, uv).rgb;
  vec2 px = 4.0 / uResolution; vec3 acc = vec3(0.0); float wsum = 0.0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      vec3 s = texture(uTexture, uv + vec2(float(i), float(j)) * px).rgb;
      float w = exp(-dot(s - c0, s - c0) * 40.0);
      acc += s * w; wsum += w;
    }
  }
  float face = ell(q, vec2(0.0, 0.05), vec2(0.62, 0.85)) * (1.0 - ell(q, vec2(0.0, 0.35), vec2(0.22, 0.09)));
  vec3 col = mix(c0, acc / wsum, face * 0.85);
  col = mix(col, col * vec3(1.04, 1.0, 0.97), face * 0.5);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
