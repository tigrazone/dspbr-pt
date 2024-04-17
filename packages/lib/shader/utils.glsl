/* @license
 * Copyright 2020  Dassault Systemes - All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

struct Geometry {
  vec3 n, t, b;
};

float saturate(float val) {
  return clamp(val, 0.0, 1.0);
}

float saturate_cos(float val) {
  return clamp(val, EPS_COS, 1.0);
}

vec3 saturate(vec3 v) {
  return vec3(saturate(v.x), saturate(v.y), saturate(v.z));
}

float sqr(float x) {
  return x * x;
}

float sum(vec3 v) {
  return v.x + v.y + v.z;
}

vec3 flip(vec3 v, vec3 n) {
  return normalize(v - (n + n) * abs(dot(v, n)));
}

float luminance(vec3 rgb) {
  return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

bool isNan(float val) {
  return (val <= 0.0 || 0.0 <= val) ? false : true;
}

vec3 to_local(vec3 v, Geometry g) {
  return vec3(dot(v, g.t), dot(v, g.b), dot(v, g.n));
}

vec3 to_world(vec3 v, Geometry g) {
  return g.t * v.x + g.b * v.y + g.n * v.z;
}

bool has_flag(int flags, int mask) {
  return (flags & mask) > 0;
}
// wi points towards surface
vec3 refractIt(vec3 wi, vec3 n, float inv_eta, out bool tir) {
  tir = false;
  float cosi = dot(-wi, n);
  float cost2 = 1.0 - inv_eta * inv_eta * (1.0 - cosi * cosi);
  if (cost2 <= 0.0) {
    tir = true;
    return reflect(wi, n);
  }
  return inv_eta * wi + ((inv_eta * cosi - sqrt(abs(cost2))) * n);
}

// Bends shading normal n into the direction of the geometry normal ng
// such that incident direction wi reflected at n does not change
// hemisphere
vec3 clamp_normal(vec3 n, vec3 ng, vec3 wi) {
  vec3 ns_new = n;
  vec3 r = reflect(-wi, n);
  float v_dot_ng = dot(wi, ng);
  float r_dot_ng = dot(r, ng);

  // if wi and r are in different hemisphere in respect of geometry normal
  if (v_dot_ng * r_dot_ng < 0.0) {
    float ns_dot_ng = abs(dot(n, ng));
    vec3 offset_vec = n * (-r_dot_ng / ns_dot_ng);
    vec3 r_corrected = normalize(r + offset_vec); // move r on horizon
    r_corrected =
        normalize(r_corrected + (ng * EPS_COS) * ((v_dot_ng > 0.0) ? 1.0 : -1.0)); // to avoid precision problems
    ns_new = normalize(wi + r_corrected);
    ns_new *= (dot(ns_new, n) < 0.0) ? -1.0 : 1.0;
  }
  return ns_new;
}

// Flips normal n and geometry normal ng such that they point into
// the direction of the given incident direction wi.
// This function should be called in each sample/eval function to prepare
// the tangent space in a way that the BSDF looks the same from top and
// bottom (two-sided materials).
bool fix_normals(inout vec3 n, inout vec3 ng, in vec3 wi) {
  bool backside = false;
  if (dot(wi, ng) < 0.0) {
    ng = -ng;
    backside = true;
  }
  if (dot(ng, n) < 0.0) {
    n = -n;
  }
  return backside;
}

vec3 fix_normal(in vec3 n, in vec3 wi) {
  return dot(n, wi) < 0.0 ? -n : n;
}

mat3 get_onb(vec3 n) {
  // from Spencer, Jones "Into the Blue", eq(3)
  vec3 tangent = normalize(cross(n, vec3(-n.z, n.x, -n.y)));
  vec3 bitangent = cross(n, tangent);
  return mat3(tangent, bitangent, n);
}

mat3 get_onb(vec3 n, vec3 t) {
  vec3 b = normalize(cross(n, t));
  vec3 tt = cross(b, n);
  return mat3(tt, b, n);
}

Geometry calculateBasis(vec3 n, vec4 t) {
  Geometry g;
  g.n = n;
  g.t = t.xyz;
  g.b = cross(n, t.xyz) * t.w;
  return g;
}

float computeTheta(vec3 dir) {
  return acos(max(-1.0, min(1.0, dir.y)));
}

float computePhi(vec3 dir) {
  float temp = atan(dir.z, dir.x);
  if (temp < 0.0)
    return TWO_PI + temp;
  else
    return temp;
}

vec3 fromThetaPhi(float theta, float phi) {
  return vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

vec2 dir_to_uv(vec3 dir, out float pdf) {
  float theta = computeTheta(dir);
  pdf = 1.0 / (TWO_PI * PI * max(EPS_COS, sin(theta)));
  return vec2(computePhi(dir) * 0.5f, theta) * ONE_OVER_PI;
}

vec3 uv_to_dir(vec2 uv, out float pdf) {
  float theta = uv.y * PI;
  float phi = uv.x * TWO_PI;
  pdf = 1.0 / (TWO_PI * PI * max(EPS_COS, sin(theta)));
  return fromThetaPhi(theta, phi);
}

vec3 sampleHemisphereCosine(vec2 uv, out float pdf) {
  float phi = uv.y * TWO_PI;
  float cosTheta = sqrt(1.0 - uv.x);
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  pdf = cosTheta * ONE_OVER_PI;
  return vec3(vec2(cos(phi), sin(phi)) * sinTheta, cosTheta);
}

vec3 sampleHemisphereUniform(vec2 uv, out float pdf) {
  float phi = uv.y * TWO_PI;
  float cosTheta = 1.0 - uv.x;
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  pdf = ONE_OVER_TWO_PI;
  return vec3(vec2(cos(phi), sin(phi)) * sinTheta, cosTheta);
  }

vec3 compute_triangle_normal(in vec3 p0, in vec3 p1, in vec3 p2) {
  return normalize(cross(p1 - p0, p2 - p0));
}

float max_(vec3 v) {
  return max(v.x, max(v.y, v.z));
}


vec4 rotation_to_tangent(float angle, vec3 normal, vec4 tangent) {
  if (angle > 0.0) {
    Geometry g = calculateBasis(normal, tangent);
    return vec4(g.t * cos(angle) + g.b * sin(angle), tangent.w);
  } else {
    return tangent;
  }
}

vec3 to_linear_rgb(vec3 srgb) {
  return pow(srgb, vec3(2.2));
}

int lower_bound(sampler2D data, int row, int size, float value)
{
  int idx;
  int step;
  int count = size;
  int first = 0;
  while (count > 0)
  {
    idx = first;
    step = count >> 1;
    idx += step;
    float v = texelFetch(data, ivec2(idx, row), 0).x;
    if (v < value)
    {
      first = ++idx;
      count -= step + 1;
    }
    else
      count = step;
  }
  return first;
}

float mis_balance_heuristic(float a, float b) {
	return a / (a + b);
}

float pow5(float x) { return x*x*x*x*x; }
