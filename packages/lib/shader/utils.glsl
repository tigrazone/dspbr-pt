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

vec3 saturate(vec3 v) {
  return vec3(saturate(v.x), saturate(v.y), saturate(v.z));
}

#define saturate_cos(val) ( clamp((val), EPS_COS, 1.0) )

float sqr(float x) { return x * x; }

#define sum(v) ( (v).x + (v).y + (v).z )

#define flip(v, n) (  normalize((v) - ((n) + (n)) * abs(dot((v), (n)))) )

vec3 lumMUL = vec3(0.2126, 0.7152, 0.0722);
#define luminance(rgb) ( dot(lumMUL, (rgb)) )

#define isNan(val) ( !((val) <= 0.0 || 0.0 <= (val)) )

#define to_local(v, g) ( (v) * mat3((g).t, (g).b, (g).n) )

#define to_world(v, g) (mat3((g).t, (g).b, (g).n) * (v))

#define has_flag(flags, mask) ( ((flags) & (mask)) > 0 )

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

#define fix_normal(n, wi) ( dot((n), (wi)) < 0.0 ? -(n) : (n) )

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

vec2 dir_to_uv(vec3 dir, out float pdf) {
  float theta = acos(max(-1.0, min(1.0, dir.y)));
  pdf = 1.0 / (TWO_PI * PI * max(EPS_COS, sin(theta)));

  float phi = atan(dir.z, dir.x);
  if (phi < 0.0) phi += TWO_PI;

  return vec2(phi * 0.5f, theta) * ONE_OVER_PI;
}

vec3 uv_to_dir(vec2 uv, out float pdf) {
  float theta = uv.y * PI;
  float phi = uv.x * TWO_PI;
  pdf = 1.0 / (TWO_PI * PI * max(EPS_COS, sin(theta)));
  return vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

vec3 sampleHemisphereCosine(vec2 uv, out float pdf) {
  float cosPhi = uv.y + uv.y - 1.0;
  float cosTheta = sqrt(1.0 - uv.x);
  pdf = cosTheta * ONE_OVER_PI;
  return vec3(vec2(cosPhi, sqrt(1.0 - cosPhi * cosPhi)) * sqrt(uv.x), cosTheta);
}

#define compute_triangle_normal(p0, p1, p2) ( normalize(cross((p1) - (p0), (p2) - (p0))) )

#define max_(v) ( max((v).x, max((v).y, (v).z)) )

vec4 rotation_to_tangent(float angle, vec3 normal, vec4 tangent) {
  if (angle > 0.0) {
    Geometry g = calculateBasis(normal, tangent);
    return vec4(g.t * cos(angle) + g.b * sin(angle), tangent.w);
  } else {
    return tangent;
  }
}

#define to_linear_rgb(srgb) ( pow((srgb), vec3(2.2)) )

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

#define mis_balance_heuristic(a, b) ( (a) / ((a) + (b)) )

float pow5(float x) { return x*x*x*x*x; }
