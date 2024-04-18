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

#define E_DIFFUSE					0x00001
#define E_DELTA						0x00002
#define E_REFLECTION				0x00004
#define E_TRANSMISSION				0x00008
#define E_COATING					0x00010
#define E_STRAIGHT					0x00020
#define E_OPAQUE_DIELECTRIC			0x00040
#define E_TRANSPARENT_DIELECTRIC	0x00080
#define E_METAL						0x00100

// Convert from roughness and anisotropy to 2d anisotropy.
vec2 roughness_conversion(float roughness, float anisotropy) {
  vec2 a = vec2(roughness * (1.0 + anisotropy), roughness * (1.0 - anisotropy));
  return max(a * a, vec2(MINIMUM_ROUGHNESS));
}

bool is_specular_event(MaterialClosure c) {
  return bool(c.event_type & E_DELTA);
}

void configure_material(const in uint matIdx, in RenderState rs, out MaterialClosure c, vec4 vertexColor) {
  vec2 uv = rs.uv0;

  MaterialData matData = get_material(matIdx);

  vec4 albedo = get_texture_value(matData.albedoTextureId, uv);
  c.albedo = matData.albedo * to_linear_rgb(albedo.xyz);
  float opacity = albedo.w;

  if (length(vertexColor) > 0.0) {
    c.albedo *= vertexColor.xyz;
    opacity *= vertexColor.w;
  }

  c.cutout_opacity = matData.cutoutOpacity * opacity;
  if (matData.alphaCutoff > 0.0) { // MASK
    c.cutout_opacity = step(matData.alphaCutoff, c.cutout_opacity);
  }
  if (matData.alphaCutoff == 1.0) { // OPAQUE
    c.cutout_opacity = 1.0;
  }

  c.transparency = matData.transparency * get_texture_value(matData.transmissionTextureId, uv).x;

  c.translucency = matData.translucency * get_texture_value(matData.translucencyTextureId, uv).w;
  c.translucencyColor = matData.translucencyColor * to_linear_rgb(get_texture_value(matData.translucencyColorTextureId, uv).xyz);

  c.thin_walled = matData.thinWalled;
  // c.ior = c.thin_walled ? 1.0 : matData.ior;
  c.ior = matData.ior;

  c.double_sided = matData.doubleSided;

  vec4 occlusionRoughnessMetallic = get_texture_value(matData.metallicRoughnessTextureId, uv);
  c.metallic = matData.metallic * occlusionRoughnessMetallic.z;
  float roughness = matData.roughness * occlusionRoughnessMetallic.y;

  float anisotropy = get_texture_value(matData.anisotropyTextureId, uv).x * 2.0 - 1.0;
  c.alpha = roughness_conversion(roughness, matData.anisotropy * anisotropy);

  vec4 specularColor = get_texture_value(matData.specularColorTextureId, rs.uv0);
  c.specular_tint = matData.specularTint * to_linear_rgb(specularColor.rgb);
  vec4 specular = get_texture_value(matData.specularTextureId, rs.uv0);
  c.specular = matData.specular * specular.a;

  vec4 sheenColor = get_texture_value(matData.sheenColorTextureId, rs.uv0);
  vec4 sheenRoughness = get_texture_value(matData.sheenRoughnessTextureId, rs.uv0);
  c.sheen_roughness = matData.sheenRoughness * sheenRoughness.x;
  c.sheen_color = matData.sheenColor * sheenColor.xyz;

  c.n = rs.n;
  c.ng = rs.ng;
  c.t = vec4(rs.tangent.xyz, rs.tangent.w);

  if (matData.normalTextureId >= 0.0) {
    mat3 to_world = get_onb(c.n, c.t.xyz);
    vec3 n = normalize(get_texture_value(matData.normalTextureId, uv).xyz * 2.0 - vec3(1.0));
    n = normalize(n * vec3(matData.normalScale, matData.normalScale, 1.0));
    c.n = to_world * n;

    // ensure orthonormal tangent after changing normal
    vec3 b = normalize(cross(c.n, c.t.xyz)) * c.t.w;
    c.t.xyz = cross(b, c.n);
  }

  // ensure n and ng point into the same hemisphere as wi
  // remember whether we hit from backside
  vec3 wi = rs.wi;
  c.backside = fix_normals(c.n, c.ng, wi);

  vec3 ansiotropyDirection = matData.anisotropyDirection;
  if (matData.anisotropyDirectionTextureId >= 0.0)
    ansiotropyDirection = get_texture_value(matData.anisotropyDirectionTextureId, uv).xyz * 2.0 - vec3(1);
  ansiotropyDirection.z = 0.0;

  float anisotropyRotation = atan(ansiotropyDirection.y, ansiotropyDirection.x) + PI;
  c.t = rotation_to_tangent(anisotropyRotation, c.n, c.t);

  if (c.backside && !c.thin_walled) {
    c.f0 = ((1.0 - c.ior) / (1.0 + c.ior)) * ((1.0 - c.ior) / (1.0 + c.ior));
  } else {
    c.f0 = ((c.ior - 1.0) / (c.ior + 1.0)) * ((c.ior - 1.0) / (c.ior + 1.0));
  }
  c.specular_f0 = mix(c.specular * c.f0 * c.specular_tint, c.albedo, c.metallic);
  c.specular_f90 = vec3(mix(c.specular, 1.0, c.metallic));

  vec3 emission = get_texture_value(matData.emissionTextureId, uv).xyz;
  c.emission = matData.emission.xyz * to_linear_rgb(emission);

  vec4 clearcoat = get_texture_value(matData.clearcoatTextureId, uv);
  c.clearcoat = matData.clearcoat * clearcoat.y;
  vec4 clearcoatRoughness = get_texture_value(matData.clearcoatRoughnessTextureId, uv);
  float clearcoat_alpha =
      matData.clearcoatRoughness * matData.clearcoatRoughness * clearcoatRoughness.x * clearcoatRoughness.x;
  c.clearcoat_alpha = max(clearcoat_alpha, MINIMUM_ROUGHNESS);

  c.attenuationColor = matData.attenuationColor;
  c.attenuationDistance = matData.attenuationDistance;

  c.iridescence = matData.iridescence * get_texture_value(matData.iridescenceTextureId, uv).x;
  c.iridescence_ior = matData.iridescenceIor;
  c.iridescence_thickness = mix(matData.iridescenceThicknessMinimum, matData.iridescenceThicknessMaximum,
                                get_texture_value(matData.iridescenceThicknessTextureId, uv).y);
}
