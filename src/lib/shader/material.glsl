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
#include <fresnel>
#include <iridescence>

struct MaterialTextureInfo {
  TexInfo albedoTexture;
  TexInfo metallicRoughnessTexture;
  TexInfo normalTexture;
  TexInfo emissionTexture;
  TexInfo specularTexture;
  TexInfo specularColorTexture;
  TexInfo transmissionTexture;
  TexInfo clearcoatTexture;
  TexInfo clearcoatRoughnessTexture;
  // TexInfo clearcoatNormalTexture;
  TexInfo sheenColorTexture;
  TexInfo sheenRoughnessTexture;
  TexInfo anisotropyTexture;
  TexInfo anisotropyDirectionTexture;
  TexInfo iridescenceTexture;
  TexInfo iridescenceThicknessTexture;
};

void unpackMaterialData(in uint idx, out MaterialData matData) {
  vec4 val;
  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 0u, MATERIAL_SIZE), 0);
  matData.albedo = val.xyz;
  matData.metallic = val.w;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 1u, MATERIAL_SIZE), 0);
  matData.roughness = val.x;
  matData.anisotropy = val.y;
  matData.anisotropyRotation = val.z * 2.0 * PI;
  matData.transparency = val.w;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 2u, MATERIAL_SIZE), 0);
  matData.cutoutOpacity = val.x;
  matData.doubleSided = bool(val.y);
  matData.normalScale = val.z;
  matData.ior = val.w;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 3u, MATERIAL_SIZE), 0);
  matData.specular = val.x;
  matData.specularTint = val.yzw;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 4u, MATERIAL_SIZE), 0);
  matData.sheenRoughness = val.x;
  matData.sheenColor = val.yzw;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 5u, MATERIAL_SIZE), 0);
  matData.normalScaleClearcoat = val.x;
  matData.emission = val.yzw;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 6u, MATERIAL_SIZE), 0);
  matData.clearcoat = val.x;
  matData.clearcoatRoughness = val.y;
  matData.translucency = val.z;
  matData.alphaCutoff = val.w;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 7u, MATERIAL_SIZE), 0);
  matData.attenuationDistance = val.x;
  matData.attenuationColor = val.yzw;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 8u, MATERIAL_SIZE), 0);
  matData.subsurfaceColor = val.xyz;
  matData.thinWalled = bool(val.w);

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 9u, MATERIAL_SIZE), 0);
  matData.anisotropyDirection = val.xyz;

  val = texelFetch(u_sampler2D_MaterialData, getStructParameterTexCoord(idx, 10u, MATERIAL_SIZE), 0);
  matData.iridescence = val.x;
  matData.iridescenceIor = val.y;
  matData.iridescenceThicknessMinimum = val.z;
  matData.iridescenceThicknessMaximum = val.w;
}

TexInfo getTextureInfo(ivec2 texInfoIdx, ivec2 transformInfoIdx) {
  ivec4 texArrayInfo = ivec4(texelFetch(u_sampler2D_MaterialTexInfoData, texInfoIdx, 0));
  vec4 transformInfo = texelFetch(u_sampler2D_MaterialTexInfoData, transformInfoIdx, 0);

  TexInfo texInfo;
  texInfo.texArrayIdx = texArrayInfo.x;
  texInfo.texIdx = texArrayInfo.y;
  texInfo.texCoordSet = texArrayInfo.z;
  texInfo.texOffset = transformInfo.xy;
  texInfo.texScale = transformInfo.zw;

  return texInfo;
}

void unpackMaterialTexInfo(in uint idx, out MaterialTextureInfo matTexInfo) {
  uint tex_info_stride = MATERIAL_TEX_INFO_SIZE * TEX_INFO_SIZE;

  ivec2 albedoTexInfoIdx = getStructParameterTexCoord(idx, 0u, tex_info_stride);
  ivec2 albedoTexTransformsIdx = getStructParameterTexCoord(idx, 1u, tex_info_stride);
  matTexInfo.albedoTexture = getTextureInfo(albedoTexInfoIdx, albedoTexTransformsIdx);

  ivec2 metallicRoughnessTexInfoIdx = getStructParameterTexCoord(idx, 2u, tex_info_stride);
  ivec2 metallicRoughnessTexTransformsIdx = getStructParameterTexCoord(idx, 3u, tex_info_stride);
  matTexInfo.metallicRoughnessTexture = getTextureInfo(metallicRoughnessTexInfoIdx, metallicRoughnessTexTransformsIdx);

  ivec2 normalTexInfoIdx = getStructParameterTexCoord(idx, 4u, tex_info_stride);
  ivec2 normalTexTexTransformsIdx = getStructParameterTexCoord(idx, 5u, tex_info_stride);
  matTexInfo.normalTexture = getTextureInfo(normalTexInfoIdx, normalTexTexTransformsIdx);

  ivec2 emissionTexInfoIdx = getStructParameterTexCoord(idx, 6u, tex_info_stride);
  ivec2 emissionTexTransformsIdx = getStructParameterTexCoord(idx, 7u, tex_info_stride);
  matTexInfo.emissionTexture = getTextureInfo(emissionTexInfoIdx, emissionTexTransformsIdx);

  ivec2 specularTexInfoIdx = getStructParameterTexCoord(idx, 8u, tex_info_stride);
  ivec2 specularTexTransformsIdx = getStructParameterTexCoord(idx, 9u, tex_info_stride);
  matTexInfo.specularTexture = getTextureInfo(specularTexInfoIdx, specularTexTransformsIdx);

  ivec2 specularColorTexInfoIdx = getStructParameterTexCoord(idx, 10u, tex_info_stride);
  ivec2 specularColorTexTransformsIdx = getStructParameterTexCoord(idx, 11u, tex_info_stride);
  matTexInfo.specularColorTexture = getTextureInfo(specularColorTexInfoIdx, specularColorTexTransformsIdx);

  ivec2 transmissionTexInfoIdx = getStructParameterTexCoord(idx, 12u, tex_info_stride);
  ivec2 transmissionTexTransformsIdx = getStructParameterTexCoord(idx, 13u, tex_info_stride);
  matTexInfo.transmissionTexture = getTextureInfo(transmissionTexInfoIdx, transmissionTexTransformsIdx);

  ivec2 clearcoatTexInfoIdx = getStructParameterTexCoord(idx, 14u, tex_info_stride);
  ivec2 clearcoatTexTransformsIdx = getStructParameterTexCoord(idx, 15u, tex_info_stride);
  matTexInfo.clearcoatTexture = getTextureInfo(clearcoatTexInfoIdx, clearcoatTexTransformsIdx);

  ivec2 clearcoatRoughnessTexInfoIdx = getStructParameterTexCoord(idx, 16u, tex_info_stride);
  ivec2 clearcoatRoughnessTexTransformsIdx =
      getStructParameterTexCoord(idx, 17u, tex_info_stride);
  matTexInfo.clearcoatRoughnessTexture = getTextureInfo(clearcoatRoughnessTexInfoIdx, clearcoatRoughnessTexTransformsIdx);

  ivec2 sheenColorTexInfoIdx = getStructParameterTexCoord(idx, 18u, tex_info_stride);
  ivec2 sheenColorTexTransformsIdx = getStructParameterTexCoord(idx, 19u, tex_info_stride);
  matTexInfo.sheenColorTexture = getTextureInfo(sheenColorTexInfoIdx, sheenColorTexTransformsIdx);

  ivec2 sheenRoughnessTexInfoIdx = getStructParameterTexCoord(idx, 20u, tex_info_stride);
  ivec2 sheenRoughnessTexTransformsIdx = getStructParameterTexCoord(idx, 21u, tex_info_stride);
  matTexInfo.sheenRoughnessTexture = getTextureInfo(sheenRoughnessTexInfoIdx, sheenRoughnessTexTransformsIdx);

  // ivec2 clearcoatNormalTexInfoIdx = getStructParameterTexCoord(idx, 8u,
  // MATERIAL_TEX_INFO_SIZE*TEX_INFO_SIZE); ivec2 clearcoatNormalTexTransformsIdx =
  // getStructParameterTexCoord(idx, 9u, MATERIAL_TEX_INFO_SIZE*TEX_INFO_SIZE);
  // matTexInfo.clearcoatNormalTexture = getTextureInfo(clearcoatNormalTexInfoIdx,
  // clearcoatNormalTexTransformsIdx);

  ivec2 anisotropyTexInfoIdx = getStructParameterTexCoord(idx, 22u, tex_info_stride);
  ivec2 anisotropyTexTransformsIdx = getStructParameterTexCoord(idx, 23u, tex_info_stride);
  matTexInfo.anisotropyTexture = getTextureInfo(anisotropyTexInfoIdx, anisotropyTexTransformsIdx);

  ivec2 anisotropyDirectionTexInfoIdx = getStructParameterTexCoord(idx, 24u, tex_info_stride);
  ivec2 anisotropyDirectionTexTransformsIdx = getStructParameterTexCoord(idx, 25u, tex_info_stride);
  matTexInfo.anisotropyDirectionTexture = getTextureInfo(anisotropyDirectionTexInfoIdx, anisotropyDirectionTexTransformsIdx);

  ivec2 iridescenceTexInfoIdx = getStructParameterTexCoord(idx, 26u, tex_info_stride);
  ivec2 iridescenceTexTransformsIdx = getStructParameterTexCoord(idx, 27u, tex_info_stride);
  matTexInfo.iridescenceTexture = getTextureInfo( iridescenceTexInfoIdx, iridescenceTexTransformsIdx);

  ivec2 iridescenceThicknessTexInfoIdx = getStructParameterTexCoord(idx, 28u, tex_info_stride);
  ivec2 iridescenceThicknessTexTransformsIdx = getStructParameterTexCoord(idx, 29u, tex_info_stride);
  matTexInfo.iridescenceThicknessTexture = getTextureInfo( iridescenceThicknessTexInfoIdx,  iridescenceThicknessTexTransformsIdx);
}

// Convert from roughness and anisotropy to 2d anisotropy.
vec2 roughness_conversion(float roughness, float anisotropy) {
  vec2 a = vec2(roughness * (1.0 + anisotropy), roughness * (1.0 - anisotropy));
  return max(a * a, vec2(MINIMUM_ROUGHNESS));
}

void configure_material(const in uint matIdx, inout RenderState rs, out MaterialClosure c, vec4 vertexColor) {
  vec2 uv = rs.uv0;

  MaterialData matData;
  MaterialTextureInfo matTexInfo;

  unpackMaterialData(matIdx, matData);
  unpackMaterialTexInfo(matIdx, matTexInfo);

  vec4 albedo = evaluateMaterialTextureValue(matTexInfo.albedoTexture, uv);
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

  vec4 transmission = evaluateMaterialTextureValue(matTexInfo.transmissionTexture, uv);
  c.transparency = matData.transparency * transmission.x;

  c.translucency = matData.translucency;

  c.thin_walled = matData.thinWalled;
  c.ior = matData.ior;

  c.double_sided = matData.doubleSided;

  vec4 occlusionRoughnessMetallic = evaluateMaterialTextureValue(matTexInfo.metallicRoughnessTexture, uv);
  c.metallic = matData.metallic * occlusionRoughnessMetallic.z;
  float roughness = matData.roughness * occlusionRoughnessMetallic.y;

  float anisotropy = evaluateMaterialTextureValue(matTexInfo.anisotropyTexture, uv).x * 2.0 - 1.0;
  c.alpha = roughness_conversion(roughness, matData.anisotropy * anisotropy);

  vec4 specularColor = evaluateMaterialTextureValue(matTexInfo.specularColorTexture, rs.uv0);
  c.specular_tint = matData.specularTint * pow(specularColor.rgb, vec3(2.2));
  vec4 specular = evaluateMaterialTextureValue(matTexInfo.specularTexture, rs.uv0);
  c.specular = matData.specular * specular.a;

  vec4 sheenColor = evaluateMaterialTextureValue(matTexInfo.sheenColorTexture, rs.uv0);
  vec4 sheenRoughness = evaluateMaterialTextureValue(matTexInfo.sheenRoughnessTexture, rs.uv0);
  c.sheen_roughness = matData.sheenRoughness * sheenRoughness.x;
  c.sheen_color = matData.sheenColor * sheenColor.xyz;

  c.n = rs.normal;
  c.ng = rs.geometryNormal;
  c.t = vec4(rs.tangent.xyz, rs.tangent.w);

  if (matTexInfo.normalTexture.texIdx >= 0) {
    mat3 to_world = get_onb(c.n, c.t.xyz);
    vec3 n = normalize(evaluateMaterialTextureValue(matTexInfo.normalTexture, uv).xyz * 2.0 - vec3(1.0));
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
  if(matTexInfo.anisotropyDirectionTexture.texArrayIdx >= 0)
    ansiotropyDirection = evaluateMaterialTextureValue(matTexInfo.anisotropyDirectionTexture, uv).xyz * 2.0 - vec3(1);
  ansiotropyDirection.z = 0.0;

  float anisotropyRotation = atan(ansiotropyDirection.y, ansiotropyDirection.x) + PI;
  c.t = rotation_to_tangent(anisotropyRotation, c.n, c.t);

  if (c.backside) {
    c.f0 = ((1.0 - c.ior) / (1.0 + c.ior)) * ((1.0 - c.ior) / (1.0 + c.ior));
  } else {
    c.f0 = ((c.ior - 1.0) / (c.ior + 1.0)) * ((c.ior - 1.0) / (c.ior + 1.0));
  }
  c.specular_f0 = (1.0 - c.metallic) * c.specular * c.f0 * c.specular_tint + c.metallic * c.albedo;
  c.specular_f90 = vec3((1.0 - c.metallic) * c.specular + c.metallic);

  vec3 emission = evaluateMaterialTextureValue(matTexInfo.emissionTexture, uv).xyz;
  c.emission = matData.emission * to_linear_rgb(emission);

  vec4 clearcoat = evaluateMaterialTextureValue(matTexInfo.clearcoatTexture, uv);
  c.clearcoat = matData.clearcoat * clearcoat.y;
  vec4 clearcoatRoughness = evaluateMaterialTextureValue(matTexInfo.clearcoatRoughnessTexture, uv);
  float clearcoat_alpha =
      matData.clearcoatRoughness * matData.clearcoatRoughness * clearcoatRoughness.x * clearcoatRoughness.x;
  c.clearcoat_alpha = max(clearcoat_alpha, MINIMUM_ROUGHNESS);

  c.attenuationColor =  matData.attenuationColor;
  c.attenuationDistance = matData.attenuationDistance;

  c.iridescence = matData.iridescence * evaluateMaterialTextureValue(matTexInfo.iridescenceTexture, uv).x;
  c.iridescence_ior = matData.iridescenceIor;
  c.iridescence_thickness = mix(matData.iridescenceThicknessMinimum, matData.iridescenceThicknessMaximum, evaluateMaterialTextureValue(matTexInfo.iridescenceThicknessTexture, uv).y);
}
