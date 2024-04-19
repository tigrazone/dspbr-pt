
#define ll(x) ( a / (1.0 + b * pow(abs((x)), c)) + d * (x) + e )

float directional_albedo_sheen(float cos_theta, float alpha) {
  float c = 1.0 - cos_theta;
  return 0.65584461 * c * c * c + 1.0 / (4.16526551 + exp(-7.97291361 * sqrt(alpha) + 6.33516894));
}

// Michael Ashikhmin, Simon Premoze – “Distribution-based BRDFs”, 2007
float ashikhminV(float alpha, float cos_theta_i, float cos_theta_o) {
  return 1.0 / (4.0 * (cos_theta_o + cos_theta_i - cos_theta_o * cos_theta_i));
}

// Alejandro Conty Estevez, Christopher Kulla. Production Friendly Microfacet
// Sheen BRDF, SIGGRAPH 2017.
// http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf
float charlieV(float alpha, float cos_theta_i, float cos_theta_o) {
  //prepare variables
  float oneMinusAlphaSq = (1.0 - alpha) * (1.0 - alpha);
  float a = mix(21.5473, 25.3245, oneMinusAlphaSq);
  float b = mix(3.82987, 3.32435, oneMinusAlphaSq);
  float c = mix(0.19823, 0.16801, oneMinusAlphaSq);
  float d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
  float e = mix(-4.32054, -4.85967, oneMinusAlphaSq);

  float l_0_5_2 = ll(0.5);
  l_0_5_2 += l_0_5_2;

  float lambda_sheen_i = abs(cos_theta_i) < 0.5 ? exp(ll(cos_theta_i)) : exp(l_0_5_2 - ll(1.0 - cos_theta_i));
  float lambda_sheen_o = abs(cos_theta_o) < 0.5 ? exp(ll(cos_theta_o)) : exp(l_0_5_2 - ll(1.0 - cos_theta_o));
  return 1.0 / (1.0 + lambda_sheen_i + lambda_sheen_o);
}

// https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2022x.md.html#components/sheen
vec3 sheen_layer(out float base_weight, vec3 sheen_color, float sheen_roughness, vec3 wi,
                 vec3 wo, vec3 wh, Geometry g) {
  // We clamp the roughness to range[0.07; 1] to avoid numerical issues and
  // because we observed that the directional albedo at grazing angles becomes
  // larger than 1 if roughness is below 0.07
  float alpha = max(sheen_roughness, 0.07);
  alpha = alpha*alpha;
  float inv_alpha = 1.0 / alpha;

  float cos_theta_i = saturate_cos(dot(wi, g.n));
  float cos_theta_o = saturate_cos(dot(wo, g.n));
  float sin_theta_h_2 = max(1.0 - sqr(dot(wh, g.n)), 0.001);
  float D = (2.0 + inv_alpha) * pow(abs(sin_theta_h_2), 0.5 * inv_alpha) * ONE_OVER_TWO_PI;

  float G = 1.0;

  if (int(u_sheen_G) == 0) {
    G = charlieV(alpha, cos_theta_i, cos_theta_o);
  } else {
    G = ashikhminV(alpha, cos_theta_i, cos_theta_o);
  }

  float sheen = G * D / (4.0 * cos_theta_i * cos_theta_o);

  float Ewi = max_(sheen_color) * directional_albedo_sheen(cos_theta_i, alpha);
  float Ewo = max_(sheen_color) * directional_albedo_sheen(cos_theta_o, alpha);

  base_weight = min(1.0 - Ewi, 1.0 - Ewo);

  return sheen_color * sheen;
}