vec3 fresnel_schlick(vec3 f0, vec3 f90, float theta) {
  return f0 + (f90 - f0) * pow5(abs(1.0 - theta));
}

float fresnel_schlick(float f0, float f90, float theta) {
  return f0 + (f90 - f0) * pow5(abs(1.0 - theta));
}

vec3 fresnel_schlick_dielectric(float cos_theta, vec3 f0, vec3 f90, float ni, float nt, bool thin_walled) {
  if (ni > nt && !thin_walled) {
    float inv_eta = ni / nt;
    float sin_theta2 = sqr(inv_eta) * (1.0 - sqr(cos_theta));
    if (sin_theta2 >= 1.0) {
      return vec3(1.0); // TIR
    }

    //     // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/,
    cos_theta = sqrt(1.0 - sin_theta2);
  }

  return fresnel_schlick(f0, f90, cos_theta);
}


vec3 schlick_to_f0(vec3 f, vec3 f90, float VdotH) {
    float x = clamp(1.0 - VdotH, 0.0, 1.0);
    float x5 = clamp(pow5(x), 0.0, 0.9999);

    return (f - f90 * x5) / (1.0 - x5);
}