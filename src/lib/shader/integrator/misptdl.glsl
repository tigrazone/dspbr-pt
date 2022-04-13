vec4 trace_misptdl(bvh_ray r)
{
  bvh_hit hit;
  vec3 path_weight = vec3(1.0);

  bool last_bounce_specular = true; // pinhole camera is considered singular
  float last_bounce_pdf = 0.0;
  float last_bsdf_selection_pdf = 0.0;

  int bounce = 0;
  vec3 radiance = vec3(0.0);
  while(bounce <= int(u_max_bounces) || last_bounce_specular)
  {
    if(check_russian_roulette_path_termination(bounce, path_weight)) break;

    RenderState rs;
    if (bvh_intersect_nearest(r, hit)) { // primary camera ray
      fillRenderState(r, hit, rs);
      last_bsdf_selection_pdf = rs.closure.bsdf_selection_pdf;
       // Absorption
      if(rs.closure.backside && !rs.closure.thin_walled) {
        vec3 absorption_sigma = -log(rs.closure.attenuationColor) / rs.closure.attenuationDistance;
        path_weight *= exp(-absorption_sigma*hit.tfar);
      }

      radiance += rs.closure.emission * path_weight;
      last_bounce_specular = bool(rs.closure.event_type & E_SINGULAR);

      if(!last_bounce_specular) { // We can't evaluate direct light conribution for singular events
        // radiance += sampleAndEvaluatePointLight(rs); // point light contribution is always evaluated

        float ibl_sample_pdf;
        vec3 ibl_sample_dir;
        vec3 ibl_radiance = vec3(0);

        if(bool(u_bool_UseIBL))
        {
          vec3 ibl_sample_dir = sample_ibl_dir_importance(rng_float(), rng_float(), ibl_sample_pdf);

          float cosNL = dot(ibl_sample_dir, rs.closure.n);
          if(bool(rs.closure.event_type & (E_REFLECTION | E_DIFFUSE)) && cosNL > EPS_COS) {
            if(!isOccluded(rs.hitPos + rs.closure.n * u_float_ray_eps, ibl_sample_dir)) {
              ibl_radiance =  eval_dspbr(rs.closure, rs.wi, ibl_sample_dir) * eval_ibl(ibl_sample_dir) * cosNL;
            }
          }
          else if(bool(rs.closure.event_type & E_TRANSMISSION) && cosNL < EPS_COS) {
            if(!isOccluded(rs.hitPos - rs.closure.n * u_float_ray_eps, ibl_sample_dir)) {
              ibl_radiance = eval_dspbr(rs.closure, rs.wi, ibl_sample_dir) * eval_ibl(ibl_sample_dir) * -cosNL;
            }
          }
          //sampleAndEvaluateEnvironmentLight(rs, rng_float(), rng_float(), ibl_sample_dir, ibl_radiance, ibl_sample_pdf)
          float brdf_sample_pdf = dspbr_pdf(rs.closure, rs.wi, ibl_sample_dir) * rs.closure.bsdf_selection_pdf;
          if(ibl_sample_pdf > EPS_PDF && brdf_sample_pdf > EPS_PDF) {
            float mis_weight = mis_balance_heuristic(ibl_sample_pdf, brdf_sample_pdf);
            radiance += ibl_radiance * path_weight * mis_weight;
          }
        }
      }

      vec3 bounce_weight;
      if(!sample_bsdf_bounce(rs, bounce_weight, last_bounce_pdf)) return vec4(radiance, 1.0); //absorped
      path_weight *= bounce_weight;

      r = bvh_create_ray(rs.wo, rs.hitPos + fix_normal(rs.ng, rs.wo) * u_float_ray_eps, TFAR_MAX);
      bounce++;
    }
    else {
      if(bool(u_bool_UseIBL)) {
        if(bounce == 0) {
            if (bool(u_bool_ShowBackground)) {
              return vec4(eval_ibl(r.dir), 1.0);
            } else {
              return vec4(pow(u_BackgroundColor.xyz, vec3(2.2)), u_BackgroundColor.w);
            }
        } else {
          float ibl_sample_pdf = sampleEnvironmentLightPdf(r.dir);
          float mis_weight = last_bounce_specular ? 1.0 : mis_balance_heuristic(last_bounce_pdf *last_bsdf_selection_pdf, ibl_sample_pdf);
          radiance += eval_ibl(r.dir) * path_weight * mis_weight;
        }
      }
      break;
    }

  }

  return vec4(radiance, 1.0);
}