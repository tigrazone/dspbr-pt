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

// @ts-ignore
import { SimpleTriangleBVH } from './bvh';
import { PathtracingSceneData } from './scene_data'
import { PathtracingSceneDataWebGL2 } from './scene_data_webgl2'

import * as glu from './gl_utils';

import shader_constants from './shader/constants.glsl';
import copy_shader from './shader/copy.glsl';
import display_shader from './shader/display.frag';
import tonemap_shader from './shader/tonemap.frag';
import fxaa_shader from './shader/fxaa.frag';

import structs_shader from './shader/structs.glsl';
import rng_shader from './shader/rng.glsl';
import utils_shader from './shader/utils.glsl';
import material_shader from './shader/material.glsl';
import dspbr_shader from './shader/dspbr.glsl';
import bvh_shader from './shader/bvh.glsl';
import lighting_shader from './shader/lighting.glsl';
import diffuse_shader from './shader/bsdfs/diffuse.glsl';
import microfacet_shader from './shader/bsdfs/microfacet.glsl';
import sheen_shader from './shader/bsdfs/sheen.glsl';
import fresnel_shader from './shader/bsdfs/fresnel.glsl';
import iridescence_shader from './shader/bsdfs/iridescence.glsl';

import render_shader from './shader/renderer.frag';
import debug_integrator_shader from './shader/integrator/debug.glsl';
import pt_integrator_shader from './shader/integrator/pt.glsl';
import misptdl_integrator_shader from './shader/integrator/misptdl.glsl';

let vertexShader = ` #version 300 es
layout(location = 0) in vec4 position;
out vec2 v_uv;

void main()
{
  v_uv = position.xy;
  gl_Position = position;
}`;

enum BufferType {
  Front = 0,
  Back = 1
}

class IBLImportanceSamplingData {
  pdf: WebGLTexture | null = null;
  cdf: WebGLTexture | null = null;
  yPdf: WebGLTexture | null = null;
  yCdf: WebGLTexture | null = null;
  width: number = 0;
  height: number = 0;
  totalSum: number = 0;
}

export interface PathtracingRendererParameters {
  canvas?: HTMLCanvasElement;
  context?: WebGL2RenderingContext;
}

export enum RenderResMode {
  High = 0,
  Low = 1
}
export class PathtracingRenderer {
  private gl: WebGL2RenderingContext;
  private canvas: any | undefined;

  private scene?: PathtracingSceneData;
  private gpu_scene?: PathtracingSceneDataWebGL2;

  private ibl: WebGLTexture | null = null;
  private iblImportanceSamplingData: IBLImportanceSamplingData = new IBLImportanceSamplingData();

  private fbos = new Map<string, WebGLFramebuffer>();
  private renderBuffers = new Map<string, WebGLFramebuffer>();

  private quadVao: WebGLVertexArrayObject | null = null;
  private renderPrograms = new Map<string, WebGLProgram>();
  private renderProgram: WebGLProgram | null = null;
  private copyProgram: WebGLProgram | null = null;
  private fxaaProgram: WebGLProgram | null = null;
  private tonemapProgram: WebGLProgram | null = null;
  private displayProgram: WebGLProgram | null = null;
  private quadVertexBuffer: WebGLBuffer | null = null;

  private renderRes = [0, 0]; // [highres, lowres]
  private displayRes = [0, 0];

  private _frameCount = 1;
  private _isRendering = false;
  private _currentAnimationFrameId = -1;
  private _resetAccumulation = false;

  private _exposure = 1.0;
  public get exposure() {
    return this._exposure;
  }
  public set exposure(val) {
    this._exposure = val;
    this.resetAccumulation();
  }

  public debugModes = ["None", "Albedo", "Metalness", "Roughness", "Normals", "Tangents", "Bitangents", "Transparency", "UV0", "Clearcoat", "IBL PDF", "IBL CDF", "Specular", "SpecularTint", "Fresnel_Schlick", "Translucency", "Fresnel_Iridescence"];
  private _debugMode: string = "None";
  public get debugMode() {
    return this._debugMode;
  }
  public set debugMode(val) {
    this._debugMode = val;
    console.log(val);
    this.resetAccumulation();
  }

  public integratorTypes = ["PT", "MISPTDL"];
  private _integrator: string = "MISPTDL";
  public get integrator() {
    return this._integrator;
  }
  public set integrator(val) {
    this._integrator = val;
    this.resetAccumulation();
  }

  public tonemappingModes = ["None", "Reinhard", "Cineon", "AcesFilm", "Uncharted2"];
  private _tonemapping: string = "None";
  public get tonemapping() {
    return this._tonemapping;
  }
  public set tonemapping(val) {
    this._tonemapping = val;
    // this.resetAccumulation();
  }

  public sheenGModes = ["Charlie", "Ashikhmin"];
  private _sheenG: string = "Charlie";
  public get sheenG() {
    return this._sheenG;
  }
  public set sheenG(val) {
    this._sheenG = val;
    this.resetAccumulation();
  }

  private _maxBounces = 8;
  public get maxBounces() {
    return this._maxBounces;
  }
  public set maxBounces(val) {
    this._maxBounces = val;
    this.resetAccumulation();
  }

  private _useIBL = true;
  public get useIBL() {
    return this._useIBL;
  }
  public set useIBL(val) {
    this._useIBL = val;
    this.resetAccumulation();
  }

  private _showBackground = true;
  public get showBackground() {
    return this._showBackground;
  }
  public set showBackground(val) {
    this._showBackground = val;
    this.resetAccumulation();
  }

  private _forceIBLEval = false;
  public get forceIBLEval() {
    return this._forceIBLEval;
  }
  public set forceIBLEval(val) {
    this._forceIBLEval = val;
    this.resetAccumulation();
  }

  private _enableGamma = true;
  public get enableGamma() {
    return this._enableGamma;
  }
  public set enableGamma(val) {
    this._enableGamma = val;
    this.resetAccumulation();
  }

  private _iblRotation = 0.0;
  public get iblRotation() {
    return this._iblRotation / Math.PI * 180.0;
  }
  public set iblRotation(val) {
    this._iblRotation = val / 180.0 * Math.PI;
    this.resetAccumulation();
  }

  private _iblImportanceSampling = true;
  public get iblImportanceSampling() {
    return this._iblImportanceSampling;
  }
  public set iblImportanceSampling(val) {
    this._iblImportanceSampling = val;
    this.resetAccumulation();
  }

  private _pixelRatio = 1.0;
  public get pixelRatio() {
    return this._pixelRatio;
  }
  public set pixelRatio(val) {
    this._pixelRatio = val;
    this.resize(this.canvas.width, this.canvas.height);
  }

  private _backgroundColor = [0.0, 0.0, 0.0, 1.0];
  public get backgroundColor() {
    return this._backgroundColor;
  }
  public set backgroundColor(val) {
    this._backgroundColor = val;
    this.resetAccumulation();
  }

  private _rayEps = 0.0001;
  public get rayEps() {
    return this._rayEps;
  }
  public set rayEps(val) {
    this._rayEps = val;
    this.resetAccumulation();
  }

  private _enableFxaa = false;
  public get enableFxaa() {
    return this._enableFxaa;
  }
  public set enableFxaa(val) {
    this._enableFxaa = val;
  }

  private _tileRes = 1;
  public set tileRes(val: number) {
    this._tileRes = val;
  }
  public get tileRes() {
    return this._tileRes;
  }

  private _clampThreshold = 3.0;
  public set clampThreshold(val: number) {
    this._clampThreshold = val;
    this.resetAccumulation();
  }
  public get clampThreshold() {
    return this._clampThreshold;
  }


  constructor(parameters: PathtracingRendererParameters = {}) {
    this.canvas = parameters.canvas ? parameters.canvas : document.createElementNS('http://www.w3.org/1999/xhtml', 'canvas');

    const gl = parameters.context ? parameters.context : this.canvas.getContext('webgl2', { alpha: true, powerPreference: "high-performance" });
    this.gl = gl;
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear');

    this.pathtracingUniformBuffer = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.pathtracingUniformBuffer);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.pathtracingUniformBuffer);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);

    this.resize(Math.floor(this.canvas.width), Math.floor(this.canvas.height));
    this.initFullscreenQuad();
  }

  resetAccumulation() {
    this._resetAccumulation = true;
  }

  stopRendering() {
    cancelAnimationFrame(this._currentAnimationFrameId);
    this._isRendering = false;
  }

  resize(width: number, height: number) {
    this.displayRes = [width, height];
    this.renderRes[0] = Math.ceil(this.displayRes[0] * this._pixelRatio);
    this.renderRes[1] = Math.ceil(this.displayRes[1] * this._pixelRatio);

    this.initFramebuffers();
    this.resetAccumulation();
  }

  private pathtracingUniformBuffer: WebGLBuffer | null;
  private pathTracingUniforms = {
    "u_u_view_mat": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "u_background_color": [0, 0, 0, 1],
    "u_camera_pos": [0, 0, 0, 0],
    "u_inv_render_res": [0, 0],
    "u_ibl_resolution": [0, 0],
    "u_frame_count": 0,
    "u_debug_mode": 0,
    "u_sheen_G": 0,
    "u_use_ibl": 0,
    "u_ibl_rotation": 0,
    "u_background_from_ibl": 0,
    "u_max_bounces": 0,
    "u_focal_length": 0,
    "u_ibl_pdf_total_sum": 0,
    "u_ray_eps": 0,
    "u_render_mode": 0,
    "u_clamp_threshold": 0
  };

  private updatePathracingUniforms(camera: any) {
    const renderRes = this.renderRes;
    let filmHeight = Math.tan(camera.fov * 0.5 * Math.PI / 180.0) * camera.near;

    this.pathTracingUniforms["u_u_view_mat"] = camera.matrixWorld.elements;
    this.pathTracingUniforms["u_camera_pos"] = [camera.position.x, camera.position.y, camera.position.z, filmHeight];
    this.pathTracingUniforms["u_frame_count"] = this._frameCount;
    this.pathTracingUniforms["u_debug_mode"] = this.debugModes.indexOf(this._debugMode);
    this.pathTracingUniforms["u_sheen_G"] = this.sheenGModes.indexOf(this._sheenG);
    this.pathTracingUniforms["u_use_ibl"] = this.useIBL ? 1 : 0;
    this.pathTracingUniforms["u_ibl_rotation"] = this._iblRotation;
    this.pathTracingUniforms["u_background_from_ibl"] = this.showBackground ? 1 : 0;
    this.pathTracingUniforms["u_background_color"] = this.backgroundColor;
    this.pathTracingUniforms["u_max_bounces"] = this.maxBounces;
    this.pathTracingUniforms["u_ibl_resolution"] = [this.iblImportanceSamplingData.width, this.iblImportanceSamplingData.height];
    this.pathTracingUniforms["u_inv_render_res"] = [1.0 / renderRes[0], 1.0 / renderRes[1]];
    this.pathTracingUniforms["u_focal_length"] = camera.near;
    this.pathTracingUniforms["u_ray_eps"] = this.rayEps;
    this.pathTracingUniforms["u_render_mode"] = this.integratorTypes.indexOf(this._integrator);
    this.pathTracingUniforms["u_ibl_pdf_total_sum"] = this.iblImportanceSamplingData.totalSum;
    this.pathTracingUniforms["u_clamp_threshold"] = this.clampThreshold;

    const uniformValues = new Float32Array(Object.values(this.pathTracingUniforms).flat());
    this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.pathtracingUniformBuffer);
    this.gl.bufferData(this.gl.UNIFORM_BUFFER, uniformValues, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, null);

    if (this.scene && this.gpu_scene) {
      for (let i = 0; i < this.scene.materials.length; i++) {
        if (this.scene.materials[i].dirty) {
          this.gpu_scene.updateMaterial(i);
          this.scene.materials[i].dirty = false;
          this.resetAccumulation();
        }
      }
    }
  }

  private bindTextures(startSlot: number): number {
    let gl = this.gl;

    const program = this.renderProgram!;
    let slot = startSlot;
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, this.ibl);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_env_map"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, this.iblImportanceSamplingData.pdf);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_env_map_pdf"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, this.iblImportanceSamplingData.cdf);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_env_map_cdf"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, this.iblImportanceSamplingData.yPdf);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_env_map_yPdf"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, this.iblImportanceSamplingData.yCdf);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_env_map_yCdf"), slot++);

    for (let t in this.gpu_scene?.texArrayTextures) {
      gl.uniform1i(gl.getUniformLocation(program, t), slot);
      gl.activeTexture(gl.TEXTURE0 + slot++);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.gpu_scene?.texArrayTextures[t]!);
    }

    return slot;
  }


  render(camera: any, num_samples: number, tileFinishedCB: () => void, frameFinishedCB: (frameCount: number) => void, renderingFinishedCB?: () => void) {
    this._isRendering = true;
    this.resetAccumulation();
    this.renderFrame(camera, num_samples, 0, tileFinishedCB, frameFinishedCB, renderingFinishedCB);
  }

  renderFrame(camera: any, num_samples: number, tile: number,
    tileFinishedCB: () => void, frameFinishedCB: (frameCount: number) => void, renderingFinishedCB?: () => void) {
    let gl = this.gl;
    if (!this._isRendering) {
      console.log("Cancel");
      return;
    }

    if (this.debugMode != "None") {
      this.renderProgram = this.renderPrograms.get("debug_program")!;
    } else {
      if (this.iblImportanceSampling) {
        this.renderProgram = this.renderPrograms.get("misptdl_program")!;
      } else {
        this.renderProgram = this.renderPrograms.get("pt_program")!;
      }
    }

    const fbo = this.fbos.get("render")!;
    const copyFbo = this.fbos.get("render_")!;
    const renderBuffer = this.renderBuffers.get("render")!;
    const copyBuffer = this.renderBuffers.get("render_")!;
    const renderRes = this.renderRes;

    if (this._resetAccumulation) {
      this._frameCount = 1;
      tile = 0;
      this._resetAccumulation = false;
      gl.clearColor(0, 0, 0, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, copyFbo);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    let slot = 3; // first 3 slots used are blocked by bvh and mesh textures
    slot = this.bindTextures(slot);
    this.updatePathracingUniforms(camera);

    gl.bindVertexArray(this.quadVao);
    gl.viewport(0, 0, renderRes[0], renderRes[1]);

    // pathtracing render pass
    gl.enable(gl.SCISSOR_TEST);
    const tx = this._tileRes || 1;
    const ty = this._tileRes || 1;
    const tilex = tile % this._tileRes;
    const tiley = Math.floor(tile / this._tileRes);

    gl.scissor(
      Math.ceil(tilex * renderRes[0] / tx),
      Math.ceil((ty - tiley - 1) * renderRes[1] / ty),
      Math.ceil(renderRes[0] / tx),
      Math.ceil(renderRes[1] / ty)
    );

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, copyBuffer!);
    gl.uniform1i(gl.getUniformLocation(this.renderProgram!, "u_sampler2D_PreviousTexture"), slot);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);

    // copy pathtracing render buffer
    // to be used as accumulation input for next frames raytracing pass
    gl.useProgram(this.copyProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, copyFbo);
    gl.activeTexture(gl.TEXTURE0 + slot)
    gl.bindTexture(gl.TEXTURE_2D, renderBuffer!);
    gl.uniform1i(gl.getUniformLocation(this.copyProgram!, "tex"), slot);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.get("postprocess")!);
    gl.useProgram(this.tonemapProgram!);
    gl.uniform1i(gl.getUniformLocation(this.tonemapProgram!, "tex"), slot); // renderbuffer
    gl.uniform1f(gl.getUniformLocation(this.tonemapProgram!, "exposure"), this._exposure);
    gl.uniform1i(gl.getUniformLocation(this.tonemapProgram!, "gamma"), this._enableGamma ? 1.0 : 0.0);
    gl.uniform1i(gl.getUniformLocation(this.tonemapProgram!, "tonemappingMode"),
      this.tonemappingModes.indexOf(this._tonemapping));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    let displayBuffer = this.renderBuffers.get("postprocess")!;
    if (this.enableFxaa) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.get("postprocess_")!);
      gl.useProgram(this.fxaaProgram!);
      gl.bindTexture(gl.TEXTURE_2D, this.renderBuffers.get("postprocess")!);
      gl.uniform1i(gl.getUniformLocation(this.fxaaProgram!, "tex"), slot);
      gl.uniform2f(gl.getUniformLocation(this.fxaaProgram!, "u_inv_res"), 1.0 / this.renderRes[0], 1.0 / this.renderRes[1]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      displayBuffer = this.renderBuffers.get("postprocess_")!;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.displayProgram!);
    gl.viewport(0, 0, this.displayRes[0], this.displayRes[1]);
    gl.bindTexture(gl.TEXTURE_2D, displayBuffer);
    gl.uniform1i(gl.getUniformLocation(this.displayProgram!, "tex"), slot);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.useProgram(null);
    gl.bindVertexArray(null);

    tileFinishedCB();

    if (tile == (this._tileRes * this._tileRes) - 1) {
      this._frameCount++;
      frameFinishedCB(this._frameCount);
    }

    if (num_samples !== -1 && this._frameCount >= num_samples) {
      if(renderingFinishedCB)
        renderingFinishedCB(); // finished rendering num_samples
      this._isRendering = false;
    }

    this._currentAnimationFrameId = requestAnimationFrame(() => {
      this.renderFrame(camera, num_samples, ++tile % (this._tileRes * this._tileRes), tileFinishedCB, frameFinishedCB, renderingFinishedCB)
    });
  };

  private initFramebuffers() {
    const gl = this.gl;

    // cleanup renderbuffers and fbos
    for (const rb of this.renderBuffers.values()) {
      if (rb) gl.deleteTexture(rb);
    }
    this.renderBuffers.clear();

    for (const fbo of this.fbos.values()) {
      if (fbo) gl.deleteFramebuffer(fbo);
    }
    this.fbos.clear();

    function createFbo(textures: WebGLTexture[]): WebGLFramebuffer {
      let drawBuffers: any = [];
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

      for (let i = 0; i < textures.length; i++) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, textures[i], 0);
        drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
      }
      gl.drawBuffers(drawBuffers);
      return fbo!;
    }

    this.renderBuffers.set("render", glu.createRenderBufferTexture(gl, null, this.renderRes[0], this.renderRes[1])!);
    this.renderBuffers.set("render_", glu.createRenderBufferTexture(gl, null, this.renderRes[0], this.renderRes[1])!);
    this.fbos.set("render", createFbo([this.renderBuffers.get("render")!]));
    this.fbos.set("render_", createFbo([this.renderBuffers.get("render_")!]));

    this.renderBuffers.set("postprocess", glu.createRenderBufferTexture(gl, null, this.renderRes[0], this.renderRes[1])!);
    this.renderBuffers.set("postprocess_", glu.createRenderBufferTexture(gl, null, this.renderRes[0], this.renderRes[1])!);
    this.fbos.set("postprocess", createFbo([this.renderBuffers.get("postprocess")!]));
    this.fbos.set("postprocess_", createFbo([this.renderBuffers.get("postprocess_")!]));

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private async initFullscreenQuad() {
    let gl = this.gl;
    // glu.printGLInfo(gl);

    // fullscreen quad position buffer
    const positions = new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    this.quadVertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // fullscreen quad vao
    this.quadVao = gl.createVertexArray();
    gl.bindVertexArray(this.quadVao);
    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  private precompute1DPdfAndCdf(data: Float32Array, pdf: Float32Array,
    cdf: Float32Array, row: number, num: number) {

    const rowIdx = row * num;
    let sum = 0;
    for (let i = 0; i < num; i++) {
      sum += data[rowIdx + i];
    }

    if (sum == 0)
      sum = 1;

    for (let i = 0; i < num; i++) {
      pdf[rowIdx + i] = data[rowIdx + i] / sum;
    }

    cdf[rowIdx] = pdf[rowIdx];
    for (let i = 1; i < num; i++) {
      cdf[rowIdx + i] = cdf[rowIdx + i - 1] + pdf[rowIdx + i];
    }

    cdf[rowIdx + num - 1] = 1;
    return sum;
  }

  private async precomputeIBLImportanceSamplingData(texture: any) {
    console.time("Precomputing IBL importance sampling data...");
    const image = texture.image;
    const w = image.width;
    const h = image.height;

    const f = new Float32Array(w * h);
    const sumX = new Float32Array(h);

    const pdf = new Float32Array(w * h);
    const cdf = new Float32Array(w * h);
    const yPdf = new Float32Array(h);
    const yCdf = new Float32Array(h);

    const numChannels = 4;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w * numChannels; x++) {
        // relative luminance of rbg pixel value
        const rgbIdx = (x + y * w) * numChannels;
        f[x + y * w] = image.data[rgbIdx] * 0.299 + image.data[rgbIdx + 1] * 0.587 + image.data[rgbIdx + 2] * 0.114;
      }
    }

    for (let y = 0; y < h; y++) {
      sumX[y] = this.precompute1DPdfAndCdf(f, pdf, cdf, y, w);
    }

    const totalSum = this.precompute1DPdfAndCdf(sumX, yPdf, yCdf, 0, h);

    texture["pdf"] = pdf;
    texture["cdf"] = cdf;
    texture["yPdf"] = yPdf;
    texture["yCdf"] = yCdf;
    texture["totalSum"] = totalSum;

    console.timeEnd("Precomputing IBL importance sampling data...");
  }

  setIBL(texture: any) {
    let gl = this.gl;
    if (this.ibl !== undefined) {
      this.gl.deleteTexture(this.ibl);
      this.gl.deleteTexture(this.iblImportanceSamplingData.pdf);
      this.gl.deleteTexture(this.iblImportanceSamplingData.cdf);
      this.gl.deleteTexture(this.iblImportanceSamplingData.yCdf);
      this.gl.deleteTexture(this.iblImportanceSamplingData.yPdf);
    }

    this.ibl = glu.createTexture(gl, gl.TEXTURE_2D, gl.RGBA32F, texture.image.width,
      texture.image.height, gl.RGBA, gl.FLOAT, texture.image.data, 0);
    this.iblImportanceSamplingData.width = texture.image.width;
    this.iblImportanceSamplingData.height = texture.image.height;

    this.precomputeIBLImportanceSamplingData(texture);

    //Importance sampling buffers
    this.iblImportanceSamplingData.pdf = glu.createTexture(gl, gl.TEXTURE_2D, gl.R32F, texture.image.width, texture.image.height, gl.RED, gl.FLOAT, texture.pdf, 0);
    this.iblImportanceSamplingData.cdf = glu.createTexture(gl, gl.TEXTURE_2D, gl.R32F, texture.image.width, texture.image.height, gl.RED, gl.FLOAT, texture.cdf, 0);
    this.iblImportanceSamplingData.yPdf = glu.createTexture(gl, gl.TEXTURE_2D, gl.R32F, texture.image.height, 1, gl.RED, gl.FLOAT, texture.yPdf, 0);
    this.iblImportanceSamplingData.yCdf = glu.createTexture(gl, gl.TEXTURE_2D, gl.R32F, texture.image.height, 1, gl.RED, gl.FLOAT, texture.yCdf, 0);
    this.iblImportanceSamplingData.totalSum = texture.totalSum;

    this.resetAccumulation();
  }

  private _bvhDataTexture?: WebGLTexture;
  private _bvhIndexTexture?: WebGLTexture;

  private initBvh(sceneData: PathtracingSceneData) {
    const gl = this.gl;
    console.time("BvhGeneration");
    const bvh = new SimpleTriangleBVH(20);
    bvh.build(sceneData.triangleBuffer!);
    console.timeEnd("BvhGeneration");

    if (this._bvhDataTexture) {
      gl.deleteTexture(this._bvhDataTexture!);
      gl.deleteTexture(this._bvhIndexTexture!);
    }

    this._bvhDataTexture = glu.createDataTexture(gl, bvh.createAndCopyToFlattenedArray_StandardFormat());

    const maxTextureSize = glu.getMaxTextureSize(gl);
    const numIndices = bvh.m_pTriIndices.length;
    const h = Math.ceil(numIndices / maxTextureSize);
    let w = Math.min(numIndices, maxTextureSize);

    const paddedIndexBuffer = new Int32Array(w * h);
    for (let i = 0; i < bvh.m_pTriIndices.length; i++) {
      paddedIndexBuffer[i] = bvh.m_pTriIndices[i];
    }

    this._bvhIndexTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this._bvhIndexTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32I, w, h, 0, gl.RED_INTEGER, gl.INT, paddedIndexBuffer);
  }

  public async setScene(scene: PathtracingSceneData) {
    const gl = this.gl
    this.stopRendering();

    this.scene = scene;
    this.gpu_scene = new PathtracingSceneDataWebGL2(this.gl, scene);
    await this.gpu_scene.init();

    this.initBvh(scene);
    await this.initializeShaders();

    this.renderPrograms.forEach(program => {
      this.bindGPUBuffersAndTextures(program);
    })

    this.resetAccumulation();
  }

  private bindGPUBuffersAndTextures(program: WebGLProgram) {
    const gl = this.gl;

    gl.useProgram(program);

    let slot = 0;
    gl.activeTexture(gl.TEXTURE0 + slot);
    gl.bindTexture(gl.TEXTURE_2D, this._bvhDataTexture!);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_bvh"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot);
    gl.bindTexture(gl.TEXTURE_2D, this._bvhIndexTexture!);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_bvh_index"), slot++);

    gl.activeTexture(gl.TEXTURE0 + slot);
    gl.bindTexture(gl.TEXTURE_2D, this.gpu_scene!.triangleDataTexture!);
    gl.uniform1i(gl.getUniformLocation(program, "u_sampler_triangle_data"), slot++);

    let pathtracingUniformBlockIdx = gl.getUniformBlockIndex(program, "PathTracingUniformBlock");
    gl.uniformBlockBinding(program, pathtracingUniformBlockIdx, 0);

    if (this.gpu_scene!.sceneData.num_textures > 0) {
      let texInfoBlockIdx = gl.getUniformBlockIndex(program, "TextureInfoBlock");
      gl.uniformBlockBinding(program, texInfoBlockIdx, 1);
    }

    const materialUniformBuffers = this.gpu_scene?.materialUniformBuffers!;
    for (let i = 0; i < materialUniformBuffers.length; i++) {
      let matBlockIdx = gl.getUniformBlockIndex(program, `MaterialBlock${i}`);
      gl.uniformBlockBinding(program, matBlockIdx, i + 2);
    }

    gl.useProgram(null);
  }

  private async initializeShaders() {
    if (!this.gpu_scene) throw new Error("Scene not initialized");
    console.time("Pathtracing shader generation");

    const bufferAccessSnippet = `
    const uint MAX_TEXTURE_SIZE = ${glu.getMaxTextureSize(this.gl)}u;
    ivec2 getStructParameterTexCoord(uint structIdx, uint paramIdx, uint structStride) {
    return ivec2((structIdx * structStride + paramIdx) % MAX_TEXTURE_SIZE,
                (structIdx * structStride + paramIdx) / MAX_TEXTURE_SIZE);
    }
    `;

    const meshConstants = `
    const uint VERTEX_STRIDE = 5u;
    const uint TRIANGLE_STRIDE = 3u*VERTEX_STRIDE;
    const uint POSITION_OFFSET = 0u;
    const uint NORMAL_OFFSET = 1u;
    const uint UV_OFFSET = 2u;
    const uint TANGENT_OFFSET = 3u;
    const uint COLOR_OFFSET = 4u;
    const uint NUM_TRIANGLES = ${this.gpu_scene.sceneData.num_triangles}u;
    `;

    const pt_trace = `
      vec4 contribution = trace_pt(r);
    `
    const debug_trace = `
      vec4 contribution = trace_debug(r);
    `
    const misptdl_trace = `
      vec4 contribution = trace_misptdl(r);
    `

    // console.log(this.scene.materialBufferShaderChunk);
    const shaderChunks = new Map<string, string>([
      ['structs', structs_shader],
      ['rng', rng_shader],
      ['constants', shader_constants],
      ['lights', this.gpu_scene.lightShaderChunk],
      ['utils', utils_shader],
      ['material', material_shader],
      ['buffer_accessor', bufferAccessSnippet],
      ['texture_accessor', this.gpu_scene.texAccessorShaderChunk],
      ['material_block', this.gpu_scene.materialBufferShaderChunk],
      ['dspbr', dspbr_shader],
      ['bvh', bvh_shader],
      ['lighting', lighting_shader],
      ['diffuse', diffuse_shader],
      ['microfacet', microfacet_shader],
      ['sheen', sheen_shader],
      ['fresnel', fresnel_shader],
      ['iridescence', iridescence_shader],
      ['mesh_constants', meshConstants]
    ]);

    const debugShaderMap = new Map<string, string>(shaderChunks);
    debugShaderMap.set('integrator', debug_integrator_shader);
    const ptShaderMap = new Map<string, string>(shaderChunks);
    ptShaderMap.set('integrator', pt_integrator_shader);
    const misptdlShaderMap = new Map<string, string>(shaderChunks);
    misptdlShaderMap.set('integrator', misptdl_integrator_shader);

    this.copyProgram = await glu.createProgramFromSource(this.gl, vertexShader, copy_shader);
    this.displayProgram = await glu.createProgramFromSource(this.gl, vertexShader, display_shader);
    this.tonemapProgram = await glu.createProgramFromSource(this.gl, vertexShader, tonemap_shader);
    this.fxaaProgram = await glu.createProgramFromSource(this.gl, vertexShader, fxaa_shader);
    this.renderPrograms.set("pt_program", await glu.createProgramFromSource(this.gl, vertexShader, render_shader, ptShaderMap));
    this.renderPrograms.set("debug_program", await glu.createProgramFromSource(this.gl, vertexShader, render_shader, debugShaderMap));
    this.renderPrograms.set("misptdl_program", await glu.createProgramFromSource(this.gl, vertexShader, render_shader, misptdlShaderMap));

    this.renderProgram = this.renderPrograms.get("misptdl_program")!;
    // this.renderProgram = this.renderPrograms.get("pt_program")!;

    console.timeEnd("Pathtracing shader generation");
  }
}
