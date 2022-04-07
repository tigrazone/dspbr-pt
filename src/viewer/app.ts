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

import Stats from 'three/examples/jsm/libs/stats.module.js';
import { GUI } from 'dat.GUI';
import { SimpleDropzone } from 'simple-dropzone';
import { ThreeRenderer } from './three_renderer';
import { PathtracingRenderer, Loader } from '../lib/index';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import * as Assets from './assets/asset_index';

if (!window.File || !window.FileReader || !window.FileList || !window.Blob) {
  alert('The File APIs are not fully supported in this browser.');
}

class App {
  _gui: any;
  _stats: any | null;
  canvas: HTMLCanvasElement;
  canvas_three: HTMLCanvasElement;
  canvas_pt: HTMLCanvasElement;
  spinner: Element;
  container: HTMLElement | null;
  startpage: HTMLElement | null;
  status: HTMLElement | null;
  loadscreen: HTMLElement | null;
  scene: string;
  ibl: string;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;

  renderer: any;
  three_renderer: ThreeRenderer;

  useControls: true;
  pathtracing = true;
  autoScaleScene = false;
  autoRotate = false;
  interactionScale = 0.2;

  sceneBoundingBox: THREE.Box3;

  constructor() {
    this.scene = Assets.getScene(0).name;
    this.ibl = Assets.getIBL(1).name;

    this.container = document.createElement('div');
    document.body.appendChild(this.container);
    this.canvas = document.createElement('canvas');
    this.container.appendChild(this.canvas);
    this.startpage = document.getElementById("startpage");
    this.loadscreen = document.getElementById("loadscreen");
    this.status = document.getElementById("status");
    this.spinner = document.getElementsByClassName('spinner')[0];

    this.canvas_pt = document.createElement('canvas');
    this.canvas_three = document.createElement('canvas');

    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.canvas_pt.width = window.innerWidth;
    this.canvas_pt.height = window.innerHeight;
    this.canvas_three.width = window.innerWidth;
    this.canvas_three.height = window.innerHeight;

    let aspect = window.innerWidth / window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 1000);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.screenSpacePanning = true;

    this.controls.addEventListener('change', () => {
      this.camera.updateMatrixWorld();
      this.renderer.resetAccumulation();
    });

    this.controls.addEventListener('start', () => {
      this.renderer.interruptFrame();
      this["pixelRatio"] = this.renderer.pixelRatio;
      this.renderer.pixelRatio = this.interactionScale;
      this.startPathtracing();
    });

    this.controls.addEventListener('end', () => {
      this.renderer.pixelRatio = this["pixelRatio"];
    });

    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY
    }

    this.renderer = new PathtracingRenderer({ canvas: this.canvas_pt });
    this.three_renderer = new ThreeRenderer({ canvas: this.canvas_three, powerPreference: "high-performance", alpha: true });

    this.renderer.pixelRatio = 0.5;
    this.renderer.maxBounces = 8;
    // this.renderer.iblRotation = 180.0;
    // this.renderer.exposure = Assets.getIBL(0).intensity || 1.4;

    window.addEventListener('resize', () => {
      this.resize();
    }, false);

    const input = document.createElement('input');
    const dropCtrlOverlay = new SimpleDropzone(this.startpage, input);
    dropCtrlOverlay.on('dropstart', () => {
      this.showLoadscreen();
      this.hideStartpage();
      GUI.toggleHide();
    });
    dropCtrlOverlay.on('drop', ({ files }) => this.load(files));


    const dropCtrlCanvas = new SimpleDropzone(this.canvas, input);
    dropCtrlCanvas.on('drop', ({ files }) => {
      this.showLoadscreen();
      this.load(files);
    });
    // dropCtrl.on('droperror', () => this.hideSpinner());
    this.container.addEventListener('dragover', function (e) {
      e.stopPropagation();
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });

    this.initUI();
    this.initStats();

    // this.hideStartpage();
    // this.loadScene("/assets/scenes/metal-roughness-0.05.gltf");
  }

  private load(fileMap) {
    const files: [string, File][] = Array.from(fileMap)
    if (files.length == 1 && files[0][1].name.match(/\.hdr$/)) {
      this.status.innerHTML = "Loading HDR...";
      // const url = URL.createObjectURL(e.dataTransfer.getData('text/html'));
      Loader.loadIBL(URL.createObjectURL(files[0][1])).then((ibl) => {
        this.renderer.setIBL(ibl);
        this.three_renderer.setIBL(ibl);
        const iblNode = document.getElementById("ibl-info");
        iblNode.innerHTML = '';
        this.hideLoadscreen();
      });
    } else {
      this.status.innerHTML = "Loading Scene...";
      const scenePromise = Loader.loadSceneFromBlobs(files, this.autoScaleScene);
      const iblPromise = Loader.loadIBL(Assets.getIBLByName(this.ibl).url);

      Promise.all([scenePromise, iblPromise]).then(([gltf, ibl]) => {
        this.sceneBoundingBox = new THREE.Box3().setFromObject(gltf.scene);
        this.updateCameraFromBoundingBox();
        this.renderer.setIBL(ibl);
        let that = this;
        this.renderer.setScene(gltf.scene, gltf).then(() => {
          this.startPathtracing();
          document.getElementById("scene-info").innerHTML = '';
        });

        this.three_renderer.setScene(new THREE.Scene().add(gltf.scene));
        this.three_renderer.setIBL(ibl);
        this.centerView();

        this.hideLoadscreen();
      });
    }
  }

  private initStats() {
    this._stats = new (Stats as any)();
    this._stats.domElement.style.position = 'absolute';
    this._stats.domElement.style.top = '0px';
    this._stats.domElement.style.cursor = "default";
    this._stats.domElement.style.webkitUserSelect = "none";
    this._stats.domElement.style.MozUserSelect = "none";
    this._stats.domElement.style.zIndex = 1;
    this.container.appendChild(this._stats.domElement);
  }

  private startRasterizer() {
    this.stopPathtracing();
    this.three_renderer.render(this.camera, () => {
      var destCtx = this.canvas.getContext("2d");
      destCtx.drawImage(this.canvas_three, 0, 0);
    });
  }

  private stopRasterizer() {
    this.three_renderer.stopRendering();
  }

  private stopPathtracing() {
    this.renderer.stopRendering();
  }

  private startPathtracing() {
    this.stopRasterizer();

    this.renderer.render(this.camera, -1, () => {
      this.controls.update();
      this._stats.update();
      if (this.pathtracing) {
        var destCtx = this.canvas.getContext("2d");
        destCtx.drawImage(this.canvas_pt, 0, 0);
      }
    })
  }

  private resize() {
    console.log("resizing", window.innerWidth, window.innerHeight);
    let res = [window.innerWidth, window.innerHeight];
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.canvas_pt.width = window.innerWidth;
    this.canvas_pt.height = window.innerHeight;
    this.canvas_three.width = window.innerWidth;
    this.canvas_three.height = window.innerHeight;

    this.renderer.resize(window.innerWidth, window.innerHeight);
    this.three_renderer.resize(window.innerWidth, window.innerHeight);

    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
  }

  private updateCameraFromBoundingBox() {
    this.controls.reset();
    let diag = this.sceneBoundingBox.max.distanceTo(this.sceneBoundingBox.min);
    let dist = diag * 2 / Math.tan(45.0 * Math.PI / 180.0);

    let center = new THREE.Vector3();
    this.sceneBoundingBox.getCenter(center);

    let pos = center.clone();
    pos.add(new THREE.Vector3(0, 0, dist));
    pos.add(new THREE.Vector3(0, diag, 0));

    this.camera.position.set(pos.x, pos.y, pos.z);
    this.camera.lookAt(center);
    this.camera.updateMatrixWorld();
    this.controls.update();
  }

  private loadScene(sceneUrl) {
    const scenePromise = Loader.loadSceneFromUrl(sceneUrl, this.autoScaleScene);
    const iblPromise = Loader.loadIBL(Assets.getIBLByName(this.ibl).url);

    Promise.all([scenePromise, iblPromise]).then(([gltf, ibl]) => {
      this.sceneBoundingBox = new THREE.Box3().setFromObject(gltf.scene);
      this.updateCameraFromBoundingBox();

      this.renderer.setIBL(ibl);
      this.renderer.setScene(gltf.scene, gltf).then(() => {
        if (this.pathtracing)
          this.startPathtracing();
      });

      this.three_renderer.setScene(new THREE.Scene().add(gltf.scene));
      this.three_renderer.setIBL(ibl);
      if (!this.pathtracing) {
        this.startRasterizer();
      }
    });
  }

  private centerView() {
    console.log("center view");
    if (this.controls) {
      let center = new THREE.Vector3();
      this.sceneBoundingBox.getCenter(center);
      this.controls.target = center;
      this.controls.update();
      this.renderer.resetAccumulation();
    }
  }

  initUI() {
    if (this._gui)
      return;

    this._gui = new GUI();
    GUI.toggleHide();
    this._gui.domElement.classList.add("hidden");
    this._gui.width = 300;

    let reload_obj = {
      reload: () => {
        console.log("Reload");
        this.loadScene(Assets.getSceneByName(this.scene).url);
      }
    };
    this._gui.add(reload_obj, 'reload').name('Reload');

    const center_obj = {
      centerView: this.centerView.bind(this)
    };
    this._gui.add(center_obj, 'centerView').name('Center View');

    const save_img = {
      save_img: () => {
        console.log("Save Image");
        var dataURL = this.canvas.toDataURL('image/png');
        const link = document.createElement("a");
        link.download = 'capture.png';
        link.href = dataURL;
        link.click();
      }
    };
    this._gui.add(save_img, 'save_img').name('Save PNG');

    let scene = this._gui.addFolder('Scene');
    // scene.add(this, "scene", Assets.scene_names).name('Scene').onChange((value) => {
    //   const sceneInfo = Assets.getSceneByName(value);
    //   console.log(`Loading ${sceneInfo.name}`);
    //   this.loadScene(sceneInfo.url);
    //   this.setSceneInfo(sceneInfo);
    // }).setValue(Assets.getScene(0).name);

    // this._gui.add(this, 'autoScaleScene').name('Autoscale Scene');

    scene.add(this, 'autoRotate').name('Auto Rotate').onChange((value) => {
      this.controls.autoRotate = value;
      this.renderer.resetAccumulation();
    });

    let lighting = this._gui.addFolder('Lighting');
    lighting.add(this, "ibl", Assets.ibl_names).name('IBL').onChange((value) => {
      const iblInfo = Assets.getIBLByName(value);
      console.log(`Loading ${iblInfo.name}`);
      this.setIBLInfo(iblInfo);
      if (iblInfo.name == "None") {
        this.renderer.useIBL = false;
        this.three_renderer.showBackground = false;
        this.renderer.showBackground = false;
      } else {
        Loader.loadIBL(iblInfo.url).then((ibl) => {
          this.renderer.setIBL(ibl);
          this.renderer.exposure = iblInfo.intensity ?? 1.0;
          this.renderer.iblRotation = iblInfo.rotation ?? 180.0;
          this.three_renderer.setIBL(ibl);
          this.renderer.useIBL = true;
          this.three_renderer.showBackground = true;
          this.renderer.showBackground = true;
        });
      }
    }).setValue(Assets.getIBL(1).name);

    lighting.add(this.renderer, 'iblRotation').name('IBL Rotation').min(-180.0).max(180.0).step(0.1).listen();
    // lighting.add(this.renderer, 'forceIBLEval').name('Force IBL Eval');
    lighting.open();

    let interator = this._gui.addFolder('Integrator');
    interator.add(this, 'pathtracing').name('Use Pathtracing').onChange((value) => {
      if (value == false) {
        this.startRasterizer();
      } else {
        this.startPathtracing();
      }
    });

    interator.add(this.renderer, 'debugMode', this.renderer.debugModes).name('Debug Mode');
    interator.add(this.renderer, 'renderMode', this.renderer.renderModes).name('Integrator');
    interator.add(this.renderer, 'maxBounces').name('Bounce Depth').min(0).max(32).step(1);
    // interator.add(this.renderer, 'sheenG', this.renderer.sheenGModes).name('Sheen G');
    interator.add(this.renderer, 'rayEps').name('Ray Offset');
    interator.open();

    let display = this._gui.addFolder('Display');
    display.add(this.renderer, 'exposure').name('Display Exposure').min(0).max(10).step(0.01).onChange((value) => {
      this.three_renderer.exposure = value;
    }).listen();

    display.add(this.renderer, 'tonemapping', this.renderer.tonemappingModes).name('Tonemapping').onChange(val => {
      this.three_renderer.tonemapping = val;
    });
    display.add(this.renderer, 'enableGamma').name('Gamma');

    display.add(this.renderer, 'pixelRatio').name('Pixel Ratio').min(0.1).max(1.0);
    display.add(this, 'interactionScale').name('Interaction Ratio').min(0.1).max(1.0).step(0.1);
    display.open();

    let background = this._gui.addFolder('Background');
    background.add(this.renderer, 'showBackground').name('Background from IBL').onChange((value) => {
      this.three_renderer.showBackground = value;
    });

    background.color = [0, 0, 0];
    background.addColor(background, 'color').name('Background Color').onChange((value) => {
      this.renderer.backgroundColor = [value[0] / 255.0, value[1] / 255.0, value[2] / 255.0, 1.0];
      this.three_renderer.backgroundColor = [value[0] / 255.0, value[1] / 255.0, value[2] / 255.0];
    });
  }

  showStartpage() {
    this.startpage.style.visibility = "visible";
  }

  hideStartpage() {
    this.startpage.style.visibility = "hidden";
  }

  showLoadscreen() {
    this.loadscreen.style.visibility = "visible";
    this.spinner.style.visibility = "visible";
  }

  hideLoadscreen() {
    this.loadscreen.style.visibility = "hidden";
    this.spinner.style.visibility = "hidden";
  }

  setIBLInfo(ibl: any) {
    const html = `
      IBL: ${ibl.name} by ${ibl.author}
      from <a href="${ibl.source_url}"> ${ibl.source} </a>
      <a href="${ibl.license_url}">(${ibl.license})</a>
      `;
    document.getElementById("ibl-info").innerHTML = html;
  }

  setSceneInfo(scene: any) {
    const html = `
      Scene: ${scene.name} by ${scene.author}
      from <a href="${scene.source_url}"> ${scene.source} </a>
      <a href="${scene.license_url}">(${scene.license})</a>
      `;
    document.getElementById("scene-info").innerHTML = html;
  }
}

let app = new App();