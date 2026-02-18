declare module "three" {
  const THREE: any;
  export = THREE;
}

declare module "three/examples/jsm/controls/OrbitControls.js" {
  export const OrbitControls: any;
}

declare module "three/examples/jsm/loaders/PLYLoader.js" {
  export const PLYLoader: any;
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": any;
    }
  }
}

export {};
