import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { FBXLoader } from "./loaders/FastFBXLoader.js";
import { retargetClipToCanonicalBones } from "./lib/animation-retarget";
import { repairAnimationLoop } from "./lib/animation-loop-fix";
import { repairBinaryFbxAnimationLoop } from "./lib/fbx-animation-loop-fix";
import {
  readBinaryFbx,
  writeBinaryFbx,
  type BinaryFbxDocument,
} from "./lib/binary-fbx";
import {
  analyzeFbxExportContents,
  buildFbxExportDocument,
  type FbxExportAvailability,
  type FbxExportSelection,
} from "./lib/fbx-export";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TGALoader } from "three/examples/jsm/loaders/TGALoader.js";
import { unzipSync } from "three/examples/jsm/libs/fflate.module.js";

type LoadState = "empty" | "loading" | "ready" | "error";

type BoneNode = {
  id: string;
  name: string;
  children: BoneNode[];
};

type TreeCommand = {
  expanded: boolean;
  version: number;
};

type AnimationImportPreview = {
  fileName: string;
  clips: Array<{ name: string; duration: number }>;
  selectedClipIndex: number;
  matchedBones: string[];
  unmatchedBones: string[];
  matchedRootNodes: string[];
  unmatchedRootNodes: string[];
  matchedTrackCount: number;
  error?: string;
};

type AnimationTimeline = {
  clipName: string;
  time: number;
  duration: number;
  isPlaying: boolean;
};

type DropChoice = {
  file: File;
  status: "analyzing" | "ready" | "error";
  hasModel: boolean;
  hasAnimation: boolean;
  replaceModel: boolean;
  importAnimation: boolean;
  resources?: File[];
  error?: string;
};

type LoadModelOptions = {
  importEmbeddedAnimation?: boolean;
  resources?: File[];
};

type AssetFolderChoice = {
  candidates: File[];
  selectedFiles: File[];
};

type MaterialRenderMode = "material" | "solid";

const MAX_RENDERED_MORPH_TARGETS = 256;
const FBX_RESOURCE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tga",
]);

type BrowserResourceFile = File & {
  webkitRelativePath?: string;
  resourcePath?: string;
};

function normalizeResourcePath(value: string) {
  let normalized = value;
  try {
    normalized = decodeURIComponent(normalized);
  } catch {
    // Keep the original value if an exporter wrote malformed URL escapes.
  }

  normalized = normalized
    .split(/[?#]/, 1)[0]
    .replace(/\\/g, "/")
    .replace(/^file:\/\//i, "")
    .replace(/^[a-z]:\//i, "")
    .replace(/^\.\//, "")
    .replace(/^\/+/, "");

  return normalized.toLowerCase();
}

function resourceBasename(value: string) {
  const normalized = normalizeResourcePath(value);
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function getMissingTextureFallback(url: string) {
  const name = resourceBasename(url);
  const color = /(?:normal|_n(?:\.|_|$))/i.test(name)
    ? "#8080ff"
    : /(?:specular|spec|_s(?:\.|_|$))/i.test(name)
      ? "#000000"
      : "#ffffff";
  return `data:image/svg+xml,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1" fill="${color}"/></svg>`,
  )}`;
}

function getResourceMimeType(path: string) {
  const lower = path.toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".bmp")) return "image/bmp";
  if (lower.endsWith(".tga")) return "image/x-tga";
  return "application/octet-stream";
}

async function expandResourceArchives(files: File[]) {
  const expanded: File[] = [];

  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      expanded.push(file);
      continue;
    }

    const archive = unzipSync(new Uint8Array(await file.arrayBuffer()));
    Object.entries(archive).forEach(([path, bytes]) => {
      const dot = path.lastIndexOf(".");
      if (dot < 0 || !FBX_RESOURCE_EXTENSIONS.has(path.slice(dot).toLowerCase())) return;
      const extracted = new File([bytes], resourceBasename(path), {
        type: getResourceMimeType(path),
      }) as BrowserResourceFile;
      Object.defineProperty(extracted, "resourcePath", {
        configurable: false,
        enumerable: false,
        value: path,
      });
      expanded.push(extracted);
    });
  }

  return expanded;
}

function getDirectoryParts(value: string) {
  const normalized = normalizeResourcePath(value);
  const parts = normalized.split("/").filter(Boolean);
  return parts.slice(0, -1);
}

function pathSegmentEditDistance(a: string[], b: string[]) {
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 0; i < a.length; i += 1) {
    const current = [i + 1];
    for (let j = 0; j < b.length; j += 1) {
      current.push(Math.min(
        current[j] + 1,
        previous[j + 1] + 1,
        previous[j] + Number(a[i] !== b[j]),
      ));
    }
    previous.splice(0, previous.length, ...current);
  }
  return previous[b.length];
}

function commonDirectorySuffixLength(a: string[], b: string[]) {
  let count = 0;
  while (
    count < a.length &&
    count < b.length &&
    a[a.length - 1 - count] === b[b.length - 1 - count]
  ) {
    count += 1;
  }
  return count;
}

function createBrowserResourceManager(files: File[]) {
  const manager = new THREE.LoadingManager();
  const resourceFiles = files.filter((file) => {
    const dot = file.name.lastIndexOf(".");
    return dot >= 0 && FBX_RESOURCE_EXTENSIONS.has(file.name.slice(dot).toLowerCase());
  });
  const resources = resourceFiles.map((file) => {
    const browserFile = file as BrowserResourceFile;
    const path = normalizeResourcePath(
      browserFile.resourcePath || browserFile.webkitRelativePath || file.name,
    );
    return {
      file,
      path,
      basename: resourceBasename(path),
      directories: getDirectoryParts(path),
    };
  });
  const byExactPath = new Map<string, typeof resources>();
  const byBasename = new Map<string, typeof resources>();
  const objectUrls = new Map<File, string>();
  const resolutionLog: Array<{
    request: string;
    status: "exact" | "recovered" | "ambiguous" | "missing";
    resolvedPath?: string;
  }> = [];

  const addIndex = (map: Map<string, typeof resources>, key: string, entry: typeof resources[number]) => {
    const bucket = map.get(key) ?? [];
    bucket.push(entry);
    map.set(key, bucket);
  };
  resources.forEach((entry) => {
    addIndex(byExactPath, entry.path, entry);
    addIndex(byBasename, entry.basename, entry);
  });

  const resolveResource = (url: string) => {
    const requestPath = normalizeResourcePath(url);
    const requestBasename = resourceBasename(requestPath);
    const exact = byExactPath.get(requestPath) ?? [];
    if (exact.length === 1) {
      resolutionLog.push({ request: url, status: "exact", resolvedPath: exact[0].path });
      return exact[0].file;
    }
    if (exact.length > 1) {
      resolutionLog.push({ request: url, status: "ambiguous" });
      return null;
    }

    const sameName = byBasename.get(requestBasename) ?? [];
    if (sameName.length === 0) {
      resolutionLog.push({ request: url, status: "missing" });
      return null;
    }
    if (sameName.length === 1) {
      resolutionLog.push({ request: url, status: "recovered", resolvedPath: sameName[0].path });
      return sameName[0].file;
    }

    const requestDirectories = getDirectoryParts(requestPath);
    const ranked = sameName
      .map((entry) => ({
        entry,
        commonSuffix: commonDirectorySuffixLength(requestDirectories, entry.directories),
        editDistance: pathSegmentEditDistance(requestDirectories, entry.directories),
      }))
      .sort((a, b) =>
        b.commonSuffix - a.commonSuffix ||
        a.editDistance - b.editDistance ||
        a.entry.path.localeCompare(b.entry.path),
      );
    const best = ranked[0];
    const tied = ranked.filter((candidate) =>
      candidate.commonSuffix === best.commonSuffix &&
      candidate.editDistance === best.editDistance,
    );
    if (tied.length !== 1) {
      resolutionLog.push({ request: url, status: "ambiguous" });
      return null;
    }

    resolutionLog.push({ request: url, status: "recovered", resolvedPath: best.entry.path });
    return best.entry.file;
  };

  manager.setURLModifier((url) => {
    if (/^(?:blob:|data:|https?:)/i.test(url)) return url;
    const file = resolveResource(url);
    if (!file) return getMissingTextureFallback(url);

    let objectUrl = objectUrls.get(file);
    if (!objectUrl) {
      objectUrl = URL.createObjectURL(file);
      objectUrls.set(file, objectUrl);
    }
    return objectUrl;
  });

  manager.addHandler(/\.tga$/i, new TGALoader(manager));
  manager.onError = (url) => {
    console.warn(`[FBX Viewer] Texture resource failed to load: ${url}`);
  };
  manager.onLoad = () => {
    if (resourceFiles.length > 0 || resolutionLog.length > 0) {
      const counts = resolutionLog.reduce<Record<string, number>>((result, item) => {
        result[item.status] = (result[item.status] ?? 0) + 1;
        return result;
      }, {});
      console.info("[FBX Viewer] Texture resource matching", {
        supplied: resourceFiles.length,
        counts,
        resolutions: resolutionLog,
      });
    }
  };

  return {
    manager,
    release: () => {
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
      objectUrls.clear();
    },
  };
}

function getTrackBoneName(trackName: string) {
  try {
    const parsed = THREE.PropertyBinding.parseTrackName(trackName);
    if (/^bones?$/i.test(parsed.objectName) && parsed.objectIndex) {
      return parsed.objectIndex;
    }
    return parsed.nodeName || null;
  } catch {
    return null;
  }
}

function cloneTrackForObject(
  track: THREE.KeyframeTrack,
  targetObject: THREE.Object3D,
) {
  try {
    const parsed = THREE.PropertyBinding.parseTrackName(track.name);
    if (!parsed.propertyName) return null;

    const propertyIndex = parsed.propertyIndex
      ? `[${parsed.propertyIndex}]`
      : "";
    const clonedTrack = track.clone();
    clonedTrack.name = `${targetObject.uuid}.${parsed.propertyName}${propertyIndex}`;
    return clonedTrack;
  } catch {
    return null;
  }
}

function collectRootAncestorNodeNames(object: THREE.Object3D) {
  const names = new Set<string>();

  object.traverse((child) => {
    if (!(child instanceof THREE.Bone) || child.parent instanceof THREE.Bone) {
      return;
    }

    let ancestor: THREE.Object3D | null = child.parent;
    while (ancestor) {
      if (!(ancestor instanceof THREE.Bone) && ancestor.name) {
        names.add(ancestor.name);
      }
      if (ancestor === object) break;
      ancestor = ancestor.parent;
    }
  });

  return names;
}

function collectUniqueNonBoneNodes(object: THREE.Object3D) {
  const nodes = new Map<string, THREE.Object3D>();
  const duplicateNames = new Set<string>();

  object.traverse((child) => {
    if (child instanceof THREE.Bone || !child.name) return;
    if (nodes.has(child.name)) {
      nodes.delete(child.name);
      duplicateNames.add(child.name);
    } else if (!duplicateNames.has(child.name)) {
      nodes.set(child.name, child);
    }
  });

  return nodes;
}

function isTransformTrack(track: THREE.KeyframeTrack) {
  try {
    const propertyName =
      THREE.PropertyBinding.parseTrackName(track.name).propertyName;
    return ["position", "quaternion", "rotation", "scale"].includes(
      propertyName,
    );
  } catch {
    return false;
  }
}

function collectBoneHierarchy(object: THREE.Object3D): BoneNode[] {
  return object.children.flatMap((child) => {
    const children = collectBoneHierarchy(child);

    if (child instanceof THREE.Bone) {
      return [{
        id: child.uuid,
        name: child.name || "Unnamed bone",
        children,
      }];
    }

    return children;
  });
}

function filterBoneHierarchy(nodes: BoneNode[], query: string): BoneNode[] {
  if (!query) return nodes;

  return nodes.flatMap((node) => {
    const children = filterBoneHierarchy(node.children, query);
    if (!node.name.toLowerCase().includes(query) && children.length === 0) {
      return [];
    }
    return [{ ...node, children }];
  });
}

function countBoneMatches(nodes: BoneNode[], query: string): number {
  if (!query) return nodes.length;
  return nodes.reduce(
    (count, node) =>
      count +
      Number(node.name.toLowerCase().includes(query)) +
      countBoneMatches(node.children, query),
    0,
  );
}

function BoneTreeNode({
  node,
  depth,
  command,
  forceOpen,
  selectedBoneId,
  onSelect,
}: {
  node: BoneNode;
  depth: number;
  command: TreeCommand | null;
  forceOpen: boolean;
  selectedBoneId: string | null;
  onSelect: (bone: BoneNode) => void;
}) {
  const [isOpen, setIsOpen] = useState(depth < 1);
  const nodeIsOpen = forceOpen || isOpen;

  useEffect(() => {
    if (command) setIsOpen(command.expanded);
  }, [command]);

  return (
    <li role="none">
      <div
        className={`bone-row ${selectedBoneId === node.id ? "is-selected" : ""}`}
        role="treeitem"
        aria-selected={selectedBoneId === node.id}
        aria-expanded={node.children.length ? nodeIsOpen : undefined}
      >
        {node.children.length ? (
          <button
            className={`bone-expander ${nodeIsOpen ? "is-open" : ""}`}
            aria-label={`${nodeIsOpen ? "Collapse" : "Expand"} ${node.name}`}
            disabled={forceOpen}
            onClick={() => setIsOpen((current) => !current)}
          >
            ›
          </button>
        ) : (
          <span className="bone-expander-spacer" />
        )}
        <button
          className="bone-select"
          title={node.name}
          onClick={() => onSelect(node)}
        >
          <span className="bone-icon" aria-hidden="true" />
          <span>{node.name}</span>
        </button>
      </div>
      {node.children.length > 0 && nodeIsOpen && (
        <ul role="group">
          {node.children.map((child) => (
            <BoneTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              command={command}
              forceOpen={forceOpen}
              selectedBoneId={selectedBoneId}
              onSelect={onSelect}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry?.dispose();
    const materials = Array.isArray(child.material)
      ? child.material
      : [child.material];
    materials.forEach((material) => {
      Object.values(material).forEach((value) => {
        if (value instanceof THREE.Texture) value.dispose();
      });
      material.dispose();
    });
  });
}

function getSkeletonMaterials(helper: THREE.SkeletonHelper) {
  return Array.isArray(helper.material) ? helper.material : [helper.material];
}

function disposeSkeletonHelper(helper: THREE.SkeletonHelper) {
  helper.geometry.dispose();
  getSkeletonMaterials(helper).forEach((material) => material.dispose());
}

function getErrorDetail(error: unknown) {
  if (error instanceof Error) return error.message || error.name;
  return String(error);
}

function getBoneWorldBounds(object: THREE.Object3D) {
  object.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3();
  const worldPosition = new THREE.Vector3();

  object.traverse((child) => {
    if (child instanceof THREE.Bone) {
      worldPosition.setFromMatrixPosition(child.matrixWorld);
      bounds.expandByPoint(worldPosition);
    }
  });

  return bounds;
}

function getBoundsIncludingBones(object: THREE.Object3D) {
  object.updateWorldMatrix(true, true);

  const bounds = new THREE.Box3().setFromObject(object, false);
  bounds.union(getBoneWorldBounds(object));

  if (bounds.isEmpty()) {
    bounds.expandByPoint(object.getWorldPosition(new THREE.Vector3()));
  }

  return bounds;
}

function getAnimationFrameTimes(clips: THREE.AnimationClip[]) {
  const duration = Math.max(...clips.map((clip) => clip.duration), 0);
  if (duration <= 0) return [0];

  let smallestStep = Number.POSITIVE_INFINITY;
  clips.forEach((clip) => {
    clip.tracks.forEach((track) => {
      for (let index = 1; index < track.times.length; index += 1) {
        const step = track.times[index] - track.times[index - 1];
        if (step > 1e-6) smallestStep = Math.min(smallestStep, step);
      }
    });
  });

  const inferredFrameStep = Number.isFinite(smallestStep) ? smallestStep : 1 / 30;
  const maxSamples = 900;
  const sampleCount = Math.max(
    1,
    Math.min(maxSamples, Math.ceil(duration / inferredFrameStep)),
  );
  return Array.from(
    { length: sampleCount + 1 },
    (_, index) => (duration * index) / sampleCount,
  );
}

export default function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const boneLabelRef = useRef<HTMLDivElement>(null);
  const panelResizeRef = useRef<{ startX: number; startWidth: number } | null>(
    null,
  );
  const inputRef = useRef<HTMLInputElement>(null);
  const assetFolderInputRef = useRef<HTMLInputElement>(null);
  const currentModelFileRef = useRef<File | null>(null);
  const loadModelRef = useRef(
    (_file: File, _options?: LoadModelOptions) => undefined,
  );
  const loadAnimationRef = useRef<(file: File) => void>(() => undefined);
  const selectAnimationClipRef = useRef<(index: number) => void>(
    () => undefined,
  );
  const applyAnimationRef = useRef<(clipIndex: number) => void>(
    () => undefined,
  );
  const cancelAnimationImportRef = useRef<() => void>(() => undefined);
  const selectBoneRef = useRef<(boneId: string) => void>(() => undefined);
  const frameObjectRef = useRef<() => void>(() => undefined);
  const saveFbxRef = useRef<(selection: FbxExportSelection) => void>(() => undefined);
  const fixAnimationLoopRef = useRef<() => void>(() => undefined);
  const seekAnimationRef = useRef<(time: number) => void>(() => undefined);
  const setAnimationPlayingRef = useRef<(playing: boolean) => void>(
    () => undefined,
  );
  const setBonesVisibilityRef = useRef<(visible: boolean) => void>(
    () => undefined,
  );
  const setBoneNameVisibilityRef = useRef<(visible: boolean) => void>(
    () => undefined,
  );
  const setMaterialRenderModeRef = useRef<(mode: MaterialRenderMode) => void>(
    () => undefined,
  );
  const [loadState, setLoadState] = useState<LoadState>("empty");
  const [fileName, setFileName] = useState("");
  const [exportAvailability, setExportAvailability] =
    useState<FbxExportAvailability>({ character: false, animation: false });
  const [saveDialog, setSaveDialog] = useState<FbxExportSelection | null>(null);
  const [message, setMessage] = useState("Drop an FBX file here");
  const [isDragging, setIsDragging] = useState(false);
  const [hasBones, setHasBones] = useState(false);
  const [showBones, setShowBones] = useState(false);
  const [showBoneName, setShowBoneName] = useState(false);
  const [materialRenderMode, setMaterialRenderMode] =
    useState<MaterialRenderMode>("material");
  const [boneHierarchy, setBoneHierarchy] = useState<BoneNode[]>([]);
  const [boneCount, setBoneCount] = useState(0);
  const [panelOpen, setPanelOpen] = useState(false);
  const [panelWidth, setPanelWidth] = useState(260);
  const [boneSearch, setBoneSearch] = useState("");
  const [treeCommand, setTreeCommand] = useState<TreeCommand | null>(null);
  const [displayMenuOpen, setDisplayMenuOpen] = useState(false);
  const [animationImport, setAnimationImport] =
    useState<AnimationImportPreview | null>(null);
  const [animationModelAlternative, setAnimationModelAlternative] = useState<{
    file: File;
    resources: File[];
  } | null>(null);
  const [animationTimeline, setAnimationTimeline] =
    useState<AnimationTimeline | null>(null);
  const [mappingDetailsOpen, setMappingDetailsOpen] = useState(false);
  const [selectedBoneId, setSelectedBoneId] = useState<string | null>(null);
  const [dropChoice, setDropChoice] = useState<DropChoice | null>(null);
  const [assetFolderChoice, setAssetFolderChoice] =
    useState<AssetFolderChoice | null>(null);
  const normalizedBoneSearch = boneSearch.trim().toLowerCase();
  const filteredBoneHierarchy = useMemo(
    () => filterBoneHierarchy(boneHierarchy, normalizedBoneSearch),
    [boneHierarchy, normalizedBoneSearch],
  );
  const boneMatchCount = useMemo(
    () => countBoneMatches(boneHierarchy, normalizedBoneSearch),
    [boneHierarchy, normalizedBoneSearch],
  );

  useEffect(() => {
    setMappingDetailsOpen(false);
  }, [animationImport?.fileName, animationImport?.selectedClipIndex]);

  useEffect(() => {
    if (!displayMenuOpen) return;
    const closeDisplayMenu = () => setDisplayMenuOpen(false);
    window.addEventListener("pointerdown", closeDisplayMenu);
    return () => window.removeEventListener("pointerdown", closeDisplayMenu);
  }, [displayMenuOpen]);

  useEffect(() => {
    const viewport = viewportRef.current;
    const boneLabel = boneLabelRef.current;
    if (!viewport || !boneLabel) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
    camera.position.set(4, 3, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    viewport.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x39424e, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3);
    keyLight.position.set(5, 8, 6);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const grid = new THREE.GridHelper(20, 20, 0x555b63, 0x252a30);
    scene.add(grid);

    const selectionMarker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 18, 12),
      new THREE.MeshBasicMaterial({
        color: 0xffd45c,
        depthTest: false,
        depthWrite: false,
      }),
    );
    selectionMarker.visible = false;
    selectionMarker.renderOrder = 1000;
    scene.add(selectionMarker);

    let model: THREE.Group | null = null;
    let loadedBinaryDocument: BinaryFbxDocument | null = null;
    let loadedBinaryFileName = "";
    let releaseModelResources: () => void = () => {};
    let mixer: THREE.AnimationMixer | null = null;
    let animationActions: THREE.AnimationAction[] = [];
    let animationDuration = 0;
    let animationClipName = "";
    let animationFrameBounds: THREE.Box3 | null = null;
    let lastTimelineUpdate = 0;
    let skeletonHelper: THREE.SkeletonHelper | null = null;
    let selectedBone: THREE.Bone | null = null;
    let showSelectedBoneName = false;
    const projectedBonePosition = new THREE.Vector3();
    let referenceTransforms = new Map<
      string,
      {
        position: THREE.Vector3;
        quaternion: THREE.Quaternion;
        scale: THREE.Vector3;
      }
    >();
    let pendingAnimationSource: THREE.Group | null = null;
    let pendingAnimationFileName = "";
    let pendingAnimationBinaryDocument: BinaryFbxDocument | null = null;
    let activeAnimationBinaryDocument: BinaryFbxDocument | null = null;
    let activeAnimationFileName = "";
    let externalAnimationApplied = false;
    let animationLoadVersion = 0;
    let lastFrame = performance.now();
    let frameId = 0;
    let activeMaterialRenderMode: MaterialRenderMode = "material";
    const originalMeshMaterials = new Map<
      THREE.Mesh,
      THREE.Material | THREE.Material[]
    >();
    const solidMaterial = new THREE.MeshLambertMaterial({
      color: 0xd8dde1,
    });

    saveFbxRef.current = (selection: FbxExportSelection) => {
      try {
        const animationDocument = externalAnimationApplied
          ? activeAnimationBinaryDocument
          : null;
        const exportDocument = buildFbxExportDocument(
          loadedBinaryDocument,
          animationDocument,
          selection,
        );
        const output = writeBinaryFbx(exportDocument);
        const blob = new Blob([output], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const stem = (name: string, fallback: string) => {
          const source = name || fallback;
          return source.toLowerCase().endsWith(".fbx") ? source.slice(0, -4) : source;
        };
        const characterStem = stem(loadedBinaryFileName, "character");
        const animationStem = stem(
          activeAnimationFileName || loadedBinaryFileName,
          "animation",
        );
        const downloadName = selection.character && selection.animation
          ? `${characterStem}_with_animation.fbx`
          : selection.character
            ? `${characterStem}_character.fbx`
            : `${animationStem}_animation.fbx`;
        const link = document.createElement("a");
        link.href = url;
        link.download = downloadName;
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
        setMessage(`FBX download started: ${downloadName}`);
      } catch (error) {
        const detail = getErrorDetail(error);
        console.error(`[FBX Viewer] FBX save failed: ${detail}`, error);
        setMessage(`FBX save failed: ${detail}`);
      }
    };

    const resize = () => {
      const { clientWidth, clientHeight } = viewport;
      renderer.setSize(clientWidth, clientHeight, false);
      camera.aspect = clientWidth / Math.max(clientHeight, 1);
      camera.updateProjectionMatrix();
    };

    const sampleAnimationBounds = (clips: THREE.AnimationClip[]) => {
      if (!model || !mixer || clips.length === 0) return null;

      const savedMixerTime = mixer.time;
      const savedActions = animationActions.map((action) => ({
        time: action.time,
        paused: action.paused,
        enabled: action.enabled,
      }));

      try {
        model.updateWorldMatrix(true, true);
        const staticBounds = new THREE.Box3().setFromObject(model, false);
        const initialBoneBounds = getBoneWorldBounds(model);
        const animatedBoneBounds = new THREE.Box3();

        getAnimationFrameTimes(clips).forEach((time) => {
          mixer!.setTime(time);
          model!.updateWorldMatrix(true, true);
          animatedBoneBounds.union(getBoneWorldBounds(model!));
        });

        if (animatedBoneBounds.isEmpty()) {
          return staticBounds.isEmpty() ? null : staticBounds;
        }

        const bounds = animatedBoneBounds.clone();
        if (!staticBounds.isEmpty() && !initialBoneBounds.isEmpty()) {
          const minMargin = new THREE.Vector3(
            Math.max(0, initialBoneBounds.min.x - staticBounds.min.x),
            Math.max(0, initialBoneBounds.min.y - staticBounds.min.y),
            Math.max(0, initialBoneBounds.min.z - staticBounds.min.z),
          );
          const maxMargin = new THREE.Vector3(
            Math.max(0, staticBounds.max.x - initialBoneBounds.max.x),
            Math.max(0, staticBounds.max.y - initialBoneBounds.max.y),
            Math.max(0, staticBounds.max.z - initialBoneBounds.max.z),
          );
          bounds.min.sub(minMargin);
          bounds.max.add(maxMargin);
        }

        const padding = bounds.getSize(new THREE.Vector3()).multiplyScalar(0.04);
        bounds.min.sub(padding);
        bounds.max.add(padding);
        return bounds.isEmpty() ? null : bounds;
      } finally {
        mixer.setTime(savedMixerTime);
        animationActions.forEach((action, index) => {
          const saved = savedActions[index];
          if (!saved) return;
          action.time = saved.time;
          action.paused = saved.paused;
          action.enabled = saved.enabled;
        });
        model.updateWorldMatrix(true, true);
      }
    };

    const frameObject = () => {
      if (!model) return;
      const box = animationFrameBounds?.clone() ?? getBoundsIncludingBones(model);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.length() * 0.5, 0.1);
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
      const viewDirection = camera.position.clone().sub(controls.target);
      if (viewDirection.lengthSq() < 0.000001) {
        viewDirection.set(0.7, 0.45, 1);
      }
      viewDirection.normalize();

      controls.target.copy(center);
      camera.position.copy(center).addScaledVector(viewDirection, distance);
      camera.near = Math.max(radius / 1000, 0.001);
      camera.far = Math.max(radius * 100, 1000);
      camera.updateProjectionMatrix();
      controls.update();
    };

    frameObjectRef.current = frameObject;
    const getAnimationTime = () => animationActions[0]?.time ?? 0;
    const syncAnimationTimeline = (force = false) => {
      if (!mixer || animationDuration <= 0 || animationActions.length === 0) {
        setAnimationTimeline(null);
        return;
      }

      const now = performance.now();
      if (!force && now - lastTimelineUpdate < 80) return;
      lastTimelineUpdate = now;

      setAnimationTimeline({
        clipName: animationClipName,
        time: Math.min(getAnimationTime(), animationDuration),
        duration: animationDuration,
        isPlaying: animationActions.some((action) => !action.paused),
      });
    };

    const setActiveAnimation = (
      nextMixer: THREE.AnimationMixer,
      clips: THREE.AnimationClip[],
    ) => {
      mixer = nextMixer;
      animationActions = clips.map((clip) =>
        nextMixer.clipAction(clip).reset().play(),
      );
      animationDuration = Math.max(...clips.map((clip) => clip.duration), 0);
      animationClipName =
        clips.length === 1
          ? clips[0].name || "Animation"
          : `${clips.length} animations`;
      nextMixer.update(0);
      try {
        animationFrameBounds = sampleAnimationBounds(clips);
      } catch (error) {
        animationFrameBounds = null;
        console.warn(
          `[FBX Viewer] Animation framing failed; falling back to current pose: ${getErrorDetail(error)}`,
          error,
        );
      }
      syncAnimationTimeline(true);
      frameObject();
    };

    fixAnimationLoopRef.current = () => {
      if (!model || animationActions.length === 0) return;
      const wasPlaying = animationActions.some((action) => !action.paused);
      const previousTime = getAnimationTime();
      let repairedPositionTracks = 0;
      let repairedQuaternionTracks = 0;
      const repairedClips = animationActions.map((action) => {
        const result = repairAnimationLoop(action.getClip(), (track) => {
          try {
            const parsed = THREE.PropertyBinding.parseTrackName(track.name);
            const target =
              model?.getObjectByProperty("uuid", parsed.nodeName) ??
              model?.getObjectByName(parsed.nodeName);
            return target instanceof THREE.Bone;
          } catch {
            return false;
          }
        });
        repairedPositionTracks += result.report.repairedPositionTracks;
        repairedQuaternionTracks += result.report.repairedQuaternionTracks;
        return result.clip;
      });
      if (repairedPositionTracks + repairedQuaternionTracks === 0) return;

      const binaryAnimationDocument = externalAnimationApplied
        ? activeAnimationBinaryDocument
        : loadedBinaryDocument;
      const binaryReport = binaryAnimationDocument
        ? repairBinaryFbxAnimationLoop(binaryAnimationDocument)
        : null;

      clearCurrentAnimation();
      const repairedMixer = new THREE.AnimationMixer(model);
      setActiveAnimation(repairedMixer, repairedClips);
      seekAnimationRef.current(Math.min(previousTime, animationDuration));
      setAnimationPlayingRef.current(wasPlaying);
      console.info("[FBX Viewer] Animation loop repaired", {
        repairedPositionTracks,
        repairedQuaternionTracks,
        binary: binaryReport,
      });
    };

    seekAnimationRef.current = (time: number) => {
      if (!mixer || animationDuration <= 0) return;
      const pausedStates = animationActions.map((action) => action.paused);
      animationActions.forEach((action) => {
        action.paused = false;
      });
      mixer.setTime(THREE.MathUtils.clamp(time, 0, animationDuration));
      mixer.update(0);
      animationActions.forEach((action, index) => {
        action.paused = pausedStates[index] ?? false;
      });
      syncAnimationTimeline(true);
    };
    setAnimationPlayingRef.current = (playing: boolean) => {
      animationActions.forEach((action) => {
        action.paused = !playing;
      });
      syncAnimationTimeline(true);
    };

    setBonesVisibilityRef.current = (visible: boolean) => {
      if (skeletonHelper) skeletonHelper.visible = visible;
    };
    setBoneNameVisibilityRef.current = (visible: boolean) => {
      showSelectedBoneName = visible;
      boneLabel.hidden = !visible || !selectedBone;
    };
    const restoreOriginalMaterials = (object: THREE.Object3D) => {
      object.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const originalMaterial = originalMeshMaterials.get(child);
        if (originalMaterial) child.material = originalMaterial;
      });
    };
    const applyMaterialRenderMode = () => {
      if (!model) return;
      model.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        if (!originalMeshMaterials.has(child)) {
          originalMeshMaterials.set(child, child.material);
        }
        child.material =
          activeMaterialRenderMode === "solid"
            ? solidMaterial
            : originalMeshMaterials.get(child) ?? child.material;
      });
    };
    setMaterialRenderModeRef.current = (mode: MaterialRenderMode) => {
      activeMaterialRenderMode = mode;
      applyMaterialRenderMode();
    };
    selectBoneRef.current = (boneId: string) => {
      if (!model) return;
      const object = model.getObjectByProperty("uuid", boneId);
      if (!(object instanceof THREE.Bone)) return;

      selectedBone = object;
      const position = object.getWorldPosition(new THREE.Vector3());
      const bounds = getBoundsIncludingBones(model);
      const modelRadius = Math.max(
        bounds.getSize(new THREE.Vector3()).length() * 0.5,
        0.1,
      );

      selectionMarker.visible = true;
      selectionMarker.position.copy(position);
      selectionMarker.scale.setScalar(modelRadius * 0.022);
      boneLabel.textContent = object.name || "Unnamed bone";
      boneLabel.hidden = !showSelectedBoneName;
    };

    const discardPendingAnimation = () => {
      animationLoadVersion += 1;
      if (pendingAnimationSource) {
        disposeObject(pendingAnimationSource);
        pendingAnimationSource = null;
      }
      pendingAnimationFileName = "";
      pendingAnimationBinaryDocument = null;
      setAnimationImport(null);
    };

    const clearCurrentAnimation = () => {
      if (!model) return;

      if (mixer) {
        mixer.stopAllAction();
        mixer.uncacheRoot(model);
        mixer = null;
      }
      animationActions = [];
      animationDuration = 0;
      animationClipName = "";
      animationFrameBounds = null;
      setAnimationTimeline(null);

      model.traverse((child) => {
        if (child instanceof THREE.SkinnedMesh) {
          child.skeleton.pose();
        }
      });
      model.traverse((child) => {
        const reference = referenceTransforms.get(child.uuid);
        if (!reference) return;
        child.position.copy(reference.position);
        child.quaternion.copy(reference.quaternion);
        child.scale.copy(reference.scale);
      });
      model.updateWorldMatrix(true, true);
    };

    const buildAnimationPreview = (clipIndex: number) => {
      if (!model || !pendingAnimationSource) return;

      const clips = pendingAnimationSource.animations;
      const clip = clips[clipIndex];
      if (!clip) return;

      const targetBoneNames = new Set<string>();
      model.traverse((child) => {
        if (child instanceof THREE.Bone && child.name) {
          targetBoneNames.add(child.name);
        }
      });
      const targetRootNodes = collectUniqueNonBoneNodes(model);

      const sourceBoneNames = new Set<string>();
      pendingAnimationSource.traverse((child) => {
        if (child instanceof THREE.Bone && child.name) {
          sourceBoneNames.add(child.name);
        }
      });
      const sourceRootNodeNames =
        collectRootAncestorNodeNames(pendingAnimationSource);

      const animatedBoneNames = new Set<string>();
      const animatedRootNodeNames = new Set<string>();
      let matchedTrackCount = 0;

      clip.tracks.forEach((track) => {
        const targetName = getTrackBoneName(track.name);
        if (!targetName) return;

        if (sourceBoneNames.has(targetName) && isTransformTrack(track)) {
          animatedBoneNames.add(targetName);
          if (targetBoneNames.has(targetName)) matchedTrackCount += 1;
          return;
        }

        if (
          sourceRootNodeNames.has(targetName) &&
          isTransformTrack(track)
        ) {
          animatedRootNodeNames.add(targetName);
          if (targetRootNodes.has(targetName)) matchedTrackCount += 1;
        }
      });

      const matchedBones = [...animatedBoneNames]
        .filter((name) => targetBoneNames.has(name))
        .sort();
      const unmatchedBones = [...animatedBoneNames]
        .filter((name) => !targetBoneNames.has(name))
        .sort();
      const matchedRootNodes = [...animatedRootNodeNames]
        .filter((name) => targetRootNodes.has(name))
        .sort();
      const unmatchedRootNodes = [...animatedRootNodeNames]
        .filter((name) => !targetRootNodes.has(name))
        .sort();

      setAnimationImport({
        fileName: pendingAnimationFileName,
        clips: clips.map((item, index) => ({
          name: item.name || `Animation ${index + 1}`,
          duration: item.duration,
        })),
        selectedClipIndex: clipIndex,
        matchedBones,
        unmatchedBones,
        matchedRootNodes,
        unmatchedRootNodes,
        matchedTrackCount,
        error: animatedBoneNames.size + animatedRootNodeNames.size
          ? undefined
          : "No animated bone or root-parent transform tracks were found in this FBX.",
      });
    };

    cancelAnimationImportRef.current = discardPendingAnimation;
    selectAnimationClipRef.current = buildAnimationPreview;

    loadAnimationRef.current = (file: File) => {
      if (!model || !file.name.toLowerCase().endsWith(".fbx")) return;

      discardPendingAnimation();
      const loadVersion = animationLoadVersion;
      const reader = new FileReader();
      reader.onerror = () => {
        if (loadVersion !== animationLoadVersion) return;
        setAnimationImport({
          fileName: file.name,
          clips: [],
          selectedClipIndex: 0,
          matchedBones: [],
          unmatchedBones: [],
          matchedRootNodes: [],
          unmatchedRootNodes: [],
          matchedTrackCount: 0,
          error: "Could not read this animation FBX.",
        });
      };
      reader.onload = () => {
        if (loadVersion !== animationLoadVersion) return;
        try {
          const sourceBuffer = reader.result as ArrayBuffer;
          pendingAnimationBinaryDocument = null;
          try {
            pendingAnimationBinaryDocument = readBinaryFbx(sourceBuffer);
          } catch (error) {
            console.info(
              `[FBX Viewer] Animation export unavailable for ${file.name}: ${getErrorDetail(error)}`,
            );
          }
          pendingAnimationSource = new FBXLoader().parse(sourceBuffer, "");
          pendingAnimationFileName = file.name;

          if (!pendingAnimationSource.animations.length) {
            setAnimationImport({
              fileName: file.name,
              clips: [],
              selectedClipIndex: 0,
              matchedBones: [],
              unmatchedBones: [],
              matchedRootNodes: [],
              unmatchedRootNodes: [],
              matchedTrackCount: 0,
              error: "This FBX does not contain any animation clips.",
            });
            return;
          }

          buildAnimationPreview(0);
        } catch (error) {
          console.error(
            `[FBX Viewer] Animation FBX parse failed: ${getErrorDetail(error)}`,
            error,
          );
          setAnimationImport({
            fileName: file.name,
            clips: [],
            selectedClipIndex: 0,
            matchedBones: [],
            unmatchedBones: [],
            matchedRootNodes: [],
            unmatchedRootNodes: [],
            matchedTrackCount: 0,
            error: "This animation FBX could not be parsed.",
          });
        }
      };
      reader.readAsArrayBuffer(file);
    };

    applyAnimationRef.current = (clipIndex: number) => {
      if (!model || !pendingAnimationSource) return;

      const sourceClip = pendingAnimationSource.animations[clipIndex];
      if (!sourceClip) return;

      const { clip: importedClip } = retargetClipToCanonicalBones(
        pendingAnimationSource,
        model,
        sourceClip,
      );
      if (!importedClip.tracks.length) return;

      clearCurrentAnimation();
      mixer = new THREE.AnimationMixer(model);
      setActiveAnimation(mixer, [importedClip]);
      mixer.update(0);
      activeAnimationBinaryDocument = pendingAnimationBinaryDocument;
      activeAnimationFileName = pendingAnimationFileName;
      externalAnimationApplied = true;
      const baseAvailability = analyzeFbxExportContents(loadedBinaryDocument);
      const externalAvailability = analyzeFbxExportContents(
        activeAnimationBinaryDocument,
      );
      setExportAvailability({
        character: baseAvailability.character,
        animation: externalAvailability.animation,
      });
      discardPendingAnimation();
    };

    loadModelRef.current = (file: File, options?: LoadModelOptions) => {
      const importEmbeddedAnimation = options?.importEmbeddedAnimation ?? true;
      const resources = options?.resources ?? [];

      if (!file.name.toLowerCase().endsWith(".fbx")) {
        setLoadState("error");
        setMessage("Please choose a .fbx file");
        return;
      }

      setLoadState("loading");
      setFileName(file.name);
      setExportAvailability({ character: false, animation: false });
      setSaveDialog(null);
      loadedBinaryDocument = null;
      loadedBinaryFileName = "";
      activeAnimationBinaryDocument = null;
      activeAnimationFileName = "";
      externalAnimationApplied = false;
      setMessage("Loading model…");
      setHasBones(false);
      setShowBones(false);
      setShowBoneName(false);
      setBoneHierarchy([]);
      setBoneCount(0);
      setPanelOpen(false);
      setBoneSearch("");
      setTreeCommand(null);
      setSelectedBoneId(null);
      setAnimationTimeline(null);
      selectedBone = null;
      selectionMarker.visible = false;
      showSelectedBoneName = false;
      boneLabel.hidden = true;
      discardPendingAnimation();
      if (skeletonHelper) skeletonHelper.visible = false;

      const reader = new FileReader();
      reader.onerror = () => {
        setLoadState("error");
        setMessage("Could not read this file");
      };
      reader.onload = () => {
        let loadStage = "FBX parsing";
        try {
          if (skeletonHelper) {
            scene.remove(skeletonHelper);
            disposeSkeletonHelper(skeletonHelper);
            skeletonHelper = null;
          }
          if (model) {
            restoreOriginalMaterials(model);
            scene.remove(model);
            disposeObject(model);
            originalMeshMaterials.clear();
          }
          mixer = null;
          animationActions = [];
          animationDuration = 0;
          animationClipName = "";
          animationFrameBounds = null;
          releaseModelResources();
          const localResources = createBrowserResourceManager(resources);
          releaseModelResources = localResources.release;
          const sourceBuffer = reader.result as ArrayBuffer;
          let binaryDocument: BinaryFbxDocument | null = null;
          try {
            binaryDocument = readBinaryFbx(sourceBuffer);
          } catch (error) {
            console.info(
              `[FBX Viewer] Save FBX unavailable for ${file.name}: ${getErrorDetail(error)}`,
            );
          }
          const loader = new FBXLoader(localResources.manager)
            .setIncludeMorphTargets(true)
            .setMaxMorphTargets(MAX_RENDERED_MORPH_TARGETS);
          model = loader.parse(sourceBuffer, "");
          loadedBinaryDocument = binaryDocument;
          loadedBinaryFileName = file.name;
          setExportAvailability(analyzeFbxExportContents(binaryDocument));
          loadStage = "model scene setup";
          referenceTransforms = new Map();
          model.traverse((child) => {
            referenceTransforms.set(child.uuid, {
              position: child.position.clone(),
              quaternion: child.quaternion.clone(),
              scale: child.scale.clone(),
            });
          });
          let modelHasBones = false;
          let modelHasMesh = false;
          let modelBoneCount = 0;
          model.traverse((child) => {
            if (child instanceof THREE.Bone) {
              modelHasBones = true;
              modelBoneCount += 1;
            }
            if (child instanceof THREE.Mesh) {
              modelHasMesh = true;
              originalMeshMaterials.set(child, child.material);
              child.castShadow = true;
              child.receiveShadow = true;
            }
          });
          applyMaterialRenderMode();
          scene.add(model);

          if (modelHasBones) {
            skeletonHelper = new THREE.SkeletonHelper(model);
            skeletonHelper.visible = !modelHasMesh;
            getSkeletonMaterials(skeletonHelper).forEach((material) => {
              material.depthTest = false;
              material.transparent = true;
              material.opacity = 0.92;
            });
            skeletonHelper.renderOrder = 999;
            scene.add(skeletonHelper);
            setShowBones(!modelHasMesh);
          }

          if (importEmbeddedAnimation && model.animations.length) {
            loadStage = "embedded animation setup";
            setActiveAnimation(
              new THREE.AnimationMixer(model),
              model.animations,
            );
          }

          frameObject();
          setHasBones(modelHasBones);
          setBoneHierarchy(collectBoneHierarchy(model));
          setBoneCount(modelBoneCount);
          setPanelOpen(
            modelHasBones && !window.matchMedia("(max-width: 640px)").matches,
          );
          setLoadState("ready");
          setMessage("Model ready");
        } catch (error) {
          const detail = getErrorDetail(error);
          console.error(`[FBX Viewer] ${loadStage} failed: ${detail}`, error);
          setLoadState("error");
          setMessage(`${loadStage} failed: ${detail}`);
        }
      };
      reader.readAsArrayBuffer(file);
    };

    const animate = (time: number) => {
      const delta = Math.min((time - lastFrame) / 1000, 0.1);
      lastFrame = time;
      if (mixer && animationActions.some((action) => !action.paused)) {
        mixer.update(delta);
        syncAnimationTimeline();
      }
      if (selectedBone) {
        selectedBone.getWorldPosition(selectionMarker.position);
      }
      controls.update();
      if (selectedBone && showSelectedBoneName) {
        projectedBonePosition
          .copy(selectionMarker.position)
          .project(camera);
        const isVisible =
          projectedBonePosition.z >= -1 && projectedBonePosition.z <= 1;
        boneLabel.hidden = !isVisible;
        if (isVisible) {
          boneLabel.style.left =
            `${(projectedBonePosition.x * 0.5 + 0.5) * viewport.clientWidth}px`;
          boneLabel.style.top =
            `${(-projectedBonePosition.y * 0.5 + 0.5) * viewport.clientHeight}px`;
        }
      }
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(viewport);
    resize();
    frameId = requestAnimationFrame(animate);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(frameId);
      controls.dispose();
      if (skeletonHelper) {
        disposeSkeletonHelper(skeletonHelper);
      }
      if (pendingAnimationSource) disposeObject(pendingAnimationSource);
      if (model) {
        restoreOriginalMaterials(model);
        disposeObject(model);
      }
      releaseModelResources();
      solidMaterial.dispose();
      selectionMarker.geometry.dispose();
      if (Array.isArray(selectionMarker.material)) {
        selectionMarker.material.forEach((material) => material.dispose());
      } else {
        selectionMarker.material.dispose();
      }
      saveFbxRef.current = () => undefined;
      fixAnimationLoopRef.current = () => undefined;
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  const openFile = useCallback((file?: File, options?: LoadModelOptions) => {
    if (file) loadModelRef.current(file, options);
  }, []);

  const openModelFile = useCallback((file?: File, options?: LoadModelOptions) => {
    if (!file) return;
    currentModelFileRef.current = file;
    openFile(file, options);
  }, [openFile]);


  const routeFbxFile = useCallback(async (file: File, resources: File[] = []) => {
    if (!file.name.toLowerCase().endsWith(".fbx")) {
      setDropChoice({
        file,
        status: "error",
        hasModel: false,
        hasAnimation: false,
        replaceModel: false,
        importAnimation: false,
        resources,
        error: "Please choose a .fbx file.",
      });
      return;
    }

    setDropChoice(null);
    setAnimationModelAlternative(null);
    try {
      const source = new FBXLoader().parse(await file.arrayBuffer(), "");
      let hasModel = false;
      source.traverse((child) => {
        if (child instanceof THREE.Mesh) hasModel = true;
      });
      const hasAnimation = source.animations.length > 0;
      disposeObject(source);

      if (!hasModel && !hasAnimation) {
        setDropChoice({
          file,
          status: "error",
          hasModel: false,
          hasAnimation: false,
          replaceModel: false,
          importAnimation: false,
          resources,
          error: "This FBX does not contain a model or animation clips.",
        });
        return;
      }

      const hasCurrentModel = currentModelFileRef.current !== null;
      const canImportAnimation = hasCurrentModel && hasBones;

      // Context wins over incidental FBX contents: once a rigged model is open,
      // any animation-bearing FBX is treated as an animation source first.
      // This is the safety invariant that prevents a Mixamo-style animation FBX
      // with an embedded/reference mesh from silently replacing the current model.
      if (hasAnimation && canImportAnimation) {
        setAnimationModelAlternative(hasModel ? { file, resources } : null);
        loadAnimationRef.current(file);
        return;
      }

      if (hasModel && hasAnimation && hasCurrentModel) {
        // A current model exists but cannot accept animation (for example, it has
        // no skeleton). Preserve it and require an explicit replacement decision.
        setAnimationModelAlternative(null);
        setDropChoice({
          file,
          status: "ready",
          hasModel: true,
          hasAnimation: true,
          replaceModel: false,
          importAnimation: false,
          resources,
          error: "The current model cannot accept animation from this FBX.",
        });
        return;
      }

      if (hasModel) {
        setAnimationModelAlternative(null);
        openModelFile(file, {
          resources,
          importEmbeddedAnimation: hasAnimation,
        });
        return;
      }

      setDropChoice({
        file,
        status: "ready",
        hasModel: false,
        hasAnimation: true,
        replaceModel: false,
        importAnimation: false,
        resources,
        error: "Open a rigged model first to use this animation FBX.",
      });
    } catch (error) {
      console.error(`[FBX Viewer] FBX analysis failed: ${getErrorDetail(error)}`, error);
      setDropChoice({
        file,
        status: "error",
        hasModel: false,
        hasAnimation: false,
        replaceModel: false,
        importAnimation: false,
        resources,
        error: `FBX analysis failed: ${getErrorDetail(error)}`,
      });
    }
  }, [hasBones, loadState, openModelFile]);

  const openFileSelection = useCallback(async (files?: FileList | File[]) => {
    if (!files?.length) return;
    const selected = Array.from(files);
    const fbx = selected.find((file) => file.name.toLowerCase().endsWith(".fbx"));
    if (!fbx) {
      setLoadState("error");
      setMessage("Choose an FBX file");
      return;
    }

    try {
      const resources = await expandResourceArchives(
        selected.filter((file) => file !== fbx),
      );
      await routeFbxFile(fbx, resources);
    } catch (error) {
      const detail = getErrorDetail(error);
      console.error(`[FBX Viewer] Resource archive failed: ${detail}`, error);
      setLoadState("error");
      setMessage(`Texture archive failed: ${detail}`);
    }
  }, [routeFbxFile]);

  const loadAssetFolderModel = useCallback(async (modelFile: File, selected: File[]) => {
    try {
      const resources = await expandResourceArchives(
        selected.filter((file) => !file.name.toLowerCase().endsWith(".fbx")),
      );
      await routeFbxFile(modelFile, resources);
    } catch (error) {
      const detail = getErrorDetail(error);
      console.error(`[FBX Viewer] Asset folder load failed: ${detail}`, error);
      setLoadState("error");
      setMessage(`Asset folder failed: ${detail}`);
    }
  }, [routeFbxFile]);

  const openAssetFolderSelection = useCallback((files?: FileList | File[]) => {
    if (!files?.length) return;
    const selected = Array.from(files);
    const fbxFiles = selected
      .filter((file) => file.name.toLowerCase().endsWith(".fbx"))
      .sort((a, b) => {
        const aPath = (a as BrowserResourceFile).webkitRelativePath || a.name;
        const bPath = (b as BrowserResourceFile).webkitRelativePath || b.name;
        return aPath.localeCompare(bPath);
      });

    if (fbxFiles.length === 0) {
      setLoadState("error");
      setMessage("No FBX file found in the selected asset folder");
      return;
    }

    if (fbxFiles.length === 1) {
      void loadAssetFolderModel(fbxFiles[0], selected);
      return;
    }

    setAssetFolderChoice({ candidates: fbxFiles, selectedFiles: selected });
  }, [loadAssetFolderModel]);

  const analyzeDroppedFile = useCallback((file: File) => {
    void routeFbxFile(file);
  }, [routeFbxFile]);

  const formatAnimationTime = useCallback((time: number) => {
    if (!Number.isFinite(time)) return "0:00.00";
    const minutes = Math.floor(time / 60);
    const seconds = time - minutes * 60;
    return `${minutes}:${seconds.toFixed(2).padStart(5, "0")}`;
  }, []);
  const animationProgress = animationTimeline
    ? Math.min(
        100,
        Math.max(
          0,
          (animationTimeline.time / Math.max(animationTimeline.duration, 0.001)) *
            100,
        ),
      )
    : 0;
  const seekAnimationByFrame = useCallback(
    (direction: -1 | 1) => {
      if (!animationTimeline) return;
      const frameDuration = 1 / 30;
      seekAnimationRef.current(animationTimeline.time + frameDuration * direction);
    },
    [animationTimeline],
  );
  const seekAnimationFromPointer = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!animationTimeline) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const progress = Math.min(
        1,
        Math.max(0, (event.clientX - rect.left) / Math.max(rect.width, 1)),
      );
      seekAnimationRef.current(progress * animationTimeline.duration);
    },
    [animationTimeline],
  );
  const seekAnimationFromKeyboard = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (!animationTimeline) return;

      const smallStep = animationTimeline.duration / 200;
      const largeStep = animationTimeline.duration / 20;
      const step = event.shiftKey ? largeStep : smallStep;

      if (event.key === "ArrowLeft") {
        seekAnimationRef.current(animationTimeline.time - step);
      } else if (event.key === "ArrowRight") {
        seekAnimationRef.current(animationTimeline.time + step);
      } else if (event.key === "Home") {
        seekAnimationRef.current(0);
      } else if (event.key === "End") {
        seekAnimationRef.current(animationTimeline.duration);
      } else {
        return;
      }

      event.preventDefault();
    },
    [animationTimeline],
  );

  const startPanelResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const panel = event.currentTarget.nextElementSibling;
      if (!(panel instanceof HTMLElement)) return;

      panelResizeRef.current = {
        startX: event.clientX,
        startWidth: panel.getBoundingClientRect().width,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.classList.add("is-active");
      event.preventDefault();
    },
    [],
  );

  const resizePanel = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const resize = panelResizeRef.current;
      const viewer = event.currentTarget.parentElement;
      if (!resize || !viewer) return;

      const maxWidth = Math.max(180, viewer.getBoundingClientRect().width * 0.6);
      const nextWidth = resize.startWidth + resize.startX - event.clientX;
      setPanelWidth(Math.round(Math.min(maxWidth, Math.max(180, nextWidth))));
    },
    [],
  );

  const stopPanelResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      panelResizeRef.current = null;
      event.currentTarget.classList.remove("is-active");
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [],
  );

  const matchedBoneCount = animationImport?.matchedBones.length ?? 0;
  const unmatchedBoneCount = animationImport?.unmatchedBones.length ?? 0;
  const totalAnimatedBoneCount = matchedBoneCount + unmatchedBoneCount;
  const unmatchedRootCount = animationImport?.unmatchedRootNodes.length ?? 0;
  const hasMappingIssues = unmatchedBoneCount + unmatchedRootCount > 0;
  const dropCanImportAnimation = Boolean(
    dropChoice?.hasAnimation && loadState === "ready" && hasBones,
  );
  const dropCanApplyChoice = Boolean(
    dropChoice?.status === "ready" &&
      ((dropChoice.replaceModel && dropChoice.hasModel) ||
        (dropChoice.importAnimation && dropCanImportAnimation)),
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-left">
          <a className="brand" href="./" aria-label="FBX Viewer home">
            <span className="brand-mark">F</span>
            <span>FBX Viewer</span>
          </a>
          <div className="file-actions">
            <details className="open-model-menu">
              <summary className="primary-button open-model-trigger">Open</summary>
              <div className="open-model-menu-panel">
                <button
                  type="button"
                  className="open-model-menu-item"
                  onClick={(event) => {
                    assetFolderInputRef.current?.click();
                    const details = event.currentTarget.closest("details");
                    if (details instanceof HTMLDetailsElement) details.open = false;
                  }}
                >
                  <span className="open-model-menu-title">
                    Asset Folder
                    <span className="open-model-menu-badge">Recommended</span>
                  </span>
                  <span className="open-model-menu-description">
                    FBX + textures, matched automatically
                  </span>
                </button>
                <button
                  type="button"
                  className="open-model-menu-item"
                  onClick={(event) => {
                    inputRef.current?.click();
                    const details = event.currentTarget.closest("details");
                    if (details instanceof HTMLDetailsElement) details.open = false;
                  }}
                >
                  <span className="open-model-menu-title">FBX File</span>
                  <span className="open-model-menu-description">Single FBX file</span>
                </button>
              </div>
            </details>
              <button
                type="button"
                className="secondary-button"
                disabled={!exportAvailability.character && !exportAvailability.animation}
                title={
                  exportAvailability.character || exportAvailability.animation
                    ? "Choose which FBX contents to export"
                    : "Load an exportable character or animation first"
                }
                onClick={() =>
                  setSaveDialog({
                    character: exportAvailability.character,
                    animation: exportAvailability.animation,
                  })
                }
              >
                Save FBX
              </button>
          </div>
        </div>
        <div className="viewport-display" aria-label="Viewport display controls">
          {loadState === "ready" && (
            <div
              className="display-menu"
              onPointerDown={(event) => event.stopPropagation()}
            >
              <button
                className={`display-menu-trigger ${
                  displayMenuOpen ? "is-open" : ""
                }`}
                type="button"
                aria-expanded={displayMenuOpen}
                aria-haspopup="menu"
                onClick={() => setDisplayMenuOpen((current) => !current)}
              >
                Display
                <span className="display-menu-chevron" aria-hidden="true" />
              </button>
              {displayMenuOpen && (
                <div className="display-menu-popover" role="menu">
                  <button
                    className={`display-menu-item ${
                      showBones ? "is-active" : ""
                    }`}
                    type="button"
                    role="menuitemcheckbox"
                    aria-checked={showBones}
                    disabled={!hasBones}
                    title={
                      hasBones
                        ? "Toggle skeleton overlay"
                        : "This model has no bones"
                    }
                    onClick={() => {
                      const nextValue = !showBones;
                      setShowBones(nextValue);
                      setBonesVisibilityRef.current(nextValue);
                    }}
                  >
                    <span className="display-check" aria-hidden="true" />
                    Bones
                  </button>
                  <button
                    className={`display-menu-item ${
                      showBoneName ? "is-active" : ""
                    }`}
                    type="button"
                    role="menuitemcheckbox"
                    aria-checked={showBoneName}
                    title={
                      selectedBoneId
                        ? "Show the selected bone name in the viewport"
                        : "Show selected bone names when a bone is selected"
                    }
                    onClick={() => {
                      const nextValue = !showBoneName;
                      setShowBoneName(nextValue);
                      setBoneNameVisibilityRef.current(nextValue);
                    }}
                  >
                    <span className="display-check" aria-hidden="true" />
                    Bone Names
                  </button>
                  <div className="display-menu-separator" role="separator" />
                  <div className="display-menu-label">Material</div>
                  <button
                    className={`display-menu-item ${
                      materialRenderMode === "material" ? "is-active" : ""
                    }`}
                    type="button"
                    role="menuitemradio"
                    aria-checked={materialRenderMode === "material"}
                    onClick={() => {
                      setMaterialRenderMode("material");
                      setMaterialRenderModeRef.current("material");
                    }}
                  >
                    <span className="display-radio" aria-hidden="true" />
                    Original Materials
                  </button>
                  <button
                    className={`display-menu-item ${
                      materialRenderMode === "solid" ? "is-active" : ""
                    }`}
                    type="button"
                    role="menuitemradio"
                    aria-checked={materialRenderMode === "solid"}
                    onClick={() => {
                      setMaterialRenderMode("solid");
                      setMaterialRenderModeRef.current("solid");
                    }}
                  >
                    <span className="display-radio" aria-hidden="true" />
                    Solid Color
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      <section
        className={`viewer ${isDragging ? "is-dragging" : ""} ${
          panelOpen ? "has-panel" : ""
        }`}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          const droppedFiles = Array.from(event.dataTransfer.files);
          if (droppedFiles.length === 0) return;
          if (droppedFiles.length > 1) {
            openFileSelection(droppedFiles);
            return;
          }
          analyzeDroppedFile(droppedFiles[0]);
        }}
      >
        <div className={`viewport-stage ${animationTimeline ? "has-timeline" : ""}`}> 
          <div ref={viewportRef} className="viewport" />
          <div ref={boneLabelRef} className="selected-bone-label" hidden />

          {loadState !== "ready" && (
            <button
              className="drop-card"
              type="button"
              onClick={() => inputRef.current?.click()}
            >
              <span className="upload-icon" aria-hidden="true" />
              <strong>{message}</strong>
              <span>
                {loadState === "loading"
                  ? fileName
                  : "or click to choose a file from your device"}
              </span>
            </button>
          )}

          {loadState === "ready" && (
            <>
              <div className="file-pill">
                <span className="status-dot" />
                <span>{fileName}</span>
              </div>
              {animationTimeline && (
                <div className="timeline-control" aria-label="Animation timeline">
                  <div className="timeline-transport">
                    <button
                      className="timeline-step timeline-step-back"
                      type="button"
                      aria-label="Step back one frame"
                      title="Step back one frame"
                      onClick={() => seekAnimationByFrame(-1)}
                    >
                      <span className="timeline-step-icon" aria-hidden="true" />
                    </button>
                    <button
                      className="timeline-play"
                      type="button"
                      aria-label={
                        animationTimeline.isPlaying
                          ? "Pause animation"
                          : "Play animation"
                      }
                      onClick={() =>
                        setAnimationPlayingRef.current(
                          !animationTimeline.isPlaying,
                        )
                      }
                    >
                      <span
                        className={`timeline-play-icon ${
                          animationTimeline.isPlaying ? "is-pause" : "is-play"
                        }`}
                        aria-hidden="true"
                      />
                    </button>
                    <button
                      className="timeline-step timeline-step-forward"
                      type="button"
                      aria-label="Step forward one frame"
                      title="Step forward one frame"
                      onClick={() => seekAnimationByFrame(1)}
                    >
                      <span className="timeline-step-icon" aria-hidden="true" />
                    </button>
                    <button
                      className="timeline-loop-fix"
                      type="button"
                      title="Smooth the active animation across the loop boundary"
                      onClick={() => fixAnimationLoopRef.current()}
                    >
                      Fix Loop
                    </button>
                  </div>
                  <div className="timeline-meta">
                    <span title={animationTimeline.clipName}>
                      {animationTimeline.clipName}
                    </span>
                    <span>
                      {formatAnimationTime(animationTimeline.time)} /{" "}
                      {formatAnimationTime(animationTimeline.duration)}
                    </span>
                  </div>
                  <div
                    id="animation-timeline"
                    className="timeline-scrubber"
                    role="slider"
                    aria-label="Animation time"
                    aria-valuemin={0}
                    aria-valuemax={Number(animationTimeline.duration.toFixed(3))}
                    aria-valuenow={Number(animationTimeline.time.toFixed(3))}
                    aria-valuetext={`${formatAnimationTime(
                      animationTimeline.time,
                    )} of ${formatAnimationTime(animationTimeline.duration)}`}
                    tabIndex={0}
                    onPointerDown={(event) => {
                      event.currentTarget.setPointerCapture(event.pointerId);
                      setAnimationPlayingRef.current(false);
                      seekAnimationFromPointer(event);
                    }}
                    onPointerMove={(event) => {
                      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                        seekAnimationFromPointer(event);
                      }
                    }}
                    onKeyDown={seekAnimationFromKeyboard}
                  >
                    <span className="timeline-track">
                      <span
                        className="timeline-fill"
                        style={{ width: `${animationProgress}%` }}
                      />
                    </span>
                    <span
                      className="timeline-thumb"
                      style={{ left: `${animationProgress}%` }}
                    />
                  </div>
                </div>
              )}
              <div className="help">Drag to orbit · Scroll to zoom · Right-drag to pan</div>
              <div className="viewport-tools">
                <button
                  className="frame-view"
                  type="button"
                  aria-label="Frame object"
                  title="Center and fit the object without changing the viewing angle"
                  onClick={() => frameObjectRef.current()}
                >
                  <span aria-hidden="true" />
                  Frame
                </button>
                {hasBones && !panelOpen && (
                  <button
                    className="hierarchy-reopen"
                    type="button"
                    aria-label="Open bone hierarchy"
                    title="Open bone hierarchy"
                    onClick={() => setPanelOpen(true)}
                  >
                    <span aria-hidden="true" />
                  </button>
                )}
              </div>
            </>
          )}
        </div>

        {loadState === "ready" && hasBones && panelOpen && (
          <>
            <div
              className="panel-resizer"
              role="separator"
              aria-label="Resize bone hierarchy panel"
              aria-orientation="vertical"
              aria-valuemin={180}
              aria-valuenow={panelWidth}
              tabIndex={0}
              onPointerDown={startPanelResize}
              onPointerMove={resizePanel}
              onPointerUp={stopPanelResize}
              onPointerCancel={stopPanelResize}
              onLostPointerCapture={(event) => {
                panelResizeRef.current = null;
                event.currentTarget.classList.remove("is-active");
              }}
              onKeyDown={(event) => {
                if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
                  return;
                }
                const viewerWidth =
                  event.currentTarget.parentElement?.getBoundingClientRect()
                    .width ?? window.innerWidth;
                const maxWidth = Math.max(180, viewerWidth * 0.6);
                const delta = event.key === "ArrowLeft" ? 16 : -16;
                setPanelWidth((current) =>
                  Math.round(
                    Math.min(maxWidth, Math.max(180, current + delta)),
                  ),
                );
                event.preventDefault();
              }}
            />
            <aside
              className="bone-panel"
              aria-label="Bone hierarchy"
              style={{ width: panelWidth }}
            >
            <div className="bone-panel-header">
              <div>
                <h2>Bone hierarchy</h2>
                <span>{boneCount} bones</span>
              </div>
              <button
                className="close-panel"
                aria-label="Close bone hierarchy"
                onClick={() => setPanelOpen(false)}
              >
                ×
              </button>
            </div>
            <div className="bone-panel-search">
              <span className="bone-search-icon" aria-hidden="true" />
              <input
                type="search"
                value={boneSearch}
                placeholder="Search bones"
                aria-label="Search bones by name"
                onChange={(event) => setBoneSearch(event.target.value)}
              />
              {normalizedBoneSearch && (
                <>
                  <span className="bone-search-count">
                    {boneMatchCount}
                  </span>
                  <button
                    className="bone-search-clear"
                    aria-label="Clear bone search"
                    onClick={() => setBoneSearch("")}
                  >
                    ×
                  </button>
                </>
              )}
            </div>
            <div className="bone-panel-actions">
              <button
                disabled={Boolean(normalizedBoneSearch)}
                onClick={() =>
                  setTreeCommand((current) => ({
                    expanded: true,
                    version: (current?.version ?? 0) + 1,
                  }))
                }
              >
                Expand all
              </button>
              <button
                disabled={Boolean(normalizedBoneSearch)}
                onClick={() =>
                  setTreeCommand((current) => ({
                    expanded: false,
                    version: (current?.version ?? 0) + 1,
                  }))
                }
              >
                Collapse all
              </button>
            </div>
            <div className="bone-tree">
              {filteredBoneHierarchy.length > 0 ? (
                <ul role="tree">
                  {filteredBoneHierarchy.map((bone) => (
                    <BoneTreeNode
                      key={bone.id}
                      node={bone}
                      depth={0}
                      command={treeCommand}
                      forceOpen={Boolean(normalizedBoneSearch)}
                      selectedBoneId={selectedBoneId}
                      onSelect={(bone) => {
                        setSelectedBoneId(bone.id);
                        selectBoneRef.current(bone.id);
                      }}
                    />
                  ))}
                </ul>
              ) : (
                <div className="bone-search-empty">No matching bones</div>
              )}
            </div>
            </aside>
          </>
        )}

        {assetFolderChoice && (
          <div className="asset-model-picker-backdrop">
            <section
              className="asset-model-picker"
              role="dialog"
              aria-modal="true"
              aria-labelledby="asset-model-picker-title"
            >
              <div className="asset-model-picker-header">
                <div>
                  <h2 id="asset-model-picker-title">Choose FBX</h2>
                  <p>{assetFolderChoice.candidates.length} FBX files found in this folder</p>
                </div>
                <button
                  type="button"
                  className="asset-model-picker-close"
                  aria-label="Cancel model selection"
                  onClick={() => setAssetFolderChoice(null)}
                >
                  ×
                </button>
              </div>
              <div className="asset-model-picker-list">
                {assetFolderChoice.candidates.map((file, index) => {
                  const relativePath =
                    (file as BrowserResourceFile).webkitRelativePath || file.name;
                  return (
                    <button
                      key={`${relativePath}-${index}`}
                      type="button"
                      className="asset-model-picker-item"
                      onClick={() => {
                        const selectedFiles = assetFolderChoice.selectedFiles;
                        setAssetFolderChoice(null);
                        void loadAssetFolderModel(file, selectedFiles);
                      }}
                    >
                      <span className="asset-model-picker-name">{file.name}</span>
                      <span className="asset-model-picker-path">{relativePath}</span>
                    </button>
                  );
                })}
              </div>
            </section>
          </div>
        )}

        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept=".fbx"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void routeFbxFile(file);
            event.currentTarget.value = "";
          }}
        />
        <input
          ref={assetFolderInputRef}
          className="visually-hidden"
          type="file"
          multiple
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(event) => {
            openAssetFolderSelection(event.target.files ?? undefined);
            event.currentTarget.value = "";
          }}
        />
      </section>

      {saveDialog && (
        <div className="import-backdrop">
          <section
            className="drop-choice-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="save-fbx-title"
          >
            <div className="drop-choice-header">
              <div>
                <h2 id="save-fbx-title">Save FBX</h2>
                <p>Choose the contents to include.</p>
              </div>
              <button
                className="close-panel"
                aria-label="Cancel FBX save"
                onClick={() => setSaveDialog(null)}
              >
                ×
              </button>
            </div>
            <div className="drop-choice-options">
              <label className="drop-choice-option">
                <input
                  type="checkbox"
                  checked={saveDialog.character}
                  disabled={!exportAvailability.character}
                  onChange={(event) =>
                    setSaveDialog((current) =>
                      current ? { ...current, character: event.target.checked } : current,
                    )
                  }
                />
                <span className="drop-choice-option-copy">
                  <span>Character</span>
                  {!exportAvailability.character && <small>No character loaded</small>}
                </span>
              </label>
              <label className="drop-choice-option">
                <input
                  type="checkbox"
                  checked={saveDialog.animation}
                  disabled={!exportAvailability.animation}
                  onChange={(event) =>
                    setSaveDialog((current) =>
                      current ? { ...current, animation: event.target.checked } : current,
                    )
                  }
                />
                <span className="drop-choice-option-copy">
                  <span>Animation</span>
                  {!exportAvailability.animation && <small>No animation loaded</small>}
                </span>
              </label>
            </div>
            <div className="drop-choice-actions">
              <button
                className="primary-button"
                disabled={!saveDialog.character && !saveDialog.animation}
                onClick={() => {
                  saveFbxRef.current(saveDialog);
                  setSaveDialog(null);
                }}
              >
                Save
              </button>
              <button
                className="secondary-button"
                onClick={() => setSaveDialog(null)}
              >
                Cancel
              </button>
            </div>
          </section>
        </div>
      )}

      {dropChoice && (
        <div className="import-backdrop">
          <section
            className="drop-choice-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="drop-choice-title"
            aria-describedby="drop-choice-description"
          >
            <div className="drop-choice-header">
              <div>
                <h2 id="drop-choice-title">Choose how to use this FBX</h2>
                <p>{dropChoice.file.name}</p>
              </div>
              <button
                className="close-panel"
                aria-label="Cancel"
                onClick={() => setDropChoice(null)}
              >
                ×
              </button>
            </div>
            {dropChoice.status === "analyzing" ? (
              <p id="drop-choice-description" className="drop-choice-description">
                Inspecting FBX contents...
              </p>
            ) : (
              <div
                id="drop-choice-description"
                className="drop-choice-options"
              >
                <label className="drop-choice-option">
                  <input
                    type="radio"
                    name="fbx-use"
                    checked={dropChoice.replaceModel}
                    disabled={!dropChoice.hasModel}
                    onChange={() =>
                      setDropChoice((current) =>
                        current
                          ? { ...current, replaceModel: true, importAnimation: false }
                          : current,
                      )
                    }
                  />
                  <span className="drop-choice-option-copy">
                    <span>Replace current model</span>
                    {!dropChoice.hasModel && (
                      <small>No model mesh found</small>
                    )}
                  </span>
                </label>
                <label className="drop-choice-option">
                  <input
                    type="radio"
                    name="fbx-use"
                    checked={dropChoice.importAnimation}
                    disabled={!dropCanImportAnimation}
                    onChange={() =>
                      setDropChoice((current) =>
                        current
                          ? { ...current, replaceModel: false, importAnimation: true }
                          : current,
                      )
                    }
                  />
                  <span className="drop-choice-option-copy">
                    <span>Import animations</span>
                    {dropChoice.hasAnimation ? (
                      !dropCanImportAnimation && (
                        <small>Open a rigged model first</small>
                      )
                    ) : (
                      <small>No animation clips found</small>
                    )}
                  </span>
                </label>
              </div>
            )}
            {dropChoice.error ? (
              <p className="drop-choice-hint">{dropChoice.error}</p>
            ) : null}
            <div className="drop-choice-actions">
              <button
                className="primary-button"
                disabled={!dropCanApplyChoice}
                onClick={() => {
                  if (!dropChoice) return;
                  const { file, replaceModel, importAnimation, resources } = dropChoice;
                  setDropChoice(null);
                  if (replaceModel) {
                    openModelFile(file, {
                      resources,
                      importEmbeddedAnimation: true,
                    });
                    return;
                  }
                  if (importAnimation) loadAnimationRef.current(file);
                }}
              >
                Continue
              </button>
              <button
                className="secondary-button"
                onClick={() => setDropChoice(null)}
              >
                Cancel
              </button>
            </div>
          </section>
        </div>
      )}

      {animationImport && (
        <div className="import-backdrop">
          <section
            className="import-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="import-dialog-title"
          >
            <div className="import-dialog-header">
              <div>
                <h2 id="import-dialog-title">Import FBX Animation</h2>
                <p>{animationImport.fileName}</p>
              </div>
              <button
                className="close-panel"
                aria-label="Cancel animation import"
                onClick={() => {
                  setAnimationModelAlternative(null);
                  cancelAnimationImportRef.current();
                }}
              >
                ×
              </button>
            </div>

            <div className="import-dialog-body">
              {animationImport.clips.length > 0 && (
                <label className="clip-field">
                  <span>Animation clip</span>
                  <select
                    value={animationImport.selectedClipIndex}
                    onChange={(event) =>
                      selectAnimationClipRef.current(Number(event.target.value))
                    }
                  >
                    {animationImport.clips.map((clip, index) => (
                      <option key={`${clip.name}-${index}`} value={index}>
                        {clip.name} · {clip.duration.toFixed(2)}s
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {animationImport.error ? (
                <div className="import-error">{animationImport.error}</div>
              ) : (
                <>
                  <div
                    className={`import-summary ${
                      hasMappingIssues ? "is-warning" : "is-ready"
                    }`}
                  >
                    <span className="import-summary-icon" aria-hidden="true">
                      {hasMappingIssues ? "!" : "✓"}
                    </span>
                    <div>
                      <strong>
                        {totalAnimatedBoneCount > 0
                          ? unmatchedBoneCount > 0
                            ? `${matchedBoneCount} of ${totalAnimatedBoneCount} bones matched`
                            : `All ${totalAnimatedBoneCount} bones matched`
                          : hasMappingIssues
                            ? "Animation has mapping issues"
                            : "Animation is ready to import"}
                      </strong>
                      {unmatchedBoneCount > 0 && (
                        <span>
                          {unmatchedBoneCount} unmatched{" "}
                          {unmatchedBoneCount === 1 ? "bone will" : "bones will"}{" "}
                          remain in reference pose.
                        </span>
                      )}
                      {unmatchedRootCount > 0 && (
                        <span>
                          {unmatchedRootCount} unmatched root{" "}
                          {unmatchedRootCount === 1
                            ? "transform will"
                            : "transforms will"}{" "}
                          be ignored.
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="import-rule-compact">
                    <span>Bones are matched by exact name.</span>
                    <span
                      className="import-info"
                      aria-describedby="animation-matching-tooltip"
                      tabIndex={0}
                    >
                      ⓘ
                      <span
                        id="animation-matching-tooltip"
                        className="import-tooltip"
                        role="tooltip"
                      >
                        Root-parent transforms are imported only when a unique,
                        identically named target node exists. Unmatched bones
                        remain in reference pose.
                      </span>
                    </span>
                  </div>

                  {hasMappingIssues && (
                    <div className="mapping-issues">
                      {animationImport.unmatchedBones.length > 0 && (
                        <section>
                          <h3>
                            Unmatched bones
                            <span>{animationImport.unmatchedBones.length}</span>
                          </h3>
                          <p>These bones will stay in reference pose.</p>
                          <ul>
                            {animationImport.unmatchedBones.map((name) => (
                              <li key={name} title={name}>{name}</li>
                            ))}
                          </ul>
                        </section>
                      )}
                      {animationImport.unmatchedRootNodes.length > 0 && (
                        <section>
                          <h3>
                            Unmatched root transforms
                            <span>{animationImport.unmatchedRootNodes.length}</span>
                          </h3>
                          <ul>
                            {animationImport.unmatchedRootNodes.map((name) => (
                              <li key={name} title={name}>{name}</li>
                            ))}
                          </ul>
                        </section>
                      )}
                    </div>
                  )}

                  <button
                    className={`mapping-details-toggle ${
                      mappingDetailsOpen ? "is-open" : ""
                    }`}
                    aria-expanded={mappingDetailsOpen}
                    onClick={() => setMappingDetailsOpen((current) => !current)}
                  >
                    <span>Bone mapping details</span>
                    <span className="mapping-details-chevron" aria-hidden="true">
                      ›
                    </span>
                  </button>

                  {mappingDetailsOpen && (
                    <div className="mapping-details">
                      {animationImport.matchedBones.length > 0 && (
                        <section>
                          <h3>
                            Matched bones
                            <span>{animationImport.matchedBones.length}</span>
                          </h3>
                          <ul>
                            {animationImport.matchedBones.map((name) => (
                              <li key={name} title={name}>{name}</li>
                            ))}
                          </ul>
                        </section>
                      )}
                      {animationImport.matchedRootNodes.length > 0 && (
                        <section>
                          <h3>
                            Matched root transforms
                            <span>{animationImport.matchedRootNodes.length}</span>
                          </h3>
                          <ul>
                            {animationImport.matchedRootNodes.map((name) => (
                              <li key={name} title={name}>{name}</li>
                            ))}
                          </ul>
                        </section>
                      )}
                      {hasMappingIssues && (
                        <section>
                          <h3>
                            Unmatched
                            <span>
                              {unmatchedBoneCount + unmatchedRootCount}
                            </span>
                          </h3>
                          <p>Shown above because they require attention.</p>
                        </section>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="import-dialog-actions">
              <button
                className="dialog-cancel"
                onClick={() => cancelAnimationImportRef.current()}
              >
                Cancel
              </button>
              <button
                className="primary-button"
                disabled={
                  Boolean(animationImport.error) ||
                  animationImport.matchedTrackCount === 0
                }
                onClick={() => {
                  setAnimationModelAlternative(null);
                  applyAnimationRef.current(animationImport.selectedClipIndex);
                }}
              >
                Import &amp; Play
              </button>
              {animationModelAlternative && (
                <button
                  className="secondary-button"
                  onClick={() => {
                    const alternative = animationModelAlternative;
                    setAnimationModelAlternative(null);
                    cancelAnimationImportRef.current();
                    openModelFile(alternative.file, {
                      resources: alternative.resources,
                      importEmbeddedAnimation: true,
                    });
                  }}
                >
                  Open as Model Instead
                </button>
              )}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
