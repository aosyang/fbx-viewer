import {
  BinaryFbxDocument,
  type FbxNode,
} from "./binary-fbx";
import {
  repairLoopScalarSamples,
  repairLoopScalarSamplesInertial,
  type AnimationLoopFixMode,
} from "./animation-loop-fix";

export type BinaryFbxLoopFixReport = {
  repairedTranslationCurves: number;
  repairedRotationCurves: number;
  skippedRootCurves: number;
  skippedCurves: number;
};

export type BinaryFbxLoopFixOptions = {
  mode?: AnimationLoopFixMode;
  inertialHalfLife?: number;
};

type CurveTarget = {
  modelName: string;
  propertyName: "Lcl Translation" | "Lcl Rotation";
};

const FBX_TIME_TICKS_PER_SECOND = 46186158000;
const DEFAULT_INERTIAL_HALF_LIFE = 0.09;

function bigintProperty(node: FbxNode, index: number): bigint | null {
  const property = node.properties[index];
  if (!property || property.code !== "L") return null;
  return BigInt(property.value as bigint | number);
}

function stringProperty(node: FbxNode, index: number): string {
  const property = node.properties[index];
  return property?.code === "S" ? String(property.value ?? "") : "";
}

function visibleObjectName(node: FbxNode): string {
  const raw = stringProperty(node, 1);
  const nul = raw.indexOf("\u0000");
  const withoutSuffix = nul >= 0 ? raw.slice(0, nul) : raw;
  return withoutSuffix.replace(/^[^:]+::/, "");
}

function isRootModelName(name: string): boolean {
  const leaf = name.split(":").pop()?.trim().toLowerCase() ?? "";
  return leaf === "root" || leaf === "rootnode";
}

function childProperty(node: FbxNode, childName: string) {
  return node.children.find((child) => child.name === childName)?.properties[0];
}

/** Restore only animation key values, preserving unrelated edits in the working FBX document. */
export function restoreBinaryFbxAnimationCurves(
  target: BinaryFbxDocument,
  pristine: BinaryFbxDocument,
): number {
  const sourceCurves = new Map<bigint, FbxNode>();
  pristine.findNodes("AnimationCurve").forEach((curve) => {
    const id = bigintProperty(curve, 0);
    if (id != null) sourceCurves.set(id, curve);
  });

  let restored = 0;
  target.findNodes("AnimationCurve").forEach((curve) => {
    const id = bigintProperty(curve, 0);
    if (id == null) return;
    const source = sourceCurves.get(id);
    if (!source) return;
    const targetValues = childProperty(curve, "KeyValueFloat");
    const sourceValues = childProperty(source, "KeyValueFloat");
    if (!targetValues || !sourceValues) return;
    targetValues.replaceArray(sourceValues.readArray());
    restored += 1;
  });
  return restored;
}

export function repairBinaryFbxAnimationLoop(
  document: BinaryFbxDocument,
  options: BinaryFbxLoopFixOptions = {},
): BinaryFbxLoopFixReport {
  const mode = options.mode ?? "cyclic";
  const halfLife = options.inertialHalfLife ?? DEFAULT_INERTIAL_HALF_LIFE;
  const report: BinaryFbxLoopFixReport = {
    repairedTranslationCurves: 0,
    repairedRotationCurves: 0,
    skippedRootCurves: 0,
    skippedCurves: 0,
  };

  const models = new Map<bigint, FbxNode>();
  const curveNodes = new Map<bigint, FbxNode>();
  const curves = new Map<bigint, FbxNode>();
  document.findNodes("Model").forEach((node) => {
    const id = bigintProperty(node, 0);
    if (id != null) models.set(id, node);
  });
  document.findNodes("AnimationCurveNode").forEach((node) => {
    const id = bigintProperty(node, 0);
    if (id != null) curveNodes.set(id, node);
  });
  document.findNodes("AnimationCurve").forEach((node) => {
    const id = bigintProperty(node, 0);
    if (id != null) curves.set(id, node);
  });

  const curveNodeTargets = new Map<bigint, CurveTarget>();
  const curveToNode = new Map<bigint, bigint>();
  document.findNodes("C").forEach((connection) => {
    if (stringProperty(connection, 0) !== "OP") return;
    const source = bigintProperty(connection, 1);
    const destination = bigintProperty(connection, 2);
    if (source == null || destination == null) return;
    const propertyName = stringProperty(connection, 3);

    if (curveNodes.has(source) && models.has(destination)) {
      if (propertyName !== "Lcl Translation" && propertyName !== "Lcl Rotation") return;
      curveNodeTargets.set(source, {
        modelName: visibleObjectName(models.get(destination)!),
        propertyName,
      });
    } else if (curves.has(source) && curveNodes.has(destination)) {
      curveToNode.set(source, destination);
    }
  });

  curves.forEach((curve, curveId) => {
    const curveNodeId = curveToNode.get(curveId);
    const target = curveNodeId == null ? undefined : curveNodeTargets.get(curveNodeId);
    if (!target) {
      report.skippedCurves += 1;
      return;
    }
    if (isRootModelName(target.modelName)) {
      report.skippedRootCurves += 1;
      return;
    }

    const valuesProperty = childProperty(curve, "KeyValueFloat");
    if (!valuesProperty || valuesProperty.code !== "f") {
      report.skippedCurves += 1;
      return;
    }
    const values = valuesProperty.readArray().map(Number);
    if (values.length < 4) {
      report.skippedCurves += 1;
      return;
    }

    let repaired: Float32Array;
    if (mode === "inertial") {
      const keyTimeProperty = childProperty(curve, "KeyTime");
      if (!keyTimeProperty || keyTimeProperty.code !== "l") {
        report.skippedCurves += 1;
        return;
      }
      const keyTimes = keyTimeProperty.readArray().map((value) => Number(value) / FBX_TIME_TICKS_PER_SECOND);
      repaired = repairLoopScalarSamplesInertial(values, keyTimes, halfLife);
    } else {
      repaired = repairLoopScalarSamples(values);
    }

    valuesProperty.replaceArray(repaired);
    if (target.propertyName === "Lcl Translation") report.repairedTranslationCurves += 1;
    else report.repairedRotationCurves += 1;
  });

  return report;
}
