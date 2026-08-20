import {
  BinaryFbxDocument,
  type FbxNode,
} from "./binary-fbx";
import { repairLoopScalarSamples } from "./animation-loop-fix";

export type BinaryFbxLoopFixReport = {
  repairedTranslationCurves: number;
  repairedRotationCurves: number;
  skippedRootCurves: number;
  skippedCurves: number;
};

type CurveTarget = {
  modelName: string;
  propertyName: "Lcl Translation" | "Lcl Rotation";
};

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

function connectionEndpoint(node: FbxNode, index: number): bigint | null {
  return bigintProperty(node, index);
}

/**
 * Repairs raw FBX AnimationCurve key values so Save FBX contains the loop fix.
 *
 * The browser preview repairs quaternion tracks in SO(3). FBX stores Lcl Rotation as
 * Euler component curves, so this export path applies the same endpoint C0/C1/C2
 * polynomial directly in native FBX curve space. Root curves are deliberately preserved.
 */
export function repairBinaryFbxAnimationLoop(
  document: BinaryFbxDocument,
): BinaryFbxLoopFixReport {
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
    const source = connectionEndpoint(connection, 1);
    const destination = connectionEndpoint(connection, 2);
    if (source == null || destination == null) return;
    const propertyName = stringProperty(connection, 3);

    if (curveNodes.has(source) && models.has(destination)) {
      if (propertyName !== "Lcl Translation" && propertyName !== "Lcl Rotation") return;
      curveNodeTargets.set(source, {
        modelName: visibleObjectName(models.get(destination)!),
        propertyName,
      });
      return;
    }

    if (curves.has(source) && curveNodes.has(destination)) {
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

    const keyValues = curve.children.find((child) => child.name === "KeyValueFloat")?.properties[0];
    if (!keyValues || (keyValues.code !== "f" && keyValues.code !== "d")) {
      report.skippedCurves += 1;
      return;
    }

    const values = keyValues.readArray().map(Number);
    if (values.length < 4) {
      report.skippedCurves += 1;
      return;
    }
    const repaired = repairLoopScalarSamples(values);
    keyValues.replaceArray(repaired, keyValues.arrayEncoding === 1);

    if (target.propertyName === "Lcl Translation") report.repairedTranslationCurves += 1;
    else report.repairedRotationCurves += 1;
  });

  return report;
}
