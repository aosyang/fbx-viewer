import {
  BinaryFbxDocument,
  BinaryFbxError,
  FbxNode,
  FbxProperty,
} from "./binary-fbx";

export type FbxExportSelection = {
  character: boolean;
  animation: boolean;
};

export type FbxExportAvailability = {
  character: boolean;
  animation: boolean;
};

const ANIMATION_OBJECT_TYPES = new Set([
  "AnimationStack",
  "AnimationLayer",
  "AnimationCurveNode",
  "AnimationCurve",
]);

function cloneValue(value: unknown): unknown {
  if (value instanceof Uint8Array) return value.slice();
  if (Array.isArray(value)) return value.slice();
  return value;
}

function cloneProperty(property: FbxProperty): FbxProperty {
  return new FbxProperty(
    property.code,
    property.raw.slice(),
    cloneValue(property.value),
    property.arrayCount,
    property.arrayEncoding,
  );
}

function cloneNode(node: FbxNode): FbxNode {
  return new FbxNode(
    node.name,
    node.properties.map(cloneProperty),
    node.children.map(cloneNode),
    node.hasChildSentinel,
  );
}

export function cloneBinaryFbxDocument(
  document: BinaryFbxDocument,
): BinaryFbxDocument {
  return new BinaryFbxDocument(
    document.version,
    document.nodes.map(cloneNode),
    document.footer.slice(),
  );
}

function topNode(document: BinaryFbxDocument, name: string): FbxNode | undefined {
  return document.nodes.find((node) => node.name === name);
}

function scalarId(property: FbxProperty | undefined): bigint | null {
  if (!property) return null;
  if (typeof property.value === "bigint") return property.value;
  if (typeof property.value === "number" && Number.isFinite(property.value)) {
    return BigInt(Math.trunc(property.value));
  }
  return null;
}

function objectId(node: FbxNode): bigint | null {
  return scalarId(node.properties[0]);
}

function objectType(node: FbxNode): string {
  return typeof node.properties[2]?.value === "string"
    ? node.properties[2].value
    : "";
}

function visibleObjectName(node: FbxNode): string {
  const raw =
    typeof node.properties[1]?.value === "string" ? node.properties[1].value : "";
  const marker = raw.indexOf("\u0000\u0001");
  const withoutClass = marker >= 0 ? raw.slice(0, marker) : raw;
  const namespace = withoutClass.lastIndexOf("::");
  return namespace >= 0 ? withoutClass.slice(namespace + 2) : withoutClass;
}

function connectionIds(node: FbxNode): [bigint, bigint] | null {
  if (node.name !== "C") return null;
  const source = scalarId(node.properties[1]);
  const target = scalarId(node.properties[2]);
  if (source === null || target === null) return null;
  return [source, target];
}

function setScalarId(property: FbxProperty | undefined, value: bigint): void {
  if (!property) throw new BinaryFbxError("FBX object is missing its ID property");
  property.replaceScalar(value);
}

function updateDefinitions(document: BinaryFbxDocument): void {
  const objects = topNode(document, "Objects");
  const definitions = topNode(document, "Definitions");
  if (!objects || !definitions) return;

  const counts = new Map<string, number>();
  for (const child of objects.children) {
    counts.set(child.name, (counts.get(child.name) ?? 0) + 1);
  }

  const objectTypes = definitions.children.filter((child) => child.name === "ObjectType");
  for (const definition of objectTypes) {
    const type =
      typeof definition.properties[0]?.value === "string"
        ? definition.properties[0].value
        : "";
    const countNode = definition.children.find((child) => child.name === "Count");
    if (countNode?.properties[0]) {
      countNode.properties[0].replaceScalar(counts.get(type) ?? 0);
    }
  }

  const definitionsCount = definitions.children.find((child) => child.name === "Count");
  if (definitionsCount?.properties[0]) {
    definitionsCount.properties[0].replaceScalar(objectTypes.length);
  }
}

function replaceTakesWithEmpty(document: BinaryFbxDocument): void {
  const takes = topNode(document, "Takes");
  if (!takes) return;
  takes.children = takes.children.filter((child) => child.name === "Current");
  const current = takes.children.find((child) => child.name === "Current");
  if (current?.properties[0]?.code === "S") current.properties[0].replaceString("");
}

function cloneTakesFrom(
  target: BinaryFbxDocument,
  source: BinaryFbxDocument,
): void {
  const sourceTakes = topNode(source, "Takes");
  if (!sourceTakes) return;
  const targetIndex = target.nodes.findIndex((node) => node.name === "Takes");
  if (targetIndex >= 0) target.nodes[targetIndex] = cloneNode(sourceTakes);
  else target.nodes.push(cloneNode(sourceTakes));
}

function mergeDefinitionTemplates(
  target: BinaryFbxDocument,
  source: BinaryFbxDocument,
  types: Set<string>,
): void {
  const targetDefinitions = topNode(target, "Definitions");
  const sourceDefinitions = topNode(source, "Definitions");
  if (!targetDefinitions || !sourceDefinitions) return;

  const existing = new Set(
    targetDefinitions.children
      .filter((child) => child.name === "ObjectType")
      .map((child) =>
        typeof child.properties[0]?.value === "string"
          ? child.properties[0].value
          : "",
      ),
  );

  for (const child of sourceDefinitions.children) {
    if (child.name !== "ObjectType") continue;
    const type =
      typeof child.properties[0]?.value === "string" ? child.properties[0].value : "";
    if (!types.has(type) || existing.has(type)) continue;
    targetDefinitions.children.push(cloneNode(child));
    existing.add(type);
  }
}

export function analyzeFbxExportContents(
  document: BinaryFbxDocument | null,
): FbxExportAvailability {
  if (!document) return { character: false, animation: false };
  const objects = topNode(document, "Objects");
  if (!objects) return { character: false, animation: false };

  const character = objects.children.some(
    (node) => node.name === "Geometry" || (node.name === "Model" && objectType(node) === "Mesh"),
  );
  const animation = objects.children.some(
    (node) => node.name === "AnimationStack" || node.name === "AnimationCurve",
  );
  return { character, animation };
}

export function createCharacterOnlyDocument(
  source: BinaryFbxDocument,
): BinaryFbxDocument {
  const document = cloneBinaryFbxDocument(source);
  const objects = topNode(document, "Objects");
  const connections = topNode(document, "Connections");
  if (!objects) throw new BinaryFbxError("FBX Objects node is missing");

  const removedIds = new Set<string>();
  objects.children = objects.children.filter((node) => {
    if (!ANIMATION_OBJECT_TYPES.has(node.name)) return true;
    const id = objectId(node);
    if (id !== null) removedIds.add(id.toString());
    return false;
  });

  if (connections) {
    connections.children = connections.children.filter((node) => {
      const ids = connectionIds(node);
      if (!ids) return true;
      return !removedIds.has(ids[0].toString()) && !removedIds.has(ids[1].toString());
    });
  }

  replaceTakesWithEmpty(document);
  updateDefinitions(document);
  return document;
}

export function createAnimationOnlyDocument(
  source: BinaryFbxDocument,
): BinaryFbxDocument {
  const document = cloneBinaryFbxDocument(source);
  const objects = topNode(document, "Objects");
  const connections = topNode(document, "Connections");
  if (!objects) throw new BinaryFbxError("FBX Objects node is missing");

  const originalConnections = connections?.children ?? [];
  const keptModelIds = new Set<string>();
  for (const node of objects.children) {
    if (node.name !== "Model" || objectType(node) === "Mesh") continue;
    const id = objectId(node);
    if (id !== null) keptModelIds.add(id.toString());
  }

  const keptNodeAttributeIds = new Set<string>();
  for (const connection of originalConnections) {
    const ids = connectionIds(connection);
    if (!ids) continue;
    if (keptModelIds.has(ids[1].toString())) keptNodeAttributeIds.add(ids[0].toString());
  }

  const keepIds = new Set<string>();
  objects.children = objects.children.filter((node) => {
    const id = objectId(node);
    const idKey = id?.toString() ?? "";
    let keep = false;
    if (node.name === "Model") keep = keptModelIds.has(idKey);
    else if (node.name === "NodeAttribute") keep = keptNodeAttributeIds.has(idKey);
    else if (ANIMATION_OBJECT_TYPES.has(node.name)) keep = true;
    if (keep && id !== null) keepIds.add(idKey);
    return keep;
  });

  if (connections) {
    connections.children = originalConnections.filter((node) => {
      const ids = connectionIds(node);
      if (!ids) return false;
      const sourceKept = ids[0] === 0n || keepIds.has(ids[0].toString());
      const targetKept = ids[1] === 0n || keepIds.has(ids[1].toString());
      return sourceKept && targetKept;
    });
  }

  updateDefinitions(document);
  return document;
}

export function mergeCharacterAndAnimationDocuments(
  characterSource: BinaryFbxDocument,
  animationSource: BinaryFbxDocument,
): BinaryFbxDocument {
  const target = createCharacterOnlyDocument(characterSource);
  const source = createAnimationOnlyDocument(animationSource);
  const targetObjects = topNode(target, "Objects");
  const sourceObjects = topNode(source, "Objects");
  const targetConnections = topNode(target, "Connections");
  const sourceConnections = topNode(source, "Connections");
  if (!targetObjects || !sourceObjects || !targetConnections || !sourceConnections) {
    throw new BinaryFbxError("FBX Objects/Connections nodes are required for animation merge");
  }

  const targetModelIds = new Map<string, bigint>();
  const duplicateTargetModelNames = new Set<string>();
  for (const node of targetObjects.children) {
    if (node.name !== "Model" || objectType(node) === "Mesh") continue;
    const id = objectId(node);
    const name = visibleObjectName(node);
    if (id === null || !name) continue;
    if (targetModelIds.has(name)) duplicateTargetModelNames.add(name);
    else targetModelIds.set(name, id);
  }
  for (const name of duplicateTargetModelNames) targetModelIds.delete(name);

  const sourceModelsById = new Map<string, string>();
  for (const node of sourceObjects.children) {
    if (node.name !== "Model") continue;
    const id = objectId(node);
    if (id !== null) sourceModelsById.set(id.toString(), visibleObjectName(node));
  }

  let maxId = 0n;
  for (const node of targetObjects.children) {
    const id = objectId(node);
    if (id !== null && id > maxId) maxId = id;
  }

  const animationIdMap = new Map<string, bigint>();
  const animationNodes = sourceObjects.children.filter((node) =>
    ANIMATION_OBJECT_TYPES.has(node.name),
  );
  for (const node of animationNodes) {
    const sourceId = objectId(node);
    if (sourceId === null) throw new BinaryFbxError(`${node.name} is missing an object ID`);
    maxId += 1n;
    const cloned = cloneNode(node);
    setScalarId(cloned.properties[0], maxId);
    animationIdMap.set(sourceId.toString(), maxId);
    targetObjects.children.push(cloned);
  }

  const mapEndpoint = (id: bigint): bigint | null => {
    if (id === 0n) return 0n;
    const animationId = animationIdMap.get(id.toString());
    if (animationId !== undefined) return animationId;
    const modelName = sourceModelsById.get(id.toString());
    if (modelName !== undefined) return targetModelIds.get(modelName) ?? null;
    return null;
  };

  for (const connection of sourceConnections.children) {
    const ids = connectionIds(connection);
    if (!ids) continue;
    const touchesAnimation =
      animationIdMap.has(ids[0].toString()) || animationIdMap.has(ids[1].toString());
    if (!touchesAnimation) continue;

    const mappedSource = mapEndpoint(ids[0]);
    const mappedTarget = mapEndpoint(ids[1]);
    if (mappedSource === null || mappedTarget === null) {
      const missingModelId = sourceModelsById.has(ids[0].toString())
        ? ids[0]
        : sourceModelsById.has(ids[1].toString())
          ? ids[1]
          : null;
      if (missingModelId !== null) {
        const name = sourceModelsById.get(missingModelId.toString()) ?? "unknown";
        throw new BinaryFbxError(
          `animation target ${name} is missing or ambiguous in the character FBX`,
        );
      }
      continue;
    }

    const cloned = cloneNode(connection);
    setScalarId(cloned.properties[1], mappedSource);
    setScalarId(cloned.properties[2], mappedTarget);
    targetConnections.children.push(cloned);
  }

  mergeDefinitionTemplates(target, source, ANIMATION_OBJECT_TYPES);
  cloneTakesFrom(target, source);
  updateDefinitions(target);
  return target;
}

export function buildFbxExportDocument(
  characterDocument: BinaryFbxDocument | null,
  animationDocument: BinaryFbxDocument | null,
  selection: FbxExportSelection,
): BinaryFbxDocument {
  if (!selection.character && !selection.animation) {
    throw new BinaryFbxError("Select Character, Animation, or both before saving");
  }

  if (selection.character && !characterDocument) {
    throw new BinaryFbxError("No exportable character is loaded");
  }

  const embeddedAnimationAvailable =
    characterDocument !== null && analyzeFbxExportContents(characterDocument).animation;
  const resolvedAnimation =
    animationDocument ?? (embeddedAnimationAvailable ? characterDocument : null);

  if (selection.animation && !resolvedAnimation) {
    throw new BinaryFbxError("No exportable animation is loaded");
  }

  if (selection.character && selection.animation) {
    if (!characterDocument || !resolvedAnimation) {
      throw new BinaryFbxError("Character and animation are both required");
    }
    if (resolvedAnimation === characterDocument) {
      return cloneBinaryFbxDocument(characterDocument);
    }
    return mergeCharacterAndAnimationDocuments(characterDocument, resolvedAnimation);
  }

  if (selection.character) return createCharacterOnlyDocument(characterDocument!);
  return createAnimationOnlyDocument(resolvedAnimation!);
}
