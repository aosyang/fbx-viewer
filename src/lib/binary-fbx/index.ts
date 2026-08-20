import { unzlibSync, zlibSync } from "three/examples/jsm/libs/fflate.module.js";

const MAGIC = new Uint8Array([
  0x4b,0x61,0x79,0x64,0x61,0x72,0x61,0x20,0x46,0x42,0x58,0x20,0x42,0x69,0x6e,0x61,0x72,0x79,0x20,0x20,0x00,0x1a,0x00,
]);

const textDecoder = new TextDecoder('utf-8');
const textEncoder = new TextEncoder();

type ScalarCode = 'Y' | 'C' | 'I' | 'F' | 'D' | 'L';
type ArrayCode = 'f' | 'd' | 'l' | 'i' | 'b';
export type FbxArrayScalar = number | bigint | boolean;
export type PropertyCode = ScalarCode | 'S' | 'R' | ArrayCode;

export class BinaryFbxError extends Error {}

export class FbxProperty {
  constructor(
    public code: PropertyCode,
    public raw: Uint8Array,
    public value: unknown = undefined,
    public arrayCount: number | null = null,
    public arrayEncoding: number | null = null,
  ) {}

  replaceString(value: string): void {
    if (this.code !== 'S') throw new BinaryFbxError(`replaceString requires S property, got ${this.code}`);
    this.value = value;
    this.raw = encodeStringProperty(value);
  }

  replaceScalar(value: number | bigint | boolean): void {
    if (!['Y','C','I','F','D','L'].includes(this.code)) {
      throw new BinaryFbxError(`replaceScalar requires scalar property, got ${this.code}`);
    }
    this.value = value;
    this.raw = encodeScalarProperty(this.code as ScalarCode, value);
  }

  readArray(): FbxArrayScalar[] {
    if (!['f','d','l','i','b'].includes(this.code)) {
      throw new BinaryFbxError(`readArray requires array property, got ${this.code}`);
    }
    return decodeArrayProperty(this);
  }

  replaceArray(values: ArrayLike<FbxArrayScalar>, compress = this.arrayEncoding === 1): void {
    if (!['f','d','l','i','b'].includes(this.code)) {
      throw new BinaryFbxError(`replaceArray requires array property, got ${this.code}`);
    }
    const copied = Array.from(values);
    this.value = copied;
    this.arrayCount = copied.length;
    this.arrayEncoding = compress ? 1 : 0;
    this.raw = encodeArrayProperty(this.code as ArrayCode, copied, compress);
  }
}

export class FbxNode {
  constructor(
    public name: string,
    public properties: FbxProperty[] = [],
    public children: FbxNode[] = [],
    public hasChildSentinel = false,
  ) {}

  *walk(): Iterable<FbxNode> {
    yield this;
    for (const child of this.children) yield* child.walk();
  }
}

export class BinaryFbxDocument {
  constructor(
    public version: number,
    public nodes: FbxNode[],
    public footer: Uint8Array,
  ) {}

  get wideOffsets(): boolean { return this.version >= 7500; }
  get nodeHeaderSize(): number { return this.wideOffsets ? 25 : 13; }

  *walk(): Iterable<FbxNode> {
    for (const node of this.nodes) yield* node.walk();
  }

  findNodes(name: string): FbxNode[] {
    return [...this.walk()].filter((node) => node.name === name);
  }

  findModelByVisibleName(visibleName: string): FbxNode | undefined {
    return this.findNodes('Model').find((node) => {
      const prop = node.properties[1];
      return prop?.code === 'S' && modelVisibleName(String(prop.value ?? '')) === visibleName;
    });
  }

  renameModel(oldVisibleName: string, newVisibleName: string): boolean {
    const node = this.findModelByVisibleName(oldVisibleName);
    if (!node) return false;
    const prop = node.properties[1];
    if (!prop || prop.code !== 'S') return false;
    const current = String(prop.value ?? '');
    const marker = current.indexOf('\u0000\u0001Model');
    const suffix = marker >= 0 ? current.slice(marker) : '\u0000\u0001Model';
    prop.replaceString(newVisibleName + suffix);
    return true;
  }
}

class Reader {
  readonly bytes: Uint8Array;
  readonly view: DataView;

  constructor(input: ArrayBuffer | Uint8Array) {
    this.bytes = input instanceof Uint8Array
      ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
      : new Uint8Array(input);
    this.view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
  }

  require(offset: number, size: number): void {
    if (offset < 0 || size < 0 || offset + size > this.bytes.byteLength) {
      throw new BinaryFbxError(`read outside file at ${offset}+${size}; file=${this.bytes.byteLength}`);
    }
  }

  slice(offset: number, size: number): Uint8Array {
    this.require(offset, size);
    return this.bytes.slice(offset, offset + size);
  }
}

function headerSize(version: number): number { return version >= 7500 ? 25 : 13; }

function bytesEqualAt(reader: Reader, offset: number, pattern: Uint8Array): boolean {
  if (offset + pattern.length > reader.bytes.length) return false;
  for (let i = 0; i < pattern.length; i++) if (reader.bytes[offset + i] !== pattern[i]) return false;
  return true;
}

function isNullRecord(reader: Reader, offset: number, version: number): boolean {
  const size = headerSize(version);
  if (offset + size > reader.bytes.length) return false;
  for (let i = 0; i < size; i++) if (reader.bytes[offset + i] !== 0) return false;
  return true;
}

function readUint64(view: DataView, offset: number): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new BinaryFbxError(`64-bit FBX offset exceeds JS safe integer: ${value}`);
  return Number(value);
}

function parseProperty(reader: Reader, start: number): [FbxProperty, number] {
  let offset = start;
  reader.require(offset, 1);
  const code = String.fromCharCode(reader.bytes[offset]) as PropertyCode;
  offset += 1;
  const v = reader.view;
  let value: unknown;

  switch (code) {
    case 'Y': value = v.getInt16(offset, true); offset += 2; break;
    case 'C': value = v.getUint8(offset) !== 0; offset += 1; break;
    case 'I': value = v.getInt32(offset, true); offset += 4; break;
    case 'F': value = v.getFloat32(offset, true); offset += 4; break;
    case 'D': value = v.getFloat64(offset, true); offset += 8; break;
    case 'L': value = v.getBigInt64(offset, true); offset += 8; break;
    case 'S':
    case 'R': {
      const len = v.getUint32(offset, true); offset += 4;
      const payload = reader.slice(offset, len); offset += len;
      value = code === 'S' ? textDecoder.decode(payload) : payload;
      break;
    }
    case 'f': case 'd': case 'l': case 'i': case 'b': {
      const count = v.getUint32(offset, true);
      const encoding = v.getUint32(offset + 4, true);
      const storedLength = v.getUint32(offset + 8, true);
      offset += 12;
      reader.require(offset, storedLength);
      offset += storedLength;
      return [new FbxProperty(code, reader.slice(start, offset - start), undefined, count, encoding), offset];
    }
    default:
      throw new BinaryFbxError(`unsupported FBX property type ${JSON.stringify(code)} at ${start}`);
  }

  return [new FbxProperty(code, reader.slice(start, offset - start), value), offset];
}

function parseNodeHeader(reader: Reader, offset: number, version: number): [number, number, number, number] {
  const v = reader.view;
  if (version >= 7500) {
    return [readUint64(v, offset), readUint64(v, offset + 8), readUint64(v, offset + 16), v.getUint8(offset + 24)];
  }
  return [v.getUint32(offset, true), v.getUint32(offset + 4, true), v.getUint32(offset + 8, true), v.getUint8(offset + 12)];
}

function parseNodeList(reader: Reader, start: number, limit: number, version: number): [FbxNode[], number, boolean] {
  const nodes: FbxNode[] = [];
  let offset = start;
  let sawSentinel = false;

  while (offset < limit) {
    if (isNullRecord(reader, offset, version)) {
      offset += headerSize(version);
      sawSentinel = true;
      break;
    }

    const nodeStart = offset;
    const [endOffset, propertyCount, propertyBytes, nameLength] = parseNodeHeader(reader, offset, version);
    if (endOffset <= nodeStart || endOffset > reader.bytes.length) {
      throw new BinaryFbxError(`invalid node end offset ${endOffset} at ${nodeStart}`);
    }
    offset += headerSize(version);
    const name = textDecoder.decode(reader.slice(offset, nameLength));
    offset += nameLength;

    const propertyStart = offset;
    const properties: FbxProperty[] = [];
    for (let i = 0; i < propertyCount; i++) {
      const [property, next] = parseProperty(reader, offset);
      properties.push(property);
      offset = next;
    }
    if (offset - propertyStart !== propertyBytes) {
      throw new BinaryFbxError(`node ${name} property length mismatch: header=${propertyBytes}, parsed=${offset - propertyStart}`);
    }

    let children: FbxNode[] = [];
    let childSentinel = false;
    if (offset < endOffset) {
      const parsed = parseNodeList(reader, offset, endOffset, version);
      children = parsed[0];
      offset = parsed[1];
      childSentinel = parsed[2];
      if (offset !== endOffset) throw new BinaryFbxError(`node ${name} child region ended at ${offset}, expected ${endOffset}`);
    }

    nodes.push(new FbxNode(name, properties, children, childSentinel));
    offset = endOffset;
  }

  return [nodes, offset, sawSentinel];
}

export function readBinaryFbx(input: ArrayBuffer | Uint8Array): BinaryFbxDocument {
  const reader = new Reader(input);
  if (!bytesEqualAt(reader, 0, MAGIC)) throw new BinaryFbxError('not a supported binary FBX');
  const version = reader.view.getUint32(MAGIC.length, true);
  const topStart = MAGIC.length + 4;
  const [nodes, footerStart, sawSentinel] = parseNodeList(reader, topStart, reader.bytes.length, version);
  if (!sawSentinel) throw new BinaryFbxError('top-level FBX node sentinel not found');
  return new BinaryFbxDocument(version, nodes, reader.bytes.slice(footerStart));
}

function concat(parts: Uint8Array[]): Uint8Array {
  const size = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
}

function encodeStringProperty(value: string): Uint8Array {
  const payload = textEncoder.encode(value);
  const out = new Uint8Array(1 + 4 + payload.length);
  out[0] = 'S'.charCodeAt(0);
  new DataView(out.buffer).setUint32(1, payload.length, true);
  out.set(payload, 5);
  return out;
}

function encodeScalarProperty(code: ScalarCode, value: number | bigint | boolean): Uint8Array {
  const sizes: Record<ScalarCode, number> = {Y:2,C:1,I:4,F:4,D:8,L:8};
  const out = new Uint8Array(1 + sizes[code]);
  out[0] = code.charCodeAt(0);
  const v = new DataView(out.buffer);
  switch (code) {
    case 'Y': v.setInt16(1, Number(value), true); break;
    case 'C': v.setUint8(1, value ? 1 : 0); break;
    case 'I': v.setInt32(1, Number(value), true); break;
    case 'F': v.setFloat32(1, Number(value), true); break;
    case 'D': v.setFloat64(1, Number(value), true); break;
    case 'L': v.setBigInt64(1, BigInt(value), true); break;
  }
  return out;
}


const ARRAY_ITEM_SIZE: Record<ArrayCode, number> = { f: 4, d: 8, l: 8, i: 4, b: 1 };

function decodeArrayProperty(property: FbxProperty): FbxArrayScalar[] {
  const code = property.code as ArrayCode;
  const count = property.arrayCount;
  const encoding = property.arrayEncoding;
  if (count == null || encoding == null) throw new BinaryFbxError('array metadata missing');
  if (property.raw.length < 13) throw new BinaryFbxError('array property payload is truncated');

  let payload = property.raw.slice(13);
  if (encoding === 1) payload = unzlibSync(payload);
  else if (encoding !== 0) throw new BinaryFbxError(`unsupported array encoding ${encoding}`);

  const itemSize = ARRAY_ITEM_SIZE[code];
  const expected = count * itemSize;
  if (payload.byteLength !== expected) {
    throw new BinaryFbxError(`array decoded bytes ${payload.byteLength} != expected ${expected}`);
  }

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const values: FbxArrayScalar[] = new Array(count);
  for (let i = 0; i < count; i += 1) {
    const offset = i * itemSize;
    switch (code) {
      case 'f': values[i] = view.getFloat32(offset, true); break;
      case 'd': values[i] = view.getFloat64(offset, true); break;
      case 'l': values[i] = view.getBigInt64(offset, true); break;
      case 'i': values[i] = view.getInt32(offset, true); break;
      case 'b': values[i] = view.getUint8(offset) !== 0; break;
    }
  }
  return values;
}

function encodeArrayProperty(
  code: ArrayCode,
  values: readonly FbxArrayScalar[],
  compress: boolean,
): Uint8Array {
  const itemSize = ARRAY_ITEM_SIZE[code];
  const unpacked = new Uint8Array(values.length * itemSize);
  const view = new DataView(unpacked.buffer);
  values.forEach((value, index) => {
    const offset = index * itemSize;
    switch (code) {
      case 'f': view.setFloat32(offset, Number(value), true); break;
      case 'd': view.setFloat64(offset, Number(value), true); break;
      case 'l': view.setBigInt64(offset, BigInt(value), true); break;
      case 'i': view.setInt32(offset, Number(value), true); break;
      case 'b': view.setUint8(offset, value ? 1 : 0); break;
    }
  });

  const payload = compress ? zlibSync(unpacked) : unpacked;
  const out = new Uint8Array(13 + payload.length);
  out[0] = code.charCodeAt(0);
  const header = new DataView(out.buffer);
  header.setUint32(1, values.length, true);
  header.setUint32(5, compress ? 1 : 0, true);
  header.setUint32(9, payload.length, true);
  out.set(payload, 13);
  return out;
}

function nodeSerializedSize(node: FbxNode, version: number): number {
  const nameBytes = textEncoder.encode(node.name);
  const propertyBytes = node.properties.reduce((sum, property) => sum + property.raw.length, 0);
  const childBytes = node.children.reduce((sum, child) => sum + nodeSerializedSize(child, version), 0);
  return headerSize(version) + nameBytes.length + propertyBytes + childBytes + (node.hasChildSentinel ? headerSize(version) : 0);
}

function writeNode(node: FbxNode, version: number, absoluteStart: number): Uint8Array {
  const nameBytes = textEncoder.encode(node.name);
  if (nameBytes.length > 255) throw new BinaryFbxError(`node name too long: ${node.name}`);
  const propertyBytes = concat(node.properties.map((property) => property.raw));
  const totalSize = nodeSerializedSize(node, version);
  const endOffset = absoluteStart + totalSize;
  const header = new Uint8Array(headerSize(version));
  const view = new DataView(header.buffer);

  if (version >= 7500) {
    view.setBigUint64(0, BigInt(endOffset), true);
    view.setBigUint64(8, BigInt(node.properties.length), true);
    view.setBigUint64(16, BigInt(propertyBytes.length), true);
    view.setUint8(24, nameBytes.length);
  } else {
    if (endOffset > 0xffffffff || propertyBytes.length > 0xffffffff) throw new BinaryFbxError('FBX < 7500 cannot represent output offsets');
    view.setUint32(0, endOffset, true);
    view.setUint32(4, node.properties.length, true);
    view.setUint32(8, propertyBytes.length, true);
    view.setUint8(12, nameBytes.length);
  }

  const parts: Uint8Array[] = [header, nameBytes, propertyBytes];
  let cursor = absoluteStart + header.length + nameBytes.length + propertyBytes.length;
  for (const child of node.children) {
    const bytes = writeNode(child, version, cursor);
    parts.push(bytes);
    cursor += bytes.length;
  }
  if (node.hasChildSentinel) parts.push(new Uint8Array(headerSize(version)));
  const out = concat(parts);
  if (out.length !== totalSize) throw new BinaryFbxError(`serialized node size mismatch for ${node.name}`);
  return out;
}

export function writeBinaryFbx(document: BinaryFbxDocument): ArrayBuffer {
  const versionBytes = new Uint8Array(4);
  new DataView(versionBytes.buffer).setUint32(0, document.version, true);
  const parts: Uint8Array[] = [MAGIC, versionBytes];
  let cursor = MAGIC.length + 4;
  for (const node of document.nodes) {
    const bytes = writeNode(node, document.version, cursor);
    parts.push(bytes);
    cursor += bytes.length;
  }
  parts.push(new Uint8Array(headerSize(document.version)));
  parts.push(document.footer);
  const output = concat(parts);
  const result = new ArrayBuffer(output.byteLength);
  new Uint8Array(result).set(output);
  return result;
}

export function modelVisibleName(encoded: string): string {
  const marker = encoded.indexOf('\u0000\u0001Model');
  return marker >= 0 ? encoded.slice(0, marker) : encoded;
}

export function buffersEqual(a: ArrayBuffer | Uint8Array, b: ArrayBuffer | Uint8Array): boolean {
  const aa = a instanceof Uint8Array ? a : new Uint8Array(a);
  const bb = b instanceof Uint8Array ? b : new Uint8Array(b);
  if (aa.length !== bb.length) return false;
  for (let i = 0; i < aa.length; i++) if (aa[i] !== bb[i]) return false;
  return true;
}
