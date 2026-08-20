import * as THREE from "three";

export type AnimationLoopFixMode = "cyclic" | "inertial";

export type AnimationLoopFixOptions = {
  mode?: AnimationLoopFixMode;
  inertialHalfLife?: number;
};

export type AnimationLoopFixReport = {
  repairedPositionTracks: number;
  repairedQuaternionTracks: number;
  skippedTracks: number;
};

type TrackFilter = (track: THREE.KeyframeTrack) => boolean;

const DEFAULT_INERTIAL_HALF_LIFE = 0.09;

function quinticCorrection(
  c0: number,
  d1: number,
  d2: number,
  sampleCount: number,
  t: number,
) {
  const span = Math.max(1, sampleCount - 1);
  const derivative = d1 * span;
  const acceleration = d2 * span * span;
  const a3 = 10 * c0 - 4 * derivative + 0.5 * acceleration;
  const a4 = -15 * c0 + 7 * derivative - acceleration;
  const a5 = 6 * c0 - 3 * derivative + 0.5 * acceleration;
  const t2 = t * t;
  const t3 = t2 * t;
  return a3 * t3 + a4 * t3 * t + a5 * t3 * t2;
}

export function repairLoopScalarSamples(source: ArrayLike<number>): Float32Array {
  const count = source.length;
  const corrected = new Float32Array(count);
  for (let i = 0; i < count; i += 1) corrected[i] = Number(source[i]);
  if (count < 4) return corrected;

  const at = (index: number) => Number(source[index]);
  const c0 = at(0) - at(count - 1);
  const startD1 = at(1) - at(0);
  const endD1 = at(count - 1) - at(count - 2);
  const startD2 = at(2) - 2 * at(1) + at(0);
  const endD2 = at(count - 1) - 2 * at(count - 2) + at(count - 3);
  const d1 = startD1 - endD1;
  const d2 = startD2 - endD2;

  for (let i = 0; i < count; i += 1) {
    const t = i / (count - 1);
    corrected[i] = at(i) + quinticCorrection(c0, d1, d2, count, t);
  }
  return corrected;
}

function finiteSupportDecay(time: number, duration: number, halfLife: number) {
  const lambda = Math.LN2 / Math.max(1e-4, halfLife);
  const horizon = Math.max(1e-4, Math.min(duration, halfLife * 6));
  const s = THREE.MathUtils.clamp(time / horizon, 0, 1);
  const cutoff = 1 - (3 * s * s - 2 * s * s * s);
  return { lambda, cutoff };
}

export function repairLoopScalarSamplesInertial(
  source: ArrayLike<number>,
  times: ArrayLike<number>,
  halfLife = DEFAULT_INERTIAL_HALF_LIFE,
): Float32Array {
  const count = source.length;
  const corrected = new Float32Array(count);
  for (let i = 0; i < count; i += 1) corrected[i] = Number(source[i]);
  if (count < 3 || times.length !== count) return corrected;

  const t0 = Number(times[0]);
  const duration = Math.max(1e-4, Number(times[count - 1]) - t0);
  const startDt = Math.max(1e-6, Number(times[1]) - Number(times[0]));
  const endDt = Math.max(1e-6, Number(times[count - 1]) - Number(times[count - 2]));
  const start = Number(source[0]);
  const end = Number(source[count - 1]);
  const startVelocity = (Number(source[1]) - start) / startDt;
  const endVelocity = (end - Number(source[count - 2])) / endDt;
  const positionOffset = end - start;
  const velocityOffset = endVelocity - startVelocity;
  const lambda = Math.LN2 / Math.max(1e-4, halfLife);
  const linearTerm = velocityOffset + lambda * positionOffset;

  for (let i = 0; i < count; i += 1) {
    const time = Math.max(0, Number(times[i]) - t0);
    const { cutoff } = finiteSupportDecay(time, duration, halfLife);
    const correction = (positionOffset + linearTerm * time) * Math.exp(-lambda * time) * cutoff;
    corrected[i] = Number(source[i]) + correction;
  }
  return corrected;
}

function repairVectorTrack(
  track: THREE.VectorKeyframeTrack,
  mode: AnimationLoopFixMode,
  inertialHalfLife: number,
) {
  const count = track.times.length;
  if (count < 4 || track.values.length !== count * 3) return false;
  const values = track.values;
  const corrected = new Float32Array(values.length);
  corrected.set(values as ArrayLike<number>);

  for (let axis = 0; axis < 3; axis += 1) {
    const axisValues = Array.from({ length: count }, (_, index) => Number(values[index * 3 + axis]));
    const repaired = mode === "inertial"
      ? repairLoopScalarSamplesInertial(axisValues, track.times, inertialHalfLife)
      : repairLoopScalarSamples(axisValues);
    for (let i = 0; i < count; i += 1) corrected[i * 3 + axis] = repaired[i];
  }

  track.values = corrected;
  return true;
}

function quaternionAt(values: ArrayLike<number>, index: number) {
  const offset = index * 4;
  return new THREE.Quaternion(
    Number(values[offset]),
    Number(values[offset + 1]),
    Number(values[offset + 2]),
    Number(values[offset + 3]),
  ).normalize();
}

function quaternionStep(from: THREE.Quaternion, to: THREE.Quaternion) {
  const delta = from.clone().invert().multiply(to).normalize();
  if (delta.w < 0) {
    delta.x *= -1;
    delta.y *= -1;
    delta.z *= -1;
    delta.w *= -1;
  }
  const sinHalf = Math.hypot(delta.x, delta.y, delta.z);
  if (sinHalf < 1e-8) return new THREE.Vector3();
  const angle = 2 * Math.atan2(sinHalf, THREE.MathUtils.clamp(delta.w, -1, 1));
  return new THREE.Vector3(delta.x, delta.y, delta.z).multiplyScalar(angle / sinHalf);
}

function quaternionExp(rotationVector: THREE.Vector3) {
  const angle = rotationVector.length();
  if (angle < 1e-8) {
    return new THREE.Quaternion(
      rotationVector.x * 0.5,
      rotationVector.y * 0.5,
      rotationVector.z * 0.5,
      1,
    ).normalize();
  }
  const half = angle * 0.5;
  const scale = Math.sin(half) / angle;
  return new THREE.Quaternion(
    rotationVector.x * scale,
    rotationVector.y * scale,
    rotationVector.z * scale,
    Math.cos(half),
  );
}

function continuousQuaternions(track: THREE.QuaternionKeyframeTrack) {
  const quaternions = Array.from(
    { length: track.times.length },
    (_, index) => quaternionAt(track.values, index),
  );
  for (let i = 1; i < quaternions.length; i += 1) {
    if (quaternions[i - 1].dot(quaternions[i]) < 0) {
      quaternions[i].x *= -1;
      quaternions[i].y *= -1;
      quaternions[i].z *= -1;
      quaternions[i].w *= -1;
    }
  }
  return quaternions;
}

function repairQuaternionTrackCyclic(track: THREE.QuaternionKeyframeTrack) {
  const count = track.times.length;
  if (count < 4 || track.values.length !== count * 4) return false;
  const quaternions = continuousQuaternions(track);

  const startVelocity = quaternionStep(quaternions[0], quaternions[1]);
  const nextVelocity = quaternionStep(quaternions[1], quaternions[2]);
  const previousVelocity = quaternionStep(quaternions[count - 3], quaternions[count - 2]);
  const endVelocity = quaternionStep(quaternions[count - 2], quaternions[count - 1]);
  const c0 = quaternionStep(quaternions[count - 1], quaternions[0]);
  const d1 = startVelocity.clone().sub(endVelocity);
  const startAcceleration = nextVelocity.clone().sub(startVelocity);
  const endAcceleration = endVelocity.clone().sub(previousVelocity);
  const d2 = startAcceleration.sub(endAcceleration);

  const corrected = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const t = i / (count - 1);
    const correction = new THREE.Vector3(
      quinticCorrection(c0.x, d1.x, d2.x, count, t),
      quinticCorrection(c0.y, d1.y, d2.y, count, t),
      quinticCorrection(c0.z, d1.z, d2.z, count, t),
    );
    const q = quaternions[i].clone().multiply(quaternionExp(correction)).normalize();
    const offset = i * 4;
    corrected[offset] = q.x;
    corrected[offset + 1] = q.y;
    corrected[offset + 2] = q.z;
    corrected[offset + 3] = q.w;
  }
  track.values = corrected;
  return true;
}

function repairQuaternionTrackInertial(
  track: THREE.QuaternionKeyframeTrack,
  halfLife: number,
) {
  const count = track.times.length;
  if (count < 3 || track.values.length !== count * 4) return false;
  const quaternions = continuousQuaternions(track);
  const t0 = Number(track.times[0]);
  const duration = Math.max(1e-4, Number(track.times[count - 1]) - t0);
  const startDt = Math.max(1e-6, Number(track.times[1]) - Number(track.times[0]));
  const endDt = Math.max(1e-6, Number(track.times[count - 1]) - Number(track.times[count - 2]));
  const poseOffset = quaternionStep(quaternions[0], quaternions[count - 1]);
  const startVelocity = quaternionStep(quaternions[0], quaternions[1]).multiplyScalar(1 / startDt);
  const endVelocity = quaternionStep(quaternions[count - 2], quaternions[count - 1]).multiplyScalar(1 / endDt);
  const velocityOffset = endVelocity.sub(startVelocity);
  const lambda = Math.LN2 / Math.max(1e-4, halfLife);
  const linearTerm = velocityOffset.addScaledVector(poseOffset, lambda);

  const corrected = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const time = Math.max(0, Number(track.times[i]) - t0);
    const { cutoff } = finiteSupportDecay(time, duration, halfLife);
    const correction = poseOffset.clone()
      .addScaledVector(linearTerm, time)
      .multiplyScalar(Math.exp(-lambda * time) * cutoff);
    const q = quaternions[i].clone().multiply(quaternionExp(correction)).normalize();
    const offset = i * 4;
    corrected[offset] = q.x;
    corrected[offset + 1] = q.y;
    corrected[offset + 2] = q.z;
    corrected[offset + 3] = q.w;
  }
  track.values = corrected;
  return true;
}

/** Repairs one pristine clip for the requested loop strategy. */
export function repairAnimationLoop(
  source: THREE.AnimationClip,
  shouldRepairTrack: TrackFilter = () => true,
  options: AnimationLoopFixOptions = {},
): { clip: THREE.AnimationClip; report: AnimationLoopFixReport } {
  const mode = options.mode ?? "cyclic";
  const inertialHalfLife = options.inertialHalfLife ?? DEFAULT_INERTIAL_HALF_LIFE;
  const clip = source.clone();
  const report: AnimationLoopFixReport = {
    repairedPositionTracks: 0,
    repairedQuaternionTracks: 0,
    skippedTracks: 0,
  };

  clip.tracks.forEach((track) => {
    if (!shouldRepairTrack(track)) {
      report.skippedTracks += 1;
      return;
    }
    if (track instanceof THREE.QuaternionKeyframeTrack) {
      const repaired = mode === "inertial"
        ? repairQuaternionTrackInertial(track, inertialHalfLife)
        : repairQuaternionTrackCyclic(track);
      if (repaired) report.repairedQuaternionTracks += 1;
      else report.skippedTracks += 1;
      return;
    }
    if (track instanceof THREE.VectorKeyframeTrack && track.name.endsWith(".position")) {
      if (repairVectorTrack(track, mode, inertialHalfLife)) report.repairedPositionTracks += 1;
      else report.skippedTracks += 1;
      return;
    }
    report.skippedTracks += 1;
  });

  const suffix = mode === "inertial" ? "Inertial Loop Fixed" : "Cyclic Loop Fixed";
  clip.name = source.name ? `${source.name} (${suffix})` : suffix;
  clip.resetDuration();
  return { clip, report };
}
