// Shared math stays free of DOM dependencies so it can be checked against Python.
export const FEATURE_VERSION = 'wrist-scale-orientation-xy-v2';
export const DEPTH_FEATURE_VERSION = 'wrist-scale-image-xyz-v3';
export const WORLD_FEATURE_VERSION = 'wrist-scale-image-xyz-world-distances-v3';
export const ALIGN_FEATURE_VERSION = 'wrist-scale-image-xyz-palm-xy-v4';
export const WORLD_ALIGN_FEATURE_VERSION = 'wrist-scale-image-xyz-world-palm-v4';
export const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export function normalize(points) {
  if (!Array.isArray(points)) throw new Error('Expected hand landmarks.');
  let p = points;
  if (points.length === 42)
    p = Array.from({ length: 21 }, (_, i) => [points[i * 2], points[i * 2 + 1]]);
  if (points.length === 63)
    p = Array.from({ length: 21 }, (_, i) => [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]]);
  if (p.length !== 21) throw new Error('Expected 21 hand landmarks.');
  const xy = p.map((v) => (Array.isArray(v) ? [v[0], v[1]] : [v.x, v.y]));
  if (xy.some((v) => v.some((n) => !Number.isFinite(n))))
    throw new Error('Invalid hand coordinates.');
  const relative = xy.map(([x, y]) => [x - xy[0][0], y - xy[0][1]]);
  const [vx, vy] = relative[9];
  const squared = vx * vx + vy * vy;
  if (squared < 1e-12) throw new Error('Hand is too small to read.');
  const cross = relative[5][0] * relative[17][1] - relative[5][1] * relative[17][0];
  const mirror = cross < 0 ? -1 : 1,
    size = Math.sqrt(squared);
  return relative.flatMap(([x, y]) => [(x * mirror) / size, y / size]);
}

function xyz(points) {
  if (!Array.isArray(points)) throw new Error('Expected XYZ hand landmarks.');
  const rows =
    points.length === 63
      ? Array.from({ length: 21 }, (_, i) => points.slice(i * 3, i * 3 + 3))
      : points.map((p) => (Array.isArray(p) ? p : [p.x, p.y, p.z]));
  if (
    rows.length !== 21 ||
    rows.some((p) => p.length < 3 || p.slice(0, 3).some((v) => !Number.isFinite(v)))
  )
    throw new Error('21 finite XYZ hand landmarks are required.');
  return rows;
}

export function staticFeatures(points, worldPoints, version = FEATURE_VERSION) {
  const xy = normalize(points);
  if (version === FEATURE_VERSION) return xy;
  if (
    ![
      DEPTH_FEATURE_VERSION,
      WORLD_FEATURE_VERSION,
      ALIGN_FEATURE_VERSION,
      WORLD_ALIGN_FEATURE_VERSION,
    ].includes(version)
  )
    throw new Error('Unknown static feature version.');
  const p = xyz(points),
    palm = Math.hypot(p[9][0] - p[0][0], p[9][1] - p[0][1]);
  const features = [...xy, ...p.map((row) => (row[2] - p[0][2]) / palm)];
  if (version === ALIGN_FEATURE_VERSION) {
    const vx = xy[18],
      vy = xy[19];
    for (let i = 0; i < 21; i++)
      features.push(xy[i * 2] * vy - xy[i * 2 + 1] * vx, xy[i * 2] * vx + xy[i * 2 + 1] * vy);
  }
  if ([WORLD_FEATURE_VERSION, WORLD_ALIGN_FEATURE_VERSION].includes(version)) {
    const world = xyz(worldPoints),
      scale = Math.hypot(...world[9].map((v, i) => v - world[0][i]));
    if (scale < 1e-6) throw new Error('Degenerate world hand geometry.');
    for (let a = 0; a < 21; a++)
      for (let b = a + 1; b < 21; b++)
        features.push(Math.hypot(...world[a].map((v, i) => v - world[b][i])) / scale);
  }
  if (version === WORLD_ALIGN_FEATURE_VERSION) {
    const vx = xy[18],
      vy = xy[19];
    for (let i = 0; i < 21; i++)
      features.push(xy[i * 2] * vy - xy[i * 2 + 1] * vx, xy[i * 2] * vx + xy[i * 2 + 1] * vy);
  }
  return features;
}

// The small frozen expert is only consulted for selected ambiguous letters.
export function denseProbabilities(model, features) {
  let x = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  if (model.kernel_basis) {
    const input = x;
    x = model.kernel_basis.centers.map((center) => {
      let squared = 0;
      for (let i = 0; i < input.length; i++) squared += (input[i] - center[i]) ** 2;
      return Math.exp(-model.kernel_basis.gamma * squared);
    });
  }
  model.layers.forEach((layer, index) => {
    x = layer.bias.map((bias, j) => {
      let value = bias;
      for (let i = 0; i < x.length; i++) value += x[i] * layer.kernel[i][j];
      return index < model.layers.length - 1 ? Math.max(0, value) : value;
    });
  });
  const maximum = Math.max(...x),
    exp = x.map((v) => Math.exp(v - maximum)),
    total = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / total);
}

export function nearestDistance(features, support) {
  if (!support?.length) return Infinity;
  return Math.min(
    ...support.map((row) =>
      Math.sqrt(row.reduce((sum, v, i) => sum + (v - features[i]) ** 2, 0) / features.length),
    ),
  );
}

export function topologyFeatures(xy) {
  if (!Array.isArray(xy) || xy.length !== 42 || xy.some((v) => !Number.isFinite(v)))
    throw new Error('Expected 42 finite canonical XY features.');
  const p = Array.from({ length: 21 }, (_, i) => xy.slice(2 * i, 2 * i + 2)),
    [vx, vy] = p[9];
  const ax = p.map(([x, y]) => x * vy - y * vx),
    ay = p.map(([x, y]) => x * vx + y * vy);
  const radius = (i) => Math.hypot(...p[i]),
    distance = (a, b) => Math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]);
  return [
    (ax[8] - ax[12]) * (ax[5] - ax[9]),
    ax[8] - ax[12],
    ax[6] - ax[10],
    ax[5] - ax[9],
    ...[
      [8, 6],
      [12, 10],
      [16, 14],
      [20, 18],
    ].map(([a, b]) => radius(a) - radius(b)),
    distance(4, 5),
    distance(4, 9),
    distance(8, 12),
    ay[8] - ay[12],
  ];
}

export function topologyProbability(expert, xy) {
  if (
    expert.version !== 'finger-crossing-r-v1' ||
    !Number.isFinite(expert.bias) ||
    !Number.isFinite(expert.threshold) ||
    expert.threshold < 0.85 ||
    expert.threshold > 1 ||
    !Array.isArray(expert.allowed_primary_labels) ||
    expert.allowed_primary_labels.join(',') !== 'R,U,V,D' ||
    ['mean', 'scale', 'weights'].some(
      (key) =>
        !Array.isArray(expert[key]) ||
        expert[key].length !== 12 ||
        expert[key].some((v) => !Number.isFinite(v)),
    ) ||
    expert.scale.some((v) => v <= 0)
  )
    throw new Error('Invalid finger topology expert.');
  const features = topologyFeatures(xy);
  const value = features.reduce(
    (sum, v, i) => sum + ((v - expert.mean[i]) / expert.scale[i]) * expert.weights[i],
    expert.bias,
  );
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

export function gatePrediction(probabilities, features, model, threshold) {
  const rejected = (reason) => ({ label: null, confidence: 0, candidate: null, reason });
  if (
    !Array.isArray(features) ||
    features.length !== model.mean.length ||
    features.some((v) => !Number.isFinite(v)) ||
    !Array.isArray(probabilities) ||
    probabilities.length !== model.labels.length ||
    !probabilities.length ||
    probabilities.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1
  )
    return rejected('invalid_input');
  const index = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[index];
  const candidate = model.labels[index];
  const distance = nearestDistance(features, model.support[candidate]);
  let reason =
    candidate === '_'
      ? 'unsupported'
      : confidence < threshold
        ? 'confidence'
        : !(distance <= model.distance_thresholds[candidate])
          ? 'distance'
          : null;
  // A second, training-only support set models reviewed non-letter shapes.
  // Keep the existing class network and confidence/distance checks intact.
  if (!reason && model.rejection) {
    const gate = model.rejection;
    if (
      gate.version !== 'negative-distance-ratio-v1' ||
      !Number.isFinite(gate.ratio) ||
      gate.ratio <= 0 ||
      !Array.isArray(gate.support) ||
      !gate.support.length ||
      gate.support.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== features.length ||
          row.some((v) => !Number.isFinite(v)),
      )
    )
      return rejected('invalid_rejection_model');
    if (nearestDistance(features, gate.support) <= distance * gate.ratio)
      reason = 'unknown_handshape';
  }
  let result = { label: reason ? null : candidate, confidence, candidate, reason };
  if (model.rescue) {
    const rescue = model.rescue;
    const minima = rescue.min_primary_probabilities;
    if (
      rescue.version !== 'gated-legacy-agreement-v1' ||
      !Array.isArray(rescue.labels) ||
      !minima ||
      rescue.labels.some(
        (label) =>
          !model.labels.includes(label) ||
          !Number.isFinite(minima[label]) ||
          minima[label] < 0 ||
          minima[label] > 1,
      ) ||
      rescue.model?.feature_version !== FEATURE_VERSION ||
      rescue.model.rescue
    )
      return rejected('invalid_rescue_model');
    if (
      rescue.labels.some((label) => probabilities[model.labels.indexOf(label)] >= minima[label])
    ) {
      try {
        const legacyFeatures = features.slice(0, 42),
          legacy = gatePrediction(
            denseProbabilities(rescue.model, legacyFeatures),
            legacyFeatures,
            rescue.model,
            threshold,
          );
        if (
          rescue.labels.includes(legacy.label) &&
          probabilities[model.labels.indexOf(legacy.label)] >= minima[legacy.label]
        )
          result = { ...legacy, candidate, expert: 'legacy_agreement' };
      } catch {
        return rejected('invalid_rescue_model');
      }
    }
  }
  if (model.topology_expert) {
    const expert = model.topology_expert;
    try {
      const probability = topologyProbability(expert, features.slice(0, 42));
      if (
        expert.allowed_primary_labels.includes(candidate) &&
        (result.label === null || expert.allowed_primary_labels.includes(result.label)) &&
        probability >= Math.max(threshold, expert.threshold)
      ) {
        result = {
          label: 'R',
          confidence: probability,
          candidate,
          reason: null,
          expert: 'finger_topology',
        };
      }
    } catch {
      return rejected('invalid_topology_model');
    }
  }
  return result;
}

export class Stabilizer {
  constructor(holdMs = 900) {
    this.holdMs = holdMs;
    this.reset();
  }
  reset() {
    this.frames = [];
    this.since = null;
    this.candidate = null;
    this.latched = null;
    this.releaseSince = null;
  }
  update(label, now) {
    // Three hundred milliseconds of a different/absent sign re-arms a repeated letter.
    if (this.latched && label !== this.latched) {
      this.releaseSince ??= now;
      if (now - this.releaseSince >= 300) this.latched = null;
    } else this.releaseSince = null;
    this.frames.push(label);
    if (this.frames.length > 6) this.frames.shift();
    const stable =
      label &&
      this.frames.length >= 4 &&
      this.frames.filter((x) => x === label).length / this.frames.length >= 0.8;
    if (!stable) {
      this.candidate = null;
      this.since = null;
      return { stable: null, progress: 0, commit: null };
    }
    if (this.candidate !== label) {
      this.candidate = label;
      this.since = now;
    }
    const progress = Math.min((now - this.since) / this.holdMs, 1);
    let commit = null;
    if (progress >= 1 && this.latched !== label) {
      commit = label;
      this.latched = label;
    }
    return { stable: label, progress, commit };
  }
}

export class LandmarkClassifier {
  constructor(model, tf, featureVersion = null) {
    if (
      featureVersion
        ? model.feature_version !== featureVersion
        : ![
            FEATURE_VERSION,
            DEPTH_FEATURE_VERSION,
            WORLD_FEATURE_VERSION,
            ALIGN_FEATURE_VERSION,
            WORLD_ALIGN_FEATURE_VERSION,
          ].includes(model.feature_version)
    )
      throw new Error('Model feature version does not match this app.');
    this.data = model;
    this.tf = tf;
    this.mean = tf.tensor1d(model.mean);
    this.scale = tf.tensor1d(model.scale);
    this.layers = model.layers.map((l) => ({
      kernel: tf.tensor2d(l.kernel),
      bias: tf.tensor1d(l.bias),
    }));
  }
  predict(points, threshold, worldPoints = null) {
    let features;
    try {
      features = staticFeatures(points, worldPoints, this.data.feature_version);
    } catch {
      return { label: null, confidence: 0, candidate: null, reason: 'invalid_geometry' };
    }
    return this.predictFeatures(features, threshold);
  }
  probabilities(features) {
    // Double precision avoids cancellation in the exported kernel basis.
    if (this.data.kernel_basis) return denseProbabilities(this.data, features);
    return this.tf.tidy(() => {
      let x = this.tf.tensor2d([features]).sub(this.mean).div(this.scale);
      this.layers.forEach((layer, i) => {
        x = x.matMul(layer.kernel).add(layer.bias);
        if (i < this.layers.length - 1) x = x.relu();
      });
      return Array.from(x.softmax().dataSync());
    });
  }
  predictFeatures(features, threshold) {
    const probabilities = this.probabilities(features);
    return gatePrediction(probabilities, features, this.data, threshold);
  }
}

export function drawHand(canvas, points, { reference = false } = {}) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!points) return;
  let xy =
    points.length === 42
      ? Array.from({ length: 21 }, (_, i) => [points[i * 2], points[i * 2 + 1]])
      : points.map((v) => (Array.isArray(v) ? v : [v.x, v.y]));
  if (reference) {
    const normalized = normalize(points);
    xy = Array.from({ length: 21 }, (_, i) => [normalized[i * 2], normalized[i * 2 + 1]]);
    const xs = xy.map((p) => p[0]),
      ys = xy.map((p) => p[1]);
    const minX = Math.min(...xs),
      minY = Math.min(...ys);
    const scale = Math.min(
      (canvas.width - 40) / (Math.max(...xs) - minX),
      (canvas.height - 25) / (Math.max(...ys) - minY),
    );
    const ox = (canvas.width - (Math.max(...xs) - minX) * scale) / 2,
      oy = (canvas.height - (Math.max(...ys) - minY) * scale) / 2;
    xy = xy.map(([x, y]) => [ox + (x - minX) * scale, oy + (y - minY) * scale]);
  } else xy = xy.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = reference ? 3 : 3;
  ctx.strokeStyle = reference ? '#b4a3d7' : '#c5acef';
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(...xy[a]);
    ctx.lineTo(...xy[b]);
    ctx.stroke();
  }
  for (let i = 0; i < xy.length; i++) {
    ctx.beginPath();
    ctx.arc(...xy[i], reference ? 3.4 : 4, 0, Math.PI * 2);
    ctx.fillStyle = reference ? '#fcfaff' : '#f2ecff';
    ctx.fill();
    ctx.strokeStyle = '#8e75b7';
    ctx.lineWidth = reference ? 1.2 : 1.5;
    ctx.stroke();
  }
}
