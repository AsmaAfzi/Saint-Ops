const SENSOR_NAME = "AVG_ANNULUS_PRESS";

const FALLBACK_ENV = [
  "AVG_DOWNHOLE_PRESSURE",
  "AVG_DOWNHOLE_TEMPERATURE",
  "BORE_OIL_VOL",
  "AVG_WHP_P",
];
const FALLBACK_SENSOR = [SENSOR_NAME];

/** Derive env/sensor feature groups from API feature list or stream info. */
export function getFeatureGroups(featureNames) {
  if (!featureNames?.length) {
    return { envFeatures: FALLBACK_ENV, sensorFeatures: FALLBACK_SENSOR };
  }

  const sensorFeatures = featureNames.filter((f) => f === SENSOR_NAME);
  const envFeatures = featureNames.filter((f) => f !== SENSOR_NAME);

  if (!sensorFeatures.length && featureNames.length) {
    sensorFeatures.push(featureNames[featureNames.length - 1]);
    return {
      envFeatures: featureNames.slice(0, -1),
      sensorFeatures,
    };
  }

  return {
    envFeatures: envFeatures.length ? envFeatures : FALLBACK_ENV,
    sensorFeatures: sensorFeatures.length ? sensorFeatures : FALLBACK_SENSOR,
  };
}

export function filterExplanation(explanation, driftType, groups) {
  if (!explanation?.length) return [];
  const list =
    driftType === "Sensor Drift" ? groups.sensorFeatures : groups.envFeatures;
  return explanation.filter((exp) => list.includes(exp.feature_name));
}

/** Map env/sensor feature names to column indices in parallel API arrays. */
export function getChannelIndices(featureNames) {
  if (!featureNames?.length) {
    return { sensorIdx: -1, envIdxs: [] };
  }
  const groups = getFeatureGroups(featureNames);
  const sensorIdx = featureNames.indexOf(groups.sensorFeatures[0]);
  const envIdxs = groups.envFeatures
    .map((name) => featureNames.indexOf(name))
    .filter((idx) => idx >= 0);
  return { sensorIdx, envIdxs };
}

export function aggregateChannelValues(values, indices, reducer = "mean") {
  if (!values?.length || !indices?.length) return 0;
  const picked = indices
    .map((i) => values[i])
    .filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (!picked.length) return 0;
  if (reducer === "max") return Math.max(...picked);
  return picked.reduce((sum, v) => sum + v, 0) / picked.length;
}

/** v8 ranking: CUSUM S+ primary, then fixed norm error, then raw reconstruction error. */
export function explanationScore(exp) {
  if (exp?.cusum != null && !Number.isNaN(exp.cusum)) return exp.cusum;
  if (exp?.norm_error != null && !Number.isNaN(exp.norm_error)) return exp.norm_error;
  return exp?.error ?? 0;
}

export function sortExplanationsByDetection(explanations) {
  return [...(explanations ?? [])].sort(
    (a, b) => explanationScore(b) - explanationScore(a)
  );
}

export function formatTopFeatures(explanations, topK) {
  return sortExplanationsByDetection(explanations)
    .slice(0, topK)
    .map((f) => {
      const bits = [f.feature_name];
      if (f.norm_error != null) bits.push(`norm ${f.norm_error.toFixed(2)}x`);
      if (f.cusum != null) bits.push(`S+ ${f.cusum.toFixed(1)}`);
      else bits.push(`err ${f.error.toFixed(4)}`);
      return bits.join(": ");
    })
    .join(" · ");
}

export const CHART_METRICS = {
  raw: { label: "Reconstruction error", yTitle: "Error", key: "raw" },
  norm: { label: "Normalised error (val_p99)", yTitle: "Norm error (×)", key: "norm" },
  cusum: { label: "CUSUM S+", yTitle: "CUSUM S+", key: "cusum" },
};

export function formatDetectorConfig(config) {
  if (!config) return "—";
  const parts = [];
  if (config.cusum_k != null) parts.push(`k=${config.cusum_k}`);
  if (config.cusum_h != null) parts.push(`h=${config.cusum_h}`);
  if (config.normalisation) parts.push(config.normalisation);
  return parts.length ? parts.join(", ") : "—";
}
