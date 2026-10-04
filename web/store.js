// Client store: the whole UI state is one frame index, a set of layer booleans and a selection.
const listeners = new Set();

export const state = {
  clipId: null,
  frame: 0, // index into the run's frame array, not the clip frame number
  playing: false,
  rate: 1, // playback speed
  layers: { skeleton: true, board: true, com: true, bbox: true, matEdge: true },
  loop: true, // playback wraps around by default
  view: "timeline", // "timeline" | "compare"
};

export function set(patch) {
  Object.assign(state, patch);
  for (const fn of listeners) fn(state, patch);
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// Speeds are stored in m/s (SI, also in the exports); everything on screen shows km/h.
export function toKmh(series, units) {
  const s = series?.speed_along;
  if (!s || units.speed_along === "km/h") return;
  s.value = s.value.map((v) => (v == null ? v : v * 3.6));
  s.err = s.err.map((v) => (v == null ? v : v * 3.6));
  units.speed_along = "km/h";
}
