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
