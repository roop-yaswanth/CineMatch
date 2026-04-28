/**
 * Centralized PWA back-navigation stack.
 */

type Handler = () => void;

const stack: Handler[] = [];
let skipNextPop = false;
let initialized = false;

function onPop(): void {
  if (skipNextPop) {
    skipNextPop = false;
    return;
  }
  const handler = stack[stack.length - 1];
  if (handler) {
    stack.pop();
    handler();
  }
}

function init(): void {
  if (initialized || typeof window === "undefined") return;
  window.addEventListener("popstate", onPop);
  initialized = true;
}

/**
 * Register a back handler and push a fake history entry so the browser
 * has a state to pop when the user swipes back or hits the back button.
 *
 * @returns cleanup — call this when the modal closes via UI (not back gesture).
 *   It removes the handler and suppresses the resulting synthetic popstate so
 *   nothing below the current modal accidentally fires.
 */
export function pushBackHandler(fn: Handler): () => void {
  init();
  window.history.pushState({ __cinematch_bs: stack.length + 1 }, "");
  stack.push(fn);

  let cleaned = false;
  return function cleanup(): void {
    if (cleaned) return;
    cleaned = true;

    const idx = stack.lastIndexOf(fn);
    if (idx === -1) return; // already fired by back gesture — nothing to do

    stack.splice(idx, 1);
    // Suppress the popstate that history.back() is about to fire
    skipNextPop = true;
    window.history.back();
  };
}
