/**
 * Centralized PWA back-navigation stack.
 *
 * iOS PWA quirk: `history.pushState({...}, "")` with the SAME URL doesn't
 * reliably enable the edge-swipe-back gesture in standalone mode. Safari's
 * gesture recognizer wants a URL that visibly differs from the current one
 * before it'll let the swipe register as a "go back" intent. So we push
 * with a hash like `#m1`, `#m2`, … which is enough URL-difference to
 * unlock the gesture without triggering Next.js route changes (Next.js
 * App Router ignores hash-only changes).
 */

type Handler = () => void;

const stack: Handler[] = [];
let skipNextPop = false;
let initialized = false;
let counter = 0;

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
 * Register a back handler and push a hashed history entry so the browser
 * has a state to pop when the user swipes back or hits the back button.
 *
 * @returns cleanup — call this when the modal closes via UI (not back gesture).
 *   It removes the handler and suppresses the resulting synthetic popstate so
 *   nothing below the current modal accidentally fires.
 */
export function pushBackHandler(fn: Handler): () => void {
  init();
  counter += 1;
  // Hash-only push: Next.js App Router does not re-render on hash changes,
  // so this is invisible to the rest of the app, but iOS PWA Safari treats
  // it as real navigable history and enables the edge-swipe gesture.
  const url = `${window.location.pathname}${window.location.search}#m${counter}`;
  window.history.pushState({ __cinematch_bs: stack.length + 1, c: counter }, "", url);
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
