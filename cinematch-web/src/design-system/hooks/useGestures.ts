"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useAnimationFrame } from "framer-motion";

export interface SwipeDirection {
  x: number;
  y: number;
  direction: "left" | "right" | "up" | "down" | null;
  distance: number;
  velocity: number;
}

export interface SwipeConfig {
  threshold?: number;
  velocityThreshold?: number;
  preventScroll?: boolean;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onSwipe?: (direction: SwipeDirection) => void;
}

export function useSwipe(config: SwipeConfig = {}) {
  const {
    threshold = 50,
    velocityThreshold = 0.3,
    preventScroll = false,
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onSwipe,
  } = config;

  const [swipeState, setSwipeState] = useState<SwipeDirection>({
    x: 0,
    y: 0,
    direction: null,
    distance: 0,
    velocity: 0,
  });

  const startX = useRef(0);
  const startY = useRef(0);
  const startTime = useRef(0);
  const elementRef = useRef<HTMLElement>(null);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0];
    startX.current = touch.clientX;
    startY.current = touch.clientY;
    startTime.current = Date.now();
    setSwipeState({ x: 0, y: 0, direction: null, distance: 0, velocity: 0 });
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (preventScroll) e.preventDefault();
    const touch = e.touches[0];
    const x = touch.clientX - startX.current;
    const y = touch.clientY - startY.current;
    const distance = Math.sqrt(x * x + y * y);

    let direction: SwipeDirection["direction"] = null;
    if (distance > threshold) {
      if (Math.abs(x) > Math.abs(y)) {
        direction = x > 0 ? "right" : "left";
      } else {
        direction = y > 0 ? "down" : "up";
      }
    }

    setSwipeState({ x, y, direction, distance, velocity: 0 });
  }, [preventScroll, threshold]);

  const handleTouchEnd = useCallback(() => {
    const { x, y, direction, distance } = swipeState;
    const time = Date.now() - startTime.current;
    const velocity = distance / time;

    if (velocity >= velocityThreshold && direction) {
      setSwipeState((prev) => ({ ...prev, velocity }));
      onSwipe?.({ x, y, direction, distance, velocity });
      switch (direction) {
        case "left":
          onSwipeLeft?.();
          break;
        case "right":
          onSwipeRight?.();
          break;
        case "up":
          onSwipeUp?.();
          break;
        case "down":
          onSwipeDown?.();
          break;
      }
    }

    setSwipeState({ x: 0, y: 0, direction: null, distance: 0, velocity: 0 });
  }, [swipeState, velocityThreshold, onSwipe, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    startX.current = e.clientX;
    startY.current = e.clientY;
    startTime.current = Date.now();
    setSwipeState({ x: 0, y: 0, direction: null, distance: 0, velocity: 0 });
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    const x = e.clientX - startX.current;
    const y = e.clientY - startY.current;
    const distance = Math.sqrt(x * x + y * y);

    let direction: SwipeDirection["direction"] = null;
    if (distance > threshold) {
      if (Math.abs(x) > Math.abs(y)) {
        direction = x > 0 ? "right" : "left";
      } else {
        direction = y > 0 ? "down" : "up";
      }
    }

    setSwipeState({ x, y, direction, distance, velocity: 0 });
  }, [threshold]);

  const handleMouseUp = useCallback(() => {
    const { x, y, direction, distance } = swipeState;
    const time = Date.now() - startTime.current;
    const velocity = distance / time;

    if (velocity >= velocityThreshold && direction) {
      setSwipeState((prev) => ({ ...prev, velocity }));
      onSwipe?.({ x, y, direction, distance, velocity });
      switch (direction) {
        case "left":
          onSwipeLeft?.();
          break;
        case "right":
          onSwipeRight?.();
          break;
        case "up":
          onSwipeUp?.();
          break;
        case "down":
          onSwipeDown?.();
          break;
      }
    }

    setSwipeState({ x: 0, y: 0, direction: null, distance: 0, velocity: 0 });
  }, [swipeState, velocityThreshold, onSwipe, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown]);

  return {
    ref: elementRef,
    swipeState,
    handlers: {
      onTouchStart: handleTouchStart,
      onTouchMove: handleTouchMove,
      onTouchEnd: handleTouchEnd,
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
    },
  };
}

export interface DragState {
  x: number;
  y: number;
  isDragging: boolean;
  velocityX: number;
  velocityY: number;
}

export interface DragConfig {
  axis?: "x" | "y" | "both";
  bounds?: { left?: number; right?: number; top?: number; bottom?: number };
  elastic?: number;
  momentum?: boolean;
  onDragStart?: () => void;
  onDrag?: (state: DragState) => void;
  onDragEnd?: (state: DragState) => void;
}

export function useDrag(config: DragConfig = {}) {
  const {
    axis = "both",
    bounds,
    elastic = 0.5,
    momentum = true,
    onDragStart,
    onDrag,
    onDragEnd,
  } = config;

  const [dragState, setDragState] = useState<DragState>({
    x: 0,
    y: 0,
    isDragging: false,
    velocityX: 0,
    velocityY: 0,
  });

  const startX = useRef(0);
  const startY = useRef(0);
  const lastX = useRef(0);
  const lastY = useRef(0);
  const lastTime = useRef(0);
  const elementRef = useRef<HTMLElement>(null);
  const animationFrame = useRef<number | null>(null);

  const handleDragStart = useCallback((clientX: number, clientY: number) => {
    startX.current = clientX;
    startY.current = clientY;
    lastX.current = clientX;
    lastY.current = clientY;
    lastTime.current = Date.now();
    setDragState((prev) => ({ ...prev, isDragging: true }));
    onDragStart?.();
  }, [onDragStart]);

  const handleDragMove = useCallback((clientX: number, clientY: number) => {
    let x = clientX - startX.current;
    let y = clientY - startY.current;

    const now = Date.now();
    const dt = now - lastTime.current;
    if (dt > 0) {
      const vx = (clientX - lastX.current) / dt;
      const vy = (clientY - lastY.current) / dt;
      setDragState((prev) => ({ ...prev, velocityX: vx, velocityY: vy }));
    }

    lastX.current = clientX;
    lastY.current = clientY;
    lastTime.current = now;

    if (axis !== "y") {
      if (bounds?.left !== undefined && x < bounds.left) {
        x = bounds.left + (x - bounds.left) * elastic;
      }
      if (bounds?.right !== undefined && x > bounds.right) {
        x = bounds.right + (x - bounds.right) * elastic;
      }
    }

    if (axis !== "x") {
      if (bounds?.top !== undefined && y < bounds.top) {
        y = bounds.top + (y - bounds.top) * elastic;
      }
      if (bounds?.bottom !== undefined && y > bounds.bottom) {
        y = bounds.bottom + (y - bounds.bottom) * elastic;
      }
    }

    setDragState((prev) => ({ ...prev, x, y }));
    onDrag?.({ x, y, isDragging: true, velocityX: 0, velocityY: 0 });
  }, [axis, bounds, elastic, onDrag]);

  const handleDragEnd = useCallback(() => {
    const { x, y, velocityX, velocityY } = dragState;
    setDragState((prev) => ({ ...prev, isDragging: false }));

    if (momentum && (Math.abs(velocityX) > 0.01 || Math.abs(velocityY) > 0.01)) {
      let finalX = x;
      let finalY = y;
      let vx = velocityX;
      let vy = velocityY;

      const animate = () => {
        finalX += vx * 16;
        finalY += vy * 16;
        vx *= 0.95;
        vy *= 0.95;

        if (bounds) {
          if (bounds.left !== undefined && finalX < bounds.left) finalX = bounds.left;
          if (bounds.right !== undefined && finalX > bounds.right) finalX = bounds.right;
          if (bounds.top !== undefined && finalY < bounds.top) finalY = bounds.top;
          if (bounds.bottom !== undefined && finalY > bounds.bottom) finalY = bounds.bottom;
        }

        setDragState((prev) => ({ ...prev, x: finalX, y: finalY }));

        if (Math.abs(vx) > 0.01 || Math.abs(vy) > 0.01) {
          animationFrame.current = requestAnimationFrame(animate);
        } else {
          onDragEnd?.({ x: finalX, y: finalY, isDragging: false, velocityX: 0, velocityY: 0 });
        }
      };

      animationFrame.current = requestAnimationFrame(animate);
    } else {
      onDragEnd?.({ x, y, isDragging: false, velocityX, velocityY });
    }
  }, [dragState, bounds, momentum, onDragEnd]);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    handleDragStart(e.touches[0].clientX, e.touches[0].clientY);
  }, [handleDragStart]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    handleDragMove(e.touches[0].clientX, e.touches[0].clientY);
  }, [handleDragMove]);

  const handleTouchEnd = useCallback(() => {
    handleDragEnd();
  }, [handleDragEnd]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    handleDragStart(e.clientX, e.clientY);
  }, [handleDragStart]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    handleDragMove(e.clientX, e.clientY);
  }, [handleDragMove]);

  const handleMouseUp = useCallback(() => {
    handleDragEnd();
  }, [handleDragEnd]);

  useEffect(() => {
    return () => {
      if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
    };
  }, []);

  return {
    ref: elementRef,
    dragState,
    handlers: {
      onTouchStart: handleTouchStart,
      onTouchMove: handleTouchMove,
      onTouchEnd: handleTouchEnd,
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
    },
  };
}

export function useSpring(initialValue: number, config: { stiffness?: number; damping?: number; mass?: number } = {}) {
  const { stiffness = 300, damping = 20, mass = 1 } = config;
  const [value, setValue] = useState(initialValue);
  const velocity = useRef(0);
  const target = useRef(initialValue);
  const frame = useRef<number | null>(null);
  const animateRef = useRef<() => void>(() => {});

  const animate = useCallback(() => {
    const displacement = target.current - value;
    const springForce = stiffness * displacement;
    const dampingForce = damping * velocity.current;
    const acceleration = (springForce - dampingForce) / mass;
    velocity.current += acceleration * 0.016;
    setValue((v) => {
      const newValue = v + velocity.current * 0.016;
      if (Math.abs(target.current - newValue) < 0.01 && Math.abs(velocity.current) < 0.01) {
        if (frame.current) cancelAnimationFrame(frame.current);
        return target.current;
      }
      // Use ref to avoid circular declaration warning
      frame.current = requestAnimationFrame(() => animateRef.current());
      return newValue;
    });
  }, [stiffness, damping, mass, value]);

  // Keep ref in sync
  useEffect(() => {
    animateRef.current = animate;
  }, [animate]);

  const setTarget = useCallback((newTarget: number) => {
    target.current = newTarget;
    if (frame.current) cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(animate);
  }, [animate]);

  useEffect(() => {
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current);
    };
  }, []);

  return [value, setTarget] as const;
}

export function useParallax(speed: number = 0.5) {
  const scrollY = useRef(0);
  const [offset, setOffset] = useState(0);

  useAnimationFrame(() => {
    if (typeof window === "undefined") return;
    const currentScrollY = window.scrollY;
    const diff = currentScrollY - scrollY.current;
    scrollY.current = currentScrollY;
    setOffset((prev) => prev + diff * speed);
  });

  return offset;
}

export function useScrollProgress() {
  const [progress, setProgress] = useState(0);

  useAnimationFrame(() => {
    if (typeof window === "undefined") return;
    const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (scrollHeight <= 0) return;
    setProgress(Math.min(window.scrollY / scrollHeight, 1));
  });

  return progress;
}

export function useElementSize(ref: React.RefObject<HTMLElement>) {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(ref.current);
    setSize({ width: ref.current.offsetWidth, height: ref.current.offsetHeight });
    return () => observer.disconnect();
  }, [ref]);

  return size;
}