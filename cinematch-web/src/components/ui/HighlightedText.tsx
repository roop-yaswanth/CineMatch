"use client";

/**
 * Renders text with the user's search term emphasised. Case-insensitive
 * substring match on the literal query — no regex, no smart tokenization,
 * just enough polish to make results feel responsive.
 */

import React from "react";

interface Props {
  text: string;
  query: string;
  className?: string;
}

export default function HighlightedText({ text, query, className }: Props) {
  if (!query) return <span className={className}>{text}</span>;
  const q = query.trim();
  if (!q) return <span className={className}>{text}</span>;

  const lowerText = text.toLowerCase();
  const lowerQuery = q.toLowerCase();
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  while (cursor < text.length) {
    const idx = lowerText.indexOf(lowerQuery, cursor);
    if (idx === -1) {
      parts.push(text.slice(cursor));
      break;
    }
    if (idx > cursor) parts.push(text.slice(cursor, idx));
    parts.push(
      <mark
        key={`m${key++}`}
        style={{ background: "rgba(255, 230, 120, 0.22)", color: "inherit", padding: 0, borderRadius: 2 }}
      >
        {text.slice(idx, idx + q.length)}
      </mark>
    );
    cursor = idx + q.length;
  }

  return <span className={className}>{parts}</span>;
}
