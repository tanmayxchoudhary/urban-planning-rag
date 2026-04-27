"use client";

import { useRef, useCallback } from "react";
import { CitationCandidate } from "@/lib/sse-types";

interface CitationChipProps {
  candidate: CitationCandidate;
  index: number;
  onClick?: (index: number) => void;
}

export default function CitationChip({ candidate, index, onClick }: CitationChipProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleClick = useCallback(() => {
    onClick?.(index);
  }, [onClick, index]);

  // Handle keyboard activation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <button
      ref={buttonRef}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-colors text-left min-w-0 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      aria-label={`Citation ${index}: ${candidate.section_title || candidate.page_id}`}
    >
      <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center text-sm font-medium">
        {index}
      </span>
      <span className="truncate text-sm text-gray-700">
        {candidate.section_title || candidate.page_id}
      </span>
      {candidate.score > 0 && (
        <span className="text-xs text-gray-400 flex-shrink-0">
          {candidate.score.toFixed(2)}
        </span>
      )}
    </button>
  );
}
