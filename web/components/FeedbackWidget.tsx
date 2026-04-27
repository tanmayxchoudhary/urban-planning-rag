"use client";

import { useState, useRef, useCallback } from "react";

interface FeedbackWidgetProps {
  queryId: string;
  onFeedbackSubmitted?: (vote: "up" | "down", comment?: string) => void;
}

type Vote = "up" | "down";

export default function FeedbackWidget({
  queryId,
  onFeedbackSubmitted,
}: FeedbackWidgetProps) {
  const [vote, setVote] = useState<Vote | null>(null);
  const [comment, setComment] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const MAX_COMMENT_LENGTH = 200;

  const handleVote = useCallback(
    async (selectedVote: Vote) => {
      if (isSubmitting || isSubmitted) return;

      setVote(selectedVote);
      setIsSubmitting(true);
      setError(null);

      try {
        const response = await fetch("/v1/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query_id: queryId,
            vote: selectedVote,
            comment: comment.trim() || undefined,
          }),
        });

        if (response.status === 204) {
          setIsSubmitted(true);
          onFeedbackSubmitted?.(selectedVote, comment.trim() || undefined);
        } else {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.error?.message || `HTTP ${response.status}`
          );
        }
      } catch (err) {
        setError((err as Error).message);
        setVote(null);
        setIsSubmitting(false);
      }
    },
    [queryId, comment, isSubmitting, isSubmitted, onFeedbackSubmitted]
  );

  const handleCommentChange = (
    e: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    const value = e.target.value;
    // Truncate to MAX_COMMENT_LENGTH
    if (value.length <= MAX_COMMENT_LENGTH) {
      setComment(value);
    } else {
      setComment(value.slice(0, MAX_COMMENT_LENGTH));
    }
  };

  if (isSubmitted) {
    return (
      <div
        className="flex items-center gap-2 text-green-700 bg-green-50 px-4 py-3 rounded-lg border border-green-200"
        role="status"
        aria-live="polite"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>
        <span className="text-sm font-medium">
          Thank you for your feedback!
        </span>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <p className="text-sm text-gray-700 font-medium mb-3">
        Was this answer helpful?
      </p>

      <div className="flex gap-2 mb-3">
        {/* Thumbs up */}
        <button
          onClick={() => handleVote("up")}
          disabled={isSubmitting}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
            vote === "up"
              ? "bg-green-100 border-green-300 text-green-700"
              : "bg-white border-gray-300 text-gray-600 hover:bg-gray-50"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          aria-pressed={vote === "up"}
          aria-label="Thumbs up - helpful"
        >
          <svg
            className="w-5 h-5"
            fill={vote === "up" ? "currentColor" : "none"}
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.007L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"
            />
          </svg>
          <span className="text-sm font-medium">Yes</span>
        </button>

        {/* Thumbs down */}
        <button
          onClick={() => handleVote("down")}
          disabled={isSubmitting}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
            vote === "down"
              ? "bg-red-100 border-red-300 text-red-700"
              : "bg-white border-gray-300 text-gray-600 hover:bg-gray-50"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          aria-pressed={vote === "down"}
          aria-label="Thumbs down - not helpful"
        >
          <svg
            className="w-5 h-5"
            fill={vote === "down" ? "currentColor" : "none"}
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v2a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0 .714.211 1.412.608 2.007L17 17v9m-7-10h2m7-4H5a2 2 0 00-2 2v6a2 2 0 002 2h2.5"
            />
          </svg>
          <span className="text-sm font-medium">No</span>
        </button>
      </div>

      {/* Comment textarea */}
      <div className="mb-3">
        <label htmlFor="feedback-comment" className="sr-only">
          Optional feedback comment
        </label>
        <textarea
          ref={textareaRef}
          id="feedback-comment"
          value={comment}
          onChange={handleCommentChange}
          placeholder="Optional: Tell us more (max 200 characters)"
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          rows={3}
          aria-describedby="comment-count"
          disabled={isSubmitting}
        />
        <p
          id="comment-count"
          className="text-xs text-gray-500 text-right mt-1"
          aria-live="polite"
        >
          {comment.length}/{MAX_COMMENT_LENGTH}
        </p>
      </div>

      {/* Submit button */}
      {vote && !isSubmitted && (
        <button
          onClick={() => handleVote(vote)}
          disabled={isSubmitting}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? "Submitting..." : "Submit feedback"}
        </button>
      )}

      {/* Error message */}
      {error && (
        <p className="text-sm text-red-600 mt-2" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
