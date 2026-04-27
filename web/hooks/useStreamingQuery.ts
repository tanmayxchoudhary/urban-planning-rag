"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  StreamEvent,
  consumeSSEStream,
  CitationCandidate,
  Citation,
  GenerationCompletedEvent,
} from "@/lib/sse-types";

interface UseStreamingQueryOptions {
  onRetrievalCompleted?: (candidates: CitationCandidate[]) => void;
  onToken?: (text: string) => void;
  onGenerationCompleted?: (event: GenerationCompletedEvent) => void;
  onError?: (message: string) => void;
  onRefused?: (reason: string, message: string) => void;
  onDone?: () => void;
}

interface StreamingQueryState {
  isStreaming: boolean;
  answerText: string;
  citations: Citation[];
  candidates: CitationCandidate[];
  error: string | null;
  refused: { reason: string; message: string } | null;
  queryId: string | null;
}

export function useStreamingQuery(options: UseStreamingQueryOptions = {}) {
  const router = useRouter();
  const [state, setState] = useState<StreamingQueryState>({
    isStreaming: false,
    answerText: "",
    citations: [],
    candidates: [],
    error: null,
    refused: null,
    queryId: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const submitQuery = useCallback(
    async (question: string, mode: "fast" | "deep" = "fast") => {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      // Reset state
      setState({
        isStreaming: true,
        answerText: "",
        citations: [],
        candidates: [],
        error: null,
        refused: null,
        queryId: null,
      });

      try {
        // Submit query
        const submitResponse = await fetch("/v1/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, mode }),
          signal: abortControllerRef.current.signal,
        });

        if (!submitResponse.ok) {
          const errorData = await submitResponse.json();
          throw new Error(errorData.error?.message || `HTTP ${submitResponse.status}`);
        }

        const { query_id, stream_url } = await submitResponse.json();

        setState((prev) => ({ ...prev, queryId: query_id }));

        // Navigate to query page (defer to after render to avoid React setState-during-render error)
        queueMicrotask(() => router.push(`/q/${query_id}`));

        // Connect to SSE stream
        const streamResponse = await fetch(stream_url, {
          signal: abortControllerRef.current.signal,
        });

        if (!streamResponse.ok) {
          throw new Error(`Stream error: HTTP ${streamResponse.status}`);
        }

        // Consume SSE events
        for await (const event of consumeSSEStream(streamResponse)) {
          switch (event.type) {
            case "retrieval_completed":
              setState((prev) => ({
                ...prev,
                candidates: event.data.candidates,
              }));
              options.onRetrievalCompleted?.(event.data.candidates);
              break;

            case "token":
              setState((prev) => ({
                ...prev,
                answerText: prev.answerText + event.data.text,
              }));
              options.onToken?.(event.data.text);
              break;

            case "generation_completed":
              setState((prev) => ({
                ...prev,
                citations: event.data.citations,
              }));
              options.onGenerationCompleted?.(event.data);
              break;

            case "error":
              setState((prev) => ({
                ...prev,
                error: event.data.message,
              }));
              options.onError?.(event.data.message);
              break;

            case "refused":
              setState((prev) => ({
                ...prev,
                refused: { reason: event.data.reason, message: event.data.message },
              }));
              options.onRefused?.(event.data.reason, event.data.message);
              break;

            case "done":
              // Store result in localStorage for reload persistence (VAL-WEB-008)
              if (state.queryId) {
                try {
                  localStorage.setItem(
                    query_id,
                    JSON.stringify({
                      question,
                      answerText: state.answerText,
                      citations: state.citations,
                      candidates: state.candidates,
                      mode,
                    })
                  );
                } catch {
                  // localStorage might be full or unavailable
                }
              }
              options.onDone?.();
              break;
          }
        }

        setState((prev) => ({ ...prev, isStreaming: false }));
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          // Request was cancelled, don't show error
          return;
        }
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          error: (err as Error).message,
        }));
        options.onError?.((err as Error).message);
      }
    },
    [router, options, state.queryId, state.answerText, state.citations, state.candidates]
  );

  const cancelQuery = useCallback(() => {
    abortControllerRef.current?.abort();
    setState((prev) => ({ ...prev, isStreaming: false }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return {
    ...state,
    submitQuery,
    cancelQuery,
  };
}
