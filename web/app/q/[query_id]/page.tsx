"use client";

import { useState, useEffect, use, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useStreamingQuery } from "@/hooks/useStreamingQuery";
import CitationChip from "@/components/CitationChip";
import CitationLightbox from "@/components/CitationLightbox";
import FeedbackWidget from "@/components/FeedbackWidget";
import { Citation, CitationCandidate } from "@/lib/sse-types";

interface QueryPageProps {
  params: Promise<{ query_id: string }>;
}

interface CachedResult {
  question: string;
  answerText: string;
  citations: Citation[];
  candidates: CitationCandidate[];
  mode: "fast" | "deep";
}

export default function QueryPage({ params }: QueryPageProps) {
  const { query_id } = use(params);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [cachedResult, setCachedResult] = useState<CachedResult | null>(null);
  const [question, setQuestion] = useState("");
  const [selectedCitationIdx, setSelectedCitationIdx] = useState<number | null>(null);
  const chipRefs = useRef<Record<number, HTMLButtonElement | null>>({});

  // Try to load from localStorage first (VAL-WEB-008)
  useEffect(() => {
    try {
      const cached = localStorage.getItem(query_id);
      if (cached) {
        const parsed = JSON.parse(cached) as CachedResult;
        setCachedResult(parsed);
        setQuestion(parsed.question);
      }
    } catch {
      // localStorage not available or invalid data
    }
    setIsLoading(false);
  }, [query_id]);

  const {
    isStreaming,
    answerText,
    citations,
    candidates,
    error,
    refused,
  } = useStreamingQuery({
    onDone: () => {
      // Stream complete
    },
  });

  // If we have a cached result, show it immediately without re-querying
  const displayText = cachedResult
    ? cachedResult.answerText
    : answerText;
  const displayCitations = cachedResult
    ? cachedResult.citations
    : citations;
  const displayCandidates = cachedResult
    ? cachedResult.candidates
    : candidates;

  // Lightbox open/close handlers
  const handleOpenLightbox = useCallback((idx: number) => {
    setSelectedCitationIdx(idx);
  }, []);

  const handleCloseLightbox = useCallback(() => {
    setSelectedCitationIdx(null);
  }, []);

  // Keyboard accessibility
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        router.push("/");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4 flex items-center justify-between">
          <button
            onClick={() => router.push("/")}
            className="text-gray-600 hover:text-gray-900 flex items-center gap-2"
          >
            <span>←</span>
            <span>Back to search</span>
          </button>
          <span className="text-sm text-gray-500 font-mono">{query_id}</span>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Question */}
        <section aria-label="Question" className="mb-6">
          <h1 className="text-xl font-semibold text-gray-900">
            {question || cachedResult?.question || "Loading..."}
          </h1>
        </section>

        {/* Lightbox */}
        {selectedCitationIdx !== null && displayCitations[selectedCitationIdx] && (
          <CitationLightbox
            citation={displayCitations[selectedCitationIdx]}
            index={selectedCitationIdx + 1}
            isOpen={true}
            onClose={handleCloseLightbox}
            triggerRef={{ current: chipRefs.current[selectedCitationIdx] } as React.RefObject<HTMLElement>}
          />
        )}

        {/* Results */}
        {isStreaming && !cachedResult && (
          <section aria-label="Streaming results" className="mb-8">
            {/* Citation chips - appear after retrieval_completed (VAL-WEB-005) */}
            {displayCandidates.length > 0 && (
              <div className="mb-6">
                <p className="text-sm text-gray-500 mb-3">
                  Found {displayCandidates.length} relevant pages:
                </p>
                <div className="flex flex-wrap gap-2">
                  {displayCandidates.map((candidate, idx) => (
                    <CitationChip
                      key={candidate.page_id}
                      candidate={candidate}
                      index={idx + 1}
                      onClick={() => handleOpenLightbox(idx)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Skeleton while retrieving */}
            {displayCandidates.length === 0 && (
              <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
                <div className="flex items-center gap-3">
                  <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
                  <span className="text-gray-600">Retrieving relevant pages...</span>
                </div>
              </div>
            )}

            {/* Answer text streaming incrementally (VAL-WEB-006) */}
            {displayText && (
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <div className="prose prose-gray max-w-none">
                  <div
                    className="text-gray-800 leading-relaxed whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{ __html: displayText }}
                  />
                  <span className="animate-pulse">▌</span>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Completed answer */}
        {(displayText || cachedResult) && !isStreaming && (
          <section aria-label="Answer" className="mb-8">
            {/* Citation chips (VAL-WEB-005) */}
            {displayCitations.length > 0 && (
              <div className="mb-6">
                <p className="text-sm text-gray-500 mb-3">
                  Sources ({displayCitations.length}):
                </p>
                <div className="flex flex-wrap gap-2">
                  {displayCitations.map((citation, idx) => (
                    <CitationChip
                      key={citation.page_id}
                      candidate={{
                        page_id: citation.page_id,
                        score: citation.rerank_score,
                        channel_scores: {},
                        page_image_uri: citation.image_uri,
                        extracted_text_excerpt: "",
                        section_title: citation.section_path.join(" > ") || null,
                      }}
                      index={idx + 1}
                      onClick={() => handleOpenLightbox(idx)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Answer text */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="prose prose-gray max-w-none">
                <div
                  className="text-gray-800 leading-relaxed whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: displayText }}
                />
              </div>
            </div>

            {/* Feedback widget (VAL-WEB-014, VAL-WEB-015) */}
            <div className="mt-6">
              <FeedbackWidget queryId={query_id} />
            </div>

            {/* Actions */}
            <div className="mt-6 flex gap-4">
              <button
                onClick={() => router.push("/")}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Ask another question
              </button>
            </div>
          </section>
        )}

        {/* Refused state */}
        {refused && (
          <section aria-label="Query refused" className="mb-8">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
              <p className="text-yellow-800 font-medium mb-2">Query Refused</p>
              <p className="text-yellow-700">{refused.message}</p>
              <button
                onClick={() => router.push("/")}
                className="mt-4 px-4 py-2 bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200 transition-colors"
              >
                Try a different question
              </button>
            </div>
          </section>
        )}

        {/* Error state */}
        {error && (
          <section aria-label="Error state" className="mb-8">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <p className="text-red-800 font-medium mb-2">Error</p>
              <p className="text-red-700">{error}</p>
              <button
                onClick={() => router.push("/")}
                className="mt-4 px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200 transition-colors"
              >
                Try again
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
