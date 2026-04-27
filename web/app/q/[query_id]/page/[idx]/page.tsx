"use client";

import { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import CitationLightbox from "@/components/CitationLightbox";
import FeedbackWidget from "@/components/FeedbackWidget";
import { Citation, CitationCandidate } from "@/lib/sse-types";

interface QueryPageProps {
  params: Promise<{ query_id: string; idx: string }>;
}

interface CachedResult {
  question: string;
  answerText: string;
  citations: Citation[];
  candidates: CitationCandidate[];
  mode: "fast" | "deep";
}

export default function CitationPage({ params }: QueryPageProps) {
  const { query_id, idx } = use(params);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [cachedResult, setCachedResult] = useState<CachedResult | null>(null);
  const [selectedCitationIdx, setSelectedCitationIdx] = useState<number | null>(
    null
  );

  // Try to load from localStorage first
  useEffect(() => {
    try {
      const cached = localStorage.getItem(query_id);
      if (cached) {
        const parsed = JSON.parse(cached) as CachedResult;
        setCachedResult(parsed);
      }
    } catch {
      // localStorage not available or invalid data
    }
    setIsLoading(false);
  }, [query_id]);

  // If we have a cached result, show citation info + lightbox
  const citations = cachedResult?.citations || [];
  const candidateIndex = parseInt(idx, 10) - 1;
  const selectedCitation = citations[candidateIndex];

  const handleOpenLightbox = (index: number) => {
    setSelectedCitationIdx(index);
  };

  const handleCloseLightbox = () => {
    setSelectedCitationIdx(null);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!cachedResult || !selectedCitation) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 py-4">
          <div className="max-w-4xl mx-auto px-4">
            <button
              onClick={() => router.push("/")}
              className="text-gray-600 hover:text-gray-900 flex items-center gap-2"
            >
              <span>←</span>
              <span>Back to search</span>
            </button>
          </div>
        </header>
        <main className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <p className="text-yellow-800 font-medium">
              Citation not found
            </p>
            <p className="text-yellow-700 mt-2">
              The requested citation does not exist or has expired.
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4 flex items-center justify-between">
          <button
            onClick={() => router.push(`/q/${query_id}`)}
            className="text-gray-600 hover:text-gray-900 flex items-center gap-2"
          >
            <span>←</span>
            <span>Back to answer</span>
          </button>
          <span className="text-sm text-gray-500 font-mono">{query_id}</span>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Citation lightbox */}
        {selectedCitationIdx !== null && (
          <CitationLightbox
            citation={citations[selectedCitationIdx]}
            index={selectedCitationIdx + 1}
            isOpen={true}
            onClose={handleCloseLightbox}
          />
        )}

        {/* Citation list */}
        <section aria-label="Citations" className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Sources ({citations.length})
          </h2>
          <div className="space-y-4">
            {citations.map((citation, cidx) => (
              <div
                key={citation.page_id}
                className="bg-white rounded-lg border border-gray-200 p-4"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">
                      [{cidx + 1}] {citation.doc_filename}, p.{citation.page_num}
                    </p>
                    {citation.section_path.length > 0 && (
                      <p className="text-sm text-gray-500 mt-1">
                        {citation.section_path.join(" > ")}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={() => handleOpenLightbox(cidx)}
                    className="ml-4 px-3 py-1.5 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                    aria-label={`View page ${citation.page_num} of ${citation.doc_filename}`}
                  >
                    View page
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Feedback widget */}
        <section aria-label="Feedback" className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Feedback
          </h2>
          <FeedbackWidget queryId={query_id} />
        </section>

        {/* Back button */}
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Ask another question
        </button>
      </main>
    </div>
  );
}
