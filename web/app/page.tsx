"use client";

import { useState, useEffect, useRef, FormEvent } from "react";
import { useStreamingQuery } from "@/hooks/useStreamingQuery";
import CitationChip from "@/components/CitationChip";
import { apiUrl } from "@/lib/api";

const EXAMPLE_QUESTIONS = [
  "What is the FAR for residential use in Mumbai?",
  "What are the parking requirements for commercial buildings?",
  "What is theFSI for residential zone in Delhi?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [corpusStats, setCorpusStats] = useState<{ documents: number; pages: number } | null>(null);
  const [corpusLoading, setCorpusLoading] = useState(true);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    isStreaming,
    answerText,
    candidates,
    error,
    refused,
    submitQuery,
  } = useStreamingQuery();

  // Fetch corpus stats on mount
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>;

    async function fetchCorpusStats() {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), 8000);

      try {
        const response = await fetch(apiUrl("/v1/corpus"), { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!response.ok) {
          throw new Error(`Failed to fetch corpus: ${response.status}`);
        }
        const data = await response.json();
        setCorpusStats(data.totals);
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          setCorpusError("Request timed out.");
        } else {
          setCorpusError((err as Error).message);
        }
      } finally {
        clearTimeout(timeoutId);
        setCorpusLoading(false);
      }
    }
    fetchCorpusStats();

    return () => {
      clearTimeout(timeoutId);
    };
  }, []);

  // Handle form submission
  const handleSubmit = async (e: FormEvent, mode: "fast" | "deep" = "fast") => {
    e.preventDefault();
    if (!question.trim() || isStreaming) return;
    await submitQuery(question.trim(), mode);
  };

  // Keyboard accessibility: Enter to submit
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-2xl font-bold text-gray-900">
            Urban Planning RAG
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Visual search over Indian urban planning regulations
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Query Bar */}
        <section aria-label="Query input">
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about urban planning regulations..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
                disabled={isStreaming}
                aria-label="Question input"
              />
              <button
                type="submit"
                disabled={isStreaming || !question.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {isStreaming ? "Searching..." : "Search"}
              </button>
            </div>
          </form>
        </section>

        {/* Example Questions */}
        <section aria-label="Example questions" className="mb-8">
          <p className="text-sm text-gray-500 mb-3">Try these examples:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((example, idx) => (
              <button
                key={idx}
                onClick={() => setQuestion(example)}
                disabled={isStreaming}
                className="px-3 py-1.5 text-sm bg-white border border-gray-200 rounded-full text-gray-700 hover:bg-gray-100 disabled:opacity-50 transition-colors"
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        {/* Streaming Status */}
        {isStreaming && (
          <section aria-label="Loading state" className="mb-8">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
                <span className="text-gray-600">Searching corpus...</span>
              </div>

              {/* Citation chips appear as soon as retrieval_completed arrives */}
              {candidates.length > 0 && (
                <div className="mb-4">
                  <p className="text-sm text-gray-500 mb-2">
                    Found {candidates.length} relevant pages:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {candidates.map((candidate, idx) => (
                      <CitationChip
                        key={candidate.page_id}
                        candidate={candidate}
                        index={idx + 1}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Answer streaming */}
              {answerText && (
                <div className="prose prose-gray max-w-none">
                  <div
                    className="text-gray-800 leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: answerText }}
                  />
                </div>
              )}
            </div>
          </section>
        )}

        {/* Refused state */}
        {refused && (
          <section aria-label="Query refused" className="mb-8">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
              <p className="text-yellow-800 font-medium mb-2">Query refused</p>
              <p className="text-yellow-700">{refused.message}</p>
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
                onClick={() => setQuestion("")}
                className="mt-4 px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200 transition-colors"
              >
                Try again
              </button>
            </div>
          </section>
        )}

        {/* Corpus stats strip */}
        <section aria-label="Corpus statistics" className="mt-12">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            {corpusLoading ? (
              <p className="text-sm text-gray-500">Loading corpus stats...</p>
            ) : corpusError ? (
              <p className="text-sm text-gray-500">Corpus stats unavailable</p>
            ) : corpusStats ? (
              <p className="text-sm text-gray-500">
                {corpusStats.documents} documents, {corpusStats.pages} pages indexed
              </p>
            ) : null}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-auto py-6 text-center text-sm text-gray-500">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline"
        >
          GitHub
        </a>
        {" | "}
        <a href="/about" className="hover:underline">
          About
        </a>
      </footer>
    </div>
  );
}
