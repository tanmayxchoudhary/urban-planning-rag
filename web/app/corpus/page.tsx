"use client";

import { useEffect, useState } from "react";

interface CorpusDocument {
  doc_hash: string;
  filename: string;
  family: string;
  jurisdiction: string;
  page_count: number;
  ingested_at: string;
}

interface CorpusResponse {
  corpus_version: string;
  indexed_at: string;
  documents: CorpusDocument[];
  totals: {
    documents: number;
    pages: number;
  };
}

export default function CorpusPage() {
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  useEffect(() => {
    async function fetchCorpus() {
      try {
        const response = await fetch("/v1/corpus");
        if (!response.ok) {
          throw new Error(`Failed to fetch corpus: ${response.status}`);
        }
        const data = await response.json();
        setCorpus(data);
        // Auto-expand all groups initially
        const groups = new Set<string>();
        data.documents.forEach((doc: CorpusDocument) => {
          const key = `${doc.family} — ${doc.jurisdiction}`;
          groups.add(key);
        });
        setExpandedGroups(groups);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    }
    fetchCorpus();
  }, []);

  const toggleGroup = (groupKey: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupKey)) {
        next.delete(groupKey);
      } else {
        next.add(groupKey);
      }
      return next;
    });
  };

  // Group documents by family + jurisdiction
  const groupedDocuments = () => {
    if (!corpus) return [];
    const groups: Record<string, CorpusDocument[]> = {};
    corpus.documents.forEach((doc) => {
      const key = `${doc.family} — ${doc.jurisdiction}`;
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(doc);
    });
    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 py-4">
          <div className="max-w-4xl mx-auto px-4">
            <h1 className="text-2xl font-bold text-gray-900">Corpus</h1>
            <p className="text-sm text-gray-500 mt-1">Document collection</p>
          </div>
        </header>
        <main className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex items-center gap-3">
            <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
            <span className="text-gray-600">Loading corpus...</span>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 py-4">
          <div className="max-w-4xl mx-auto px-4">
            <h1 className="text-2xl font-bold text-gray-900">Corpus</h1>
          </div>
        </header>
        <main className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <p className="text-red-800 font-medium mb-2">Error loading corpus</p>
            <p className="text-red-700">{error}</p>
          </div>
        </main>
      </div>
    );
  }

  if (!corpus) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-2xl font-bold text-gray-900">Corpus</h1>
          <p className="text-sm text-gray-500 mt-1">
            {corpus.totals.documents} documents, {corpus.totals.pages} pages
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-4 text-sm text-gray-500">
          Version: {corpus.corpus_version} · Indexed: {new Date(corpus.indexed_at).toLocaleDateString()}
        </div>

        <div className="space-y-4">
          {groupedDocuments().map(([groupKey, docs]) => {
            const isExpanded = expandedGroups.has(groupKey);
            return (
              <div key={groupKey} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                <button
                  onClick={() => toggleGroup(groupKey)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
                  aria-expanded={isExpanded}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg font-semibold text-gray-900">
                      {groupKey}
                    </span>
                    <span className="px-2.5 py-0.5 bg-gray-100 text-gray-700 text-sm rounded-full">
                      {docs.length} {docs.length === 1 ? "doc" : "docs"}
                    </span>
                  </div>
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {isExpanded && (
                  <div className="border-t border-gray-100">
                    <table className="w-full">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Document
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Pages
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Ingested
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {docs.map((doc) => (
                          <tr key={doc.doc_hash} className="hover:bg-gray-50">
                            <td className="px-6 py-4">
                              <span className="text-gray-900">{doc.filename}</span>
                            </td>
                            <td className="px-6 py-4 text-gray-600">
                              {doc.page_count}
                            </td>
                            <td className="px-6 py-4 text-gray-600">
                              {new Date(doc.ingested_at).toLocaleDateString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {corpus.documents.length === 0 && (
          <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
            <p className="text-gray-500">No documents in corpus yet.</p>
            <p className="text-sm text-gray-400 mt-1">
              Ingest documents using the CLI to see them here.
            </p>
          </div>
        )}
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
