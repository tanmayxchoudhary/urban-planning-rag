/**
 * SSE event types matching the API contract (VAL-API-007)
 */

export interface RetrievalCompletedEvent {
  query_id: string;
  candidates: CitationCandidate[];
  latency_ms: number;
}

export interface CitationCandidate {
  page_id: string;
  score: number;
  channel_scores: Record<string, number>;
  page_image_uri: string;
  extracted_text_excerpt: string;
  section_title: string | null;
}

export interface TokenEvent {
  text: string;
}

export interface GenerationCompletedEvent {
  answer_markdown: string;
  citations: Citation[];
  confidence: "high" | "medium" | "low";
  diagnostics: Diagnostics;
  query_id: string;
}

export interface Citation {
  page_id: string;
  doc_hash: string;
  doc_filename: string;
  page_num: number;
  section_path: string[];
  image_uri: string;
  rerank_score: number;
}

export interface Diagnostics {
  latency_ms: {
    encode: number;
    retrieve: number;
    rerank: number;
    generate: number;
    total: number;
  };
  backends: {
    encoder: string;
    reranker: string;
    generator: string;
  };
  candidate_count: {
    visual: number;
    text: number;
    sparse: number;
    fused: number;
    reranked: number;
  };
  flags?: {
    vlm_rerank_skipped?: boolean;
    degraded_mode?: boolean;
  };
}

export interface ErrorEvent {
  code: string;
  message: string;
  stage: string;
}

export interface RefusedEvent {
  reason: string;
  message: string;
}

export interface DoneEvent {
  query_id: string;
}

export type StreamEvent =
  | { type: "retrieval_started"; data: { query_id: string; ts: string } }
  | { type: "retrieval_completed"; data: RetrievalCompletedEvent }
  | { type: "generation_started"; data: { query_id: string; model: string; ts: string } }
  | { type: "token"; data: TokenEvent }
  | { type: "generation_completed"; data: GenerationCompletedEvent }
  | { type: "error"; data: ErrorEvent }
  | { type: "refused"; data: RefusedEvent }
  | { type: "done"; data: DoneEvent };

/**
 * Parse a raw SSE message line into event type and data
 */
function parseSSEMessage(line: string): { eventType: string; data: unknown } | null {
  if (!line.startsWith("data:")) return null;
  const dataStr = line.slice(5).trim();
  try {
    return { eventType: "data", data: JSON.parse(dataStr) };
  } catch {
    return null;
  }
}

/**
 * Consume an SSE stream and yield structured events
 */
export async function* consumeSSEStream(
  response: Response
): AsyncGenerator<StreamEvent, void, unknown> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else {
          const parsed = parseSSEMessage(line);
          if (parsed) {
            yield { type: eventType || "unknown", data: parsed.data } as StreamEvent;
            eventType = "";
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
