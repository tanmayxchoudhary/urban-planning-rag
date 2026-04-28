/**
 * Analytics tracking — PLAN §9.1: PostHog for product events.
 *
 * Events tracked:
 * - query_submitted: when user submits a question
 * - answer_received: when generation completes
 * - citation_clicked: when user clicks a citation chip
 * - feedback_submitted: when user submits feedback (thumbs up/down)
 *
 * In development / when PostHog is not configured, events are logged to console.
 * In production, events are sent to PostHog via the /path endpoint (self-hosted)
 * or the cloud endpoint (if NEXT_PUBLIC_POSTHOG_KEY is set).
 */

export type AnalyticsEvent =
  | { type: "query_submitted"; question: string; mode: "fast" | "deep" }
  | { type: "answer_received"; query_id: string; confidence: string }
  | { type: "citation_clicked"; query_id: string; page_id: string; doc_filename: string }
  | { type: "feedback_submitted"; query_id: string; vote: "up" | "down"; comment?: string };

function isPostHogConfigured(): boolean {
  return !!(
    process.env.NEXT_PUBLIC_POSTHOG_KEY ||
    process.env.NEXT_PUBLIC_POSTHOG_HOST
  );
}

function getPostHogHost(): string {
  return process.env.NEXT_PUBLIC_POSTHOG_HOST || "https://app.posthog.com";
}

function getPostHogApiKey(): string {
  return process.env.NEXT_PUBLIC_POSTHOG_KEY || "";
}

/**
 * Send an analytics event.
 * In dev mode without PostHog config, logs to console.
 * In production, sends to PostHog (self-hosted or cloud).
 */
export async function trackEvent(event: AnalyticsEvent): Promise<void> {
  if (!isPostHogConfigured()) {
    // Dev mode: log to console
    if (process.env.NODE_ENV === "development") {
      console.log("[analytics]", JSON.stringify(event));
    }
    return;
  }

  try {
    const host = getPostHogHost();
    const apiKey = getPostHogApiKey();

    // PostHog /capture endpoint
    const response = await fetch(`${host}/capture`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_key: apiKey,
        event: event.type,
        properties: {
          ...event,
          // PostHog convention: distinct_id per session
          distinct_id: getAnonymousId(),
          // Timestamp
          timestamp: new Date().toISOString(),
        },
      }),
    });

    if (!response.ok) {
      console.warn("[analytics] Failed to send event:", response.status);
    }
  } catch (err) {
    // Non-blocking — analytics failures should not affect UX
    console.warn("[analytics] Error sending event:", err);
  }
}

/**
 * Get or create an anonymous session ID for analytics.
 * Stored in localStorage to persist across page loads.
 */
function getAnonymousId(): string {
  if (typeof window === "undefined") return "server";

  const KEY = "_uprag_aid";
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = `anon_${Math.random().toString(36).slice(2)}_${Date.now()}`;
    localStorage.setItem(KEY, id);
  }
  return id;
}
