## Area: Web

### VAL-WEB-001: First-visit landing renders core entry points
On first visit to `/` with empty `localStorage`, the page renders a visible query input, submit button, top-3 example questions, and a corpus stats strip without requiring authentication.
Tool: agent-browser
Evidence: URL is `/`; no auth prompt; 3 example question items visible; corpus stats values visible; screenshot captured.

### VAL-WEB-002: Keyboard-first submit works from landing
From `/`, keyboard-only flow (`Tab` to query input, type question, `Enter`) submits once and transitions to `/q/[query_id]` with a newly created `query_id`.
Tool: agent-browser
Evidence: Focus ring visible on input; single `POST /v1/ask`; response includes `query_id` and `stream_url`; browser URL changes to `/q/<id>`.

### VAL-WEB-003: Query submit request contract is correct
Submitting a question sends `POST /v1/ask` with JSON body `{question, mode}` and receives HTTP `202 Accepted` containing `query_id`, `stream_url`, and `expires_at`.
Tool: agent-browser
Evidence: Network request payload and response body fields match contract in Appendix C.

### VAL-WEB-004: Streaming event order is handled correctly
For a successful query, UI processing follows stream order: `retrieval_started` → `retrieval_completed` → `generation_started` → `token*` → `generation_completed` → `done`.
Tool: agent-browser
Evidence: Event stream transcript (network/event log) and UI state transitions align with the ordered sequence.

### VAL-WEB-005: Retrieval skeleton and citation-first render behavior
After submit, retrieval skeleton appears immediately; citation chips render as soon as `retrieval_completed` arrives, before final answer completion.
Tool: agent-browser
Evidence: Timeline capture showing skeleton state then citation chips visible before stream completion.

### VAL-WEB-006: Token streaming progressively updates answer body
During `token` events, answer text grows incrementally without full-body replacement/flicker and ends with a stable final answer after `generation_completed`.
Tool: agent-browser
Evidence: Sequential DOM snapshots or text-length progression showing incremental growth and final settled text.

### VAL-WEB-007: Query permalink route is directly renderable
Navigating directly to `/q/[query_id]` renders the corresponding question/answer view with citation chips and does not redirect to `/`.
Tool: agent-browser
Evidence: Direct URL load succeeds; answer panel and chips visible; no unexpected redirect.

### VAL-WEB-008: Done-state cache enables reload without re-query
After receiving `done`, reloading `/q/[query_id]` restores answer and citations from `localStorage` key `query_id` without issuing another `POST /v1/ask`.
Tool: agent-browser
Evidence: `localStorage` contains cached payload for `query_id`; reload shows content; network shows no new ask submission.

### VAL-WEB-009: Inline citation markers map to chips
Clicking an inline citation marker like `[1]` in the answer moves focus/viewport to the corresponding citation chip and preserves marker-to-chip index integrity.
Tool: agent-browser
Evidence: Marker click highlights/targets chip `1`; no index mismatch across visible citations.

### VAL-WEB-010: Citation chip metadata and thumbnail render correctly
Each citation chip shows lazy-loaded thumbnail, document name, page number, and section title when available.
Tool: agent-browser
Evidence: Chip UI fields populated; thumbnail request occurs on visibility; fallback behavior observed when section title is absent.

### VAL-WEB-011: Lightbox route `/q/[query_id]/page/[idx]` renders citation page details
Opening a citation from chips routes to `/q/[query_id]/page/[idx]` and shows full page image, extracted text, and filename in the lightbox context.
Tool: agent-browser
Evidence: URL matches page route; modal/side panel displays image + extracted text + filename for selected citation.

### VAL-WEB-012: Lightbox keyboard accessibility and focus return
Lightbox is keyboard-operable: focus enters modal on open, `Tab` cycles within modal controls, `Esc` closes it, and focus returns to the triggering citation chip.
Tool: agent-browser
Evidence: Focus order trace confirms trap; `Esc` closes modal; post-close focus equals original trigger element.

### VAL-WEB-013: Copy citation action produces structured markdown link
Using “Copy citation” on a chip copies a markdown citation in the form `[Doc..., p.X](https://...)` to clipboard.
Tool: agent-browser
Evidence: Clipboard readback matches expected markdown link pattern and selected citation metadata.

### VAL-WEB-014: Feedback widget submits valid payload and enforces comment limit
Thumbs up/down submission sends `POST /v1/feedback` with `{query_id, vote, comment?}`; optional comment is limited to 200 characters client-side.
Tool: agent-browser
Evidence: Network payload matches schema; over-200 input is blocked/truncated; HTTP response is `204 No Content`.

### VAL-WEB-015: Feedback controls are accessible with ARIA and keyboard
Feedback buttons expose meaningful accessible names/ARIA labels and are operable via keyboard (`Tab` + `Enter/Space`) with visible focus.
Tool: agent-browser
Evidence: Accessibility tree shows labels; keyboard activation submits chosen vote; focus ring remains visible.

### VAL-WEB-016: `/corpus` route renders grouped corpus metadata
Route `/corpus` loads and displays indexed documents grouped by family/jurisdiction with counts and last-updated context.
Tool: agent-browser
Evidence: URL `/corpus`; grouped sections visible with document counts; data presence aligns with `/v1/corpus` response.

### VAL-WEB-017: `/about` route contains required project links
Route `/about` renders one-paragraph project description, license mention, and links to GitHub and paper/reference.
Tool: agent-browser
Evidence: URL `/about`; paragraph text present; two outbound links visible and clickable.

### VAL-WEB-018: Ask validation failure is surfaced without UI crash
If `POST /v1/ask` returns `422 validation_error` (e.g., invalid/empty question), the UI shows a non-blocking error message and remains interactive for correction.
Tool: agent-browser
Evidence: Simulated invalid submit yields visible validation error; no full-page crash; input remains editable and re-submittable.

### VAL-WEB-019: Stream error event shows recoverable failure state
If stream emits `event: error` (`retrieval_timeout` or `generation_failed`), UI exits loading state, presents stage-aware failure copy, and offers retry/new-query action.
Tool: agent-browser
Evidence: Injected error event produces visible error state with actionable control; spinner/skeleton stops.

### VAL-WEB-020: Out-of-corpus refusal is explicitly represented
If stream emits `event: refused` with `reason: out_of_corpus`, UI displays explicit refusal text and avoids rendering fabricated answer content.
Tool: agent-browser
Evidence: Refusal banner/message visible; answer body contains no generated claim text; citation chips are absent or clearly marked unavailable.
