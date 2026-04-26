"""Answer generation with strict grounding, citation handling, and deep mode.

Per PART VIII §8.3:
    - Strict grounding instructions
    - Refusal on insufficient context
    - Citation requirements: [k] inline markers verified against candidates
    - Deep mode: Gemini 2.5 Pro with stronger grounding check
"""

from urban_rag.generate.gemini import (
    _GeminiStreamEvent,
    generate,
    generate_stream,
    generate_sync,
)
from urban_rag.generate.gemini import (
    smoke_test as gemini_smoke_test,
)
from urban_rag.generate.orchestrator import (
    AnswerEvent,
    AnswerStreamEvent,
    DoneEvent,
    ErrorEvent,
    GenerationCompletedEvent,
    GenerationStartedEvent,
    RefusedEvent,
    RetrievalCompletedEvent,
    RetrievalStartedEvent,
    TokenEvent,
    answer,
)
from urban_rag.generate.orchestrator import (
    smoke_test as orchestrator_smoke_test,
)
from urban_rag.generate.prompts import (
    AnswerMode,
    AnswerPromptConfig,
    PageMetadata,
    RefusalReason,
    build_answer_prompt,
    build_citations_list,
    build_refusal_prompt,
    build_self_critique_prompt,
    extract_citations_from_answer,
    validate_citations,
)
from urban_rag.generate.prompts import (
    smoke_test as prompt_smoke_test,
)

__all__ = [
    "AnswerEvent",
    "AnswerMode",
    "AnswerPromptConfig",
    "AnswerStreamEvent",
    "DoneEvent",
    "ErrorEvent",
    "GenerationCompletedEvent",
    "GenerationStartedEvent",
    "PageMetadata",
    "RefusalReason",
    "RefusedEvent",
    "RetrievalCompletedEvent",
    "RetrievalStartedEvent",
    "TokenEvent",
    "_GeminiStreamEvent",
    "answer",
    "build_answer_prompt",
    "build_citations_list",
    "build_refusal_prompt",
    "build_self_critique_prompt",
    "extract_citations_from_answer",
    "gemini_smoke_test",
    "generate",
    "generate_stream",
    "generate_sync",
    "orchestrator_smoke_test",
    "prompt_smoke_test",
    "validate_citations",
]
