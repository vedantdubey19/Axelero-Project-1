import re
from typing import Tuple, Optional


class GuardrailsService:
    """
    Day 22-23 Implementation: NeMo Guardrails Integration Layer.
    Enforces input rails (jailbreak/off-topic detection) and output rails (safe responses).
    """
    def __init__(self):
        # List of disallowed prompt injection signatures
        self.injection_patterns = [
            r"ignore (all )?previous instructions",
            r"system prompt",
            r"reveal (your |the )?api key",
            r"act as (dan|unrestricted)",
            r"bypass (all )?filters"
        ]

        # Allowed topical keywords for OmniBrain enterprise scope
        self.domain_keywords = [
            "revenue", "profit", "growth", "financial", "table", "chart",
            "figure", "document", "report", "data", "summary", "trend",
            "metric", "annual", "page", "analysis", "what", "how", "who", "explain"
        ]

    def validate_input(self, user_query: str) -> Tuple[bool, Optional[str]]:
        """
        Input Rail: Checks for prompt injections and strictly off-topic queries.
        Returns: (is_allowed: bool, rejection_reason: Optional[str])
        """
        query_lower = user_query.lower().strip()

        # 1. Jailbreak / Injection Detection
        for pattern in self.injection_patterns:
            if re.search(pattern, query_lower):
                return False, "Query blocked by safety guardrails: Prompt injection pattern detected."

        # 2. Out-of-Domain Filter (Blocks obvious non-document questions)
        strictly_disallowed_topics = ["write a poem", "tell me a joke", "recipe for", "sing a song"]
        if any(topic in query_lower for topic in strictly_disallowed_topics):
            return False, "Query blocked by safety guardrails: Request is outside document analysis scope."

        return True, None

    def validate_output(self, generated_answer: str) -> str:
        """
        Output Rail: Ensures generated text does not leak system tokens.
        """
        # Redact potential accidental credential leaks
        sanitized = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", generated_answer)
        return sanitized