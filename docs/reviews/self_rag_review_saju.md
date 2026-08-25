# Technical Review: Self-RAG Implementation & Hardening
**Author:** Vedant Dubey  
**Reviewee:** Saju  
**Date:** August 27, 2026  
**Target Scope:** `backend/app/main.py` (`/api/v1/query`), `backend/app/services/llm_service.py` (`rewrite_query`), and `tests/test_citations_and_self_rag.py`

---

## 1. Executive Summary

As part of preparing for the August 28 "test against deliberately vague queries" milestone and ahead of the scheduled August 26–28 Self-RAG cycle, this review evaluates the current Self-RAG evaluation loop in `backend/app/main.py` and the query rewriter in `backend/app/services/llm_service.py`.

While the baseline integration functions end-to-end, two critical architectural issues and one test coverage gap were identified that directly impact multi-domain retrieval robustness and confidence visibility for the UI/downstream consumers.

---

## 2. Identified Issues & Recommendations

### Issue 1: Offline Fallback Rewriter Is Hardcoded to a Single Domain

#### Current Code (`backend/app/services/llm_service.py`)
```python
def rewrite_query(self, vague_query: str) -> str:
    if not self.api_key:
        return f"{vague_query} detailed financial metrics and overview summary"
```

#### Problem Analysis
When `OPENAI_API_KEY` is not set (e.g. running in offline test environments, local developer sandboxes without keys, or during connectivity disruptions on demo machines), every single vague or low-confidence query is appended with `"detailed financial metrics and overview summary"`. 
- This was hardcoded against a specific financial test report.
- If a user uploads non-financial documents (e.g., technical specifications, legal contracts, research publications, medical reports), appending financial keywords injects strong semantic distortion into dense vector embeddings.
- Consequently, the fallback rewrite **actively degrades retrieval accuracy** on any non-financial document.

#### Resolution Applied & Long-Term Proposal
1. **Immediate Additive Fix (Shipped):** Switched the offline fallback string to domain-neutral terms:
   ```python
   f"{vague_query} detailed summary and key points"
   ```
2. **Proposed Recommendation for Saju (Aug 26–28):** Implement dynamic keyword extraction from the initial retrieved chunks (even when scores are low) or use a frequency-based TF-IDF / term-overlap heuristic to construct an adaptive domain-grounded query expansion without requiring an external LLM call.

---

### Issue 2: The Self-RAG "Loop" Retries Only Once Without Re-Checking Confidence

#### Current Code (`backend/app/main.py`)
```python
if not retrieved_data or avg_score < SIMILARITY_CONFIDENCE_THRESHOLD:
    has_retried = True
    rewritten_query_str = llm_service.rewrite_query(clean_question)
    retry_data = retriever_service.retrieve_relevant_chunks(
        query=rewritten_query_str,
        top_k=request.top_k,
        document_id=request.document_id
    )
    if retry_data:
        retrieved_data = retry_data
```

#### Problem Analysis
1. The execution path performs exactly one retry.
2. If `retry_data` returns non-empty results — even if the average similarity score is still well below `SIMILARITY_CONFIDENCE_THRESHOLD` (e.g. 0.35) — the system unconditionally accepts it as a confident result and sends it to LLM synthesis.
3. The API caller (and frontend citation UI) receives no indication that retrieval confidence remained low, leading to potential hallucination risks presented with false visual certainty.

#### Resolution Applied & Long-Term Proposal
1. **Immediate Additive Fix (Shipped):**
   - Re-computed `retry_avg_score` on `retry_data`.
   - Introduced an explicit `low_confidence: bool` flag on `QueryResponse` (set to `True` if initial or retry scores remain below `SIMILARITY_CONFIDENCE_THRESHOLD` or if chunks are empty).
   - Allows the frontend and citation component to display uncertainty banners or confidence warnings.
2. **Proposed Recommendation for Saju (Aug 26–28):**
   - Formalize the evaluation loop into a bounded multi-step loop (`max_retries = 2` or `3`) using alternative search strategies (e.g., lexical BM25 fallback, expanding chunk window, or prompting the user for clarification if score remains below confidence floor).

---

### Issue 3: Test Coverage Was Asserting Schema Presence Rather Than Self-Correction

#### Previous Test State (`tests/test_citations_and_self_rag.py`)
```python
def test_self_rag_query_execution():
    payload = {"question": "What was the growth trend in revenue?", "top_k": 2}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "retried" in data
    assert isinstance(data["retried"], bool)
```
- The test only checked that `retried` was a boolean type.
- It never verified whether the retry loop was actually entered, never verified `rewritten_query` differed from `question`, and never tested a deliberately low-confidence or unindexed document scenario.

#### Resolution Applied (Shipped):
Added `test_self_rag_retry_triggered_on_low_confidence_query` which deliberately queries unindexed content to guarantee a low-confidence score, explicitly asserting:
- `retried is True`
- `rewritten_query is not None and rewritten_query != question`
- `original_question == question`
- `low_confidence is True`

---

## 3. Discussion Points for Aug 28 Meeting

1. **Adaptive Query Rewriter:** Should we implement an offline BM25/keyword-frequency extractor as fallback when OpenAI API keys are unavailable?
2. **Confidence Threshold Calibration:** Confirm whether `SIMILARITY_CONFIDENCE_THRESHOLD = 0.65` is optimal across all dense embedding collections (`all-MiniLM-L6-v2`) or if Cosine distance requires a normalized threshold.
3. **Frontend UI Integration:** Ensure the frontend citation modal renders a distinct visual indicator when `low_confidence: true`.
