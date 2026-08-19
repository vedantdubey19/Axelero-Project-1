# OmniBrain — API Reference

Base URL (local): `http://localhost:8000`

---

## POST /api/v1/upload
Upload a PDF for processing.

**Request** (multipart/form-data):
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@sample_pdfs/example.pdf"
```

**Response** `200 OK`
```json
{
  "job_id": "b3f2e1a4-...",
  "filename": "example.pdf",
  "status": "QUEUED"
}
```

**Errors**
- `400` — file is not a valid PDF (MIME/extension check)
- `413` — file exceeds 25MB limit

---

## GET /api/v1/ingest/status/{job_id}
Poll ingestion progress for an uploaded file.

```bash
curl http://localhost:8000/api/v1/ingest/status/b3f2e1a4-...
```

**Response**
```json
{
  "job_id": "b3f2e1a4-...",
  "status": "PROCESSING"   // QUEUED | PROCESSING | DONE | FAILED
}
```

---

## POST /api/v1/query
Ask a question against ingested documents.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy in this document?"}'
```

**Response** (current — placeholder until LLM wiring is complete)
```json
{
  "answer": "<placeholder text>",
  "retrieved_chunks": [
    {"text": "...", "score": 0.83, "source": "example.pdf", "page": 4}
  ]
}
```

**Errors**
- `400` — empty query string
- `404` — referenced job_id not found (if job-scoped querying is used)

---

## Notes for integrators (Frontend team)
- Always poll `/ingest/status/{job_id}` until `status == "DONE"` before calling `/query` for that document.
- Wrap all calls in try/except with timeout — LLM calls may take several seconds.
