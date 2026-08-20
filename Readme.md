# Axlero Project 1

## Team Information

**Project:** Axlero Project 1

**Organization:** Axlero Solutions

**Team Size:** 6 Members

---

## Team Members

| Name | Role |
|------|------|
| Vedant Dubey | Team Leader |
| Dhruv Soni | Team Member |
| Saju J | Team Member |
| Venkatesh Namana | Team Member |
| Shaik Humera Thabassum | Team Member |
| Arya Rajhans Mohod | Team Member |

---

## Git Branches

| Branch | Purpose |
|---------|---------|
| `main` | Production-ready code |
| `dev` | Development and integration |
| `vedant` | Vedant's work |
| `dhruv` | Dhruv's work |
| `saju` | Saju's work |
| `venkatesh` | Venkatesh's work |
| `humera` | Humera's work |
| `arya` | Arya's work |

---


## Git Workflow

1. Create your feature on your own branch.
2. Commit your changes regularly.
3. Push your branch to GitHub.
4. Create a Pull Request to the `dev` branch.
5. The Team Leader reviews and merges the Pull Request.
6. Stable code is merged from `dev` to `main`.

---

## Team Rules

- Do not push directly to `main`.
- Work only on your assigned branch.
- Commit your work regularly.
- Keep commit messages meaningful.
- Resolve conflicts before creating a Pull Request.
- Communicate blockers early.

---

## Repository Structure

```
Project/
├── frontend/
├── backend/
├── database/
├── docs/
├── tests/
└── README.md
```

---

## Team Leader

**Vedant Dubey**

Responsible for:
- Repository management
- Branch management
- Task distribution
- Pull Request reviews
- Merge management
- Team coordination

---

## Project Status

🟢 Multi-Agent Supervisor Active & Verified

---

## Multi-Agent Supervisor Architecture

OmniBrain incorporates a LangGraph-driven **Supervisor Agent** that dynamically orchestrates specialized agent workflows:
- **`SupervisorAgent`**: Evaluates query intent and dispatches queries to target agents (`SearchAgent`, `VisionAgent`).
- **`SearchAgent`**: Executes dense vector retrieval from Qdrant and grounded synthesis via `LLMSynthesisService`.
- **`VisionAgent`**: Explicit labeled stub for multimodal chart/image reasoning.

### Running End-to-End Tests

To execute the full automated test suite (canonical retrieval pipeline + multi-agent supervisor routing + empty context handling):

```bash
pytest -v --maxfail=1
```

Or run the supervisor routing suite specifically:

```bash
pytest tests/test_supervisor_routing.py -v
```

### Running the Live Demo

Run the standalone live demonstration script showing PDF upload, supervisor intent classification, real vector search, and grounded synthesis:

```bash
python scripts/demo_supervisor.py
```

