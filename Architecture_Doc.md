# PhishGuard — System Architecture

> ET AI Hackathon · Problem Statement 5

---

## 1. High-Level Design

PhishGuard is a **multi-stage, defence-in-depth** phishing detection system. Each stage is independently optimised for a different speed/accuracy trade-off, forming a cascade that is both fast for common cases and accurate for edge cases.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client / Dashboard                          │
│              POST /api/v1/analyze  { "text": "..." }            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — SemanticRouter  (src/dsa_router/vector_triage.py)   │
│                                                                  │
│  • Embeds text with all-MiniLM-L6-v2 (384-dim, ~80 MB)         │
│  • FAISS IndexFlatL2 k=1 nearest-neighbour search              │
│  • Compares L2 distance against two thresholds:                 │
│    d ≤ 0.35 → SAFE        (fast-path return)                   │
│    d ≤ 0.75 → KNOWN_SCAM  (fast-path return)                   │
│    d >  0.75 → ANOMALY    (escalate to Stage 2)                │
│                                                                  │
│  Latency: ~ 1–5 ms (CPU)                                        │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │ SAFE /            │ ANOMALY
                      │ KNOWN_SCAM        │
                      │                   ▼
                      │  ┌────────────────────────────────────────┐
                      │  │  Stage 2 — IntentClassifier            │
                      │  │  (src/ml_pipeline/inference.py)        │
                      │  │                                        │
                      │  │  • Base: distilroberta-base (82M params)│
                      │  │  • Adapter: LoRA (r=8, ~0.3% params)   │
                      │  │  • Fine-tuned on 50 red-team +         │
                      │  │    20 safe synthetic examples          │
                      │  │  • Returns: SAFE | KNOWN_SCAM +        │
                      │  │    softmax confidence score            │
                      │  │                                        │
                      │  │  Latency: ~50–200 ms (CPU)             │
                      │  └──────────────┬─────────────────────────┘
                      │                 │
                      ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Explainability Agent  (src/agent/explainability.py)           │
│                                                                  │
│  Triggered ONLY when final verdict == KNOWN_SCAM               │
│                                                                  │
│  • Sends flagged text to Groq LLM (Mixtral / GPT-4o-mini)      │
│  • System prompt: compliance analyst role + RBI/NPCI context   │
│  • Structured output (JSON):                                    │
│    - fraud_intent:          "Authority Impersonation"          │
│    - compliance_reasoning:  1-sentence policy explanation      │
│                                                                  │
│  Non-blocking: failure returns graceful fallback strings        │
│  Latency: ~300–800 ms (network I/O to Groq API)                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   TransactionResponse JSON
          { verdict, distance, matched_template,
            fraud_intent, compliance_reasoning }
```

---

## 2. Component Breakdown

### 2.1 SemanticRouter (DSA Layer)

The fast-path router is a **two-threshold FAISS classifier**:

- A FAISS `IndexFlatL2` is pre-populated with 7 safe banking templates and 7 known-scam templates at startup.
- The `add_templates()` method allows extending the index at runtime without restarting the server.
- Thresholds are configurable per-request via the Streamlit sidebar sliders, enabling live tuning during the demo.

**Why FAISS over a simple cosine similarity loop?**
FAISS supports billion-scale indices with sub-millisecond ANN search. Even at a modest scale, it is 10–100× faster than a Python loop over numpy arrays, and it separates the embedding concern from the retrieval concern cleanly.

### 2.2 IntentClassifier (ML Layer)

The heavy-brain classifier uses **Parameter-Efficient Fine-Tuning (PEFT)** via LoRA:

- Only the query and value attention projection matrices are adapted (< 0.3% of total parameters).
- Training on 70 examples requires < 10 minutes on CPU and < 1 minute on a T4 GPU.
- The adapter is saved separately from the base model so the 82M base weights are shared and never duplicated.
- The base model is loaded once at API startup and kept in memory, making per-request inference stateless.

### 2.3 Explainability Agent (Audit Layer)

The agent implements the **Chain-of-Compliance** pattern:

1. Sends the flagged message to the LLM with a role-prompted system message that locks it to a compliance analyst persona.
2. Restricts `fraud_intent` to a closed vocabulary of 9 categories (e.g., Authority Impersonation, OTP Phishing) so the output is always structured and downstream-parseable.
3. Temperature is set to `0.2` to minimise creative variation — the output should be factual and repeatable.
4. JSON is validated server-side; malformed output triggers a graceful fallback rather than a 500 error.

---

## 3. How PhishGuard Addresses PS5 Requirements

| PS5 Requirement | PhishGuard Implementation |
|---|---|
| **Guardrails** | Stage 1 (FAISS) acts as a deterministic guardrail — known-safe traffic is never sent to the LLM, preventing hallucination risk on legitimate messages |
| **Auditable decisions** | Every `KNOWN_SCAM` verdict generates a structured audit log with `fraud_intent` (categorical) + `compliance_reasoning` (natural language) stored in the API response and visible in the dashboard |
| **Compliance context** | The LLM system prompt anchors reasoning to RBI / NPCI AML guidelines and Indian fintech context |
| **Explainability** | The matched FAISS template shows *why* Stage 1 flagged a message; the LLM reasoning shows *what policy* it violates |
| **Resilience** | The audit log is non-blocking — if the Groq API is unavailable, the triage verdict still returns correctly |

---

## 4. Data Flow Summary

```
Text Input
  → Tokenise + embed (MiniLM, 384-dim)
  → FAISS L2 search (k=1)
  → Threshold decision
        ├─ SAFE / SCAM  →  return (< 5 ms total)
        └─ ANOMALY
              → DistilRoBERTa tokenise + forward pass
              → Softmax argmax → SAFE / KNOWN_SCAM (~100 ms)
                    └─ If SCAM → Groq API call → JSON audit log (~500 ms)
  → TransactionResponse (returned to client)
```

---

## 5. Deployment Notes

- The API is stateless and horizontally scalable (multiple Uvicorn workers behind a load balancer).
- The FAISS index and both model weights are loaded once per worker process.
- For production, the FAISS index should be persisted to disk with `faiss.write_index()` and reloaded on startup to avoid index loss on restart.
- The LoRA adapter can be hot-swapped by replacing the `models/distilroberta-finetuned/` directory without changing any API code.
