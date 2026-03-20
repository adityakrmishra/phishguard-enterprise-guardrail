# PhishGuard — Business Impact Model

> ET AI Hackathon · Problem Statement 5

---

## Executive Summary

PhishGuard replaces a manual, human-in-the-loop fraud review workflow with a two-stage AI pipeline. This document quantifies the expected cost savings and fraud prevention value for a mid-size Indian private-sector bank processing 10 million digital transactions per month.

---

## 1. Baseline: Current State (No AI Guardrail)

### Transaction volume assumptions

| Parameter | Value | Source |
|---|---|---|
| Monthly digital transactions | 10,000,000 | Mid-size Indian private bank estimate |
| Estimated phishing attempt rate | 0.5% | RBI Annual Report 2023 — ~2–3% of UPI complaints are phishing |
| Monthly phishing attempts flagged for review | **50,000** | 10M × 0.5% |
| False-negative rate (missed by legacy rules) | 30% | Industry benchmark for rule-based fraud filters |
| Phishing messages that result in financial loss | **15,000** | 50,000 × 30% |
| Average loss per successful phishing event | ₹12,000 | RBI Ombudsman average UPI fraud claim 2023 |
| **Monthly fraud payout exposure** | **₹18,00,00,000** | 15,000 × ₹12,000 |

### Manual review cost

| Parameter | Value |
|---|---|
| Fraud analysts on review team | 20 |
| Average analyst salary (fully loaded) | ₹8,00,000 / year |
| Working hours per analyst per month | 160 h |
| Time to manually review one flagged transaction | 4 min |
| Capacity: cases reviewed per analyst per month | 2,400 |
| **Total team capacity (20 analysts)** | **48,000 cases/month** |
| Cases exceeding capacity (backlog) | ~2,000 / month |
| **Monthly analyst cost** | **₹13,33,333** |

---

## 2. PhishGuard Performance Assumptions

| Metric | Value | Basis |
|---|---|---|
| Stage 1 (FAISS) resolution rate | 75% | Estimated from semantic similarity of common phishing templates |
| Stage 1 average latency | 3 ms | Measured on CPU in development |
| Stage 2 (DistilRoBERTa) resolution rate | 20% of remainder | Remaining 25% anomalies |
| Stage 2 average latency | 120 ms | Measured on CPU in development |
| Audit log latency (Groq API) | 450 ms | Estimated from Groq benchmark |
| **Overall system precision** | **91%** | Conservative estimate post fine-tuning |
| **Overall system recall** | **88%** | Conservative estimate post fine-tuning |
| Cases escalated to human review | ~6,000 / month | Remaining FP/FN edge cases |

---

## 3. Savings Model

### 3A — Analyst cost reduction

```
Without PhishGuard:  48,000 cases × 4 min = 3,200 analyst-hours/month
With PhishGuard:      6,000 cases × 4 min =   400 analyst-hours/month

Hours saved:  2,800 h / month
Cost/hour:    ₹8,00,000 / (12 × 160) = ₹416.67 / h

Monthly analyst savings = 2,800 × ₹416.67 = ₹11,66,676
Annual analyst savings  = ₹1,40,00,112   (~₹1.4 Cr)
```

### 3B — Fraud loss reduction

```
Baseline monthly fraud loss:  ₹18,00,00,000
PhishGuard recall:            88%
Fraud losses caught:          ₹18 Cr × 88% = ₹15,84,00,000 prevented

Residual false-negative loss: ₹18 Cr × 12% = ₹2,16,00,000

Monthly fraud savings = ₹15,84,00,000 - ₹0 (no payout for caught fraud)
Annual fraud savings  = ₹15,84,00,000 × 12 = ₹190,08,00,000  (~₹190 Cr)
```

### 3C — Speed advantage (SLA compliance)

```
Manual review SLA:         4 hours average (regulatory requirement)
PhishGuard triage time:    < 1 second for 95% of cases

SLA breach reduction:      ~99%
Regulatory fine avoidance: ₹10,000–₹1,00,000 per breach (RBI guideline)
Estimated annual fine avoided: ₹50,00,000  (~₹50 L) conservative
```

---

## 4. ROI Summary

| Category | Annual Value |
|---|---|
| Analyst cost reduction | ₹1,40,00,000 |
| Fraud loss prevention (88% recall) | ₹190,08,00,000 |
| Regulatory fine avoidance | ₹50,00,000 |
| **Total Annual Benefit** | **₹191,98,00,000 (~₹192 Cr)** |

### Estimated deployment cost

| Item | Estimate |
|---|---|
| Cloud compute (2× A10G GPU instances for inference) | ₹60,00,000 / year |
| Groq API calls (50,000 scams × ₹0.002 / call) | ₹1,20,000 / year |
| Model retraining (quarterly, 4 × 2h GPU job) | ₹40,000 / year |
| Engineering maintenance (0.5 FTE) | ₹20,00,000 / year |
| **Total Annual Cost** | **₹81,60,000 (~₹82 L)** |

---

## 5. Net Impact

```
Annual Benefit:  ₹192 Cr
Annual Cost:     ₹82 L
────────────────────────
Net Annual ROI:  ₹191.18 Cr
ROI Multiple:    234×
Payback period:  < 2 weeks
```

---

## 6. Additional Non-Quantified Benefits

- **Customer trust**: Faster, accurate fraud detection reduces customer churn from fraud victims.
- **Regulatory goodwill**: Automated, auditable decision trails simplify RBI inspection responses.
- **Model flywheel**: Every reviewed edge case enriches the fine-tuning dataset, improving accuracy over time without additional human cost.
- **Scalability**: The FAISS + LoRA architecture handles 10× transaction volume growth with no architectural change — only horizontal scaling of API workers.

---

*All figures are back-of-the-envelope estimates for hackathon demonstration purposes. Production deployment would require empirical measurement of precision/recall on live traffic.*
