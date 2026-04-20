]# Fine-Tuning Plan for Where Is My Paisa (Local, In-App LLM)

> **Objective**: Build and ship finance-specialized, on-device LLMs that run inside the main Android app.
> 
> **Audience**: Humans + coding agents implementing data, training, evaluation, packaging, and rollout.
> 
> **Date**: April 2026

---

## 1) Scope and Success Criteria

### In-scope use cases

1. **Transaction understanding** from SMS/document text (categorization + structured extraction).
2. **Insight narration** from structured finance JSON.
3. **Budget coaching** and short actionable advice.
4. **Anomaly explanation** for detected unusual spends.
5. **Goal progress coaching** (short horizon planning, no legal claims).

### Out-of-scope use cases

- Legal/tax compliance advice as authoritative truth.
- Real-time market recommendations.
- Multi-hop planning that requires fresh web data.

### Product-level success criteria

A model is shippable only when all pass:

- **Quality**: beats template baseline on target tasks.
- **Safety**: refuses high-risk legal/tax certainty prompts.
- **Latency**: meets local runtime SLO for its device tier.
- **Stability**: no major ANR/OOM regressions.

---

## 2) Why this training approach

This plan uses **parameter-efficient fine-tuning** because it is practical, iterative, and cost-effective for small/medium models.

- **LoRA** reduces trainable params while preserving quality and avoids inference-latency overhead from classic adapters in many setups.
- **QLoRA** enables 4-bit training pipelines with strong quality/efficiency trade-offs for limited compute budgets.
- **SFT first, preference tuning optional** keeps training stable and interpretable.

Recommended order:

1. Data curation + normalization.
2. SFT (instruction tuning for finance tasks).
3. Optional DPO on preference pairs (if tone/safety quality needs improvement).
4. Quantize/export and run full on-device eval gate.

---

## 3) Model strategy for this app

### Base-model shortlist

Use 2–3 bases in parallel during research; ship only one default + one fallback:

- **Default quality/speed**: 1B-ish instruct model (e.g., Llama 3.2 1B class).
- **Low-RAM fallback**: 0.5B-ish instruct model (Qwen2.5 0.5B class).
- **Premium optional**: ~1.7B model for 8GB+ devices.

### Fine-tune policy

- Keep one **general finance assistant** checkpoint per base model.
- Avoid many narrowly specialized checkpoints at first (ops overhead).
- Use prompt routing + task tags instead of separate models unless metrics demand split models.

---

## 4) Data design (critical)

## 4.1 Data sources

1. **Existing app logic outputs**
   - transaction parser outputs
   - category rules and corrections
   - anomaly flags
   - budget summaries
2. **De-identified historical user examples** (only if consent + policy allows).
3. **Synthetic augmentation** generated from deterministic templates and controlled perturbations.

## 4.2 Privacy and de-identification pipeline

Before anything reaches training corpora:

1. Detect sensitive spans: names, phone, account numbers, UPI IDs, emails, addresses.
2. Replace with stable placeholders:
   - `<NAME_1>`, `<ACCOUNT_LAST4_1>`, `<VPA_1>`, `<PHONE_1>`
3. Keep value-preserving abstractions for utility:
   - amount buckets and exact currency values where needed.
4. Store mapping only in short-lived secure workspace (never in training artifacts).

Implementation options:

- Use deterministic regex + app-specific detectors first.
- Add optional Presidio-style entity detection for broader PII coverage.

## 4.3 Dataset splits

- Train: 80%
- Validation: 10%
- Test: 10%

Hard rule: split by **user and merchant families** to prevent leakage.

---

## 5) Task schemas and annotation specs

Use one canonical JSONL schema per sample:

```json
{
  "id": "uuid",
  "task": "txn_categorization|insight_narration|budget_coaching|anomaly_explanation|goal_coaching",
  "input": {
    "context": "...",
    "structured": {...}
  },
  "output": {
    "text": "...",
    "labels": {...}
  },
  "metadata": {
    "source": "rules|historical|synthetic",
    "language": "en",
    "safety_tags": ["tax", "advice"],
    "difficulty": "easy|medium|hard"
  }
}
```

### 5.1 Transaction categorization samples

Input includes SMS text + optional merchant hints.
Output includes:

- normalized merchant
- category (existing app enum only)
- confidence band (low/med/high)
- short rationale (<= 1 sentence)

### 5.2 Narration/coaching tasks

Output rules:

- 2–4 short sentences
- amounts always in ₹ format
- no fabricated values
- include at least one action recommendation

### 5.3 Safety annotation

Add explicit negative samples where the expected output is refusal/limitation text:

- legal certainty requests
- tax filing certainty requests
- investment guarantee requests

---

## 6) Data volume targets

Minimum viable corpus per language (English first):

- Txn categorization/extraction: **20k–40k**
- Insight narration: **8k–15k**
- Budget coaching: **6k–10k**
- Anomaly explanation: **4k–8k**
- Goal coaching: **4k–8k**
- Safety refusal pairs: **5k+**

Total target: **50k–80k** high-quality instruction examples.

Quality matters more than raw size. Remove low-signal or contradictory labels aggressively.

---

## 7) Training pipeline (implementable runbook)

### 7.1 Repo structure (recommended)

```text
research/
training/
  configs/
    sft_1b.yaml
    sft_05b.yaml
    dpo_1b.yaml
  data/
    raw/
    deid/
    curated/
    splits/
  scripts/
    01_extract.py
    02_deidentify.py
    03_build_jsonl.py
    04_validate.py
    05_train_sft.py
    06_train_dpo.py
    07_eval.py
    08_export_gguf.sh
    09_publish_manifest.py
```

### 7.2 Stage A — SFT (mandatory)

Recommended defaults (starting point):

- Method: QLoRA
- LoRA rank: 16 or 32
- LoRA alpha: 32 or 64
- Dropout: 0.05
- LR: 1e-4 to 2e-4
- Epochs: 2–4
- Sequence length: 2048 (reduce for 0.5B if memory constrained)
- Packing: ON for plain text tasks, OFF for completion-only instruction masks

Use completion-only loss for instruction data where prompt should not be learned as target.

### 7.3 Stage B — Preference tuning (optional but recommended)

When to run DPO:

- Tone inconsistency
- Excess verbosity
- Safety borderline behavior

DPO dataset format:

- prompt
- chosen response
- rejected response

Sources for preference pairs:

- in-house reviewer comparisons
- thumbs up/down from beta cohort (with consent)
- safety-focused synthetic contrast pairs

### 7.4 Stage C — Merge + export

1. Merge LoRA adapters into base (or keep adapters for archival).
2. Convert to GGUF.
3. Quantize variants:
   - Q4_K_M (default)
   - Q5_K_M (premium quality)
   - optional Q3_K_M (extreme low RAM)

---

## 8) Evaluation harness (must be automated)

## 8.1 Offline metrics by task

### Transaction categorization

- Macro F1 (primary)
- Top-1 accuracy
- Calibration error (confidence quality)

### Extraction correctness

- field-level precision/recall (amount/date/merchant)

### Narration/coaching

- Factual consistency score against source JSON
- Actionability score (rubric)
- Brevity compliance (% outputs within length policy)

### Safety

- Refusal precision/recall on disallowed prompt set
- Hallucinated-number rate (must be near zero)

## 8.2 On-device metrics

Per model × device tier matrix:

- first token latency p50/p95
- decode tokens/sec p50
- peak RAM
- thermal downgrade frequency
- battery drain per 100 requests

## 8.3 Acceptance gates

Ship candidate only if:

- Beats template baseline on 3+ primary use cases.
- No safety regression on refusal tests.
- Meets device-tier runtime SLOs.

---

## 9) Data quality controls (non-negotiable)

1. **Schema validation** for every JSONL row.
2. **Duplicate near-match removal** via semantic + lexical dedupe.
3. **Contradiction detection** for same input with conflicting labels.
4. **Leak checks** (PII token scans post-deid).
5. **Holdout sanctity** (no synthetic variants of test rows in train).

---

## 10) Continual learning strategy

Use monthly micro-releases, not continuous retraining.

Cycle:

1. Collect errors from production telemetry (privacy-safe, no raw prompts by default).
2. Curate hard-example set.
3. Retrain SFT delta.
4. Re-evaluate full benchmark.
5. Publish new version if all gates pass.

Versioning:

`<base>-fin-<major.minor.patch>-<quant>`

Example: `llama-3.2-1b-fin-1.3.0-q4_k_m`

---

## 11) Infrastructure and compute planning

### Practical starting compute

- 1× A100 80GB (fastest iteration), or
- 1–2× 4090/6000 Ada for cost-sensitive runs.

Expected per-run costs depend on corpus size and sequence length; optimize for fast iteration with strong eval gates rather than oversized training runs.

### Storage planning

- Raw + processed training data: 100–500 GB typical for iterative runs.
- Model artifacts (multiple quants + versions): plan 1–3 TB over time.
- Keep immutable artifact registry with checksums.

---

## 12) Implementation checklist for an agent

### Phase 0 — bootstrap

- [ ] Create training repo layout and config templates.
- [ ] Implement data extraction and de-identification scripts.
- [ ] Build schema validator + data card generator.

### Phase 1 — baseline model

- [ ] Prepare v1 corpus (>=50k examples).
- [ ] Train SFT for 1B and 0.5B bases.
- [ ] Run offline + on-device eval matrix.
- [ ] Pick winner + fallback.

### Phase 2 — alignment and polish

- [ ] Build preference dataset.
- [ ] Run DPO on top candidate if needed.
- [ ] Re-run full regression suite.

### Phase 3 — packaging and release

- [ ] Merge/export GGUF + quantize.
- [ ] Generate manifest with metrics, checksums, constraints.
- [ ] Publish artifacts to selected storage backend.
- [ ] Update model catalog API entries.

### Phase 4 — post-launch

- [ ] Monitor fallback rate / latency / thermal / battery.
- [ ] Collect hard cases and schedule next monthly training cycle.

---

## 12.1 Release handoff contract (must match app catalog API)

Every shipped fine-tuned model must produce a **release bundle** consumed by `POST /v1/local-llm/models/catalog` from `research/local-llm.md`.

### Required bundle files

1. `model.gguf` (or shard set)
2. `manifest.json`
3. `eval_report.json`
4. `model_card.md`

### Required `manifest.json` fields

```json
{
  "id": "llama-3.2-1b-fin-q4km-v4",
  "displayName": "Llama 3.2 1B Finance Q4_K_M",
  "baseModel": "meta-llama/Llama-3.2-1B-Instruct",
  "finetuneMethod": "QLoRA-SFT",
  "quantization": "Q4_K_M",
  "sizeBytes": 880123456,
  "sha256": "abc123...",
  "requirements": {
    "minRamMb": 6000,
    "recommendedFreeStorageMb": 2200,
    "abis": ["arm64-v8a"],
    "minAndroidApi": 26
  },
  "metrics": {
    "qualityScore": 7.9,
    "medianDecodeTokensPerSec": 14.8,
    "medianFirstTokenMs": 1800,
    "hallucinationRisk": "medium",
    "batteryImpact": "medium"
  },
  "safety": {
    "refusalRecall": 0.94,
    "hallucinatedNumberRate": 0.01
  },
  "provenance": {
    "datasetVersion": "finetune-v1.4",
    "trainRunId": "run_2026_04_20_001",
    "evalRunId": "eval_2026_04_20_004"
  }
}
```

### Definition of done for a model release

- [ ] Manifest schema validation passes.
- [ ] SHA-256 matches downloaded artifact.
- [ ] Eval report contains offline + on-device metrics.
- [ ] Safety thresholds meet release gates.
- [ ] Catalog API entry created/updated with recommendation tier + tradeoff text.

---

## 13) Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to template-like phrasing | Repetitive, low-utility responses | Increase hard negatives, diversify narration style targets |
| PII leakage in training corpus | Severe privacy risk | Strong de-id pipeline + automated leak scan + manual audit |
| Device instability (OOM/thermal) | Bad UX, crashes | strict runtime guardrails + model tier gating + fallback |
| Hallucinated financial facts | Trust erosion | factuality eval + explicit "unknown" behavior + refusal patterns |
| Annotation inconsistency | Noisy learning signal | labeling handbook + adjudication + disagreement audits |

---

## 14) Minimal prompt-format contract for training

Use one consistent chat template. Example:

```text
<|system|>
You are a concise Indian personal finance assistant.
Never fabricate numbers. If data is insufficient, say so.
</|system|>
<|user|>
TASK=insight_narration
INPUT_JSON={...}
</|user|>
<|assistant|>
...
</|assistant|>
```

Consistency in prompt format materially improves fine-tune stability and inference predictability.

---

## 15) Suggested first 30-day execution plan

### Week 1

- finalize schema + annotation handbook
- build extraction/deid/validation pipeline
- generate first 20k clean rows

### Week 2

- scale to 50k+ rows
- run first SFT (1B + 0.5B)
- execute offline eval

### Week 3

- on-device benchmarking
- error analysis + dataset fixes
- second SFT run with tuned hyperparameters

### Week 4

- optional DPO pass
- export GGUF quants
- publish manifest + rollout candidate

---

## 16) References (for implementation choices)

- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- DPO paper: https://arxiv.org/abs/2305.18290
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
- HF Hub storage limits: https://huggingface.co/docs/hub/en/storage-limits
- HF Hub rate limits: https://huggingface.co/docs/hub/en/rate-limits
- llama.cpp quantization docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- Transformers + llama.cpp GGUF integration notes: https://huggingface.co/docs/transformers/community_integrations/llama_cpp
- Microsoft Presidio (PII anonymization concepts): https://microsoft.github.io/presidio/

---

## 17) Final decision summary for this app

- Start with **QLoRA SFT** on 1B + 0.5B instruct bases.
- Keep **DPO optional** for response-quality/safety polishing.
- Ship **Q4_K_M** as default quant; maintain one smaller fallback.
- Enforce strict **privacy/eval/runtime gates** before every release.
- Keep the pipeline simple, repeatable, and monthly-iterative.