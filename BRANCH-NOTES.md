# Branch notes

## Short summary

This fork speeds up the Qwen 27B model that Junie runs locally. It combines a fresh llama.cpp base, direct-KV attention patch, three not-yet-merged upstream PRs, a deeper MTP draft, and a new acceptance gate for n-gram drafting. On real Junie sessions on an RTX 3090, decode is 30-50% faster and prefill is 14-83% faster (600-700 -> 800-1100 t/s).

Use branch **`ngram-gate`**, flags: `--spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 4 --spec-ngram-gate`

## What changes included

**Direct-KV attention patch** — tag `s1-directkv` (commit `1eadf46`)
- The KV cache is stored compressed (Q4_0/Q8_0). Before, the engine unpacked the whole cache into a temporary FP16 buffer on every step. Now the attention kernel unpacks the data while it reads it, so the extra pass over memory is gone.
- *Why:* decode speed on long context is limited by memory reads; this removes the biggest extra read.
- *Results (same build, patch on vs off):* decode +8% at 26k context, +35% at 150k; no change on short context; outputs are bit-identical.

**Upstream PR [#31533](https://github.com/ggml-org/llama.cpp/pull/31533)** — merged at `s2-prs`
- Fast tensor-core matmul for small batches of 2-24 tokens.
- *Why:* speculative decoding checks several draft tokens at once; exactly these small batches were slow before.

**Upstream PR [#30139](https://github.com/ggml-org/llama.cpp/pull/30139)** — merged at `s2-prs`
- Reads the recurrent-layer (GDN) state in place, without a copy per token.
- *Why:* our model has many recurrent layers; the copies waste memory bandwidth.

**Upstream PR [#30196](https://github.com/ggml-org/llama.cpp/pull/30196)** — merged at `s2-prs`
- The CPU waits for the GPU once per decode step instead of several times.
- *Why:* fewer waits means less dead time between tokens.

**Acceptance gate for n-gram drafting** — tag `s4-ngram-gate`, new code in this fork
- An n-gram drafter proposes tokens found in the context, on top of the MTP head. The gate pauses it when its measured acceptance is low and probes it from time to time, so it turns back on when the text becomes repetitive (file rewrites).
- *Why:* n-gram drafts are almost free on repeated text but hurt on fresh text; the gate keeps the win and removes the cost.

**Adaptive MTP depth, [#27210](https://github.com/ggml-org/llama.cpp/pull/27210)** — merged at `s3-adaptive`, off by default
- *Why it is off:* on RTX 3090 it measured slower than the fixed depth 4.

## Measurements on real Junie work (RTX 3090)

Same machine and the same kind of work on both engines: some chat, then exploring and explaining projects, then refactoring and creating files. 60+ requests per engine, contexts up to ~116k tokens. Decode values are medians over the recorded requests.

| | Junie engine before | this fork |
|---|---:|---:|
| decode, context 10-60k | ~65 t/s | **97 t/s** |
| decode, context >60k | ~55-60 t/s | **91 t/s** |
| prefill, new input chunks | 600-700 t/s | **800-1100 t/s** |


## Scope & notes

> **Important:** the measurements above assume MTP draft depth **4**. The fork does not change the default (3) — pass `--spec-draft-n-max 4` when launching llama-server, or set `"draft_tokens": 4` in the model descriptor when running through the Junie runner. Without it, expect roughly 10% lower decode speed.

- The direct-KV patch activates only on SM86 GPUs (RTX 30 series, Ampere workstation cards) and only for this model's attention shape. On any other GPU or model it silently falls back to the stock path — nothing breaks, it just does not speed up. 4090/5090 would need their own tuning.
- The n-gram gate defaults (`rate 0.4, window 32, probe 8`) were tuned on RTX 3090, where draft verification is cheap thanks to #31533. On other setups start with the defaults and adjust `--spec-ngram-gate-rate` first.
