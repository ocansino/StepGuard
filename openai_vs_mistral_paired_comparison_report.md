# StepGuard Paired Provider Comparison: OpenAI vs. Mistral

**Study status:** Exploratory paired comparison  
**OpenAI model:** `gpt-5.4-mini-2026-03-17`  
**Mistral model:** `mistral-small-2603`  
**Datasets:** StrategyQA and GSM8K  
**Records:** 20 per dataset, identical across providers  
**Subset seed:** 2603  
**Run date:** 2026-08-30 PDT / 2026-08-31 UTC

## Executive summary

On the 40 paired examples, Mistral began one answer ahead of OpenAI: 35/40
(87.5%) versus 34/40 (85%). OpenAI's risk-only repair corrected one StrategyQA
answer without causing a regression, bringing both providers to 35/40. Mistral
risk-only also corrected one answer but introduced one regression, leaving its
accuracy unchanged.

The provider comparison does not establish a statistically meaningful accuracy
winner. Only three StrategyQA questions had different correctness outcomes
between the original providers: Mistral alone was correct on two and OpenAI
alone was correct on one. All 20 GSM8K answers matched and were correct.

Operationally, OpenAI used fewer calls and tokens. Its two full live pipelines
used 119 provider calls and 39,920 tokens, versus 193 calls and 70,207 tokens
for Mistral. OpenAI's estimated API cost was $0.0699 at the published standard
token prices used for this report. Mistral ran in Free mode.

The most important shared weakness was acceptance judging. Neither provider's
judge preferred a repaired trace in any judged proposal: 0/11 for OpenAI and
0/19 for Mistral. In both providers, this original-answer bias blocked a real
StrategyQA correction.

## Controlled comparison design

The comparison held the following constant:

- exact JSONL dataset bytes and record IDs;
- task profiles and prompt builders;
- temperature 0.2 and maximum output length 800;
- weighted risk formula;
- verifier weight 0.75 and contradiction weight 0.25;
- risk threshold 0.20 and improvement threshold 0.02;
- maximum of two suffix-repair iterations;
- support tolerance 0.05 and maximum regression risk 0.35;
- concurrency 1 and five-record checkpoint chunks; and
- local `FacebookAI/roberta-large-mnli` contradiction scoring.

Each provider performed its own generation, verification, repair, and acceptance
judging. Candidate pools were frozen once per provider. Risk-only, strict judge,
updated judge, and oracle policies were then replayed offline against the same
provider-specific pool.

The OpenAI model used the dated snapshot `gpt-5.4-mini-2026-03-17`. The Mistral
model used the pinned `mistral-small-2603` identifier rather than a `-latest`
alias.

## Original-answer comparison

| Dataset | OpenAI | Mistral | Net difference |
|---|---:|---:|---:|
| StrategyQA | 14/20 (70%) | 15/20 (75%) | Mistral +1 |
| GSM8K | 20/20 (100%) | 20/20 (100%) | Tie |
| **Combined** | **34/40 (85%)** | **35/40 (87.5%)** | **Mistral +1** |

### Paired StrategyQA outcomes

| Outcome | Records |
|---|---:|
| Both correct | 13 |
| Both wrong | 4 |
| Mistral correct, OpenAI wrong | 2 |
| OpenAI correct, Mistral wrong | 1 |

With only three discordant pairs, this sample cannot support a provider-quality
claim. A two-sided exact McNemar test yields `p = 1.0`.

### StrategyQA answer disagreements

| Record | Gold | OpenAI | Mistral | Correct provider |
|---|---|---|---|---|
| `52e1b2df624813a9d66b` | no | yes | no | Mistral |
| `691c8df42c886d6db9d4` | no | no | yes | OpenAI |
| `01c3faf4915a44133f60` | yes | no | yes | Mistral |

Both providers produced the same normalized answer for every GSM8K record.

## Candidate-pool behavior

| Dataset | Provider | Proposals | Judged | Judge preferred repair | Mean original risk | Mean risky steps |
|---|---|---:|---:|---:|---:|---:|
| StrategyQA | OpenAI | 10 | 8 | 0 | 0.08738 | 0.45 |
| StrategyQA | Mistral | 25 | 13 | 0 | 0.15911 | 1.35 |
| GSM8K | OpenAI | 4 | 3 | 0 | 0.06768 | 0.25 |
| GSM8K | Mistral | 22 | 6 | 0 | 0.18386 | 1.10 |

OpenAI's verifier assigned substantially lower risk to OpenAI traces and
therefore triggered fewer repairs. This must not be interpreted directly as
proof that OpenAI reasoning is safer: each provider verified its own output, so
provider-specific calibration and self-evaluation bias are confounded with
trace quality. A cross-verifier matrix would be needed to separate them.

## Acceptance-policy results

### StrategyQA

| Provider | Policy | Original | Final | Wrong→correct | Correct→wrong | Attempts | Accepted | Risk reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | Risk-only | 70% | 75% | 1 | 0 | 10 | 8 | 0.01672 |
| OpenAI | Strict judge | 70% | 70% | 0 | 0 | 8 | 0 | 0 |
| OpenAI | Updated judge | 70% | 70% | 0 | 0 | 10 | 7 | 0.01498 |
| OpenAI | Oracle | 70% | 75% | 1 | 0 | 10 | 8 | 0.01672 |
| Mistral | Risk-only | 75% | 75% | 1 | 1 | 25 | 13 | 0.01657 |
| Mistral | Strict judge | 75% | 75% | 0 | 0 | 20 | 0 | 0 |
| Mistral | Updated judge | 75% | 75% | 0 | 0 | 25 | 10 | 0.01495 |
| Mistral | Oracle | 75% | 80% | 1 | 0 | 25 | 12 | 0.01578 |

### GSM8K

| Provider | Policy | Original | Final | Wrong→correct | Correct→wrong | Attempts | Accepted | Risk reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | Risk-only | 100% | 100% | 0 | 0 | 4 | 3 | 0.01472 |
| OpenAI | Strict judge | 100% | 100% | 0 | 0 | 4 | 0 | 0 |
| OpenAI | Updated judge | 100% | 100% | 0 | 0 | 4 | 3 | 0.01472 |
| OpenAI | Oracle | 100% | 100% | 0 | 0 | 4 | 3 | 0.01472 |
| Mistral | Risk-only | 100% | 100% | 0 | 0 | 22 | 6 | 0.00476 |
| Mistral | Strict judge | 100% | 100% | 0 | 0 | 20 | 0 | 0 |
| Mistral | Updated judge | 100% | 100% | 0 | 0 | 22 | 6 | 0.00476 |
| Mistral | Oracle | 100% | 100% | 0 | 0 | 22 | 6 | 0.00476 |

`Accepted` is the number of accepted repair iterations. No accepted GSM8K
repair changed a normalized answer.

### Combined accuracy by policy

| Policy | OpenAI | Mistral |
|---|---:|---:|
| Original | 34/40 (85%) | 35/40 (87.5%) |
| Risk-only | 35/40 (87.5%) | 35/40 (87.5%) |
| Strict judge | 34/40 (85%) | 35/40 (87.5%) |
| Updated judge | 34/40 (85%) | 35/40 (87.5%) |
| Oracle | 35/40 (87.5%) | 36/40 (90%) |

## Representative acceptance failures

### OpenAI judge blocked an OpenAI correction

For StrategyQA record `52e1b2df624813a9d66b`, OpenAI originally answered
`yes` to whether someone in the Canary Islands could fish for largemouth bass.
The gold answer was `no`. OpenAI's repair changed the answer to `no` and reduced
average risk from 0.11585 to 0.08098, an improvement of 0.03487. Despite this,
the OpenAI judge preferred the incorrect original, scoring original support
0.78, repaired support 0.42, and regression risk 0.74. Risk-only and oracle
accepted the correction; both judge guards rejected it.

### Mistral risk-only admitted a regression

For StrategyQA record `e1f93419cb9a2f1d06ca`, Mistral changed the correct
answer from `yes` to `no` while reducing measured average risk by 0.01591. The
Mistral judge correctly assigned the repair lower support and high regression
risk. Risk-only admitted the regression, while both judge-aware policies and
the oracle blocked it.

### Both judges showed original-answer bias

The OpenAI judge rejected all 11 judged repairs, including its real correction.
The Mistral judge rejected all 19 judged repairs, including a different real
correction. Strict judge therefore accepted nothing for either provider. The
updated judge recovered answer-preserving risk reductions, but it still blocked
both providers' observed corrections because their repaired-answer support did
not satisfy the configured tolerance.

## Usage, latency, and cost

| Dataset | Provider | Calls | Input tokens | Output tokens | Total tokens | Wall time | Rate-limit wait |
|---|---|---:|---:|---:|---:|---:|---:|
| StrategyQA | OpenAI | 68 | 16,879 | 5,681 | 22,560 | 108.18 s | 0 s |
| StrategyQA | Mistral | 103 | 26,738 | 9,426 | 36,164 | 624.45 s | 486.80 s |
| GSM8K | OpenAI | 51 | 12,391 | 4,969 | 17,360 | 78.92 s | 0 s |
| GSM8K | Mistral | 90 | 24,047 | 9,996 | 34,043 | 550.55 s | 423.08 s |
| **Combined** | **OpenAI** | **119** | **29,270** | **10,650** | **39,920** | **187.10 s** | **0 s** |
| **Combined** | **Mistral** | **193** | **50,785** | **19,422** | **70,207** | **1,175.00 s** | **909.88 s** |

At the OpenAI standard prices recorded for this study—$0.75 per million input
tokens and $4.50 per million output tokens—the estimated cost was:

| Dataset | Estimated OpenAI cost |
|---|---:|
| StrategyQA | $0.0382 |
| GSM8K | $0.0317 |
| **Combined** | **$0.0699** |

The OpenAI price source is the official
[`gpt-5.4-mini` model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
Mistral used Free mode, so no direct token charge is assigned here.

The wall-time ratio mainly reflects the configured account limits. Mistral used
an effective 0.166 requests per second, while OpenAI had no measured limiter
wait. These wall times are operational observations, not a controlled inference
latency benchmark.

## Conclusions

1. **Raw accuracy was effectively tied at this scale.** Mistral's one-question
   original lead disappeared under OpenAI risk-only repair. The sample is too
   small for a provider-ranking claim.
2. **OpenAI was more intervention-efficient in this sample.** It used 38% fewer
   calls and 43% fewer tokens, largely because its verifier triggered far fewer
   repairs.
3. **Verifier scores are not calibrated across providers.** The large risk-scale
   difference is confounded by self-verification and should not be used as a
   direct provider-quality metric.
4. **Strict judging is not viable without recalibration.** It eliminated every
   accepted repair for both providers.
5. **Updated judging traded corrections for safety.** It prevented Mistral's
   observed regression but also blocked both providers' observed corrections.
6. **Risk-only happened to work better for OpenAI here, but not for Mistral.**
   That asymmetry is exactly why a larger paired study and calibrated acceptance
   mechanism are needed.
7. **GSM8K remains ceiling-limited.** Both models were perfect before repair, so
   the subset primarily measured process risk, cost, and regression avoidance.

## Recommended follow-up

1. Run a cross-verifier matrix: OpenAI verifier on both providers' traces and
   Mistral verifier on both providers' traces, while keeping the same local NLI
   scores. This would separate trace quality from self-verifier calibration.
2. Calibrate acceptance-judge support and regression thresholds on frozen
   labeled candidates. Both current judges show strong original-answer bias.
3. Repeat the paired comparison over multiple fixed seeds and report paired
   confidence intervals.
4. Use a harder or error-enriched GSM8K subset so correction opportunity is not
   eliminated by a perfect baseline.
5. Commit the current code and configurations before further live runs; current
   manifests record the base commit but not the uncommitted provider changes.

## Reproducibility artifacts

### OpenAI shared pools

- [`data/runs/strategyqa_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared`](data/runs/strategyqa_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared)
- [`data/runs/gsm8k_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared`](data/runs/gsm8k_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared)

### Mistral shared pools

- [`data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_shared`](data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_shared)
- [`data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_shared`](data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_shared)

Each shared directory contains generation and candidate-pool manifests, provider
metrics, generated traces, the immutable candidate pool, risk-only results, and
per-record evaluation. Strict, updated, and oracle replays are stored in sibling
run directories with matching suffixes.
