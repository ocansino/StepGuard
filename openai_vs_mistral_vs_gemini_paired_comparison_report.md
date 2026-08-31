# StepGuard Paired Three-Provider Comparison

## Executive summary

This exploratory study compares full-provider StepGuard pipelines using OpenAI `gpt-5.4-mini-2026-03-17`, Mistral `mistral-small-2603`, and Gemini `gemini-3.5-flash-lite`. Each provider supplied generation, verification, suffix repair, and acceptance judging; RoBERTa NLI remained local. All providers ran on the same 20-question StrategyQA subset and the same 20-question GSM8K subset selected with seed 2603.

The main findings are:

- Original combined accuracy was 34/40 for OpenAI, 35/40 for Mistral, and 34/40 for Gemini.
- OpenAI's risk-only and oracle policies each corrected one StrategyQA answer without a regression, producing 35/40.
- Mistral's risk-only policy made one correction and one regression, remaining at 35/40; its oracle reached 36/40 by avoiding the regression.
- Gemini remained at 34/40 under risk-only, strict-judge, updated-judge, and oracle evaluation. No Gemini policy caused an answer transition.
- Gemini's judge preferred the repair in 11 of 20 judged proposals, unlike the OpenAI and Mistral judges, which preferred none. This made Gemini's strict policy operationally active, but did not improve final accuracy.
- All three providers scored 20/20 on GSM8K before intervention. This ceiling makes GSM8K uninformative for measuring correction gains in this sample, though it still tests regression resistance.
- The sample is too small for model-quality claims. Exact paired McNemar tests on StrategyQA all gave p = 1.0.
- Gemini completed without a 429 response. It encountered three transient HTTP 503 responses during GSM8K generation; all were recovered by retries.
- Gemini latency is not an intrinsic speed comparison: its conservative three-requests-per-minute setting accounted for about 84% of its observed wall time.

The result supports treating cross-model transfer as behaviorally meaningful rather than merely an API-porting exercise. The unchanged StepGuard architecture produced substantially different proposal volumes, judge preferences, and acceptance behavior across providers.

## Controlled design

The comparison held the following elements fixed:

- identical prepared records and record IDs for each dataset;
- the existing StrategyQA and GSM8K task profiles;
- risk formulas, thresholds, candidate-pool construction, checkpointing, and metrics;
- regression-aware acceptance policies;
- local RoBERTa NLI;
- one provider for generation, verification, suffix repair, and judge calls within each pipeline;
- concurrency 1 for the externally hosted model calls.

The evaluated acceptance policies were:

- **Risk-only:** accepts a candidate when the configured risk criteria are met.
- **Strict judge:** additionally requires the provider judge to prefer the repaired answer.
- **Updated judge:** uses the updated regression-aware judge behavior.
- **Oracle:** uses gold-label awareness for analysis only and is not a deployable policy.

The Gemini runs used the account limits reported in the provider admin panel: 15 requests per minute, 250,000 tokens per minute, and 500 requests per day. The configuration applied a conservative 0.2 request-rate multiplier, yielding an effective limit of 3 requests per minute, with concurrency 1. The study made no assumption about a monthly text-token allowance because none was displayed in the account panel.

## Original-answer accuracy

| Dataset | OpenAI | Mistral | Gemini |
|---|---:|---:|---:|
| StrategyQA (n = 20) | 14/20 (70%) | 15/20 (75%) | 14/20 (70%) |
| GSM8K (n = 20) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| **Combined (n = 40)** | **34/40 (85%)** | **35/40 (87.5%)** | **34/40 (85%)** |

### Paired StrategyQA comparisons

| Pair | First provider only correct | Second provider only correct | Answer disagreements | Exact McNemar p |
|---|---:|---:|---:|---:|
| OpenAI vs Mistral | 1 | 2 | 3 | 1.0 |
| OpenAI vs Gemini | 2 | 2 | 4 | 1.0 |
| Mistral vs Gemini | 2 | 1 | 3 | 1.0 |

Across the 20 StrategyQA records, all three providers were correct on 12 and all three were wrong on 3. The remaining 5 records had mixed correctness. On GSM8K, all normalized answers were identical across the three providers and all were correct.

These paired differences are descriptive only. With so few discordant records, the experiment cannot distinguish provider quality statistically.

## Candidate generation and judge behavior

| Dataset | Provider | Proposals | Judged proposals | Judge preferred repair | Answer-changing proposals | Potential corrections | Potential regressions |
|---|---|---:|---:|---:|---:|---:|---:|
| StrategyQA | OpenAI | 10 | 8 | 0 | 1 | 1 | 0 |
| StrategyQA | Mistral | 25 | 13 | 0 | 2 | 1 | 1 |
| StrategyQA | Gemini | 28 | 14 | 9 | 1 | 1 | 0 |
| GSM8K | OpenAI | 4 | 3 | 0 | 0 | 0 | 0 |
| GSM8K | Mistral | 22 | 6 | 0 | 0 | 0 | 0 |
| GSM8K | Gemini | 8 | 6 | 2 | 0 | 0 | 0 |

Gemini is the clearest judge-behavior outlier. Its judge preferred 9 of 14 judged StrategyQA repairs and 2 of 6 judged GSM8K repairs. Consequently, strict-judge evaluation accepted 8 StrategyQA repair iterations and 2 GSM8K repair iterations. The OpenAI and Mistral judges preferred no repair in their judged proposals, so their strict policies accepted none.

Gemini produced one StrategyQA proposal whose normalized answer changed from wrong to correct, with no answer-changing regression proposal. That available correction did not become the final chain under any evaluated policy, including the oracle analysis. This indicates that proposal availability alone is insufficient: candidate-chain placement, risk eligibility, and selection mechanics determine whether a proposal can affect the final answer. The current artifacts establish the outcome but do not justify attributing it to any one mechanism without a record-level trace audit.

## Policy outcomes

### Combined final accuracy

| Policy | OpenAI | Mistral | Gemini |
|---|---:|---:|---:|
| Original, before intervention | 34/40 | 35/40 | 34/40 |
| Risk-only | **35/40** | 35/40 | 34/40 |
| Strict judge | 34/40 | 35/40 | 34/40 |
| Updated judge | 34/40 | 35/40 | 34/40 |
| Oracle | **35/40** | **36/40** | 34/40 |

### StrategyQA details

| Provider | Policy | Final correct | Wrong→correct | Correct→wrong | Repair attempts | Accepted iterations | Records with acceptance | Mean measured risk reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | Risk-only | 15/20 | 1 | 0 | 10 | 8 | 8 | 0.016724 |
| OpenAI | Strict judge | 14/20 | 0 | 0 | 8 | 0 | 0 | 0.000000 |
| OpenAI | Updated judge | 14/20 | 0 | 0 | 10 | 7 | 7 | 0.014980 |
| OpenAI | Oracle | 15/20 | 1 | 0 | 10 | 8 | 8 | 0.016724 |
| Mistral | Risk-only | 15/20 | 1 | 1 | 25 | 13 | 12 | 0.016573 |
| Mistral | Strict judge | 15/20 | 0 | 0 | 20 | 0 | 0 | 0.000000 |
| Mistral | Updated judge | 15/20 | 0 | 0 | 25 | 10 | 9 | 0.014955 |
| Mistral | Oracle | 16/20 | 1 | 0 | 25 | 12 | 11 | 0.015778 |
| Gemini | Risk-only | 14/20 | 0 | 0 | 28 | 14 | 11 | 0.027769 |
| Gemini | Strict judge | 14/20 | 0 | 0 | 26 | 8 | 8 | 0.021894 |
| Gemini | Updated judge | 14/20 | 0 | 0 | 28 | 13 | 10 | 0.027167 |
| Gemini | Oracle | 14/20 | 0 | 0 | 28 | 14 | 11 | 0.027769 |

### GSM8K details

| Provider | Policy | Final correct | Wrong→correct | Correct→wrong | Repair attempts | Accepted iterations | Records with acceptance | Mean measured risk reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | Risk-only | 20/20 | 0 | 0 | 4 | 3 | 3 | 0.014718 |
| OpenAI | Strict judge | 20/20 | 0 | 0 | 4 | 0 | 0 | 0.000000 |
| OpenAI | Updated judge | 20/20 | 0 | 0 | 4 | 3 | 3 | 0.014718 |
| OpenAI | Oracle | 20/20 | 0 | 0 | 4 | 3 | 3 | 0.014718 |
| Mistral | Risk-only | 20/20 | 0 | 0 | 22 | 6 | 6 | 0.004763 |
| Mistral | Strict judge | 20/20 | 0 | 0 | 20 | 0 | 0 | 0.000000 |
| Mistral | Updated judge | 20/20 | 0 | 0 | 22 | 6 | 6 | 0.004763 |
| Mistral | Oracle | 20/20 | 0 | 0 | 22 | 6 | 6 | 0.004763 |
| Gemini | Risk-only | 20/20 | 0 | 0 | 8 | 6 | 6 | 0.014588 |
| Gemini | Strict judge | 20/20 | 0 | 0 | 8 | 2 | 2 | 0.009194 |
| Gemini | Updated judge | 20/20 | 0 | 0 | 8 | 6 | 6 | 0.014588 |
| Gemini | Oracle | 20/20 | 0 | 0 | 8 | 6 | 6 | 0.014588 |

Measured risk reduction is provider-relative because each full-provider pipeline used that provider's verifier. Absolute risk values and deltas should therefore not be read as calibrated cross-provider confidence scores.

## Usage, latency, and reliability

The following totals cover generation and candidate-pool construction for both 20-record datasets. Replaying the four policy evaluations did not make additional provider calls because they reused the shared candidate pools.

| Provider | Logical calls | Attempts | Successful calls | Failed attempts | Retries | Input tokens | Output tokens | Total tokens | Wall time | Configured rate-limit wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | 119 | 119 | 119 | 0 | 0 | 29,270 | 10,650 | 39,920 | 187.1 s | 0.0 s |
| Mistral | 193 | 193 | 193 | 0 | 0 | 50,785 | 19,422 | 70,207 | 1,175.0 s | 909.9 s |
| Gemini | 172 | 175 | 172 | 3 | 3 | 51,775 | 17,661 | 69,436 | 3,527.7 s | 2,960.8 s |

Gemini's 2,960.8 seconds of configured waiting were approximately 84% of its 3,527.7-second wall time. Its runtime therefore primarily reflects the deliberately conservative 3-RPM policy rather than provider response latency. Mistral was also rate-limiter dominated, while the OpenAI run recorded no configured wait.

Gemini used 21 fewer logical calls and 771 fewer tokens than Mistral, but more calls and tokens than OpenAI. Its three failed attempts were HTTP 503 responses in GSM8K generation and were all recovered. There were no Gemini 429 errors. OpenAI and Mistral recorded no failed attempts in these paired exploratory runs.

The OpenAI two-provider report estimated about $0.0699 for its paired run using the price assumptions recorded there. Mistral and Gemini were run under their respective free-access modes, so no monetary cost is assigned here. Free-tier availability and terms are operational conditions, not stable experimental properties.

## Interpretation

1. **Baseline results are close and inconclusive.** Mistral answered one more of 40 records correctly than OpenAI or Gemini, but the paired StrategyQA differences are only one or two records per comparison and are not statistically meaningful.

2. **StepGuard transferred operationally to Gemini.** Generation, verification, repair, judging, retry accounting, checkpoint reuse, policy replay, and metric aggregation all completed under the existing architecture.

3. **Provider behavior changed the intervention dynamics.** Gemini generated the most StrategyQA proposals and was the only provider whose judge regularly preferred repairs. Mistral generated the most GSM8K proposals. OpenAI used the fewest calls and tokens overall.

4. **Acceptance is not equivalent to correction.** Gemini accepted repairs and measured risk reductions under every policy, yet its final answers did not change. Mistral's risk-only policy demonstrated the complementary danger: a correction and a regression can cancel in aggregate accuracy.

5. **The oracle gap differs by provider.** OpenAI's oracle matched its risk-only outcome; Mistral's oracle avoided the risk-only regression and reached 36/40; Gemini's oracle could not convert its available proposal-level correction into a final correction. Record-level chain analysis is the most useful next diagnostic if this behavior becomes a research focus.

6. **GSM8K needs a harder sample for efficacy measurement.** The shared 20-record sample is useful for checking regressions and provider plumbing, but its 100% baseline leaves no opportunity for wrong-to-correct improvement.

## Limitations

- Each dataset contains only 20 records, selected with one seed.
- GSM8K is ceiling-limited in this sample.
- The providers' own verifiers define their risk signals, so risk magnitudes are not directly calibrated across models.
- The study does not isolate model capability from prompt sensitivity, provider serving behavior, or sampling variance.
- The observed rate-limit behavior applies only to the account state and configuration used for these runs.
- `mistral-small-2603` and `gpt-5.4-mini-2026-03-17` are dated model identifiers. `gemini-3.5-flash-lite` is a stable non-`latest` identifier but is not a dated immutable snapshot, which weakens long-term reproducibility if its backend changes.
- Oracle evaluation is diagnostic and cannot be used in a real inference-time deployment.
- Proposal-level availability does not guarantee that a candidate is eligible or selected as the final repair.

## Recommended next slice

The current three-provider comparison is sufficient as an exploratory transfer result. Before expanding sample size, the highest-value offline analysis would be a record-level trace of:

- Gemini's one answer-changing StrategyQA proposal and why it was not selected by the oracle replay;
- Mistral's StrategyQA correct-to-wrong risk-only regression and why the oracle rejected it;
- OpenAI's StrategyQA correction and which policy conditions made it selectable.

That focused comparison would directly expose where risk gating, judge gating, and candidate-chain selection differ, without spending additional API quota. If a larger live study is later desired, use a harder or deliberately error-enriched GSM8K subset and multiple seeds rather than simply enlarging this ceiling-limited sample.

## Reproducibility artifacts

Prepared paired subsets:

- `data/prepared/strategyqa_mistral_20_seed2603.jsonl`
- `data/prepared/gsm8k_mistral_20_seed2603.jsonl`
- `data/prepared/manifest.strategyqa_mistral_20_seed2603.json`
- `data/prepared/manifest.gsm8k_mistral_20_seed2603.json`

Shared generation and candidate-pool runs:

- `data/runs/strategyqa_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared`
- `data/runs/gsm8k_openai_gpt_5_4_mini_2026_03_17_comparison_20_seed2603_shared`
- `data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_shared`
- `data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_shared`
- `data/runs/strategyqa_gemini_3_5_flash_lite_comparison_20_seed2603_shared`
- `data/runs/gsm8k_gemini_3_5_flash_lite_comparison_20_seed2603_shared`

For each provider and dataset, sibling run directories ending in `_strict_judge`, `_updated_judge`, and `_oracle` contain the corresponding policy evaluation. The `_shared` directory contains the risk-only evaluation and the shared live-call artifacts.

Related reports:

- `mistral_cross_model_transfer_report.md`
- `openai_vs_mistral_paired_comparison_report.md`

No submitted StrategyQA artifact or earlier GSM8K artifact was overwritten in producing this comparison.
