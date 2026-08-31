# StepGuard Mistral Cross-Model Transfer: Exploratory Report

**Study status:** Exploratory, offline-replay complete  
**Model:** `mistral-small-2603`  
**Run date:** 2026-08-30 PDT / 2026-08-31 UTC  
**Datasets:** StrategyQA and GSM8K  
**Subset seed:** 2603

**Paired provider follow-up:**
[`openai_vs_mistral_paired_comparison_report.md`](openai_vs_mistral_paired_comparison_report.md)

## Executive summary

The Mistral integration completed the full StepGuard pipeline—trace generation,
step verification, suffix repair, and acceptance judging—with RoBERTa NLI kept
local. Across the two 20-record exploratory subsets, all 193 Mistral requests
succeeded, 70,207 tokens were reported, and there were no retries, failures, or
HTTP 429 responses.

The main result is policy-dependent:

- On StrategyQA, risk-only repair exchanged one wrong-to-correct correction for
  one correct-to-wrong regression, leaving accuracy unchanged at 75%. The
  updated judge guard prevented both answer changes while retaining most of the
  measured risk reduction. The oracle guard retained the correction, blocked
  the regression, and raised accuracy from 75% to 80%.
- On GSM8K, the initial Mistral traces were already 20/20 correct. Risk-only,
  updated-judge, and oracle policies accepted the same six answer-preserving
  repairs and kept accuracy at 100%. A proposed `48 grams` to `240 grams`
  regression increased risk and was rejected before judging.
- The strict judge guard accepted no repairs on either exploratory subset. The
  Mistral judge preferred the original answer in all 19 judgments across the
  two frozen candidate pools, making this policy too conservative in this
  sample.

These results establish operational cross-model transfer, but they are not a
confirmatory effectiveness result. The samples are small, use one selection
seed, and have no repeated stochastic runs.

## Research question

Does StepGuard's calibrated inference-time intervention pipeline transfer from
its existing providers to a pinned Mistral model without changing the task
profiles, risk formula, thresholds, candidate-pool architecture,
checkpointing, metrics, or regression-aware acceptance behavior?

## Experimental design

### Model roles

`mistral-small-2603` was used for:

1. initial trace generation;
2. per-step verification;
3. suffix repair; and
4. acceptance judging.

`FacebookAI/roberta-large-mnli` remained local for contradiction scoring.

### Preserved StepGuard settings

| Setting | Value |
|---|---:|
| Risk formula | weighted |
| Verifier weight | 0.75 |
| Contradiction weight | 0.25 |
| Risk threshold | 0.20 |
| Required improvement | 0.02 |
| Maximum repair iterations | 2 |
| Support tolerance | 0.05 |
| Maximum regression risk | 0.35 |
| Temperature | 0.2 |
| Maximum output tokens | 800 |

The primary shared candidate pools were constructed as policy-neutral,
risk-only chains. Strict judge, updated judge, and oracle policies were then
replayed offline against those immutable pools, so policy comparisons did not
consume additional API calls or resample candidates.

### Rate limiting

The account's observed Free-mode limits for `mistral-small-2603` were 0.83
requests per second and 50,000 tokens per minute. Runs used concurrency 1 and a
0.20 headroom fraction, producing an effective request rate of 0.166 requests
per second, or approximately one request every six seconds. The admin panel did
not expose a monthly text-token allowance; the configuration records that value
as `null`, which means unpublished rather than unlimited.

### Dataset selection

Each exploratory subset contains 20 records selected using
`random.Random(2603).sample` after excluding the corresponding five smoke-test
IDs, with selected records restored to source order.

| Dataset | Source pool | Eligible records | Subset | Output SHA-256 |
|---|---:|---:|---:|---|
| StrategyQA | 100 | 100 | 20 | `f0a0ec86ea890d69409734fe3805fa4bc4c2f464dd34d5683d54eb42436c1433` |
| GSM8K | 100 | 95 | 20 | `597c82a6c5827d7a46e9971229419b02ce6dba003f114d37d026aadbac46c9ad` |

The StrategyQA subset contains 12 `no` and 8 `yes` gold answers. Full selected
ID lists and source hashes are recorded in the preparation manifests.

## Smoke-test results

| Dataset | Records | Original accuracy | Final accuracy | Attempts | Accepted | Mean risk reduction | Mistral calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StrategyQA | 5 | 60% | 60% | 6 | 1 | 0.00423 | 23 | 8,214 |
| GSM8K | 5 | 100% | 100% | 6 | 2 | 0.00909 | 24 | 8,260 |
| Combined | 10 | 80% | 80% | 12 | 3 | — | 47 | 16,474 |

All smoke-test calls succeeded without retries, failures, or 429 responses.

## Exploratory execution results

### Provider usage and latency

| Dataset | Stage | Records | Mistral calls | Input tokens | Output tokens | Total tokens | Wall time | Rate-limit wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| StrategyQA | Generation | 20 | 20 | 2,643 | 2,071 | 4,714 | 116.85 s | 90.89 s |
| StrategyQA | Candidate pool | 20 | 83 | 24,095 | 7,355 | 31,450 | 507.60 s | 395.90 s |
| GSM8K | Generation | 20 | 20 | 2,694 | 2,676 | 5,370 | 125.79 s | 82.58 s |
| GSM8K | Candidate pool | 20 | 70 | 21,353 | 7,320 | 28,673 | 424.76 s | 340.50 s |
| **Combined** | **All live stages** | **40** | **193** | **50,785** | **19,422** | **70,207** | **1,174.99 s** | **909.87 s** |

The candidate-pool stages also performed nine local RoBERTa batches per
dataset. These local operations are excluded from the Mistral call counts.

### Reliability

| Measure | StrategyQA | GSM8K | Combined |
|---|---:|---:|---:|
| Mistral attempts | 103 | 90 | 193 |
| Successful calls | 103 | 90 | 193 |
| Retries | 0 | 0 | 0 |
| Failures | 0 | 0 | 0 |
| HTTP 429 errors | 0 | 0 | 0 |
| Candidate-pool generation errors | 0 | 0 | 0 |

The deliberately conservative request cadence accounted for 77% of measured
end-to-end wall time. This avoided rate-limit events, but latency should not be
treated as an intrinsic model-speed benchmark.

### Frozen candidate pools

| Dataset | Records | Proposals | Iteration 1 | Iteration 2 | Judged proposals | Judge preferred repair |
|---|---:|---:|---:|---:|---:|---:|
| StrategyQA | 20 | 25 | 20 | 5 | 13 | 0 |
| GSM8K | 20 | 22 | 20 | 2 | 6 | 0 |

## Acceptance-policy comparison

| Dataset | Policy | Original accuracy | Final accuracy | Wrong→correct | Correct→wrong | Attempts | Accepted iterations | Accepted records | Mean risk reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StrategyQA | Risk-only | 75% | 75% | 1 | 1 | 25 | 13 | 12 | 0.01657 |
| StrategyQA | Strict judge | 75% | 75% | 0 | 0 | 20 | 0 | 0 | 0 |
| StrategyQA | Updated judge | 75% | 75% | 0 | 0 | 25 | 10 | 9 | 0.01495 |
| StrategyQA | Oracle | 75% | 80% | 1 | 0 | 25 | 12 | 11 | 0.01578 |
| GSM8K | Risk-only | 100% | 100% | 0 | 0 | 22 | 6 | 6 | 0.00476 |
| GSM8K | Strict judge | 100% | 100% | 0 | 0 | 20 | 0 | 0 | 0 |
| GSM8K | Updated judge | 100% | 100% | 0 | 0 | 22 | 6 | 6 | 0.00476 |
| GSM8K | Oracle | 100% | 100% | 0 | 0 | 22 | 6 | 6 | 0.00476 |

`Accepted iterations` can exceed `accepted records` because a record may accept
more than one repair iteration.

## Representative cases

### StrategyQA correction available only to risk-only and oracle

- **Record:** `691c8df42c886d6db9d4`
- **Question:** Would John the Baptist be invited to a hypothetical cephalophore
  reunion in heaven?
- **Gold:** `no`
- **Original:** `yes`
- **Candidate:** `no`
- **Average risk:** 0.15815 → 0.15150
- **Measured improvement:** 0.00665
- **Judge:** preferred the original; original support 0.9, repaired support 0.8,
  regression risk 0.3.

The candidate corrected the answer, but the measured risk improvement was below
the configured 0.02 threshold. Risk-only accepted it as part of its frozen
chain, the updated judge guard rejected it under the support constraint, and
the oracle retained it because it knew the gold answer. This is evidence that
risk reduction and answer correctness are related imperfectly.

### StrategyQA regression caught by judge-aware and oracle policies

- **Record:** `e1f93419cb9a2f1d06ca`
- **Question:** Does Hammurabi's Code violate Christians Golden Rule?
- **Gold:** `yes`
- **Original:** `yes`
- **Candidate:** `no`
- **Average risk:** 0.17429 → 0.15838
- **Measured improvement:** 0.01591
- **Judge:** preferred the original; original support 0.9, repaired support 0.6,
  regression risk 0.7.

Risk-only accepted a lower-risk but incorrect candidate. Both judge guards and
the oracle blocked it. This is the clearest observed example of why risk
reduction alone is not a sufficient acceptance criterion.

### GSM8K regression rejected by risk before judging

- **Record:** `gsm8k_test_000043`
- **Question:** A 300 g bag of chips has five 250-calorie servings; how many
  grams fit the remaining 200-calorie allowance?
- **Gold:** `48`
- **Original:** `48 grams`
- **Candidate:** `240 grams`
- **Average risk:** 0.19806 → 0.21255
- **Measured improvement:** -0.01449
- **Judge:** not called.

The repair confused calories with grams, increased measured risk, and was
rejected by every policy before acceptance judging. This case demonstrates that
the existing risk signal can detect some arithmetic regressions directly.

## Interpretation

1. **Provider transfer succeeded operationally.** The pinned Mistral model
   completed every StepGuard role with complete usage reporting and no observed
   retry or rate-limit failures.
2. **Regression-aware acceptance remains necessary.** StrategyQA produced both
   a correction and a regression under risk-only selection, canceling its net
   accuracy gain.
3. **The updated judge guard is the strongest deployable policy in this
   sample.** It preserved 90% of StrategyQA's risk-only reduction
   (`0.01495 / 0.01657`) while eliminating the observed correct-to-wrong
   regression. On GSM8K it matched risk-only and oracle exactly.
4. **The strict judge is too conservative as currently calibrated.** Because
   the judge never preferred a repair, strict judging eliminated all measured
   intervention benefit.
5. **Oracle performance identifies policy headroom.** On StrategyQA, the same
   frozen candidates could raise accuracy by five percentage points if the
   correction were selected without admitting the regression.
6. **GSM8K has a ceiling effect.** With a 100% original baseline, this subset can
   measure regression avoidance and process-risk changes but cannot demonstrate
   wrong-to-correct improvement.

## Limitations

- Each exploratory subset contains only 20 examples and is not statistically
  powered for a general performance claim.
- Only one deterministic subset seed was used. Generation and repair still used
  temperature 0.2, with no repeated stochastic trials.
- The same Mistral model generated, verified, repaired, and judged traces. Error
  correlation between roles may limit the independence of the guard signal.
- The StrategyQA and GSM8K subsets were drawn from existing 100-record prepared
  pools, not directly from their complete benchmark test sets.
- The GSM8K subset's perfect baseline obscures correction capability.
- Free-mode limits are account-specific. No text-model monthly allowance was
  visible in the admin panel, so monthly feasibility is unknown.
- Run manifests record Git commit
  `02a11008e5b9c2a03bfeaf53f06f27f01e279621`, but the Mistral integration was
  present as uncommitted working-tree changes. The commit hash alone therefore
  does not fully identify the executed code state.

## Recommended next steps

1. Treat `judge_guard` as the primary non-oracle Mistral acceptance policy;
   retain risk-only and strict judge as ablations.
2. Review and recalibrate the strict judge prompt or decision threshold before
   spending quota on a larger strict-policy study.
3. Commit the provider integration, configurations, tests, and subset manifests
   before any new live experiment so future manifests identify an exact code
   state. First remove the trailing whitespace currently reported at
   `src/cli.py:162`.
4. If quota permits, repeat with multiple fixed seeds and include GSM8K examples
   where the baseline model is not already perfect. Report confidence intervals
   rather than relying on a single 20-record point estimate.
5. Preserve the frozen candidate pools and use offline policy replay for any
   further threshold or acceptance-policy sensitivity analysis.

## Reproducibility artifacts

### Prepared subsets

- [`data/prepared/strategyqa_mistral_20_seed2603.jsonl`](data/prepared/strategyqa_mistral_20_seed2603.jsonl)
- [`data/prepared/manifest.strategyqa_mistral_20_seed2603.json`](data/prepared/manifest.strategyqa_mistral_20_seed2603.json)
- [`data/prepared/gsm8k_mistral_20_seed2603.jsonl`](data/prepared/gsm8k_mistral_20_seed2603.jsonl)
- [`data/prepared/manifest.gsm8k_mistral_20_seed2603.json`](data/prepared/manifest.gsm8k_mistral_20_seed2603.json)

### Shared live-run artifacts

- [`data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_shared`](data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_shared)
- [`data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_shared`](data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_shared)

Each shared directory contains generation and candidate-pool manifests, metrics,
generated traces, the frozen candidate pool, the risk-only replay, and per-record
evaluation output.

### Offline policy replays

- StrategyQA: `strict_judge`, `updated_judge`, and `oracle` sibling run
  directories under `data/runs/strategyqa_mistral_small_2603_exploratory_20_seed2603_*`.
- GSM8K: `strict_judge`, `updated_judge`, and `oracle` sibling run directories
  under `data/runs/gsm8k_mistral_small_2603_exploratory_20_seed2603_*`.

All 96 offline unit tests passed after adding the Mistral policy configurations
and contract coverage.
