from pathlib import Path
from threading import local
from typing import Optional, List, Dict, Tuple, Any
from .datasets_prep import prepare_gsm8k, prepare_strategyqa_local
import typer
import json
from .config import load_config
from .acceptance import JUDGE_MODES, evaluate_repair_acceptance
from .candidate_pool import apply_candidate_pool_record
from .io_utils import read_jsonl, write_jsonl
from .manifest import make_manifest, write_manifest
from .execution_metrics import ExecutionMetrics
from .execution import map_bounded
from .parsing import split_steps, validate_score_alignment
from .risk import compute_risks, rescore_frozen_trace_score, summarize_risk
from .schemas import make_base_record
from .task_profiles import answers_match, get_task_profile, normalize_math_answer
from .candidate_pool_checkpoint import (
    delete_candidate_pool_checkpoint,
    load_candidate_pool_checkpoint,
    make_candidate_pool_checkpoint_identity,
    save_candidate_pool_checkpoint,
)

from .generation_checkpoint import (
    delete_generation_checkpoint,
    load_generation_checkpoint,
    make_generation_checkpoint_identity,
    save_generation_checkpoint,
)

from .providers.gemini_client import GeminiClient
from .providers.openai_client import OpenAIClientWrapper

from .scorers.nli import NLIScorer
from .rate_limiter import RequestRateLimiter

app = typer.Typer(add_completion=False)

def run_dir(base_output_dir: str, run_name: str) -> Path:
    return Path(base_output_dir) / run_name

def create_execution_metrics(
    cfg: Any,
    command: str,
    outdir: Path,
) -> ExecutionMetrics:
    execution_cfg = cfg.raw.get("execution", {}) or {}
    if not isinstance(execution_cfg, dict):
        raise ValueError("execution configuration must be a mapping")

    filename = f"metrics.{command.replace('-', '_')}.json"

    return ExecutionMetrics(
        run_name=cfg.run_name,
        command=command,
        output_path=outdir / filename,
        execution_config=execution_cfg,
        context={
            "dataset_path": cfg.dataset_path,
            "task_profile": cfg.raw.get("task_profile"),
            "model": dict(cfg.raw.get("model", {})),
        },
        enabled=bool(execution_cfg.get("metrics_enabled", True)),
        flush_interval_events=execution_cfg.get(
            "metrics_flush_interval_events",
            25,
        ),
    )

def resolve_openai_execution_options(
    execution_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "request_timeout_seconds": float(
            execution_cfg.get(
                "request_timeout_seconds",
                120.0,
            )
        ),
        "max_retries": int(
            execution_cfg.get("max_retries", 2)
        ),
        "retry_base_delay_seconds": float(
            execution_cfg.get(
                "retry_base_delay_seconds",
                1.0,
            )
        ),
        "retry_jitter_fraction": float(
            execution_cfg.get(
                "retry_jitter_fraction",
                0.2,
            )
        ),
    }

def create_openai_rate_limiter(
    execution_cfg: Dict[str, Any],
    metrics: ExecutionMetrics,
) -> RequestRateLimiter:
    configured_rpm = execution_cfg.get(
        "openai_requests_per_minute"
    )

    requests_per_minute = (
        None
        if configured_rpm is None
        else float(configured_rpm)
    )

    return RequestRateLimiter(
        requests_per_minute=requests_per_minute,
        headroom_fraction=float(
            execution_cfg.get(
                "rate_limit_headroom_fraction",
                0.8,
            )
        ),
        metrics=metrics,
    )

def normalize_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    return normalize_math_answer(text)


def resolve_record_task(
    record: Dict[str, Any],
    config_raw: Optional[Dict[str, Any]] = None,
) -> str:
    override = (config_raw or {}).get("task_profile")
    requested_task = override if override is not None else record.get("task")
    return get_task_profile(requested_task).name


def gsm8k_exact_match(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_gsm8k_answer(pred)
    g = normalize_gsm8k_answer(gold)
    return (p is not None) and (g is not None) and (p == g)

def answer_is_correct(task: Optional[str], pred: Optional[str], gold: Optional[str]) -> bool:
    if gold is None or pred is None:
        return False
    return answers_match(task, pred, gold)

def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def classification_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def get_original_risk_summary(rec: Dict[str, Any], tau: float) -> Dict[str, Any]:
    iterative_log = rec.get("logs", {}).get("iterative_repair", {})
    iterations = iterative_log.get("iterations", [])

    if iterations:
        first = iterations[0]
        return {
            "avg_risk": float(first.get("avg_risk", 0.0)),
            "max_risk": float(first.get("max_risk", 0.0)),
            "num_risky_steps": int(first.get("num_risky_steps", 0)),
        }

    risks = rec.get("risks", [])
    return summarize_risk(risks, tau)

def get_final_risk_summary(rec: Dict[str, Any], tau: float) -> Dict[str, Any]:
    final_risks = rec.get("final_risks")

    if final_risks is not None:
        return summarize_risk(final_risks, tau)

    risks = rec.get("risks", [])
    return summarize_risk(risks, tau)

def decide_repair_acceptance(
    *,
    mode: str,
    rec: Dict[str, Any],
    judge_client: Any,
    current_trace: str,
    current_answer: str,
    candidate_trace: str,
    candidate_answer: str,
    old_avg_risk: float,
    new_avg_risk: float,
    judge_result: Optional[Dict[str, Any]] = None,
    judge_error: Optional[str] = None,
    support_tolerance: float = 0.05,
    max_regression_risk: float = 0.35,
) -> Dict[str, Any]:
    if (
        mode in JUDGE_MODES
        and old_avg_risk - new_avg_risk >= 0.0
        and judge_result is None
        and judge_error is None
    ):
        try:
            judge_result = judge_client.judge_repair_candidate(
                question=rec["question"],
                original_trace=current_trace,
                original_answer=current_answer,
                repaired_trace=candidate_trace,
                repaired_answer=candidate_answer,
                task=resolve_record_task(rec),
            )
        except Exception as error:
            judge_error = str(error)

    return evaluate_repair_acceptance(
        mode=mode,
        task=resolve_record_task(rec),
        gold_answer=rec.get("gold_answer"),
        current_answer=current_answer,
        candidate_answer=candidate_answer,
        old_avg_risk=old_avg_risk,
        new_avg_risk=new_avg_risk,
        judge_result=judge_result,
        judge_error=judge_error,
        support_tolerance=support_tolerance,
        max_regression_risk=max_regression_risk,
    )

def assemble_trace_score(
    *,
    steps: List[str],
    judge_results: List[Dict[str, Any]],
    nli: NLIScorer,
    tau: float,
    scoring_cfg: Optional[Dict[str, Any]] = None,
    contradiction_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    scoring_cfg = scoring_cfg or {}

    verifier = [0.5] * len(steps)
    for item in judge_results:
        try:
            index = int(item.get("step_index"))
            p_wrong = float(item.get("p_wrong"))
            if 0 <= index < len(steps):
                verifier[index] = max(0.0, min(1.0, p_wrong))
        except Exception:
            continue

    if contradiction_scores is None:
        contradiction = []

        for index, step in enumerate(steps):
            if index == 0:
                contradiction.append(0.0)
                continue

            previous_steps = steps[
                max(0, index - 2):index
            ]
            premise = "\n".join(previous_steps)
            p_contradiction = (
                nli.contradiction_prob(
                    premise=premise,
                    hypothesis=step,
                )
            )
            contradiction.append(p_contradiction)
    else:
        contradiction = [
            float(value)
            for value in contradiction_scores
        ]

    validate_score_alignment(
        steps,
        verifier,
        contradiction,
    )

    risks = compute_risks(
        verifier,
        contradiction,
        scoring_cfg,
    )
    earliest = next(
        (
            index
            for index, risk in enumerate(risks)
            if risk > tau
        ),
        None,
    )

    return {
        "steps": steps,
        "scores": {
            "verifier": verifier,
            "contradiction": contradiction,
            "evidence_support": None,
        },
        "risks": risks,
        "earliest_bad_step": earliest,
        "risk_summary": summarize_risk(risks, tau),
    }

def assemble_trace_scores_batch(
    *,
    items: List[Dict[str, Any]],
    nli: NLIScorer,
    tau: float,
    scoring_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    all_pairs = []
    pair_counts = []

    for item in items:
        steps = item["steps"]
        trace_pairs = []

        for index in range(1, len(steps)):
            previous_steps = steps[
                max(0, index - 2):index
            ]
            trace_pairs.append(
                (
                    "\n".join(previous_steps),
                    steps[index],
                )
            )

        pair_counts.append(len(trace_pairs))
        all_pairs.extend(trace_pairs)

    if all_pairs:
        all_contradictions = (
            nli.contradiction_probs(all_pairs)
        )
    else:
        all_contradictions = []

    if len(all_contradictions) != len(all_pairs):
        raise ValueError(
            "Batched NLI result count does not match "
            "the requested pair count"
        )

    scores = []
    contradiction_offset = 0

    for item, pair_count in zip(
        items,
        pair_counts,
    ):
        steps = item["steps"]
        trace_contradictions = (
            [0.0]
            if steps
            else []
        )
        trace_contradictions.extend(
            all_contradictions[
                contradiction_offset:
                contradiction_offset + pair_count
            ]
        )
        contradiction_offset += pair_count

        scores.append(
            assemble_trace_score(
                steps=steps,
                judge_results=item[
                    "judge_results"
                ],
                nli=nli,
                tau=tau,
                scoring_cfg=scoring_cfg,
                contradiction_scores=(
                    trace_contradictions
                ),
            )
        )

    return scores

def score_record_trace(
    *,
    rec: Dict[str, Any],
    trace: str,
    nli: NLIScorer,
    judge_client: Any,
    tau: float,
    scoring_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    steps = split_steps(trace)

    try:
        judge_results = judge_client.judge_steps(
            question=rec["question"],
            steps=steps,
            task=resolve_record_task(rec),
        )
    except Exception:
        # Preserve the existing fail-soft verifier behavior.
        judge_results = []

    return assemble_trace_score(
        steps=steps,
        judge_results=judge_results,
        nli=nli,
        tau=tau,
        scoring_cfg=scoring_cfg,
    )

@app.command("prepare-dataset")
def prepare_dataset(
    name: str = typer.Option(..., "--name", help="gsm8k or strategyqa"),
    split: str = typer.Option("test", "--split", help="train/test/validation"),
    out_path: Optional[str] = typer.Option(None, "--out", help="Output JSONL path"),
    input_path: Optional[str] = typer.Option(
        None,
        "--input",
        help="Local StrategyQA JSON array",
    ),
    source_path: Optional[str] = typer.Option(
        None,
        "--source",
        help="Original StrategyQA train JSON used to record provenance",
    ),
    selection_seed: int = typer.Option(42, "--seed"),
):
    name = name.lower().strip()

    if name == "gsm8k":
        out = Path(out_path) if out_path else Path("data/raw") / f"{name}_{split}.jsonl"
        n = prepare_gsm8k(split=split, out_path=out)
    elif name == "strategyqa":
        if input_path is None:
            raise typer.BadParameter("StrategyQA preparation requires --input")
        local_input = Path(input_path)
        out = (
            Path(out_path)
            if out_path
            else Path("data/prepared") / f"{local_input.stem}.jsonl"
        )
        n = prepare_strategyqa_local(
            input_path=local_input,
            out_path=out,
            source_path=Path(source_path) if source_path else None,
            selection_seed=selection_seed,
        )
    else:
        raise typer.BadParameter("name must be one of: gsm8k, strategyqa")

    typer.echo(f"Wrote {n} examples to {out}")

# prompt the model and get the chain of thought trace
@app.command("generate-traces")
def generate_traces(
    config: str = typer.Option(..., "--config", "-c"),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the last compatible generation checkpoint",
    ),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    generated_path = outdir / "generated.jsonl"
    checkpoint_path = generated_path.with_suffix(
        generated_path.suffix + ".checkpoint.json"
    )

    metrics = create_execution_metrics(
        cfg,
        "generate-traces",
        outdir,
    )

    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("name", "gpt-5.4-mini")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_output_tokens = int(model_cfg.get("max_output_tokens", 800))

    execution_cfg = cfg.raw.get("execution", {}) or {}
    max_concurrency = int(execution_cfg.get("max_concurrency", 1))

    generation_chunk_size = execution_cfg.get(
        "generation_chunk_size",
        25,
    )

    if (
        not isinstance(generation_chunk_size, int)
        or isinstance(generation_chunk_size, bool)
        or generation_chunk_size < 1
    ):
        raise ValueError(
            "generation_chunk_size must be a positive integer"
        )
    
    openai_execution_options = resolve_openai_execution_options(
        execution_cfg
    )

    if provider not in {"openai", "gemini"}:
        raise ValueError(f"Unknown provider: {provider}")

    openai_rate_limiter = (
        create_openai_rate_limiter(
            execution_cfg,
            metrics,
        )
        if provider == "openai"
        else None
    )

    manifest = make_manifest(
        cfg.run_name,
        "generate-traces",
        config,
        cfg.raw,
    )
    write_manifest(outdir / "manifest.generate.json", manifest)

    with metrics.stage(
        "dataset_loading",
        metadata={"path": str(cfg.dataset_path)},
    ):
        rows = list(read_jsonl(cfg.dataset_path))

    checkpoint_identity = (
        make_generation_checkpoint_identity(
            run_name=cfg.run_name,
            input_path=cfg.dataset_path,
            config_raw=cfg.raw,
        )
    )

    if resume:
        with metrics.stage(
            "checkpoint_loading",
            metadata={"path": str(checkpoint_path)},
        ):
            checkpoint = load_generation_checkpoint(
                checkpoint_path,
                expected_identity=checkpoint_identity,
                expected_record_count=len(rows),
            )

        records = list(checkpoint["records"])
        start_source_index = checkpoint[
            "next_source_index"
        ]

        typer.echo(
            f"Resuming trace generation from "
            f"{checkpoint_path} at source index "
            f"{start_source_index}"
        )
    else:
        delete_generation_checkpoint(checkpoint_path)
        records = []
        start_source_index = 0

    thread_state = local()

    def get_generation_client() -> Any:
        client = getattr(thread_state, "generation_client", None)
        if client is not None:
            return client

        if provider == "openai":
            client = OpenAIClientWrapper(
                model=model_name,
                metrics=metrics,
                rate_limiter=openai_rate_limiter,
                **openai_execution_options,
            )
        else:
            client = GeminiClient(model=model_name)

        thread_state.generation_client = client
        return client

    def generate_record(row: Dict[str, Any]) -> Dict[str, Any]:
        client = get_generation_client()
        question = row["question"]
        task = resolve_record_task(row, cfg.raw)

        trace, answer = client.generate_trace(
            question,
            task=task,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        return make_base_record(
            rid=row["id"],
            question=question,
            model_trace=trace,
            model_answer=answer,
            evidence=row.get("evidence"),
            metadata=row.get("metadata"),
            gold_answer=row.get("gold_answer"),
            task=task,
            source=row.get("source"),
            model=model_name,
            prompt_id=model_cfg.get("prompt_id"),
        )

    for chunk_start in range(
        start_source_index,
        len(rows),
        generation_chunk_size,
    ):
        chunk_end = min(
            chunk_start + generation_chunk_size,
            len(rows),
        )
        chunk_rows = rows[chunk_start:chunk_end]

        with metrics.stage(
            "trace_generation_wave",
            metadata={
                "items": len(chunk_rows),
                "max_concurrency": max_concurrency,
            },
        ):
            chunk_records = map_bounded(
                chunk_rows,
                generate_record,
                max_workers=max_concurrency,
                item_id=lambda row: str(
                    row.get("id", "")
                ),
            )

        records.extend(chunk_records)
        metrics.set_records_processed(len(records))

        try:
            with metrics.stage(
                "checkpoint_write",
                metadata={
                    "path": str(checkpoint_path),
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "completed_records": len(records),
                    "total_records": len(rows),
                },
            ):
                save_generation_checkpoint(
                    checkpoint_path,
                    identity=checkpoint_identity,
                    next_source_index=chunk_end,
                    records=records,
                )
        finally:
            metrics.flush()

    metrics.set_records_processed(len(records))

    with metrics.stage(
        "artifact_write",
        metadata={
            "path": str(outdir / "generated.jsonl"),
            "records": len(records),
        },
    ):
        write_jsonl(outdir / "generated.jsonl", records)

    with metrics.stage(
        "checkpoint_cleanup",
        metadata={"path": str(checkpoint_path)},
    ):
        delete_generation_checkpoint(checkpoint_path)

    metrics.finish(records_processed=len(records))

    typer.echo(
        f"Wrote {len(records)} records to {generated_path}"
    )

# given the trace, score it, WIP
@app.command("score-traces")
def score_traces(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    inpath = Path(input_path) if input_path else (outdir / "generated.jsonl")
    nli = NLIScorer(model_name="FacebookAI/roberta-large-mnli")
    manifest = make_manifest(cfg.run_name, "score-traces", config, cfg.raw)
    write_manifest(outdir / "manifest.score.json", manifest)

    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    judge_model = model_cfg.get("name", "gpt-5.4-mini")  # same model for now
    if provider == "openai":
        judge_client = OpenAIClientWrapper(model=judge_model)
    elif provider == "gemini":
        judge_client = GeminiClient(model=judge_model)
    else:
        raise ValueError(...)

    scored = []
    scoring_cfg = cfg.raw.get("scoring", {})
    tau = scoring_cfg.get("risk_threshold", scoring_cfg.get("tau", 0.8))

    for rec in read_jsonl(inpath):
        task = resolve_record_task(rec, cfg.raw)
        steps = split_steps(rec["model_trace"])

        # Get verifier judgments in one call
        try:
            judge_results = judge_client.judge_steps(
                question=rec["question"],
                steps=steps,
                task=task,
            )
        except Exception:
            judge_results = []

        # Convert to per-step verifier list (p_wrong)
        verifier = [0.5] * len(steps)  # default fallback
        for item in judge_results:
            try:
                idx = int(item.get("step_index"))
                p_wrong = float(item.get("p_wrong"))
                if 0 <= idx < len(steps):
                    verifier[idx] = max(0.0, min(1.0, p_wrong))
            except Exception:
                continue
        if len(verifier) != len(steps):
            # Shouldn't happen with our mapping, but just in case
            verifier = (verifier + [0.5] * len(steps))[:len(steps)]

        # NLI contradiction score per step (compare to previous 1–2 steps)
        contradiction = []
        for i, step in enumerate(steps):
            if i == 0:
                contradiction.append(0.0)
                continue
            prev = steps[max(0, i - 2): i]
            premise = "\n".join(prev)
            p_contra = nli.contradiction_prob(premise=premise, hypothesis=step)
            contradiction.append(p_contra)

        validate_score_alignment(steps, verifier, contradiction)

        evidence_support = None  # enabled later for non-math

        # Risk + earliest bad step
        risks = compute_risks(verifier, contradiction, scoring_cfg)
        earliest = next((i for i, r in enumerate(risks) if r > tau), None)

        rec2 = dict(rec)
        rec2["task"] = task
        rec2.update(
            {
                "steps": steps,
                "scores": {
                    "verifier": verifier,
                    "contradiction": contradiction,
                    "evidence_support": evidence_support,
                },
                "risks": risks,
                "earliest_bad_step": earliest,
            }
        )
        scored.append(rec2)

    write_jsonl(outdir / "scored.jsonl", scored)
    typer.echo(f"Wrote {len(scored)} records to {outdir/'scored.jsonl'}")


@app.command("repair-traces")
def repair_traces(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    inpath = Path(input_path) if input_path else (outdir / "scored.jsonl")

    manifest = make_manifest(cfg.run_name, "repair-traces", config, cfg.raw)
    write_manifest(outdir / "manifest.repair.json", manifest)

    # Build the generation client from config (OpenAI by default)
    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("name", "gpt-5.4-mini")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_output_tokens = int(model_cfg.get("max_output_tokens", 800))

    if provider == "openai":
        gen_client = OpenAIClientWrapper(model=model_name)
    elif provider == "gemini":
        gen_client = GeminiClient(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    repaired = []
    for rec in read_jsonl(inpath):
        task = resolve_record_task(rec, cfg.raw)
        k = rec.get("earliest_bad_step")
        rec2 = dict(rec)
        rec2["task"] = task
        steps = rec.get("steps", [])

        # Default, no repair performed
        rec2["repaired_trace"] = None
        rec2["repaired_answer"] = None

        # If no risky step found, keep as-is
        if k is None or not steps:
            rec2.setdefault("logs", {})
            rec2["logs"]["repair"] = {"performed": False, "reason": "no_bad_step_or_no_steps"}
            repaired.append(rec2)
            continue

        # Prefix: keep steps before first bad step
        prefix = steps[:k]
        next_step_number = k + 1  # Steps are human-numbered starting at 1

        try:
            suffix_text, repaired_answer = gen_client.repair_suffix(
                question=rec["question"],
                prefix_steps=prefix,
                next_step_number=next_step_number,
                task=task,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            # Assemble full repaired trace
            suffix_lines = [ln.strip() for ln in suffix_text.splitlines() if ln.strip()]
            rec2["repaired_trace"] = "\n".join(prefix + suffix_lines)
            rec2["repaired_answer"] = repaired_answer

            rec2.setdefault("logs", {})
            rec2["logs"]["repair"] = {
                "performed": True,
                "k": k,
                "next_step_number": next_step_number,
            }

        except Exception as e:
            # Fail-soft: keep record, but mark repair failure
            rec2.setdefault("logs", {})
            rec2["logs"]["repair"] = {
                "performed": False,
                "k": k,
                "error": str(e),
            }

        repaired.append(rec2)
        
        

    write_jsonl(outdir / "repaired.jsonl", repaired)
    typer.echo(f"Wrote {len(repaired)} records to {outdir/'repaired.jsonl'}")


@app.command("build-candidate-pool")
def build_candidate_pool(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
    output_path: Optional[str] = typer.Option(None, "--output"),
    reuse_original_scores: bool = typer.Option(
        False,
        "--reuse-original-scores",
        help="Reuse verifier/NLI outputs from input candidate_pool.original_score",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the last compatible candidate-pool checkpoint",
    ),
):
    """Generate and freeze a policy-neutral, risk-only candidate chain."""
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = create_execution_metrics(
        cfg,
        "build-candidate-pool",
        outdir,
    )

    inpath = (
        Path(input_path)
        if input_path
        else outdir / "generated.jsonl"
    )
    outpath = (
        Path(output_path)
        if output_path
        else outdir / "candidate_pool.jsonl"
    )
    checkpoint_path = outpath.with_suffix(
        outpath.suffix + ".checkpoint.json"
    )
    manifest = make_manifest(
        cfg.run_name,
        "build-candidate-pool",
        config,
        cfg.raw,
    )
    write_manifest(
        outdir / "manifest.candidate_pool.json",
        manifest,
    )

    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("name", "gpt-5.4-mini")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_output_tokens = int(
        model_cfg.get("max_output_tokens", 800)
    )

    execution_cfg = cfg.raw.get("execution", {}) or {}
    max_concurrency = int(
        execution_cfg.get("max_concurrency", 1)
    )
    candidate_chunk_size = execution_cfg.get(
            "candidate_chunk_size",
            25,
        )
    
    if (
            not isinstance(candidate_chunk_size, int)
            or isinstance(candidate_chunk_size, bool)
            or candidate_chunk_size < 1
        ):
            raise ValueError(
                "candidate_chunk_size must be a positive integer"
            )

    nli_batch_size = execution_cfg.get(
        "nli_batch_size",
        32,
    )

    if (
        not isinstance(nli_batch_size, int)
        or isinstance(nli_batch_size, bool)
        or nli_batch_size < 1
    ):
        raise ValueError(
            "nli_batch_size must be a positive integer"
        )

    openai_execution_options = resolve_openai_execution_options(
        execution_cfg
    )

    if provider not in {"openai", "gemini"}:
        raise ValueError(f"Unknown provider: {provider}")

    openai_rate_limiter = (
        create_openai_rate_limiter(
            execution_cfg,
            metrics,
        )
        if provider == "openai"
        else None
    )

    scoring_cfg = cfg.raw.get("scoring", {})
    tau = float(
        scoring_cfg.get(
            "risk_threshold",
            scoring_cfg.get("tau", 0.8),
        )
    )
    improvement_threshold = float(
        scoring_cfg.get("improvement_threshold", 0.02)
    )
    max_iters = int(scoring_cfg.get("max_iters", 2))

    with metrics.stage(
        "nli_initialization",
        metadata={"model": "FacebookAI/roberta-large-mnli"},
    ):
        nli = NLIScorer(
            model_name="FacebookAI/roberta-large-mnli",
            metrics=metrics,
            batch_size=nli_batch_size,
        )

    # Provider clients are local to each worker thread. Metrics are shared,
    # but ExecutionMetrics protects its mutable state with a lock.
    thread_state = local()

    def get_provider_client(role: str) -> Any:
        attribute = f"{role}_client"
        client = getattr(thread_state, attribute, None)
        if client is not None:
            return client

        if provider == "openai":
            client = OpenAIClientWrapper(
                model=model_name,
                metrics=metrics,
                rate_limiter=openai_rate_limiter,
                **openai_execution_options,
            )
        else:
            client = GeminiClient(model=model_name)

        setattr(thread_state, attribute, client)
        return client

    def perform_verification(
        request: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        judge_client = get_provider_client("judge")

        try:
            return judge_client.judge_steps(
                question=request["question"],
                steps=request["steps"],
                task=request["task"],
            )
        except Exception:
            # Preserve the existing fallback of p_wrong=0.5.
            return []

    def perform_repair(
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        generation_client = get_provider_client("generation")

        try:
            suffix_text, candidate_answer = (
                generation_client.repair_suffix(
                    question=request["question"],
                    prefix_steps=request["prefix"],
                    next_step_number=request["next_step_number"],
                    task=request["task"],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )
            return {
                "suffix_text": suffix_text,
                "candidate_answer": candidate_answer,
                "error": None,
            }
        except Exception as error:
            return {
                "suffix_text": None,
                "candidate_answer": None,
                "error": str(error),
            }

    def perform_acceptance_judgment(
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        judge_client = get_provider_client("judge")

        try:
            judge_result = judge_client.judge_repair_candidate(
                question=request["question"],
                original_trace=request["parent_trace"],
                original_answer=request["parent_answer"],
                repaired_trace=request["candidate_trace"],
                repaired_answer=request["candidate_answer"],
                task=request["task"],
            )
            return {
                "judge": judge_result,
                "judge_error": None,
            }
        except Exception as error:
            return {
                "judge": None,
                "judge_error": str(error),
            }

    with metrics.stage(
        "dataset_loading",
        metadata={"path": str(inpath)},
    ):
        source_records = list(read_jsonl(inpath))

    checkpoint_identity = make_candidate_pool_checkpoint_identity(
        run_name=cfg.run_name,
        input_path=inpath,
        config_raw=cfg.raw,
        reuse_original_scores=reuse_original_scores,
    )

    def persist_checkpoint(
        *,
        phase: str,
        next_iteration: int,
        active: List[Dict[str, Any]],
        checkpoint_states: List[Dict[str, Any]],
    ) -> None:
        active_state_indexes = [
            state["state_index"]
            for state in active
        ]

        try:
            with metrics.stage(
                "checkpoint_write",
                metadata={
                    "phase": phase,
                    "next_iteration": next_iteration,
                    "active_states": len(
                        active_state_indexes
                    ),
                    "records": len(checkpoint_states),
                },
            ):
                save_candidate_pool_checkpoint(
                    checkpoint_path,
                    identity=checkpoint_identity,
                    phase=phase,
                    next_iteration=next_iteration,
                    active_state_indexes=(
                        active_state_indexes
                    ),
                    states=checkpoint_states,
                )
        finally:
            metrics.flush()

    if resume:
        with metrics.stage(
            "checkpoint_loading",
            metadata={"path": str(checkpoint_path)},
        ):
            checkpoint = load_candidate_pool_checkpoint(
                checkpoint_path,
                expected_identity=checkpoint_identity,
            )

        states = checkpoint["states"]
        active_states = [
            states[state_index]
            for state_index in checkpoint["active_state_indexes"]
        ]
        start_iteration = checkpoint["next_iteration"]
        checkpoint_phase = checkpoint["phase"]
        if (
            checkpoint_phase
            == "original_scoring_chunk_complete"
        ):
            if len(states) > len(source_records):
                raise ValueError(
                    "Original-scoring checkpoint contains "
                    "more records than the source dataset"
                )

            records_to_prepare = source_records[
                len(states):
            ]
            needs_original_scoring = True
        else:
            if len(states) != len(source_records):
                raise ValueError(
                    "Completed candidate-pool checkpoint "
                    "record count does not match the source "
                    "dataset"
                )

            records_to_prepare = []
            needs_original_scoring = False

        typer.echo(
            f"Resuming candidate-pool construction from "
            f"{checkpoint_path}; phase={checkpoint_phase}, "
            f"completed_records={len(states)}, "
            f"next_iteration={start_iteration}"
        )
    else:
        delete_candidate_pool_checkpoint(checkpoint_path)
        states: List[Dict[str, Any]] = []
        active_states: List[Dict[str, Any]] = []
        start_iteration = 1
        records_to_prepare = source_records
        needs_original_scoring = True

    for source_chunk_start in range(
        0,
        len(records_to_prepare),
        candidate_chunk_size,
    ):
        source_chunk_end = min(
            source_chunk_start + candidate_chunk_size,
            len(records_to_prepare),
        )
        source_chunk = records_to_prepare[
            source_chunk_start:source_chunk_end
        ]

        original_verification_requests = []
        chunk_states = []

        for source_record in source_chunk:
            task = resolve_record_task(
                source_record,
                cfg.raw,
            )
            record = {
                **source_record,
                "task": task,
            }
            state_index = len(states)

            state: Dict[str, Any] = {
                "state_index": state_index,
                "record": record,
                "task": task,
                "current_trace": record["model_trace"],
                "current_answer": record.get(
                    "model_answer"
                ),
                "current_score": None,
                "original_score": None,
                "original_score_source": None,
                "proposals": [],
                "generation_errors": [],
                "stop_reason": None,
                "scheduled_iteration": 1,
            }

            if reuse_original_scores:
                frozen_score = (
                    source_record
                    .get("candidate_pool", {})
                    .get("original_score")
                )

                if not isinstance(frozen_score, dict):
                    raise ValueError(
                        f"Record {record.get('id')!r} "
                        "has no frozen original score "
                        "to reuse"
                    )

                state["current_score"] = (
                    rescore_frozen_trace_score(
                        frozen_score,
                        scoring_cfg=scoring_cfg,
                        tau=tau,
                    )
                )
                state["original_score_source"] = (
                    "reused_frozen_verifier_and_nli"
                )
            else:
                original_verification_requests.append(
                    {
                        "state_index": state_index,
                        "record_id": str(
                            record.get("id", "")
                        ),
                        "work_id": (
                            f"{record.get('id', state_index)}:"
                            "original-verification"
                        ),
                        "question": record["question"],
                        "steps": split_steps(
                            record["model_trace"]
                        ),
                        "task": task,
                    }
                )
                state["original_score_source"] = (
                    "newly_scored"
                )

            states.append(state)
            chunk_states.append(state)

        if original_verification_requests:
            with metrics.stage(
                "original_verification_wave",
                metadata={
                    "items": len(
                        original_verification_requests
                    ),
                    "max_concurrency": max_concurrency,
                },
            ):
                original_verification_results = (
                    map_bounded(
                        original_verification_requests,
                        perform_verification,
                        max_workers=max_concurrency,
                        item_id=lambda request: request[
                            "work_id"
                        ],
                    )
                )
        else:
            original_verification_results = []

        original_scores = (
            assemble_trace_scores_batch(
                items=[
                    {
                        "steps": request["steps"],
                        "judge_results": judge_results,
                    }
                    for request, judge_results in zip(
                        original_verification_requests,
                        original_verification_results,
                    )
                ],
                nli=nli,
                tau=tau,
                scoring_cfg=scoring_cfg,
            )
        )

        for request, original_score in zip(
            original_verification_requests,
            original_scores,
        ):
            state = states[request["state_index"]]
            state["current_score"] = original_score

        for state in chunk_states:
            if not isinstance(
                state["current_score"],
                dict,
            ):
                raise ValueError(
                    f"Record "
                    f"{state['record'].get('id')!r} "
                    "did not produce a valid original score"
                )

            state["original_score"] = state[
                "current_score"
            ]

        active_states = list(states)

        persist_checkpoint(
            phase="original_scoring_chunk_complete",
            next_iteration=1,
            active=active_states,
            checkpoint_states=states,
        )

    if needs_original_scoring:
        active_states = list(states)

        persist_checkpoint(
            phase="original_scoring_complete",
            next_iteration=1,
            active=active_states,
            checkpoint_states=states,
        )

    for iter_idx in range(start_iteration, max_iters + 1):
        iteration_states = [
            state
            for state in active_states
            if state["scheduled_iteration"] == iter_idx
        ]

        for chunk_start in range(
            0,
            len(iteration_states),
            candidate_chunk_size,
        ):
            chunk_end = min(
                chunk_start + candidate_chunk_size,
                len(iteration_states),
            )
            chunk_states = iteration_states[
                chunk_start:chunk_end
            ]
            repair_requests = []

            # Determine which records in this chunk need repair.
            for state in chunk_states:
                current_score = state["current_score"]
                earliest_bad_step = current_score[
                    "earliest_bad_step"
                ]
                steps = current_score["steps"]

                if earliest_bad_step is None:
                    state["stop_reason"] = "no_bad_step"
                    state["scheduled_iteration"] = None
                    continue

                if not steps:
                    state["stop_reason"] = "no_steps"
                    state["scheduled_iteration"] = None
                    continue

                prefix = steps[:earliest_bad_step]
                next_step_number = earliest_bad_step + 1
                record = state["record"]
                state_index = state["state_index"]

                repair_requests.append(
                    {
                        "state_index": state_index,
                        "record_id": str(
                            record.get("id", "")
                        ),
                        "work_id": (
                            f"{record.get('id', state_index)}:"
                            f"repair:{iter_idx}"
                        ),
                        "iter": iter_idx,
                        "question": record["question"],
                        "task": state["task"],
                        "k": earliest_bad_step,
                        "prefix": prefix,
                        "next_step_number": next_step_number,
                    }
                )

            if repair_requests:
                with metrics.stage(
                    "repair_wave",
                    metadata={
                        "iteration": iter_idx,
                        "items": len(repair_requests),
                        "max_concurrency": max_concurrency,
                    },
                ):
                    repair_results = map_bounded(
                        repair_requests,
                        perform_repair,
                        max_workers=max_concurrency,
                        item_id=lambda request: request[
                            "work_id"
                        ],
                    )

                candidate_verification_requests = []

                for request, repair_result in zip(
                    repair_requests,
                    repair_results,
                ):
                    state = states[
                        request["state_index"]
                    ]

                    if repair_result["error"] is not None:
                        state["generation_errors"].append(
                            {
                                "iter": iter_idx,
                                "kind": "repair_error",
                                "error": repair_result["error"],
                            }
                        )
                        state["stop_reason"] = "repair_error"
                        state["scheduled_iteration"] = None
                        continue

                    suffix_lines = [
                        line.strip()
                        for line in repair_result[
                            "suffix_text"
                        ].splitlines()
                        if line.strip()
                    ]
                    candidate_trace = "\n".join(
                        request["prefix"] + suffix_lines
                    )

                    candidate_verification_requests.append(
                        {
                            **request,
                            "work_id": (
                                f"{request['record_id']}:"
                                "candidate-verification:"
                                f"{iter_idx}"
                            ),
                            "parent_trace": state[
                                "current_trace"
                            ],
                            "parent_answer": state[
                                "current_answer"
                            ],
                            "parent_score": state[
                                "current_score"
                            ],
                            "candidate_trace": candidate_trace,
                            "candidate_answer": repair_result[
                                "candidate_answer"
                            ],
                            "steps": split_steps(
                                candidate_trace
                            ),
                        }
                    )

                with metrics.stage(
                    "candidate_verification_wave",
                    metadata={
                        "iteration": iter_idx,
                        "items": len(
                            candidate_verification_requests
                        ),
                        "max_concurrency": max_concurrency,
                    },
                ):
                    candidate_verification_results = (
                        map_bounded(
                            candidate_verification_requests,
                            perform_verification,
                            max_workers=max_concurrency,
                            item_id=lambda request: request[
                                "work_id"
                            ],
                        )
                    )

                candidate_entries = []
                acceptance_requests = []

                candidate_scores = (
                    assemble_trace_scores_batch(
                        items=[
                            {
                                "steps": request["steps"],
                                "judge_results": judge_results,
                            }
                            for request, judge_results in zip(
                                candidate_verification_requests,
                                candidate_verification_results,
                            )
                        ],
                        nli=nli,
                        tau=tau,
                        scoring_cfg=scoring_cfg,
                    )
                )

                for request, candidate_score in zip(
                    candidate_verification_requests,
                    candidate_scores,
                ):

                    old_avg_risk = float(
                        request["parent_score"]
                        ["risk_summary"]["avg_risk"]
                    )
                    new_avg_risk = float(
                        candidate_score[
                            "risk_summary"
                        ]["avg_risk"]
                    )
                    improvement = (
                        old_avg_risk - new_avg_risk
                    )

                    candidate_index = len(
                        candidate_entries
                    )
                    entry = {
                        "state_index": request[
                            "state_index"
                        ],
                        "iter": iter_idx,
                        "k": request["k"],
                        "next_step_number": request[
                            "next_step_number"
                        ],
                        "parent_trace": request[
                            "parent_trace"
                        ],
                        "parent_answer": request[
                            "parent_answer"
                        ],
                        "parent_score": request[
                            "parent_score"
                        ],
                        "candidate_trace": request[
                            "candidate_trace"
                        ],
                        "candidate_answer": request[
                            "candidate_answer"
                        ],
                        "candidate_score": candidate_score,
                        "old_avg_risk": old_avg_risk,
                        "new_avg_risk": new_avg_risk,
                        "improvement": improvement,
                        "judge": None,
                        "judge_error": None,
                    }
                    candidate_entries.append(entry)

                    if improvement >= 0.0:
                        state = states[
                            request["state_index"]
                        ]
                        record = state["record"]

                        acceptance_requests.append(
                            {
                                "candidate_index": (
                                    candidate_index
                                ),
                                "record_id": str(
                                    record.get("id", "")
                                ),
                                "work_id": (
                                    f"{record.get('id', request['state_index'])}:"
                                    f"acceptance:{iter_idx}"
                                ),
                                "question": record[
                                    "question"
                                ],
                                "task": state["task"],
                                "parent_trace": request[
                                    "parent_trace"
                                ],
                                "parent_answer": request[
                                    "parent_answer"
                                ],
                                "candidate_trace": request[
                                    "candidate_trace"
                                ],
                                "candidate_answer": request[
                                    "candidate_answer"
                                ],
                            }
                        )

                with metrics.stage(
                    "acceptance_wave",
                    metadata={
                        "iteration": iter_idx,
                        "items": len(
                            acceptance_requests
                        ),
                        "max_concurrency": max_concurrency,
                    },
                ):
                    acceptance_results = map_bounded(
                        acceptance_requests,
                        perform_acceptance_judgment,
                        max_workers=max_concurrency,
                        item_id=lambda request: request[
                            "work_id"
                        ],
                    )

                for request, acceptance_result in zip(
                    acceptance_requests,
                    acceptance_results,
                ):
                    entry = candidate_entries[
                        request["candidate_index"]
                    ]
                    entry["judge"] = acceptance_result[
                        "judge"
                    ]
                    entry["judge_error"] = (
                        acceptance_result["judge_error"]
                    )

                # Commit this chunk in stable record order.
                for entry in candidate_entries:
                    state = states[
                        entry["state_index"]
                    ]

                    state["proposals"].append(
                        {
                            "iter": entry["iter"],
                            "k": entry["k"],
                            "next_step_number": entry[
                                "next_step_number"
                            ],
                            "parent_trace": entry[
                                "parent_trace"
                            ],
                            "parent_answer": entry[
                                "parent_answer"
                            ],
                            "parent_score": entry[
                                "parent_score"
                            ],
                            "candidate_trace": entry[
                                "candidate_trace"
                            ],
                            "candidate_answer": entry[
                                "candidate_answer"
                            ],
                            "candidate_score": entry[
                                "candidate_score"
                            ],
                            "old_avg_risk": entry[
                                "old_avg_risk"
                            ],
                            "new_avg_risk": entry[
                                "new_avg_risk"
                            ],
                            "improvement": entry[
                                "improvement"
                            ],
                            "judge": entry["judge"],
                            "judge_error": entry[
                                "judge_error"
                            ],
                        }
                    )

                    if entry["improvement"] < 0.0:
                        state["stop_reason"] = (
                            "risk_worsened"
                        )
                        state[
                            "scheduled_iteration"
                        ] = None
                        continue

                    state["current_trace"] = entry[
                        "candidate_trace"
                    ]
                    state["current_answer"] = entry[
                        "candidate_answer"
                    ]
                    state["current_score"] = entry[
                        "candidate_score"
                    ]

                    if (
                        entry["improvement"]
                        < improvement_threshold
                    ):
                        state["stop_reason"] = (
                            "improvement_below_threshold"
                        )
                        state[
                            "scheduled_iteration"
                        ] = None
                        continue

                    state["scheduled_iteration"] = (
                        iter_idx + 1
                    )

            # Keep all unfinished records, including records already
            # processed and scheduled for the next iteration.
            active_states = [
                state
                for state in states
                if state["scheduled_iteration"] is not None
            ]

            persist_checkpoint(
                phase="iteration_chunk_complete",
                next_iteration=iter_idx,
                active=active_states,
                checkpoint_states=states,
            )

        # All records scheduled for this iteration are now complete.
        # Records already processed before a resumed invocation remain
        # scheduled for the following iteration.
        active_states = [
            state
            for state in states
            if (
                state["scheduled_iteration"] is not None
                and state["scheduled_iteration"] > iter_idx
            )
        ]

        persist_checkpoint(
            phase="iteration_complete",
            next_iteration=iter_idx + 1,
            active=active_states,
            checkpoint_states=states,
        )

        if not active_states:
            break

    for state in states:
        if state["stop_reason"] is None:
            state["stop_reason"] = "max_iters"
        state["scheduled_iteration"] = None

    results = []

    for state in states:
        record = state["record"]
        result = dict(record)
        result["candidate_pool"] = {
            "construction_policy": "risk_only_chain",
            "original_score_source": state[
                "original_score_source"
            ],
            "risk_threshold": tau,
            "scoring": scoring_cfg,
            "improvement_threshold": improvement_threshold,
            "max_iters": max_iters,
            "original_score": state["original_score"],
            "proposals": state["proposals"],
            "generation_stop_reason": state["stop_reason"],
            "generation_errors": state["generation_errors"],
        }
        results.append(result)
    metrics.set_records_processed(len(results))

    with metrics.stage(
        "artifact_write",
        metadata={
            "path": str(outpath),
            "records": len(results),
        },
    ):
        write_jsonl(outpath, results)

    with metrics.stage(
        "checkpoint_cleanup",
        metadata={"path": str(checkpoint_path)},
    ):
        delete_candidate_pool_checkpoint(checkpoint_path)

    metrics.finish(records_processed=len(results))

    typer.echo(
        f"Wrote {len(results)} frozen candidate-pool records "
        f"to {outpath}"
    )


@app.command("apply-candidate-policy")
def apply_candidate_policy(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    """Apply one configured policy to a frozen pool without model or NLI calls."""
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    inpath = Path(input_path) if input_path else (outdir / "candidate_pool.jsonl")

    manifest = make_manifest(cfg.run_name, "apply-candidate-policy", config, cfg.raw)
    write_manifest(outdir / "manifest.apply_candidate_policy.json", manifest)

    scoring_cfg = cfg.raw.get("scoring", {})
    acceptance_cfg = cfg.raw.get("repair_acceptance", {})
    mode = acceptance_cfg.get("mode", "risk_only")
    support_tolerance = float(acceptance_cfg.get("support_tolerance", 0.05))
    max_regression_risk = float(acceptance_cfg.get("max_regression_risk", 0.35))
    improvement_threshold = float(scoring_cfg.get("improvement_threshold", 0.02))

    results = []
    for source_record in read_jsonl(inpath):
        task = resolve_record_task(source_record, cfg.raw)
        record = {**source_record, "task": task}
        results.append(
            apply_candidate_pool_record(
                record,
                mode=mode,
                support_tolerance=support_tolerance,
                max_regression_risk=max_regression_risk,
                improvement_threshold=improvement_threshold,
            )
        )

    write_jsonl(outdir / "iterative_repaired.jsonl", results)
    typer.echo(
        f"Applied {mode} to {len(results)} frozen records and wrote "
        f"{outdir/'iterative_repaired.jsonl'}"
    )


@app.command("iterative-repair")
def iterative_repair(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    inpath = Path(input_path) if input_path else (outdir / "generated.jsonl")

    manifest = make_manifest(cfg.run_name, "iterative-repair", config, cfg.raw)
    write_manifest(outdir / "manifest.iterative_repair.json", manifest)

    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("name", "gpt-5.4-mini")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_output_tokens = int(model_cfg.get("max_output_tokens", 800))

    scoring_cfg = cfg.raw.get("scoring", {})
    tau = float(scoring_cfg.get("risk_threshold", scoring_cfg.get("tau", 0.8)))
    improvement_threshold = float(scoring_cfg.get("improvement_threshold", 0.02))
    max_iters = int(scoring_cfg.get("max_iters", 2))
    acceptance_cfg = cfg.raw.get("repair_acceptance", {})
    acceptance_mode = acceptance_cfg.get("mode", "risk_only")
    support_tolerance = float(acceptance_cfg.get("support_tolerance", 0.05))
    max_regression_risk = float(acceptance_cfg.get("max_regression_risk", 0.35))

    if provider == "openai":
        gen_client = OpenAIClientWrapper(model=model_name)
        judge_client = OpenAIClientWrapper(model=model_name)
    elif provider == "gemini":
        gen_client = GeminiClient(model=model_name)
        judge_client = GeminiClient(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    nli = NLIScorer(model_name="FacebookAI/roberta-large-mnli")

    results = []

    for rec in read_jsonl(inpath):
        task = resolve_record_task(rec, cfg.raw)
        rec = {**rec, "task": task}
        current_trace = rec["model_trace"]
        current_answer = rec.get("model_answer")

        iteration_logs = []

        current_score = score_record_trace(
            rec=rec,
            trace=current_trace,
            nli=nli,
            judge_client=judge_client,
            tau=tau,
            scoring_cfg=scoring_cfg,
        )

        iteration_logs.append({
            "iter": 0,
            "kind": "original",
            "answer": current_answer,
            "earliest_bad_step": current_score["earliest_bad_step"],
            **current_score["risk_summary"],
        })

        final_score = current_score
        stop_reason = None

        for iter_idx in range(1, max_iters + 1):
            k = final_score["earliest_bad_step"]
            steps = final_score["steps"]

            if k is None:
                stop_reason = "no_bad_step"
                break

            if not steps:
                stop_reason = "no_steps"
                break

            prefix = steps[:k]
            next_step_number = k + 1

            try:
                suffix_text, candidate_answer = gen_client.repair_suffix(
                    question=rec["question"],
                    prefix_steps=prefix,
                    next_step_number=next_step_number,
                    task=task,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as e:
                iteration_logs.append({
                    "iter": iter_idx,
                    "kind": "repair_error",
                    "error": str(e),
                    "accepted": False,
                })
                stop_reason = "repair_error"
                break

            suffix_lines = [ln.strip() for ln in suffix_text.splitlines() if ln.strip()]
            candidate_trace = "\n".join(prefix + suffix_lines)

            candidate_score = score_record_trace(
                rec=rec,
                trace=candidate_trace,
                nli=nli,
                judge_client=judge_client,
                tau=tau,
                scoring_cfg=scoring_cfg,
            )

            old_avg_risk = final_score["risk_summary"]["avg_risk"]
            new_avg_risk = candidate_score["risk_summary"]["avg_risk"]
            improvement = old_avg_risk - new_avg_risk

            acceptance = decide_repair_acceptance(
                mode=acceptance_mode,
                rec=rec,
                judge_client=judge_client,
                current_trace=current_trace,
                current_answer=current_answer,
                candidate_trace=candidate_trace,
                candidate_answer=candidate_answer,
                old_avg_risk=old_avg_risk,
                new_avg_risk=new_avg_risk,
                support_tolerance=support_tolerance,
                max_regression_risk=max_regression_risk,
            )

            accepted = acceptance["accepted"]

            iteration_logs.append({
                "iter": iter_idx,
                "kind": "repair",
                "k": k,
                "next_step_number": next_step_number,
                "answer": candidate_answer,
                "parent_trace": current_trace,
                "parent_answer": current_answer,
                "candidate_trace": candidate_trace,
                "candidate_answer": candidate_answer,
                "candidate_score": candidate_score,
                "old_avg_risk": old_avg_risk,
                "new_avg_risk": new_avg_risk,
                "improvement": improvement,
                "accepted": accepted,
                "acceptance": acceptance,
                "continue": accepted and improvement >= improvement_threshold,
                "earliest_bad_step": candidate_score["earliest_bad_step"],
                **candidate_score["risk_summary"],
            })

            if not accepted:
                stop_reason = acceptance["reason"]
                break

            current_trace = candidate_trace
            current_answer = candidate_answer
            final_score = candidate_score

            if improvement < improvement_threshold:
                stop_reason = "improvement_below_threshold"
                break

        if stop_reason is None:
            stop_reason = "max_iters"

        rec2 = dict(rec)
        rec2.update({
            "final_trace": current_trace,
            "final_answer": current_answer,
            "final_steps": final_score["steps"],
            "final_scores": final_score["scores"],
            "final_risks": final_score["risks"],
            "final_earliest_bad_step": final_score["earliest_bad_step"],
            "logs": {
                **rec.get("logs", {}),
                "iterative_repair": {
                    "max_iters": max_iters,
                    "improvement_threshold": improvement_threshold,
                    "risk_threshold": tau,
                    "acceptance_mode": acceptance_mode,
                    "support_tolerance": support_tolerance,
                    "max_regression_risk": max_regression_risk,
                    "stop_reason": stop_reason,
                    "iterations": iteration_logs,
                },
            },
        })

        results.append(rec2)

    write_jsonl(outdir / "iterative_repaired.jsonl", results)
    typer.echo(f"Wrote {len(results)} records to {outdir/'iterative_repaired.jsonl'}")

@app.command("evaluate")
def evaluate(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    if input_path:
        inpath = Path(input_path)
    elif (outdir / "iterative_repaired.jsonl").exists():
        inpath = outdir / "iterative_repaired.jsonl"
    else:
        inpath = outdir / "repaired.jsonl"

    manifest = make_manifest(cfg.run_name, "evaluate", config, cfg.raw)
    write_manifest(outdir / "manifest.evaluate.json", manifest)

    scoring_cfg = cfg.raw.get("scoring", {})
    tau = float(scoring_cfg.get("risk_threshold", scoring_cfg.get("tau", 0.8)))

    total = 0
    answer_evaluated = 0
    attempt_tp = 0
    attempt_fp = 0
    attempt_fn = 0

    accept_tp = 0
    accept_fp = 0
    accept_fn = 0
    original_correct_count = 0
    final_correct_count = 0

    outcome_counts = {
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "missing_gold_or_answer": 0,
    }

    stop_reason_counts: Dict[str, int] = {}
    total_iteration_logs = 0
    records_with_repair_attempt = 0
    records_with_accepted_repair = 0
    total_repair_attempts = 0
    total_accepted_repair_iterations = 0

    original_avg_risks = []
    final_avg_risks = []
    risk_deltas = []

    original_max_risks = []
    final_max_risks = []

    original_risky_steps = []
    final_risky_steps = []

    per_record = []

    for rec in read_jsonl(inpath):
        total += 1

        task = resolve_record_task(rec, cfg.raw)
        gold = rec.get("gold_answer")

        original_answer = rec.get("model_answer")
        final_answer = (
            rec.get("final_answer")
            or rec.get("repaired_answer")
            or rec.get("model_answer")
        )

        has_answer_eval = gold is not None and original_answer is not None and final_answer is not None

        if has_answer_eval:
            answer_evaluated += 1
            original_correct = answer_is_correct(task, original_answer, gold)
            final_correct = answer_is_correct(task, final_answer, gold)

            if original_correct:
                original_correct_count += 1
            if final_correct:
                final_correct_count += 1

            if not original_correct and final_correct:
                outcome = "wrong_to_correct"
            elif not original_correct and not final_correct:
                outcome = "wrong_to_wrong"
            elif original_correct and final_correct:
                outcome = "correct_to_correct"
            else:
                outcome = "correct_to_wrong"

            outcome_counts[outcome] += 1
        else:
            original_correct = False
            final_correct = False
            outcome = "missing_gold_or_answer"
            outcome_counts[outcome] += 1

        original_risk = get_original_risk_summary(rec, tau)
        final_risk = get_final_risk_summary(rec, tau)

        risk_delta = original_risk["avg_risk"] - final_risk["avg_risk"]

        original_avg_risks.append(original_risk["avg_risk"])
        final_avg_risks.append(final_risk["avg_risk"])
        risk_deltas.append(risk_delta)

        original_max_risks.append(original_risk["max_risk"])
        final_max_risks.append(final_risk["max_risk"])

        original_risky_steps.append(original_risk["num_risky_steps"])
        final_risky_steps.append(final_risk["num_risky_steps"])

        iterative_log = rec.get("logs", {}).get("iterative_repair", {})
        iterations = iterative_log.get("iterations", [])
        stop_reason = iterative_log.get("stop_reason", "not_iterative")

        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        total_iteration_logs += len(iterations)

        repair_iterations = [it for it in iterations if it.get("kind") == "repair"]
        accepted_iterations = [it for it in repair_iterations if it.get("accepted") is True]

        actual_needs_repair = not original_correct
        predicted_attempt = len(repair_iterations) > 0
        predicted_accept = len(accepted_iterations) > 0

        if actual_needs_repair and predicted_attempt:
            attempt_tp += 1
        elif not actual_needs_repair and predicted_attempt:
            attempt_fp += 1
        elif actual_needs_repair and not predicted_attempt:
            attempt_fn += 1

        if actual_needs_repair and predicted_accept:
            accept_tp += 1
        elif not actual_needs_repair and predicted_accept:
            accept_fp += 1
        elif actual_needs_repair and not predicted_accept:
            accept_fn += 1

        total_repair_attempts += len(repair_iterations)
        total_accepted_repair_iterations += len(accepted_iterations)

        if repair_iterations:
            records_with_repair_attempt += 1
        if accepted_iterations:
            records_with_accepted_repair += 1

        per_record.append({
            "id": rec.get("id"),
            "task": task,
            "gold_answer": gold,
            "original_answer": original_answer,
            "final_answer": final_answer,
            "original_correct": original_correct,
            "final_correct": final_correct,
            "answer_outcome": outcome,
            "original_avg_risk": original_risk["avg_risk"],
            "final_avg_risk": final_risk["avg_risk"],
            "risk_delta": risk_delta,
            "original_max_risk": original_risk["max_risk"],
            "final_max_risk": final_risk["max_risk"],
            "original_num_risky_steps": original_risk["num_risky_steps"],
            "final_num_risky_steps": final_risk["num_risky_steps"],
            "stop_reason": stop_reason,
            "num_iteration_logs": len(iterations),
            "num_repair_attempts": len(repair_iterations),
            "num_accepted_repair_iterations": len(accepted_iterations),
        })

    original_accuracy = original_correct_count / answer_evaluated if answer_evaluated else 0.0
    final_accuracy = final_correct_count / answer_evaluated if answer_evaluated else 0.0

    metrics = {
        "input_path": str(inpath),
        "total_records": total,
        "answer_evaluated_records": answer_evaluated,
        "correctness": {
            "original_correct": original_correct_count,
            "final_correct": final_correct_count,
            "original_accuracy": original_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_delta": final_accuracy - original_accuracy,
            "outcomes": outcome_counts,
        },
        "risk": {
            "mean_original_avg_risk": mean_or_zero(original_avg_risks),
            "mean_final_avg_risk": mean_or_zero(final_avg_risks),
            "mean_risk_delta": mean_or_zero(risk_deltas),
            "mean_original_max_risk": mean_or_zero(original_max_risks),
            "mean_final_max_risk": mean_or_zero(final_max_risks),
            "mean_original_num_risky_steps": mean_or_zero(original_risky_steps),
            "mean_final_num_risky_steps": mean_or_zero(final_risky_steps),
        },
        "repair_loop": {
            "stop_reasons": stop_reason_counts,
            "records_with_repair_attempt": records_with_repair_attempt,
            "records_with_accepted_repair": records_with_accepted_repair,
            "total_repair_attempts": total_repair_attempts,
            "total_accepted_repair_iterations": total_accepted_repair_iterations,
            "repair_attempt_rate": records_with_repair_attempt / total if total else 0.0,
            "accepted_repair_record_rate": records_with_accepted_repair / total if total else 0.0,
            "accepted_repair_iteration_rate": (
                total_accepted_repair_iterations / total_repair_attempts
                if total_repair_attempts
                else 0.0
            ),
            "mean_iteration_logs_per_record": total_iteration_logs / total if total else 0.0,
        },
        "repair_detection": {
            "attempted_repair": {
                "tp": attempt_tp,
                "fp": attempt_fp,
                "fn": attempt_fn,
                **classification_metrics(attempt_tp, attempt_fp, attempt_fn),
            },
            "accepted_repair": {
                "tp": accept_tp,
                "fp": accept_fp,
                "fn": accept_fn,
                **classification_metrics(accept_tp, accept_fp, accept_fn),
            },
        },
    }

    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_jsonl(outdir / "evaluation_records.jsonl", per_record)

    typer.echo(f"Metrics saved to {outdir/'metrics.json'}")
    typer.echo(f"Per-record evaluation saved to {outdir/'evaluation_records.jsonl'}")

#wip
@app.command("run-pipeline")
def run_pipeline(config: str = typer.Option(..., "--config", "-c")):
    #generate -> iterative repair -> evaluate

    generate_traces(
        config=config,
        resume=False,
    )
    iterative_repair(config=config, input_path=None)
    evaluate(config=config, input_path=None)


def main():
    app()


if __name__ == "__main__":
    main()
