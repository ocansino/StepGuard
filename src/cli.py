from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from .datasets_prep import prepare_gsm8k, prepare_strategyqa
import typer
import re
import json
from .config import load_config
from .io_utils import read_jsonl, write_jsonl
from .manifest import make_manifest, write_manifest
from .schemas import make_base_record

from .providers.gemini_client import GeminiClient
from .providers.openai_client import OpenAIClientWrapper

from .scorers.nli import NLIScorer

app = typer.Typer(add_completion=False)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def run_dir(base_output_dir: str, run_name: str) -> Path:
    return Path(base_output_dir) / run_name

def normalize_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    """
    Returns a canonical numeric string for GSM8K-style answers.
    Examples:
      "$18" -> "18"
      "18.0" -> "18"
      "The answer is 18 dollars." -> "18"
    """
    if text is None:
        return None

    s = str(text).strip().lower()
    if not s:
        return None

    # remove common clutter
    s = s.replace(",", "")
    s = s.replace("$", "")

    nums = _NUM_RE.findall(s)
    if not nums:
        return None

    # GSM8K final answer is usually the last number mentioned
    num_str = nums[-1]

    # canonicalize: int if it's an integer, else float stripped
    try:
        val = float(num_str)
    except ValueError:
        return None

    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))

    # remove trailing zeros for floats
    out = f"{val:.10f}".rstrip("0").rstrip(".")
    return out

def split_steps(trace: str) -> List[str]:
    raw_lines = [s.strip() for s in trace.splitlines() if s.strip()]
    return [ln for ln in raw_lines if ln.lower().startswith("step ")]


def summarize_risk(risks: List[float], tau: float) -> Dict[str, Any]:
    if not risks:
        return {
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "num_risky_steps": 0,
        }

    return {
        "avg_risk": sum(risks) / len(risks),
        "max_risk": max(risks),
        "num_risky_steps": sum(1 for r in risks if r > tau),
    }


def gsm8k_exact_match(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_gsm8k_answer(pred)
    g = normalize_gsm8k_answer(gold)
    return (p is not None) and (g is not None) and (p == g)

def answer_is_correct(task: Optional[str], pred: Optional[str], gold: Optional[str]) -> bool:
    if gold is None or pred is None:
        return False

    if task == "math":
        return gsm8k_exact_match(pred, gold)

    return str(pred).strip().lower() == str(gold).strip().lower()

def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

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



def score_record_trace(
    *,
    rec: Dict[str, Any],
    trace: str,
    nli: NLIScorer,
    judge_client: Any,
    tau: float,
) -> Dict[str, Any]:
    steps = split_steps(trace)

    try:
        judge_results = judge_client.judge_steps(
            question=rec["question"],
            steps=steps,
        )
    except Exception:
        judge_results = []

    verifier = [0.5] * len(steps)
    for item in judge_results:
        try:
            idx = int(item.get("step_index"))
            p_wrong = float(item.get("p_wrong"))
            if 0 <= idx < len(steps):
                verifier[idx] = max(0.0, min(1.0, p_wrong))
        except Exception:
            continue

    contradiction = []
    for i, step in enumerate(steps):
        if i == 0:
            contradiction.append(0.0)
            continue

        prev = steps[max(0, i - 2): i]
        premise = "\n".join(prev)
        p_contra = nli.contradiction_prob(
            premise=premise,
            hypothesis=step,
        )
        contradiction.append(p_contra)

    risks = [min(1.0, v + c) for v, c in zip(verifier, contradiction)]
    earliest = next((i for i, r in enumerate(risks) if r > tau), None)

    risk_summary = summarize_risk(risks, tau)

    return {
        "steps": steps,
        "scores": {
            "verifier": verifier,
            "contradiction": contradiction,
            "evidence_support": None,
        },
        "risks": risks,
        "earliest_bad_step": earliest,
        "risk_summary": risk_summary,
    }

@app.command("prepare-dataset")
def prepare_dataset(
    name: str = typer.Option(..., "--name", help="gsm8k or strategyqa"),
    split: str = typer.Option("test", "--split", help="train/test/validation"),
    out_path: Optional[str] = typer.Option(None, "--out", help="Output JSONL path"),
):
    name = name.lower().strip()
    if out_path is None:
        out = Path("data/raw") / f"{name}_{split}.jsonl"
    else:
        out = Path(out_path)

    if name == "gsm8k":
        n = prepare_gsm8k(split=split, out_path=out)
    elif name == "strategyqa":
        n = prepare_strategyqa(split=split, out_path=out)
    else:
        raise typer.BadParameter("name must be one of: gsm8k, strategyqa")

    typer.echo(f"Wrote {n} examples to {out}")

# prompt the model and get the chain of thought trace
@app.command("generate-traces")
def generate_traces(config: str = typer.Option(..., "--config", "-c")):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.raw.get("model", {})
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("name", "gpt-5.4-mini")
    #gen_model = model_cfg.get("name", "gemini-2.0-flash")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_output_tokens = int(model_cfg.get("max_output_tokens", 800))

    if provider == "openai":
        gen_client = OpenAIClientWrapper(model=model_name)
    elif provider == "gemini":
        gen_client = GeminiClient(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    manifest = make_manifest(cfg.run_name, "generate-traces", config, cfg.raw)
    write_manifest(outdir / "manifest.generate.json", manifest)

    records = []
    for row in read_jsonl(cfg.dataset_path):
        q = row["question"]
        trace, ans = gen_client.generate_trace(
            q,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        records.append(
            make_base_record(
                rid=row["id"],
                question=q,
                model_trace=trace,
                model_answer=ans,
                evidence=row.get("evidence"),
                gold_answer=row.get("gold_answer"),
                task=row.get("task"),
                source=row.get("source"),
                model=cfg.raw.get("model", {}).get("name"),
                prompt_id=cfg.raw.get("model", {}).get("prompt_id"),
            )
        )

    write_jsonl(outdir / "generated.jsonl", records)
    typer.echo(f"Wrote {len(records)} records to {outdir/'generated.jsonl'}")

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
        raw_lines = [s.strip() for s in rec["model_trace"].splitlines() if s.strip()]

        # Keep only reasoning steps (drop the final answer line)
        steps = [ln for ln in raw_lines if ln.lower().startswith("step ")]

        # Get verifier judgments in one call
        try:
            judge_results = judge_client.judge_steps(question=rec["question"], steps=steps)
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

        evidence_support = None  # enabled later for non-math

        # Risk + earliest bad step
        risks = [min(1.0, v + c) for v, c in zip(verifier, contradiction)]
        earliest = next((i for i, r in enumerate(risks) if r > tau), None)

        rec2 = dict(rec)
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
        k = rec.get("earliest_bad_step")
        rec2 = dict(rec)
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
        current_trace = rec["model_trace"]
        current_answer = rec.get("model_answer")

        iteration_logs = []

        current_score = score_record_trace(
            rec=rec,
            trace=current_trace,
            nli=nli,
            judge_client=judge_client,
            tau=tau,
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
            )

            old_avg_risk = final_score["risk_summary"]["avg_risk"]
            new_avg_risk = candidate_score["risk_summary"]["avg_risk"]
            improvement = old_avg_risk - new_avg_risk

            accepted = improvement >= 0.0

            iteration_logs.append({
                "iter": iter_idx,
                "kind": "repair",
                "k": k,
                "next_step_number": next_step_number,
                "answer": candidate_answer,
                "old_avg_risk": old_avg_risk,
                "new_avg_risk": new_avg_risk,
                "improvement": improvement,
                "accepted": accepted,
                "continue": improvement >= improvement_threshold,
                "earliest_bad_step": candidate_score["earliest_bad_step"],
                **candidate_score["risk_summary"],
            })

            if not accepted:
                stop_reason = "risk_worsened"
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

        task = rec.get("task")
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

    generate_traces(config=config)
    iterative_repair(config=config, input_path=None)
    evaluate(config=config, input_path=None)


def main():
    app()


if __name__ == "__main__":
    main()