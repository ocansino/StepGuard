from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from .datasets_prep import prepare_gsm8k, prepare_strategyqa
import typer
import re
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


def gsm8k_exact_match(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_gsm8k_answer(pred)
    g = normalize_gsm8k_answer(gold)
    return (p is not None) and (g is not None) and (p == g)

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

# WIP
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





















# STUB, DOES NOTHING YET BUT RETURN FAKE INFO
@app.command("evaluate")
def evaluate(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    inpath = Path(input_path) if input_path else (outdir / "repaired.jsonl")

    manifest = make_manifest(cfg.run_name, "evaluate", config, cfg.raw)
    write_manifest(outdir / "manifest.evaluate.json", manifest)

    total = 0
    correct = 0
    for rec in read_jsonl(inpath):
        total += 1
        gold = rec.get("gold_answer")
        pred = rec.get("repaired_answer") or rec.get("model_answer")
        task = rec.get("task")

        if gold is None or pred is None:
            continue

        if task == "math":
            if gsm8k_exact_match(pred, gold):
                correct += 1
        else:
            # keep old string match for now (we'll improve non-math later)
            if str(pred).strip().lower() == str(gold).strip().lower():
                correct += 1

    metrics = {"total": total, "exact_match": (correct / total if total else 0.0)}
    (outdir / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"Metrics: {metrics} (saved to {outdir/'metrics.json'})")


@app.command("run-pipeline")
def run_pipeline(config: str = typer.Option(..., "--config", "-c")):
    # Phase 0: just execute the stages in order
    generate_traces(config=config)
    score_traces(config=config, input_path=None)
    repair_traces(config=config, input_path=None)
    evaluate(config=config, input_path=None)


def main():
    app()


if __name__ == "__main__":
    main()