from pathlib import Path
from typing import Optional, List, Dict, Any
from .datasets_prep import prepare_gsm8k, prepare_strategyqa
import typer

from .config import load_config
from .io_utils import read_jsonl, write_jsonl
from .manifest import make_manifest, write_manifest
from .schemas import make_base_record

app = typer.Typer(add_completion=False)


def run_dir(base_output_dir: str, run_name: str) -> Path:
    return Path(base_output_dir) / run_name

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

@app.command("generate-traces")
def generate_traces(config: str = typer.Option(..., "--config", "-c")):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest(cfg.run_name, "generate-traces", config, cfg.raw)
    write_manifest(outdir / "manifest.generate.json", manifest)

    records = []
    for row in read_jsonl(cfg.dataset_path):
        # STUB: fake a chain-of-thought style trace
        q = row["question"]
        trace = f"Step 1: Understand the question.\nStep 2: Compute or recall answer.\nStep 3: Provide final."
        ans = row.get("gold_answer", "UNKNOWN")  # stub uses gold if present
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


@app.command("score-traces")
def score_traces(
    config: str = typer.Option(..., "--config", "-c"),
    input_path: Optional[str] = typer.Option(None, "--input"),
):
    cfg = load_config(config)
    outdir = run_dir(cfg.output_dir, cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    inpath = Path(input_path) if input_path else (outdir / "generated.jsonl")

    manifest = make_manifest(cfg.run_name, "score-traces", config, cfg.raw)
    write_manifest(outdir / "manifest.score.json", manifest)

    scored = []
    tau = cfg.raw.get("scoring", {}).get("tau", 0.8)

    for rec in read_jsonl(inpath):
        # STUB: step splitting by lines
        steps = [s.strip() for s in rec["model_trace"].splitlines() if s.strip()]

        # STUB: fake scores (replace later)
        verifier = [0.1 for _ in steps]
        contradiction = [0.05 for _ in steps]
        evidence_support = None  # enabled later for non-math

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

    repaired = []
    for rec in read_jsonl(inpath):
        k = rec.get("earliest_bad_step")
        rec2 = dict(rec)

        if k is None:
            rec2["repaired_trace"] = None
            rec2["repaired_answer"] = None
        else:
            # STUB: pretend we repaired by replacing suffix with a placeholder
            prefix = rec.get("steps", [])[:k]
            suffix = ["[REPAIRED] Step regenerated with constraints.", "[REPAIRED] Final answer produced."]
            rec2["repaired_trace"] = "\n".join(prefix + suffix)
            rec2["repaired_answer"] = rec.get("model_answer")

        rec2["logs"] = {
            "iterations": 1,
            "note": "stub repair; replace with suffix regeneration later",
        }
        repaired.append(rec2)

    write_jsonl(outdir / "repaired.jsonl", repaired)
    typer.echo(f"Wrote {len(repaired)} records to {outdir/'repaired.jsonl'}")


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
        if gold is not None and pred is not None and str(pred).strip() == str(gold).strip():
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