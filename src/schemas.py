from typing import Any, Dict, List, Optional


def make_base_record(
    *,
    rid: str,
    question: str,
    model_trace: str,
    model_answer: str,
    evidence: Optional[Any] = None,
    gold_answer: Optional[str] = None,
    task: Optional[str] = None,
    source: Optional[str] = None,
    model: Optional[str] = None,
    prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": rid,
        "question": question,
        "model_trace": model_trace,
        "model_answer": model_answer,
    }
    if evidence is not None:
        rec["evidence"] = evidence
    if gold_answer is not None:
        rec["gold_answer"] = gold_answer
    if task is not None:
        rec["task"] = task
    if source is not None:
        rec["source"] = source
    if model is not None:
        rec["model"] = model
    if prompt_id is not None:
        rec["prompt_id"] = prompt_id
    return rec