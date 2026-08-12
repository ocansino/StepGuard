from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, cast


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class WorkItemFailure:
    index: int
    item_id: str
    error_type: str
    message: str


class BoundedExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        failures: List[WorkItemFailure],
        completed_items: int,
        total_items: int,
    ) -> None:
        self.failures = tuple(sorted(failures, key=lambda failure: failure.index))
        self.completed_items = completed_items
        self.total_items = total_items

        details = "; ".join(
            f"{failure.item_id}: {failure.error_type}"
            for failure in self.failures[:5]
        )
        if len(self.failures) > 5:
            details += f"; and {len(self.failures) - 5} more"

        super().__init__(
            f"{len(self.failures)} of {total_items} bounded work items failed "
            f"after {completed_items} completed successfully: {details}"
        )


def _resolve_item_id(
    *,
    index: int,
    item: InputT,
    item_id: Optional[Callable[[InputT], str]],
) -> str:
    if item_id is None:
        return str(index)

    try:
        resolved = str(item_id(item)).strip()
    except Exception:
        return str(index)

    return resolved or str(index)


def _make_failure(
    *,
    index: int,
    item: InputT,
    error: Exception,
    item_id: Optional[Callable[[InputT], str]],
) -> WorkItemFailure:
    message = str(error).replace("\r", " ").replace("\n", " ")
    return WorkItemFailure(
        index=index,
        item_id=_resolve_item_id(
            index=index,
            item=item,
            item_id=item_id,
        ),
        error_type=type(error).__name__,
        message=message[:500],
    )


def map_bounded(
    items: Iterable[InputT],
    worker: Callable[[InputT], ResultT],
    *,
    max_workers: int,
    item_id: Optional[Callable[[InputT], str]] = None,
) -> List[ResultT]:
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("max_workers must be an integer between 1 and 32")
    if max_workers < 1 or max_workers > 32:
        raise ValueError("max_workers must be an integer between 1 and 32")

    work_items = list(items)
    if not work_items:
        return []

    results: List[Optional[ResultT]] = [None] * len(work_items)
    failures: List[WorkItemFailure] = []
    completed_items = 0

    # Preserve the original direct execution path for the sequential baseline.
    if max_workers == 1:
        for index, item in enumerate(work_items):
            try:
                results[index] = worker(item)
                completed_items += 1
            except Exception as error:
                failures.append(
                    _make_failure(
                        index=index,
                        item=item,
                        error=error,
                        item_id=item_id,
                    )
                )
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="stepguard",
        ) as executor:
            futures = {
                executor.submit(worker, item): (index, item)
                for index, item in enumerate(work_items)
            }

            for future in as_completed(futures):
                index, item = futures[future]
                try:
                    results[index] = future.result()
                    completed_items += 1
                except Exception as error:
                    failures.append(
                        _make_failure(
                            index=index,
                            item=item,
                            error=error,
                            item_id=item_id,
                        )
                    )

    if failures:
        raise BoundedExecutionError(
            failures=failures,
            completed_items=completed_items,
            total_items=len(work_items),
        )

    # Every slot is populated when no failures occurred.
    return [cast(ResultT, result) for result in results]