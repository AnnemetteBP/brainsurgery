from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


def choose_num_io_workers(num_items: int, max_io_workers: int) -> int:
    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if max_io_workers < 1:
        raise ValueError("max_io_workers must be at least 1")
    return max(1, min(max_io_workers, num_items))


def run_threadpool_tasks_with_progress(
    *,
    items: Iterable[ItemT],
    worker: Callable[[ItemT], ResultT],
    num_workers: int,
    total: int,
    progress_desc: str,
    progress_unit: str,
    progress_factory: Callable[..., Any],
    on_result: Callable[[ItemT, ResultT], None] | None = None,
) -> None:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}

        progress = progress_factory(
            total=total,
            desc=progress_desc,
            unit=progress_unit,
            leave=False,
        )
        try:
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                if on_result is not None:
                    on_result(item, result)
                progress.update(1)
        finally:
            progress.close()
