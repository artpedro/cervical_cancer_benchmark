from __future__ import annotations

from collections.abc import Iterable


SOLO_DATASETS: tuple[str, ...] = ("herlev", "sipakmed", "riva")
DATASET_DISPLAY: dict[str, str] = {
    "herlev": "Herlev",
    "sipakmed": "SIPaKMeD",
    "riva": "Riva",
}


def split_mixed_slug(dataset: str, *, solo_datasets: Iterable[str] = SOLO_DATASETS) -> tuple[str, ...]:
    parts = [p for p in str(dataset).split("_") if p]
    known = set(str(x) for x in solo_datasets)
    if len(parts) < 2 or any(p not in known for p in parts):
        return tuple()
    return tuple(parts)


def is_mixed_dataset(dataset: str, *, solo_datasets: Iterable[str] = SOLO_DATASETS) -> bool:
    return bool(split_mixed_slug(dataset, solo_datasets=solo_datasets))


def dataset_regime(dataset: str, *, solo_datasets: Iterable[str] = SOLO_DATASETS) -> str:
    d = str(dataset)
    known = set(str(x) for x in solo_datasets)
    if d in known:
        return "solo"
    if is_mixed_dataset(d, solo_datasets=solo_datasets):
        return "mixed"
    return "unknown"


def canonical_mixed_slug(
    dataset: str,
    *,
    solo_datasets: Iterable[str] = SOLO_DATASETS,
) -> str:
    parts = split_mixed_slug(dataset, solo_datasets=solo_datasets)
    if not parts:
        return str(dataset)
    order = {name: idx for idx, name in enumerate(str(x) for x in solo_datasets)}
    return "_".join(sorted(parts, key=lambda p: order.get(p, 10_000)))


def infer_dataset_order(
    datasets: Iterable[str],
    *,
    solo_order: Iterable[str] = SOLO_DATASETS,
) -> list[str]:
    names = sorted({str(d) for d in datasets if str(d)})
    solo_order_list = [str(x) for x in solo_order]
    solo_seen = [d for d in solo_order_list if d in names]
    solo_extra = [d for d in names if d not in solo_order_list and "_" not in d]
    mixed = [d for d in names if d not in solo_seen and d not in solo_extra]
    mixed = sorted(mixed, key=lambda d: canonical_mixed_slug(d, solo_datasets=solo_order_list))
    return solo_seen + solo_extra + mixed


def dataset_display_name(dataset: str) -> str:
    ds = str(dataset)
    if ds in DATASET_DISPLAY:
        return DATASET_DISPLAY[ds]
    parts = split_mixed_slug(ds)
    if not parts:
        return ds
    return " + ".join(DATASET_DISPLAY.get(p, p.title()) for p in parts)
