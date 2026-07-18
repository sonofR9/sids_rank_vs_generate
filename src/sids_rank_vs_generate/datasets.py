from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


DATASET_SIZES = ("50m", "500m", "5b")
DEFAULT_DATASET_SIZE = "50m"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "sids_rank_vs_generate" / "datasets"
REQUIRED_FILES = ("interactions.parquet", "embeddings.parquet")


@dataclass(frozen=True)
class DatasetFiles:
    """Resolved files for one supported Yambda scale."""

    size: str
    interactions_path: Path
    embeddings_path: Path


def _extract_required_files(archive_path: Path, target_dir: Path) -> None:
    """Extract only the files used by the common pipeline from a zip archive."""

    with zipfile.ZipFile(archive_path) as archive:
        members_by_name = {
            Path(member.filename).name: member
            for member in archive.infolist()
            if not member.is_dir()
        }
        missing = [name for name in REQUIRED_FILES if name not in members_by_name]
        if missing:
            raise ValueError(
                f"dataset archive {archive_path} is missing: {', '.join(missing)}"
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_FILES:
            destination = target_dir / filename
            with archive.open(members_by_name[filename]) as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target)


def resolve_dataset(
    size: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    bootstrap_archive: Path | None = None,
) -> DatasetFiles:
    """Resolve a dataset scale from the shared cache.

    The repository's legacy ``dataset.zip`` can bootstrap the default 50m cache.
    A remote source is intentionally not guessed here: the reviewed project files
    use a course-specific schema that does not match the public Yambda archives.
    """

    normalized_size = size.lower()
    if normalized_size not in DATASET_SIZES:
        choices = ", ".join(DATASET_SIZES)
        raise ValueError(f"unsupported dataset size {size!r}; choose one of: {choices}")

    dataset_dir = cache_root.expanduser() / f"yambda-{normalized_size}"
    interactions_path = dataset_dir / "interactions.parquet"
    embeddings_path = dataset_dir / "embeddings.parquet"

    if not (interactions_path.is_file() and embeddings_path.is_file()):
        archive_path = bootstrap_archive
        if archive_path is None and normalized_size == DEFAULT_DATASET_SIZE:
            archive_path = Path("dataset.zip")
        if archive_path is not None and archive_path.expanduser().is_file():
            _extract_required_files(archive_path.expanduser(), dataset_dir)

    missing = [
        path.name
        for path in (interactions_path, embeddings_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Yambda-{normalized_size} is not available in {dataset_dir}. "
            f"Missing: {', '.join(missing)}. Put the prepared files in this "
            "shared cache. Automatic Kaggle download requires the team's exact "
            "dataset reference and archive schema."
        )

    return DatasetFiles(
        size=normalized_size,
        interactions_path=interactions_path,
        embeddings_path=embeddings_path,
    )
