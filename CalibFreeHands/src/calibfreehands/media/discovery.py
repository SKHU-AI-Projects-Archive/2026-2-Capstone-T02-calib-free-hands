from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
    ".gif", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv",
})


def discover_media(path: Path) -> list[Path]:
    """Return supported files directly inside *path*; never recurse."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(
        (entry for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda item: item.name.lower(),
    )
