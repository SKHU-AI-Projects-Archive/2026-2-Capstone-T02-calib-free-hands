from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # HandCalib/
REPO_ROOT = PROJECT_ROOT.parent


def resolve(p) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else PROJECT_ROOT / q
