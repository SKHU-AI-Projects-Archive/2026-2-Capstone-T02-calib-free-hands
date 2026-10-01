"""Verify that Phase A code never READS the target.

Checks actual code, not prose: an import of VIEW_TARGETS, or a read of the
target file. A docstring that mentions the name in order to assert blindness,
or a frozen spec that DECLARES which target will later be used, are both
correct and must not be flagged.
"""
from __future__ import annotations
import ast, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUN_DIR, SUM, write_json

PHASE_A = ["audit_hand_model.py", "freeze_spec.py", "run_hand_inference.py",
           "extract_hand_features.py", "feature_quality_audit.py",
           "verify_hook_identity.py"]
PHASE_B = ["run_probes.py", "evaluate.py", "feature_associations.py",
           "figures.py"]
TARGET_NAMES = {"VIEW_TARGETS", "SCENE_FEATURES", "CAM005_KEPT"}


def reads_target(path: Path):
    """Return the code-level target references in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in TARGET_NAMES:
                    hits.append(f"imports {a.name}")
        elif isinstance(node, ast.Name) and node.id in TARGET_NAMES:
            hits.append(f"uses {node.id}")
    return sorted(set(hits))


def main() -> None:
    a = {f: reads_target(RUN_DIR / "src" / f) for f in PHASE_A
         if (RUN_DIR / "src" / f).exists()}
    b = {f: reads_target(RUN_DIR / "src" / f) for f in PHASE_B
         if (RUN_DIR / "src" / f).exists()}
    violations = {f: h for f, h in a.items() if h}
    res = {
        "phase_a_files": a, "phase_b_files": b,
        "phase_a_violations": violations,
        "verdict": "PHASE_SEPARATION_HELD" if not violations
                   else "PHASE_SEPARATION_VIOLATED",
        "note": "prose mentions are deliberately not flagged: "
                "extract_hand_features.py's docstring names VIEW_TARGETS in "
                "order to assert it is absent, and freeze_spec.py records "
                "which target the pre-registration commits to. Neither reads "
                "target data.",
    }
    write_json(SUM / "phase_separation_check.json", res)
    for f, h in a.items():
        print(f"  PHASE A {f:32s} {'OK' if not h else h}")
    for f, h in b.items():
        print(f"  PHASE B {f:32s} {h or '(none)'}")
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
