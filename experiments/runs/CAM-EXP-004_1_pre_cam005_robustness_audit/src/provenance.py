"""Pin down exactly which code and which weights produced every result.

The goal is that someone in three years can rebuild this environment from the
manifest alone. Where an exact commit could not be recovered the record says so
rather than guessing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, RUN_DIR, rel, sha256, write_csv, write_json  # noqa: E402

VENV = RUN_DIR.parents[1] / ".venv-calib"
PY = VENV / "Scripts" / "python.exe"
HUB = Path.home() / ".cache" / "torch" / "hub"

CHECKPOINTS = {
    "anycalib_gen": HUB / "anycalib" / "anycalib_gen.pt",
    "anycalib_dist": HUB / "anycalib" / "anycalib_dist.pt",
    "anycalib_pinhole": HUB / "anycalib" / "anycalib_pinhole.pt",
    "geocalib_distorted": HUB / "geocalib" / "distorted.tar",
    "geocalib_pinhole": HUB / "geocalib" / "pinhole.tar",
    "pf_uncentered_rpfpp": HUB / "checkpoints" / "paramnet_360cities_edina_rpfpp.pth",
    "pf_centered_rpf": HUB / "checkpoints" / "paramnet_360cities_edina_rpf.pth",
}

CHECKPOINT_URLS = {
    "anycalib_gen": "https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_gen.pt",
    "anycalib_dist": "https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_dist.pt",
    "anycalib_pinhole": "https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_pinhole.pt",
    "geocalib_distorted": "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-distorted.tar",
    "geocalib_pinhole": "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
    "pf_uncentered_rpfpp": "torch.hub checkpoint for Paramnet-360Cities-edina-uncentered",
    "pf_centered_rpf": "torch.hub checkpoint for Paramnet-360Cities-edina-centered",
}


def direct_urls() -> dict:
    code = (
        "import importlib.metadata as md, json\n"
        "out={}\n"
        "for n in ['anycalib','geocalib','perspective2d']:\n"
        "    try:\n"
        "        d=md.distribution(n)\n"
        "        t=d.read_text('direct_url.json')\n"
        "        out[n]={'version':d.version,'direct_url':json.loads(t) if t else None}\n"
        "    except Exception as e:\n"
        "        out[n]={'error':str(e)}\n"
        "print(json.dumps(out))\n"
    )
    return json.loads(subprocess.check_output([str(PY), "-c", code], text=True))


def pf_git_commit() -> str | None:
    src = RUN_DIR.parents[1] / "cache" / "external_models" / "PerspectiveFields"
    try:
        return subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def versions() -> dict:
    code = (
        "import json,sys\n"
        "out={'python':sys.version.split()[0]}\n"
        "for m in ['torch','torchvision','numpy','cv2','kornia','scipy','PIL',"
        "'matplotlib','detectron2']:\n"
        "    try:\n"
        "        mod=__import__(m)\n"
        "        out[m]=getattr(mod,'__version__','unknown')\n"
        "    except Exception as e: out[m]='NOT_INSTALLED'\n"
        "try:\n"
        "    import torch\n"
        "    out['cuda']=torch.version.cuda\n"
        "    out['cudnn']=str(torch.backends.cudnn.version())\n"
        "    out['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'\n"
        "except Exception: pass\n"
        "print(json.dumps(out))\n"
    )
    return json.loads(subprocess.check_output([str(PY), "-c", code], text=True))


def main() -> None:
    du = direct_urls()
    ver = versions()
    pf_commit = pf_git_commit()

    ckpt = {}
    for name, path in CHECKPOINTS.items():
        if path.exists():
            ckpt[name] = {"file": str(path), "bytes": path.stat().st_size,
                          "sha256": sha256(path), "url": CHECKPOINT_URLS[name],
                          "status": "PRESENT_AND_HASHED"}
        else:
            ckpt[name] = {"file": str(path), "url": CHECKPOINT_URLS[name],
                          "status": "NOT_PRESENT_LOCALLY"}

    def commit_of(pkg):
        d = du.get(pkg, {})
        vi = (d.get("direct_url") or {}).get("vcs_info") or {}
        return vi.get("commit_id")

    models = {
        "AnyCalib": {
            "repository": "https://github.com/javrtg/AnyCalib",
            "exact_commit": commit_of("anycalib"),
            "commit_resolution": "recovered from pip direct_url.json vcs_info",
            "installed_version": du.get("anycalib", {}).get("version"),
            "checkpoints_used": ["anycalib_gen", "anycalib_dist", "anycalib_pinhole"],
            "adapter": "experiments/src/calibration/anycalib_adapter.py",
            "inference_command": "AnyCalibAdapter(model_id=..., cam_id='radial:2').predict(bgr, meta)",
            "output_convention": "intrinsics already mapped back to ORIGINAL image "
                                 "pixels by predict(); [fx, fy, cx, cy, k1, k2] for "
                                 "radial:2; normalised radial polynomial "
                                 "1 + k1 r^2 + k2 r^4 (OpenCV-compatible)",
            "license": "see repository",
            "used_in": ["CAM-EXP-003", "CAM-EXP-003.1", "CAM-EXP-004", "CAM-EXP-004.1"],
        },
        "GeoCalib": {
            "repository": "https://github.com/cvg/GeoCalib",
            "exact_commit": commit_of("geocalib"),
            "commit_resolution": "recovered from pip direct_url.json vcs_info",
            "installed_version": du.get("geocalib", {}).get("version"),
            "checkpoints_used": ["geocalib_distorted", "geocalib_pinhole"],
            "checkpoint_release": "GitHub release v1.0, fetched by the package itself "
                                  "via torch.hub.load_state_dict_from_url",
            "adapter": "experiments/src/calibration/geocalib_adapter.py",
            "inference_command": "GeoCalibAdapter(weights='distorted', camera_model='radial').predict(bgr, meta)",
            "output_convention": "camera.f already in original pixels; principal "
                                 "point fixed at the image centre and reported as "
                                 "NOT_PREDICTED; radial k1,k2 OpenCV-compatible",
            "critical_pin": "kornia==0.7.2 (newer kornia has a TorchScript "
                            "incompatibility with torch 2.1.2)",
            "license": "see repository",
            "used_in": ["CAM-EXP-003", "CAM-EXP-003.1", "CAM-EXP-004", "CAM-EXP-004.1"],
        },
        "PerspectiveFields": {
            "repository": "https://github.com/jinlinyi/PerspectiveFields",
            "exact_commit": pf_commit,
            "commit_resolution": "installed from a local clone at "
                                 "experiments/cache/external_models/PerspectiveFields; "
                                 "commit read from that clone's git HEAD",
            "installed_version": du.get("perspective2d", {}).get("version"),
            "checkpoints_used": ["pf_uncentered_rpfpp", "pf_centered_rpf"],
            "adapter": "experiments/src/calibration/perspective_fields_adapter.py",
            "inference_command": "PerspectiveFieldsAdapter(version=UNCENTERED).predict(bgr, meta)",
            "output_convention": "rel_focal = f / image_height; f_px = rel_focal * H",
            "install_note": "installed with --no-deps; albumentations pulls "
                            "stringzilla which has no Windows wheel and is not "
                            "needed for inference",
            "license": "see repository",
            "used_in": ["CAM-EXP-003", "CAM-EXP-004 (diversity baseline)"],
        },
        "AnyCam": {
            "repository": "https://github.com/Brummi/anycam",
            "exact_commit": None,
            "commit_resolution": "NOT_INSTALLED",
            "status": "NOT_APPLICABLE_REQUIRES_CAMERA_MOTION - audited in "
                      "CAM-EXP-004 and stress-tested in CAM-EXP-004.1; see "
                      "anycam_static_stress_test.md",
            "used_in": [],
        },
    }

    manifest = {
        "id": "EXTERNAL_MODEL_PROVENANCE_V1",
        "frozen_on": "2026-09-23",
        "environment": {
            "venv": "experiments/.venv-calib (git-ignored)",
            **ver,
            "os": "Windows 11 Pro 10.0.26200",
        },
        "models": models,
        "unresolved": [k for k, v in models.items()
                       if v.get("exact_commit") is None and k != "AnyCam"],
        "note": "AnyCalib and GeoCalib exact commits were recovered from pip's "
                "direct_url.json; no EXACT_COMMIT_UNRESOLVED_FOR_HISTORICAL_RUN "
                "case remained, so no re-install equivalence check was needed for "
                "provenance reasons. An independent equivalence check on the "
                "8 shared frames is reported separately.",
        "checkpoints": ckpt,
    }
    write_json(MANIFESTS / "external_model_provenance_v1.json", manifest)

    table = []
    for name, m in models.items():
        table.append({
            "model": name, "repository": m.get("repository", ""),
            "exact_commit": m.get("exact_commit") or "NOT_INSTALLED",
            "commit_resolution": m.get("commit_resolution", ""),
            "checkpoints": ";".join(m.get("checkpoints_used", [])) or "none",
            "checkpoint_sha256_first16": ";".join(
                ckpt[c]["sha256"][:16] for c in m.get("checkpoints_used", [])
                if ckpt.get(c, {}).get("sha256")),
            "adapter": m.get("adapter", ""),
            "output_convention": m.get("output_convention", ""),
            "used_in": ";".join(m.get("used_in", [])),
        })
    write_csv(RUN_DIR / "tables" / "external_model_provenance.csv", table)

    dep = [{"package": k, "version": v, "criticality":
            ("PINNED - required" if k in ("torch", "kornia", "numpy") else "recorded")}
           for k, v in ver.items()]
    write_csv(RUN_DIR / "tables" / "dependency_versions.csv", dep)

    freeze = subprocess.check_output([str(PY), "-m", "pip", "freeze"], text=True)
    (RUN_DIR / "requirements-calibration-frozen.txt").write_text(
        "# pip freeze of experiments/.venv-calib, the environment that produced\n"
        "# every CAM-EXP-003 / 003.1 / 004 / 004.1 model prediction.\n"
        "# Exact repository commits for the git-installed packages are in\n"
        "# experiments/manifests/external_model_provenance_v1.json.\n" + freeze,
        encoding="utf-8")

    print(json.dumps({k: v.get("exact_commit") for k, v in models.items()}, indent=2))
    print(f"unresolved: {manifest['unresolved']}")
    for k, v in ckpt.items():
        print(f"  {k:22s} {v['status']:24s} {v.get('sha256', '')[:16]}")


if __name__ == "__main__":
    main()
