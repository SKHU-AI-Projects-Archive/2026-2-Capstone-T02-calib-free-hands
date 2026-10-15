"""Video probing and normalisation (ffmpeg / ffprobe). Only technical stream fields are kept: no location, no device tags."""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

SAFE_STREAM_KEYS = ("codec_name", "profile", "width", "height", "pix_fmt", "color_transfer", "color_primaries", "color_space",
                    "r_frame_rate", "avg_frame_rate", "nb_frames", "duration", "bits_per_raw_sample")
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def tool_info() -> dict:
    out = {}
    for t in ("ffmpeg", "ffprobe"):
        p = shutil.which(t)
        out[t] = {"path": p, "version": subprocess.check_output([p, "-hide_banner", "-version"], text=True).splitlines()[0] if p else None}
    filt = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], text=True) if out["ffmpeg"]["path"] else ""
    out["filters"] = {f: (f" {f} " in filt) for f in ("zscale", "tonemap", "colorspace")}
    return out


def probe(path: Path) -> dict:
    """Technical fields of the first video stream (rotation from the display matrix); tags are never read."""
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=" + ",".join(SAFE_STREAM_KEYS), "-show_entries", "stream_side_data_list",
                                            "-show_entries", "format=duration,size,format_name", "-of", "json", str(path)], text=True))
    s = j["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    info = {k: s.get(k) for k in SAFE_STREAM_KEYS}
    info["rotation_deg"] = rot
    info["file_name"] = path.name
    info["file_size"] = int(j["format"].get("size", 0))
    info["container"] = j["format"].get("format_name")
    info["duration_s"] = float(j["format"].get("duration") or s.get("duration") or 0)
    info["fps"] = float(Fraction(s.get("avg_frame_rate") or s.get("r_frame_rate")))
    pix = s.get("pix_fmt") or ""
    info["bit_depth"] = int(s.get("bits_per_raw_sample") or (10 if "10" in pix else 8))
    info["hdr"] = (s.get("color_transfer") in HDR_TRANSFERS) or info["bit_depth"] > 8
    rotated = abs(rot) % 180 == 90
    info["display_w"], info["display_h"] = (s["height"], s["width"]) if rotated else (s["width"], s["height"])
    return info


def target_dims(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Aspect kept, never upscaled, even sides."""
    s = min(1.0, long_side / max(w, h))
    tw, th = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    return max(tw, 2), max(th, 2)


def build_command(src: Path, dst: Path, info: dict, cfg: dict, tools: dict) -> tuple[list, dict]:
    n = cfg["normalize"]
    tw, th = target_dims(info["display_w"], info["display_h"], int(n["long_side"]))
    vf, path = [], {"autorotate": True, "rotation_applied_deg": info["rotation_deg"]}
    if info["hdr"]:
        if tools["filters"]["zscale"] and tools["filters"]["tonemap"] and "zscale_tonemap" in n["hdr_path"]:
            vf += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709", "tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"]
            path["tonemap"] = "zscale+tonemap(hable)"
        elif tools["filters"]["colorspace"] and "colorspace" in n["hdr_path"]:
            vf += ["colorspace=all=bt709:iall=bt2020:fast=1"]
            path["tonemap"] = "colorspace (no tone curve)"
        else:
            raise RuntimeError("HDR input but neither zscale+tonemap nor colorspace is available")
    else:
        path["tonemap"] = "none (8-bit SDR input)"
    vf.append(f"scale={tw}:{th}:flags=bicubic")
    fps_in = info["fps"]
    if fps_in > float(n["max_fps"]) + 0.01:
        vf.append(f"fps={n['max_fps']}")
        path["fps"] = f"{fps_in:.3f} -> {n['max_fps']}"
    else:
        path["fps"] = f"{fps_in:.3f} (kept)"
    vf.append("format=yuv444p" if n["codec"] == "ffv1" else "format=yuv420p")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-map_metadata", "-1", "-map_chapters", "-1", "-vf", ",".join(vf), "-c:v", n["codec"], "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", str(dst)]
    path.update({"filters": vf, "out_size": [tw, th], "codec": n["codec"]})
    return cmd, path


def normalize(src: Path, dst: Path, cfg: dict) -> dict:
    tools = tool_info()
    info = probe(src)
    cmd, path = build_command(src, dst, info, cfg, tools)
    subprocess.run(cmd, check=True)
    out = probe(dst)
    if [out["display_w"], out["display_h"]] != path["out_size"]:
        raise RuntimeError(f"normalised size {out['display_w']}x{out['display_h']} != {path['out_size']}")
    return {"input": info, "output": {k: out[k] for k in ("display_w", "display_h", "fps", "duration_s", "nb_frames", "pix_fmt", "color_transfer", "rotation_deg")},
            "path": path, "command": cmd[:-1] + ["<out>"], "tools": tools}
