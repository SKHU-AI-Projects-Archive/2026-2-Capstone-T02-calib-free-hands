from __future__ import annotations

from contextlib import closing
from fractions import Fraction
from itertools import islice
import math

from calibfreehands.pipeline.service import reconstruct_frames

from .video import Recording


CALIBRATION_SECONDS = 20


def run_video(recording: Recording, calibration, pose, *, work_seconds=5.0, calibration_hz=None,
              progress=None, focal_ready=None, cancelled=None):
    if not math.isfinite(work_seconds) or work_seconds <= 0:
        raise ValueError("work duration must be finite and positive")
    if calibration_hz is not None and (not math.isfinite(calibration_hz) or calibration_hz <= 0):
        raise ValueError("calibration sample rate must be finite and positive")
    times = recording.times
    calibration_stop = CALIBRATION_SECONDS
    work_stop = calibration_stop + Fraction(str(work_seconds))
    calibration_ids = []
    sample_rate = Fraction(str(calibration_hz)) if calibration_hz is not None else None
    next_sample = 0
    for index, timestamp in enumerate(times):
        if timestamp >= calibration_stop:
            break
        if timestamp >= next_sample:
            calibration_ids.append(index)
            if sample_rate is not None:
                next_sample = (math.floor(timestamp * sample_rate) + 1) / sample_rate
    work_ids = [i for i, t in enumerate(times) if calibration_stop <= t < work_stop]
    if not calibration_ids or not work_ids:
        raise ValueError("Video needs a full 20-second calibration span and frames at or after 20 seconds")

    selected = set(calibration_ids)
    with closing(recording.frames()) as frames:
        sample_count = 0
        for index in range(work_ids[0]):
            if cancelled and cancelled():
                raise InterruptedError("Processing cancelled during calibration")
            frame = next(frames, None)
            if frame is None:
                raise ValueError("Video ended during calibration")
            if index in selected:
                calibration.add(frame.rgb)
                sample_count += 1
                if progress:
                    progress("Calibrating", sample_count, len(calibration_ids))
        focal = float(calibration.estimate())
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError("The estimator returned an invalid focal length")
        if focal_ready:
            focal_ready(focal)

        result = reconstruct_frames(islice(frames, len(work_ids)), len(work_ids), pose, focal,
                                    progress, cancelled)

    report = {
        "estimator": calibration.name,
        "pose_backend": pose.name,
        "calibration_range_s": [0.0, float(calibration_stop)],
        "calibration_source_pts_s": [float(recording.pts[i]) for i in calibration_ids],
        "calibration_sample_times_s": [float(times[i]) for i in calibration_ids],
        "work_range_s": [float(calibration_stop), float(work_stop)],
        "work_frame_times_s": result.timestamps.tolist(),
        "image_size": list(recording.size),
        "focal_px": focal,
        "pose_input_focal_px": focal,
        "work_frames": len(work_ids),
        "frames_with_hands": int(result.detected.any(axis=1).sum()),
    }
    if hasattr(calibration, "details"):
        report["estimator_details"] = calibration.details()
    return result, report
