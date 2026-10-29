"""HandCalib hand-template focal length: the former single module, now split into handfocal_estimate.py (focal-length estimation: everything from video normalisation to the f estimate, verdicts and the
experiments' analysis) and handfocal_report.py (result writing, reports, provenance). This file loads both into one namespace, so `import handfocal as hf` (run.py, tests) sees the same names as before;
a function defined in one of the two modules finds the other module's names through its own imports, not through this file."""
import handfocal_estimate as _estimate
import handfocal_report as _report

globals().update({k: v for k, v in vars(_estimate).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_report).items() if not k.startswith("__")})
