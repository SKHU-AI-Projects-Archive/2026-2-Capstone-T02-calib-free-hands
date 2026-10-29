import os
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


def _fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=APP_ROOT, env=os.environ.copy(),
        capture_output=True, text=True, timeout=30,
    )


def test_application_module_imports_in_a_fresh_process():
    result = _fresh_python("import calibfreehands.app.main; print('IMPORT_OK')")
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_small_export_and_visualization_modules_do_not_eagerly_import_pipeline_or_viewer():
    result = _fresh_python(
        "import sys; import calibfreehands.export.paths; "
        "assert 'calibfreehands.pipeline.service' not in sys.modules; "
        "import calibfreehands.pipeline.result; "
        "assert 'calibfreehands.pipeline.service' not in sys.modules; "
        "import calibfreehands.visualization.topology; "
        "assert 'calibfreehands.visualization.comparison' not in sys.modules; "
        "print('LIGHT_IMPORT_OK')"
    )
    assert result.returncode == 0, result.stderr
    assert "LIGHT_IMPORT_OK" in result.stdout
