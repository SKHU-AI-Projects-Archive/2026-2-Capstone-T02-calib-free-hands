import contextlib
import io
import importlib.util
import os
import tarfile
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "download_gigahands_demo.py"
SPEC = importlib.util.spec_from_file_location("download_gigahands_demo", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def make_archive(path: Path, include_traversal: bool = False) -> None:
    with tempfile.TemporaryDirectory() as source:
        root = Path(source) / "wrapper"
        (root / "hand_pose" / "sequence-1" / "rgb_vid" / "camera-1").mkdir(parents=True)
        (root / "object_pose" / "sequence-1").mkdir(parents=True)
        (root / "hand_pose" / "sequence-1" / "optim_params.txt").write_text("# camera\nrow\n", encoding="utf-8")
        (root / "hand_pose" / "sequence-1" / "rgb_vid" / "camera-1" / "frame.mp4").write_bytes(b"video")
        with tarfile.open(path, "w:gz") as archive:
            archive.add(root, arcname="wrapper")
            if include_traversal:
                info = tarfile.TarInfo("../escape.txt")
                info.size = 4
                archive.addfile(info, io.BytesIO(b"oops"))


class DownloaderTests(unittest.TestCase):
    def test_profile_and_wrapper_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "wrapper"
            (root / "hand_pose" / "sequence-1" / "rgb_vid" / "camera-1").mkdir(parents=True)
            (root / "object_pose").mkdir()
            (root / "hand_pose" / "sequence-1" / "optim_params.txt").write_text("# camera\nrow\n", encoding="utf-8")
            (root / "hand_pose" / "sequence-1" / "rgb_vid" / "camera-1" / "frame.mp4").write_bytes(b"video")
            profile = MODULE.inspect_dataset(root)
            self.assertEqual(profile["sequence_count"], 1)
            self.assertEqual(profile["rgb_mp4_total"], 1)
            self.assertEqual(profile["camera_rows"], 1)

    def test_download_rename_and_install_from_local_http_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "gigahands_demo_all.tar.gz"
            make_archive(archive)
            original_profile = MODULE.EXPECTED_PROFILE
            MODULE.EXPECTED_PROFILE = MODULE.inspect_dataset(base / "wrapper") if (base / "wrapper").exists() else {
                "sequence_count": 1,
                "sequences": ("sequence-1",),
                "rgb_mp4_total": 1,
                "rgb_mp4_by_sequence": {"sequence-1": 1},
                "optim_params_count": 1,
                "camera_rows": 1,
                "file_count": 2,
            }
            server_root = base / "server"
            server_root.mkdir()
            archive.replace(server_root / archive.name)
            handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(server_root), **kwargs)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                cache = base / "cache"
                target = base / "install"
                url = f"http://127.0.0.1:{server.server_port}/{archive.name}"
                MODULE.prepare_dataset(target, cache, url)
                self.assertTrue((cache / archive.name).is_file())
                self.assertFalse((cache / f"{archive.name}.part").exists())
                self.assertTrue((target / "hand_pose").is_dir())
                self.assertTrue((target / "object_pose").is_dir())
            finally:
                server.shutdown()
                server.server_close()
                MODULE.EXPECTED_PROFILE = original_profile

    def test_non_empty_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            (target / "hand_pose").mkdir(parents=True)
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.prepare_dataset(target, Path(temporary) / "cache", "http://invalid.example/archive")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.tar.gz"
            make_archive(archive, include_traversal=True)
            with self.assertRaises(RuntimeError):
                MODULE.safe_extract(archive, Path(temporary) / "extract")

    def test_dry_run_does_not_depend_on_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    MODULE.print_dry_run(Path(temporary) / "target", Path(temporary) / "cache")
                self.assertIn("gigahands_demo_all.tar.gz", output.getvalue())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
