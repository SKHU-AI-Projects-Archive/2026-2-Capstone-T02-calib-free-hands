from calibfreehands.pipeline.provenance import sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"calibfreehands")
    assert sha256_file(path) == "7e7a433d417316976adaeb9ef7120e011f40e8cf217e7b2fc34bcce700342284"
