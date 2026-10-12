import pytest

from calibfreehands.export.paths import choose_output_directory


def test_output_collision_policies(tmp_path):
    (tmp_path / "sample").mkdir()
    assert choose_output_directory(tmp_path, "sample", "auto_rename").name == "sample_1"
    with pytest.raises(FileExistsError):
        choose_output_directory(tmp_path, "sample", "ask")
