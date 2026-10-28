import pytest

from calibfreehands.export.paths import choose_output_directory, choose_output_file, result_filename


def test_output_collision_policies(tmp_path):
    (tmp_path / "sample").mkdir()
    assert choose_output_directory(tmp_path, "sample", "auto_rename").name == "sample_1"
    with pytest.raises(FileExistsError):
        choose_output_directory(tmp_path, "sample", "ask")


def test_checkerboard_filename_includes_clean_reference_focal():
    assert result_filename("clip", "Checkerboard Reference", "Original WiLoR", 1200.0) == (
        "clip__Checkerboard-Reference-1200px__Original-WiLoR.cfh"
    )
    assert result_filename("clip", "Checkerboard Reference", "Original WiLoR", 1280.5) == (
        "clip__Checkerboard-Reference-1280.5px__Original-WiLoR.cfh"
    )


def test_predicted_focal_does_not_change_anycalib_filename():
    assert result_filename("clip", "Original AnyCalib", "Original WiLoR", 1573.25) == (
        "clip__Original-AnyCalib__Original-WiLoR.cfh"
    )


def test_checkerboard_auto_rename_keeps_reference_focal(tmp_path):
    name = result_filename("clip", "Checkerboard Reference", "Original WiLoR", 1200.0)
    (tmp_path / name).touch()

    assert choose_output_file(tmp_path, name, "auto_rename").name == (
        "clip__Checkerboard-Reference-1200px__Original-WiLoR_1.cfh"
    )
