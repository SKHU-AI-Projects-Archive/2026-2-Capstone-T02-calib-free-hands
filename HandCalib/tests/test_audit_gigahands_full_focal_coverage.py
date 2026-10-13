import importlib.util
from pathlib import Path
import tempfile
import unittest


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_gigahands_full_focal_coverage.py"
)
spec = importlib.util.spec_from_file_location("gigahands_full_focal_audit", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


HEADER = (
    "# cam_id width height fx fy cx cy k1 k2 p1 p2 cam_name "
    "qvecw qvecx qvecy qvecz tvecx tvecy tvecz\n"
)
ROW = (
    "1 1280 720 900 901 640 360 0.1 0.01 0.001 0.002 cam0 "
    "1 0 0 0 0 0 0\n"
)


class FullFocalAuditTests(unittest.TestCase):
    def write_file(self, body=ROW):
        temp = tempfile.TemporaryDirectory()
        sequence = Path(temp.name) / "p041-boxing"
        sequence.mkdir()
        path = sequence / "optim_params.txt"
        path.write_text(HEADER + body, encoding="utf-8")
        self.addCleanup(temp.cleanup)
        return path

    def test_valid_file_parses_19_fields_without_splitting_numbers(self):
        path = self.write_file()
        result = audit.parse_optim_file(path, Path(path).parents[1])
        self.assertEqual(result["integrity"]["data_row_count"], 1)
        self.assertEqual(result["rows"][0]["k2"], 0.01)
        self.assertEqual(result["rows"][0]["p1"], 0.001)

    def test_wrong_field_count_reports_path_line_and_raw_text(self):
        path = self.write_file(ROW.replace("0.01 0.001", "0.010.001"))
        with self.assertRaisesRegex(audit.IntegrityError, "field_count.*line=2"):
            audit.parse_optim_file(path, Path(path).parents[1])

    def test_non_numeric_value_is_fatal(self):
        path = self.write_file(ROW.replace("0.01", "not-a-number"))
        with self.assertRaisesRegex(audit.IntegrityError, "numeric_conversion"):
            audit.parse_optim_file(path, Path(path).parents[1])

    def test_nan_and_infinity_are_rejected(self):
        for value in ("nan", "inf"):
            with self.subTest(value=value):
                path = self.write_file(ROW.replace("900", value, 1))
                with self.assertRaisesRegex(audit.IntegrityError, "non_finite"):
                    audit.parse_optim_file(path, Path(path).parents[1])

    def test_nonpositive_focal_is_rejected(self):
        path = self.write_file(ROW.replace("900", "0", 1))
        with self.assertRaisesRegex(audit.IntegrityError, "invalid_focal"):
            audit.parse_optim_file(path, Path(path).parents[1])

    def test_duplicate_cam_id_and_name_are_rejected(self):
        path = self.write_file(ROW + ROW)
        with self.assertRaisesRegex(audit.IntegrityError, "duplicate_cam_id"):
            audit.parse_optim_file(path, Path(path).parents[1])

    def test_participant_normalization_preserves_raw_value(self):
        self.assertEqual(audit.participant_labels("p041-boxing"), ("p041", "p41"))
        self.assertEqual(audit.participant_labels("p007-tea"), ("p007", "p7"))

    def test_maximum_value_is_included_in_last_bin(self):
        self.assertEqual(audit.assign_bin(2.0, [0.0, 1.0, 2.0]), 1)

    def test_balance_keeps_camera_key_and_signature_units_separate(self):
        result = audit.balance_summary([6, 228, 11], [4, 134, 9])
        self.assertEqual(result["camera_key"]["balanced_subset_size"], 18)
        self.assertEqual(result["exact_full_signature"]["balanced_subset_size"], 12)

    def test_signature_shared_participants_form_one_component(self):
        links = {"sig-a": {"p1", "p2"}, "sig-b": {"p2", "p3"}}
        components = audit.participant_components(links, {"p1", "p2", "p3"})
        self.assertEqual(components, [{"p1", "p2", "p3"}])

    def test_demo_overlap_distinguishes_all_match_definitions(self):
        full = [{
            "participant": "p41", "sequence": "p041-boxing", "camera": "cam0",
            "intrinsic_signature": "i", "full_calibration_signature": "f",
        }]
        demo = [{
            "participant": "p41", "sequence": "p41-boxing-0021", "camera": "cam0",
            "intrinsic_signature": "i", "full_calibration_signature": "f",
        }]
        result = audit.overlap_summary(full, demo)
        self.assertEqual(result["exact_full_signature_overlap_count"], 1)
        self.assertEqual(result["exact_sequence_identity_overlap_count"], 0)
        self.assertEqual(result["normalized_participant_action_overlap_count"], 1)

    def test_csv_writer_projects_rows_to_declared_metadata_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            audit.write_csv(path, [{"kept": 1, "validation_only": 2}], ["kept"])
            self.assertEqual(path.read_text(encoding="utf-8"), "kept\n1\n")


if __name__ == "__main__":
    unittest.main()
