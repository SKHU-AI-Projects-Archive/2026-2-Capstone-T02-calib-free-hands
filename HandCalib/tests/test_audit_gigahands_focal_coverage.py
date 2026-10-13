import importlib.util
from pathlib import Path
import unittest


AUDIT_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_gigahands_focal_coverage.py"
spec = importlib.util.spec_from_file_location("gigahands_focal_audit", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(key, video, participant, fx, fy, width=100, height=50, sequence="s1", camera="cam0"):
    return {
        "participant": participant,
        "sequence": sequence,
        "camera": camera,
        "camera_key": key,
        "video_path": video,
        "video_name": Path(video).name,
        "frame_count": 10,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width / 2,
        "cy": height / 2,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


class FocalCoverageAuditTests(unittest.TestCase):
    def test_duplicate_video_path_is_detected(self):
        rows = [row("s1/c0", "same.mp4", "p1", 100, 100), row("s1/c1", "same.mp4", "p1", 110, 110)]
        inventory = audit.dataset_inventory(rows)
        self.assertEqual(inventory["duplicate_video_path_count"], 1)

    def test_repeated_calibration_across_camera_keys_is_counted_once_by_signature(self):
        rows = [row("s1/c0", "a.mp4", "p1", 100, 90), row("s2/c0", "b.mp4", "p2", 100, 90, sequence="s2")]
        inventory = audit.dataset_inventory(rows)
        self.assertEqual(inventory["unique_camera_key_count"], 2)
        self.assertEqual(inventory["unique_exact_full_signature_count"], 1)

    def test_same_fx_with_different_fy_remains_distinct(self):
        rows = [row("s1/c0", "a.mp4", "p1", 100, 90), row("s1/c1", "b.mp4", "p1", 100, 95, camera="cam1")]
        inventory = audit.dataset_inventory(rows)
        self.assertEqual(inventory["unique_fx_count"], 1)
        self.assertEqual(inventory["unique_fx_fy_count"], 2)

    def test_normalized_focal_accounts_for_resolution(self):
        first = audit.derive_focal(row("s1/c0", "a.mp4", "p1", 100, 100, 100, 50))
        second = audit.derive_focal(row("s1/c1", "b.mp4", "p1", 200, 200, 200, 100, camera="cam1"))
        self.assertEqual((first["nfx"], first["nfy"]), (1.0, 2.0))
        self.assertEqual((second["nfx"], second["nfy"]), (1.0, 2.0))

    def test_internal_bin_boundary_goes_to_upper_bin(self):
        self.assertEqual(audit.assign_bin(1.0, [0.0, 1.0, 2.0]), 1)

    def test_maximum_is_included_in_last_bin(self):
        self.assertEqual(audit.assign_bin(2.0, [0.0, 1.0, 2.0]), 1)

    def test_empty_bin_reduces_balanced_subset_to_zero(self):
        feasibility = audit.balance_feasibility([3, 0, 2])
        self.assertEqual(feasibility["balanced_subset_size"], 0)
        self.assertEqual(feasibility["empty_bin_count"], 1)

    def test_single_calibration_group_is_supported(self):
        feasibility = audit.balance_feasibility([1])
        self.assertEqual(feasibility["retained_groups"], 1)
        self.assertEqual(feasibility["retention_rate"], 1.0)

    def test_signature_overlap_across_splits_is_detected(self):
        train = [row("s1/c0", "a.mp4", "p1", 100, 90)]
        test = [row("s2/c0", "b.mp4", "p2", 100, 90, sequence="s2")]
        overlap = audit.compare_splits("train", train, "test", test)
        self.assertEqual(overlap["exact_intrinsic_signature_overlap"], 1)
        self.assertEqual(overlap["exact_full_signature_overlap"], 1)

    def test_participant_split_conflict_is_reported(self):
        groups = {"train": [row("s1/c0", "a.mp4", "p1", 100, 90)], "test": [row("s2/c0", "b.mp4", "p1", 110, 100, sequence="s2")]}
        self.assertEqual(audit.participant_conflicts(groups), ["p1"])

    def test_equal_bin_counts_keep_all_groups(self):
        feasibility = audit.balance_feasibility([4, 4, 4])
        self.assertEqual(feasibility["retained_groups"], 12)
        self.assertEqual(feasibility["discarded_groups"], 0)

    def test_extreme_balancing_reports_large_discard(self):
        feasibility = audit.balance_feasibility([2, 195, 2])
        self.assertEqual(feasibility["retained_groups"], 6)
        self.assertEqual(feasibility["discarded_groups"], 193)
        self.assertAlmostEqual(feasibility["discard_rate"], 193 / 199)

    def test_three_way_capacity_requires_each_split_share(self):
        self.assertTrue(audit.three_way_capacity([6, 9, 15], 2))
        self.assertFalse(audit.three_way_capacity([5, 9, 15], 2))

    def test_macro_evaluation_rejects_empty_target_bin(self):
        result = audit.macro_evaluation_feasibility([0, 3, 43, 3, 0])
        self.assertEqual(result["status"], "TECHNICALLY_INFEASIBLE")
        self.assertEqual(result["empty_bin_count"], 2)

    def test_frame_retention_range_uses_whole_calibration_groups(self):
        result = audit.frame_retention_range([[10, 20, 30], [5, 50, 60]], 1)
        self.assertEqual(result["minimum_retained_frames"], 15)
        self.assertEqual(result["maximum_retained_frames"], 90)

    def test_compact_result_summarizes_per_bin_rows(self):
        summary = {
            "experiment": "x",
            "status": "complete",
            "bin_coverage": [
                {"variable": "fx", "method": "equal_width", "bin_count": 2, "bin_index": 1, "lower_edge": 0.0, "upper_edge": 1.0, "camera_key_count": 2, "exact_intrinsic_signature_count": 1, "exact_full_signature_count": 1},
                {"variable": "fx", "method": "equal_width", "bin_count": 2, "bin_index": 2, "lower_edge": 1.0, "upper_edge": 2.0, "camera_key_count": 3, "exact_intrinsic_signature_count": 2, "exact_full_signature_count": 2},
            ],
            "balancing_feasibility": {},
        }
        compact = audit.compact_result(summary)
        coverage = compact["bin_coverage"]["fx_equal_width_2"]
        self.assertEqual(coverage["edges"], [0.0, 1.0, 2.0])
        self.assertEqual(coverage["camera_key_counts"], [2, 3])


if __name__ == "__main__":
    unittest.main()
