import importlib.util
from pathlib import Path
import tempfile
import unittest


DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "design_gigahands_focal_balance_protocol.py"
)
spec = importlib.util.spec_from_file_location("gigahands_protocol_design", DESIGN_PATH)
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


def record(participant, sequence, camera, signature, fx=100.0, fy=100.0, width=100, height=50):
    return {
        "participant": participant,
        "participant_raw": participant,
        "sequence": sequence,
        "camera": camera,
        "camera_key": f"{sequence}/{camera}",
        "full_calibration_signature": signature,
        "intrinsic_signature": "i-" + signature,
        "fx": fx,
        "fy": fy,
        "width": width,
        "height": height,
        "nfx": fx / width,
        "nfy": fy / height,
        "hfov_deg": 45.0,
        "vfov_deg": 30.0,
        "source_path": "metadata",
    }


class ProtocolDesignTests(unittest.TestCase):
    def test_shared_calibration_across_participants_is_detected(self):
        rows = [record("p1", "s1", "c1", "a"), record("p2", "s2", "c1", "a")]
        graph = design.duplicate_structure(rows)
        self.assertEqual(graph["cross_participant_signature_count"], 1)

    def test_repeated_calibration_across_sequences_is_not_unique(self):
        rows = [record("p1", "s1", "c1", "a"), record("p1", "s2", "c1", "a")]
        self.assertEqual(design.duplicate_structure(rows)["unique_full_signature_count"], 1)

    def test_same_fx_with_different_fy_remains_distinct(self):
        rows = [record("p1", "s1", "c1", "a", fy=90), record("p1", "s1", "c2", "b", fy=95)]
        self.assertEqual(len(design.unique_rows(rows, "full_calibration_signature")), 2)

    def test_same_normalized_focal_can_have_different_raw_fx(self):
        left = record("p1", "s1", "c1", "a", fx=100, width=100)
        right = record("p2", "s2", "c2", "b", fx=200, width=200)
        self.assertNotEqual(left["fx"], right["fx"])
        self.assertEqual(left["nfx"], right["nfx"])

    def test_participant_disjoint_candidate_can_still_leak_calibration(self):
        split = {
            "train": [record("p1", "s1", "c1", "a")],
            "validation": [record("p2", "s2", "c2", "a")],
            "test": [record("p3", "s3", "c3", "b")],
        }
        overlap = design.split_overlaps(split)
        self.assertEqual(overlap["participant_overlap"], 0)
        self.assertEqual(overlap["exact_full_overlap"], 1)

    def test_calibration_disjoint_candidate_can_reuse_participant(self):
        split = {
            "train": [record("p1", "s1", "c1", "a")],
            "validation": [record("p1", "s2", "c2", "b")],
            "test": [record("p2", "s3", "c3", "c")],
        }
        overlap = design.split_overlaps(split)
        self.assertEqual(overlap["exact_full_overlap"], 0)
        self.assertEqual(overlap["participant_overlap"], 1)

    def test_connected_graph_can_be_split_after_explicit_record_exclusion(self):
        rows = [
            record("p1", "s1", "c1", "a"), record("p2", "s2", "c1", "a"),
            record("p2", "s2", "c2", "b"), record("p3", "s3", "c2", "b"),
            record("p3", "s3", "c3", "c"),
        ]
        assignment = {"p1": "train", "p2": "validation", "p3": "test"}
        owner = {"a": "train", "b": "validation", "c": "test"}
        split, excluded = design.select_matching_records(rows, assignment, owner, "full_calibration_signature")
        self.assertEqual(design.split_overlaps(split)["exact_full_overlap"], 0)
        self.assertEqual(len(excluded), 2)

    def test_minimum_bin_support_failure_is_reported(self):
        result = design.bin_support([1, 0, 2], required=1)
        self.assertFalse(result["meets_requirement"])

    def test_empty_bin_is_not_silently_removed(self):
        result = design.bin_support([3, 0, 4], required=1)
        self.assertEqual(result["empty_bin_count"], 1)

    def test_rightmost_bin_edge_is_included(self):
        self.assertEqual(design.assign_bin(2.0, [0.0, 1.0, 2.0]), 1)

    def test_repeated_signature_increases_camera_key_exposure(self):
        rows = [record("p1", "s1", "c1", "a"), record("p1", "s2", "c1", "a")]
        weights = design.signature_base_weights(rows, "S0")
        self.assertEqual(weights["a"], 2.0)

    def test_full_and_partial_bin_balance_differ(self):
        counts = [1, 9]
        full = design.bin_weight_factors(counts, alpha=1.0)
        partial = design.bin_weight_factors(counts, alpha=0.5)
        self.assertGreater(full[0] / full[1], partial[0] / partial[1])

    def test_weight_clipping_limits_concentration(self):
        clipped = design.clip_and_normalize({"a": 100.0, "b": 1.0, "c": 1.0}, max_multiple=2.0)
        self.assertLessEqual(max(clipped.values()), 2.0 / 3.0 + 1e-12)

    def test_weight_clipping_returns_deterministic_key_order(self):
        clipped = design.clip_and_normalize({"z": 100.0, "b": 1.0, "a": 1.0}, max_multiple=2.0)
        self.assertEqual(list(clipped), ["a", "b", "z"])

    def test_effective_sample_size(self):
        self.assertAlmostEqual(design.effective_sample_size([0.5, 0.5]), 2.0)
        self.assertAlmostEqual(design.effective_sample_size([1.0, 0.0]), 1.0)

    def test_intrinsic_and_full_overlap_are_reported_separately(self):
        train = record("p1", "s1", "c1", "a")
        test = record("p2", "s2", "c2", "b")
        test["intrinsic_signature"] = train["intrinsic_signature"]
        overlap = design.split_overlaps({"train": [train], "validation": [], "test": [test]})
        self.assertEqual(overlap["exact_intrinsic_overlap"], 1)
        self.assertEqual(overlap["exact_full_overlap"], 0)

    def test_camera_identifier_overlap_is_reported(self):
        split = {
            "train": [record("p1", "s1", "shared", "a")],
            "validation": [],
            "test": [record("p2", "s2", "shared", "b")],
        }
        self.assertEqual(design.split_overlaps(split)["camera_identifier_overlap"], 1)

    def test_excluded_record_is_in_no_split(self):
        rows = [record("p1", "s1", "c1", "a"), record("p2", "s2", "c2", "a")]
        split, excluded = design.select_matching_records(rows, {"p1": "train", "p2": "test"}, {"a": "train"}, "full_calibration_signature")
        used = {row["camera_key"] for group in split.values() for row in group}
        self.assertTrue(all(row["camera_key"] not in used for row in excluded))

    def test_record_cannot_appear_in_two_splits(self):
        row = record("p1", "s1", "c1", "a")
        with self.assertRaisesRegex(ValueError, "record overlap"):
            design.validate_record_partition({"train": [row], "validation": [], "test": [row]}, [])

    def test_same_seed_produces_same_assignment(self):
        items = [f"x{i}" for i in range(20)]
        first = design.deterministic_partition(items, (0.6, 0.2, 0.2), seed=42)
        second = design.deterministic_partition(items, (0.6, 0.2, 0.2), seed=42)
        self.assertEqual(first, second)

    def test_nonempty_output_is_not_overwritten_without_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                design.prepare_output(path, overwrite=False)

    def test_heuristic_failure_is_not_called_proven_infeasible(self):
        self.assertEqual(design.search_status(found=False), "NO_CANDIDATE_FOUND_WITHIN_BUDGET")

    def test_missing_rgb_does_not_invent_frame_count(self):
        row = design.rgb_requirement_row("candidate", "train", record("p1", "s1", "c1", "a"), 0)
        self.assertEqual(row["frame_count"], "unknown")
        self.assertEqual(row["video_size_bytes"], "unknown")

    def test_search_coverage_counts_only_signatures_with_matching_participant(self):
        rows = [record("p1", "s1", "c1", "a"), record("p2", "s2", "c2", "b")]
        usable = design.usable_signatures(
            rows,
            {"p1": "train", "p2": "validation"},
            {"c1": "train", "c2": "test"},
            "camera",
        )
        self.assertEqual(usable, {"a"})

    def test_active_participant_count_excludes_zero_retention_assignments(self):
        scores = {
            "p1": {"train": 2, "validation": 0, "test": 0},
            "p2": {"train": 0, "validation": 0, "test": 0},
            "p3": {"train": 0, "validation": 0, "test": 3},
        }
        assignment = {"p1": "train", "p2": "validation", "p3": "test"}
        self.assertEqual(
            design.active_participant_counts(scores, assignment),
            {"train": 1, "validation": 0, "test": 1},
        )

    def test_assigned_and_active_participants_are_reported_separately(self):
        split_rows = {
            "train": [record("p1", "s1", "c1", "a")],
            "validation": [],
            "test": [record("p3", "s3", "c3", "c")],
        }
        assignment = {"p1": "train", "p2": "validation", "p3": "test"}
        counts = design.participant_assignment_counts(split_rows, assignment)
        self.assertEqual(counts["assigned"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(counts["active"], {"train": 1, "validation": 0, "test": 1})

    def test_out_of_train_support_counts_unique_signatures(self):
        split_rows = {
            "train": [record("p1", "s1", "c1", "a", fx=100), record("p1", "s2", "c2", "b", fx=200)],
            "validation": [record("p2", "s3", "c3", "c", fx=90)],
            "test": [record("p3", "s4", "c4", "d", fx=150), record("p3", "s5", "c5", "e", fx=250)],
        }
        support = design.out_of_train_support_counts(split_rows, "fx")
        self.assertEqual(support["validation"], 1)
        self.assertEqual(support["test"], 1)

    def test_quarter_alpha_is_between_uniform_and_half_balance(self):
        counts = [1, 16]
        uniform = design.bin_weight_factors(counts, alpha=0.0)
        quarter = design.bin_weight_factors(counts, alpha=0.25)
        half = design.bin_weight_factors(counts, alpha=0.5)
        self.assertLess(uniform[0] / uniform[1], quarter[0] / quarter[1])
        self.assertLess(quarter[0] / quarter[1], half[0] / half[1])


if __name__ == "__main__":
    unittest.main()
