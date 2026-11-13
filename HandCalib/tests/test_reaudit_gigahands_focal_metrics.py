import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "reaudit_gigahands_focal_metrics.py"
)
spec = importlib.util.spec_from_file_location("focal_metric_reaudit", MODULE_PATH)
reaudit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reaudit)


def frame(camera_key, frame_index, pred_fx, pred_fy, gt_fx=100.0, gt_fy=200.0):
    sequence, camera = camera_key.split("/", 1)
    return {
        "participant": "p1",
        "sequence": sequence,
        "camera": camera,
        "camera_key": camera_key,
        "frame_index": frame_index,
        "width": 100,
        "height": 50,
        "gt_fx": gt_fx,
        "gt_fy": gt_fy,
        "gt_cx": 50.0,
        "gt_cy": 25.0,
        "pred_fx": pred_fx,
        "pred_fy": pred_fy,
        "pred_cx": 50.0,
        "pred_cy": 25.0,
        "success": True,
    }


def manifest_row(camera_key, fx=100.0, fy=200.0, participant="p1"):
    sequence, camera = camera_key.split("/", 1)
    return {
        "participant": participant,
        "sequence": sequence,
        "camera": camera,
        "camera_key": camera_key,
        "width": 100,
        "height": 50,
        "fx": fx,
        "fy": fy,
        "cx": 50.0,
        "cy": 25.0,
    }


class FocalMetricReauditTests(unittest.TestCase):
    def test_relative_focal_error_uses_gt_denominator_and_percent(self):
        result = reaudit.relative_focal_errors(120.0, 150.0, 100.0, 200.0)
        self.assertAlmostEqual(result["relative_fx_percent"], 20.0)
        self.assertAlmostEqual(result["relative_fy_percent"], 25.0)
        self.assertAlmostEqual(result["relative_focal_percent"], 25.0)

    def test_pair_aggregation_uses_componentwise_median(self):
        rows = [
            frame("s1/c1", 0, 90.0, 180.0),
            frame("s1/c1", 1, 110.0, 220.0),
            frame("s1/c1", 2, 1000.0, 2000.0),
        ]
        pair = reaudit.aggregate_frame_predictions(rows)[0]
        self.assertEqual(pair["pred_fx"], 110.0)
        self.assertEqual(pair["pred_fy"], 220.0)
        self.assertEqual(pair["frame_count"], 3)

    def test_pair_aggregation_rejects_inconsistent_gt(self):
        rows = [
            frame("s1/c1", 0, 90.0, 180.0, gt_fx=100.0),
            frame("s1/c1", 1, 110.0, 220.0, gt_fx=101.0),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent gt_fx"):
            reaudit.aggregate_frame_predictions(rows)

    def test_summary_weights_pairs_equally(self):
        stats = reaudit.distribution_stats([1.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["population_variance"], 1.0)
        self.assertEqual(stats["population_std"], 1.0)
        self.assertEqual(stats["count"], 2)

    def test_range_error_uses_axis_specific_denominators(self):
        result = reaudit.range_errors(110.0, 230.0, 100.0, 200.0, 50.0, 100.0)
        self.assertEqual(result["range_fx_percent"], 20.0)
        self.assertEqual(result["range_fy_percent"], 30.0)
        self.assertEqual(result["range_focal_percent"], 30.0)

    def test_fov_error_is_zero_for_exact_prediction(self):
        result = reaudit.fov_errors(100.0, 200.0, 100.0, 200.0, 100, 50)
        self.assertAlmostEqual(result["hfov_absolute_error_deg"], 0.0)
        self.assertAlmostEqual(result["vfov_absolute_error_deg"], 0.0)
        self.assertAlmostEqual(
            reaudit.fov_degrees(100.0, 100),
            math.degrees(2.0 * math.atan(0.5)),
        )

    def test_manifest_duplicates_collapse_when_gt_matches(self):
        rows = [manifest_row("s1/c1"), manifest_row("s1/c1")]
        self.assertEqual(len(reaudit.deduplicate_manifest(rows)), 1)

    def test_manifest_duplicates_with_different_gt_are_rejected(self):
        rows = [manifest_row("s1/c1"), manifest_row("s1/c1", fx=101.0)]
        with self.assertRaisesRegex(ValueError, "inconsistent manifest duplicate"):
            reaudit.deduplicate_manifest(rows)

    def test_train_constant_prior_uses_normalized_medians(self):
        train = [
            manifest_row("s1/c1", fx=100.0, fy=100.0),
            manifest_row("s2/c2", fx=200.0, fy=200.0),
            manifest_row("s3/c3", fx=1000.0, fy=1000.0),
        ]
        prior = reaudit.train_constant_prior(train)
        self.assertEqual(prior["median_nfx"], 2.0)
        self.assertEqual(prior["median_nfy"], 4.0)
        self.assertEqual(reaudit.constant_prediction(prior, 200, 100), (400.0, 400.0))

    def test_response_diagnostics_are_axis_specific(self):
        pairs = [
            {"gt_fx": 1.0, "pred_fx": 3.0, "gt_fy": 1.0, "pred_fy": 5.0},
            {"gt_fx": 2.0, "pred_fx": 5.0, "gt_fy": 2.0, "pred_fy": 5.0},
            {"gt_fx": 3.0, "pred_fx": 7.0, "gt_fy": 3.0, "pred_fy": 5.0},
        ]
        fx = reaudit.response_diagnostics(pairs, "fx")
        fy = reaudit.response_diagnostics(pairs, "fy")
        self.assertAlmostEqual(fx["slope"], 2.0)
        self.assertAlmostEqual(fx["intercept"], 1.0)
        self.assertAlmostEqual(fx["r_squared"], 1.0)
        self.assertAlmostEqual(fx["pred_std_over_gt_std"], 2.0)
        self.assertAlmostEqual(fy["slope"], 0.0)
        self.assertAlmostEqual(fy["pred_std_over_gt_std"], 0.0)
        self.assertIsNone(fy["pearson"])

    def test_spearman_uses_average_ranks_for_ties(self):
        value = reaudit.spearman_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])
        self.assertAlmostEqual(value, math.sqrt(3.0) / 2.0)

    def test_fixed_bins_keep_empty_bins_and_include_right_edge(self):
        groups = reaudit.fixed_bin_groups(
            [{"gt_fx": 0.0}, {"gt_fx": 2.0}],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual([len(group) for group in groups], [1, 1])
        empty = reaudit.fixed_bin_groups([{"gt_fx": 0.0}], [0.0, 1.0, 2.0])
        self.assertEqual([len(group) for group in empty], [1, 0])

    def test_prepare_output_refuses_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reaudit.prepare_output(path, overwrite=False)

    def test_legacy_reproduction_mismatch_stops_analysis(self):
        pairs = [
            {"relative_focal_percent": 10.0},
            {"relative_focal_percent": 20.0},
        ]
        metrics = {"pair_level": {"pair_max_rel_f_error_mean": 0.14}}
        with self.assertRaisesRegex(RuntimeError, "legacy focal mean mismatch"):
            reaudit.validate_legacy_reproduction(pairs, metrics, tolerance=1e-12)

    def test_model_pair_keys_must_match(self):
        models = {
            "pretrained": [{"camera_key": "s1/c1"}],
            "finetuned": [{"camera_key": "s2/c2"}],
        }
        with self.assertRaisesRegex(ValueError, "camera-key mismatch"):
            reaudit.validate_model_pair_alignment(models)

    def test_resolution_validation_rejects_multiple_sizes(self):
        rows = [{"width": 100, "height": 50}, {"width": 200, "height": 100}]
        with self.assertRaisesRegex(ValueError, "single-resolution"):
            reaudit.validate_single_resolution(rows)

    def test_source_snapshot_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("before", encoding="utf-8")
            snapshot = reaudit.source_snapshot([source])
            source.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "immutable source changed"):
                reaudit.verify_source_snapshot(snapshot)

    def test_response_fields_keep_fx_and_fy_separate(self):
        pairs = [
            {"gt_fx": 1.0, "pred_fx": 3.0, "gt_fy": 1.0, "pred_fy": 2.0},
            {"gt_fx": 2.0, "pred_fx": 5.0, "gt_fy": 2.0, "pred_fy": 4.0},
            {"gt_fx": 3.0, "pred_fx": 7.0, "gt_fy": 3.0, "pred_fy": 6.0},
        ]
        result = reaudit.response_summary_fields(pairs)
        for field in (
            "r_squared_fx", "r_squared_fy", "slope_fx", "slope_fy",
            "intercept_fx", "intercept_fy", "pearson_fx", "pearson_fy",
            "spearman_fx", "spearman_fy", "pred_std_over_gt_std_fx",
            "pred_std_over_gt_std_fy",
        ):
            self.assertIn(field, result)
        self.assertAlmostEqual(result["slope_fx"], 2.0)
        self.assertAlmostEqual(result["slope_fy"], 2.0)

    def test_bin_level_rows_keep_all_models_and_empty_bins(self):
        model_pairs = {
            name: [{
                "gt_fx": 0.0,
                "pred_fx": 0.0,
                "pred_fy": 0.0,
                "relative_focal_percent": 1.0,
                "test_range_focal_percent": 2.0,
                "global_range_focal_percent": 3.0,
            }]
            for name in ("pretrained", "finetuned", "train_constant")
        }
        rows = reaudit.bin_level_rows(model_pairs, [0.0, 1.0, 2.0])
        self.assertEqual(len(rows), 6)
        empty = [row for row in rows if row["bin_index"] == 2]
        self.assertEqual(len(empty), 3)
        self.assertTrue(all(row["pair_count"] == 0 for row in empty))
        self.assertTrue(all(row["relative_focal_mean_percent"] is None for row in empty))

    def test_identity_line_uses_joint_gt_prediction_extent(self):
        pairs = [
            {"gt_fx": 10.0, "pred_fx": 8.0},
            {"gt_fx": 20.0, "pred_fx": 30.0},
        ]
        self.assertEqual(reaudit.identity_line_extent(pairs, "fx"), (8.0, 30.0))

    def test_scatter_annotation_includes_ranges_and_response_statistics(self):
        diagnostic = {
            "slope": 0.5,
            "r_squared": 0.25,
            "gt_range": 54.0,
            "prediction_range": 67.0,
            "gt_std": 12.6,
            "prediction_std": 11.9,
        }
        annotation = reaudit.scatter_annotation(diagnostic)
        self.assertIn("slope=0.500", annotation)
        self.assertIn("R²=0.250", annotation)
        self.assertIn("GT range=54.00", annotation)
        self.assertIn("Pred range=67.00", annotation)
        self.assertIn("GT σ=12.60", annotation)
        self.assertIn("Pred σ=11.90", annotation)


if __name__ == "__main__":
    unittest.main()
