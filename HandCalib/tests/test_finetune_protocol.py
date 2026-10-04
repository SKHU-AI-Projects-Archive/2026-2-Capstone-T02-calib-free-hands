import importlib.util
from pathlib import Path
import unittest

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
spec = importlib.util.spec_from_file_location("handcalib_train", TRAIN_PATH)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class FineTuneProtocolTests(unittest.TestCase):
    def test_optimizer_groups_apply_official_backbone_scale(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2)
        model.head = torch.nn.Linear(2, 1)

        optimizer = train.build_finetune_optimizer(model, 6e-5)

        by_lr = {group["lr"]: group["params"] for group in optimizer.param_groups}
        self.assertEqual(sorted(by_lr), [6e-6, 6e-5])
        self.assertEqual(len(by_lr[6e-6]), 2)
        self.assertEqual(len(by_lr[6e-5]), 2)
        self.assertTrue(all(group["weight_decay"] == 0.01 for group in optimizer.param_groups))


    def test_scheduler_uses_official_step_boundaries(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=6e-5, weight_decay=0.01)
        scheduler = train.build_finetune_scheduler(optimizer)

        values = []
        for _ in range(1001):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(values[0], 6e-8)
        self.assertAlmostEqual(values[999], 6e-5 * 0.999001, places=10)
        self.assertAlmostEqual(values[1000], 6e-5, places=10)


    def test_cli_requires_explicit_mode(self):
        old_argv = train.sys.argv
        try:
            train.sys.argv = ["train.py", "--config", "config.yaml"]
            with self.assertRaises(SystemExit):
                train.parse_args()
        finally:
            train.sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
