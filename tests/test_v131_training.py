"""Focused CPU tests for the v1.3.1 split, objective and batching contracts."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_v131_test_module", ROOT / "05_train_cnn2d_v131.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V131TrainingTests(unittest.TestCase):
    def test_fold_ranges_are_contiguous_and_cover_calendar(self):
        calendar = pd.date_range("2009-01-01", periods=105, freq="B")
        folds = MODULE.make_fold_ranges(calendar, 5)
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0][0], calendar[0])
        self.assertEqual(folds[-1][1], calendar[-1])
        for previous, current in zip(folds, folds[1:]):
            self.assertLess(previous[1], current[0])

    def test_split_has_twenty_trading_day_purge(self):
        calendar = pd.date_range("2009-01-01", periods=120, freq="B")
        meta = pd.DataFrame(
            {
                "date": np.repeat(calendar, 2),
                "code": np.tile(["000001", "000002"], len(calendar)),
            }
        )
        fold = (calendar[40], calendar[59])
        train, valid, test, info = MODULE.split_indices(meta, fold, calendar, 20)
        self.assertEqual(len(valid), 40)
        train_dates = set(meta.iloc[train]["date"])
        valid_dates = set(meta.iloc[valid]["date"])
        self.assertFalse(train_dates.intersection(valid_dates))
        for date in calendar[20:40]:
            self.assertNotIn(date, train_dates)
        for date in calendar[60:80]:
            self.assertNotIn(date, train_dates)
        self.assertEqual(info["purge_before_start"], calendar[20].strftime("%Y-%m-%d"))
        self.assertEqual(info["purge_after_end"], calendar[79].strftime("%Y-%m-%d"))

    def test_date_sampler_emits_split_local_positions(self):
        meta = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-01", "2020-01-02"]),
                "code": ["000002", "000002", "000001", "000001"],
            }
        )
        global_indices = np.array([3, 1, 2], dtype=np.int64)
        sampler = MODULE.DateBatchSampler(meta, global_indices)
        self.assertEqual(list(sampler), [[2, 1], [0]])

    def test_winsorized_target_uses_population_std(self):
        meta = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
                "future_ret": [1.0, 2.0, 3.0, 100.0, -2.0, 0.0, 2.0, 4.0],
            }
        )
        target = MODULE.winsorized_cross_section_target(meta)
        self.assertTrue(np.isfinite(target).all())
        for start in (0, 4):
            section = target[start : start + 4]
            self.assertAlmostEqual(float(section.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(np.sqrt(np.mean(section**2))), 1.0, places=6)

    def test_three_objectives_have_finite_values_and_gradients(self):
        scores = torch.tensor([-1.0, 0.0, 0.5, 2.0], requires_grad=True)
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        target = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        for loss in MODULE.LOSS_CHOICES:
            options = MODULE.TrainOptions(loss=loss, fold_id=1, seed=42)
            value, _ = MODULE.objective_from_scores(scores, labels, target, options)
            self.assertTrue(torch.isfinite(value))
            value.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(scores.grad).all())
            scores.grad.zero_()

    def test_bce_microbatch_weighting_matches_full_date_gradient(self):
        full_scores = torch.tensor([-1.0, -0.2, 0.4, 1.5], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        full = torch.nn.functional.binary_cross_entropy_with_logits(full_scores, labels)
        full.backward()
        expected = full_scores.grad.detach().clone()
        parts = torch.tensor([-1.0, -0.2, 0.4, 1.5], requires_grad=True)
        for start, end in ((0, 2), (2, 4)):
            value = torch.nn.functional.binary_cross_entropy_with_logits(parts[start:end], labels[start:end])
            value.mul_((end - start) / 4.0).backward()
        self.assertTrue(torch.allclose(parts.grad, expected, atol=1e-7, rtol=1e-6))

    def test_huber_ic_replay_matches_full_date_gradient(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)

            def forward(self, x):
                return self.linear(x[:, :, 0, 0]).squeeze(-1)

        x = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, -1.0, 0.5], [-1.0, 0.2, 1.0], [0.5, 0.5, -2.0]]
        ).reshape(4, 3, 1, 1)
        x_original = x.clone()
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        target = torch.tensor([-0.7, 0.2, 0.8, -0.1])
        options = MODULE.TrainOptions(
            loss="huber_ic",
            fold_id=1,
            seed=42,
            micro_batch_size=2,
            amp=False,
            pin_memory=False,
            batch_workers=0,
        )

        replay_model = TinyModel()
        with torch.no_grad():
            replay_model.linear.weight.copy_(torch.tensor([[0.2, -0.4, 0.3]]))
            replay_model.linear.bias.fill_(0.1)
        reference_model = TinyModel()
        reference_model.load_state_dict(replay_model.state_dict())
        reference_score = reference_model(x / 255.0)
        reference_value, _ = MODULE.objective_from_scores(
            reference_score, labels, target, options
        )
        reference_value.backward()
        reference_grad = torch.cat([p.grad.flatten() for p in reference_model.parameters()])

        pool = MODULE.DeviceMicrobatchPool(torch.device("cpu"), 2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        replay_metrics = MODULE.hybrid_date_backward(
            replay_model,
            x,
            labels,
            target,
            pool,
            torch.device("cpu"),
            options,
            scaler,
        )
        replay_grad = torch.cat([p.grad.flatten() for p in replay_model.parameters()])
        self.assertAlmostEqual(replay_metrics["objective"], float(reference_value), places=6)
        self.assertTrue(torch.allclose(replay_grad, reference_grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(x, x_original))

    def test_early_stopping_min_delta_is_strict(self):
        self.assertTrue(MODULE.objective_improved(0.9, np.inf, 1e-3))
        self.assertTrue(MODULE.objective_improved(0.8989, 0.9, 1e-3))
        self.assertFalse(MODULE.objective_improved(0.899, 0.9, 1e-3))
        self.assertFalse(MODULE.objective_improved(0.8995, 0.9, 1e-3))


if __name__ == "__main__":
    unittest.main()
