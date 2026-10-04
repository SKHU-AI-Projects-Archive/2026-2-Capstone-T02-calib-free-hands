# 02-D.1 Fine-Tuning Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Select the base learning rate with bounded one-epoch Validation pilots, then freeze a reproducible five-epoch AnyCalib fine-tuning protocol without running production training.

**Architecture:** Extend the existing single-GPU `HandCalib/train.py` entry point with shared official optimizer/scheduler construction, a one-epoch `--lr-pilot` mode, a guarded `--train` production path, and a CPU-safe `--train-plan`. Reuse the existing GigaHands loader and result summarizer for p36 Validation; keep all raw pilot outputs under ignored `HandCalib/runs/` and track only the protocol record.

**Tech Stack:** Python, PyTorch, YAML, existing GigaHands `PairBalancedSampler`, official pinned AnyCalib model/loss.

**Spec:** `/home/junghyub/.codex/attachments/64658978-b203-489f-b49b-499f4582ab9e/Pasted text.txt`

## Global Constraints

- Never use Test/p52 or `test.txt` for selection.
- Candidate LR pilots are exactly `6e-5` and `2e-5`, with `1e-5` only if both fail to improve step-0.
- Every candidate starts from fresh `anycalib_pinhole` pretrained weights, seed 42, sampler epoch 0, batch 4, workers 4, BF16 training, FP32 Validation.
- Production training requires explicit `--train`; this turn must not execute it.
- Freeze AdamW, weight decay 0.01, backbone LR scale 0.1, clip norm 1.0, five maximum epochs, warmup 1000, milestones 10000/30000, gamma 0.3, validation every epoch, no early stopping.
- Do not modify locked manifest, split, raw dataset, or ignored run outputs in Git.
- No push, amend, rebase, reset, or history rewrite.

## Review Focus

- Scheduler off-by-one at warmup and milestone boundaries: synthetic LR table test.
- Accidental training without `--train`: CLI guard test.
- Incorrect backbone grouping: optimizer group LR test.
- Pilot contamination between candidates: fresh model construction and fixed sampler epoch assertions.
- Resume corruption: synthetic checkpoint round-trip test.

### Task 1: Protocol Primitives

**Files:**
- Modify: `HandCalib/train.py`
- Test: `HandCalib/tests/test_finetune_protocol.py`

- [ ] Write failing tests for optimizer grouping, scheduler boundary values, and no-mode CLI guard.
- [ ] Implement shared protocol helpers and make the tests pass.
- [ ] Run the focused tests and compile check.

### Task 2: Pilot and Production Paths

**Files:**
- Modify: `HandCalib/train.py`
- Modify: `HandCalib/configs/02_anycalib_finetune.yaml`
- Test: `HandCalib/tests/test_finetune_protocol.py`

- [ ] Implement one-epoch LR pilot output and p36-only FP32 Validation.
- [ ] Implement explicit `--train`, resume, atomic best/last checkpoints, and `--train-plan`.
- [ ] Run the two permitted pilots on one idle GPU, adding `1e-5` only when required.

### Task 3: Freeze and Record

**Files:**
- Create: `HandCalib/results/02d1_finetune_protocol.yaml`
- Modify: `HandCalib/README.md`

- [ ] Record pilot metrics, selection rule, frozen protocol, and runtime estimates.
- [ ] Verify locked hashes, ignored pilot outputs, compileall, diff check, and existing dry-run paths.
- [ ] Commit source and protocol record separately where practical; do not push.
