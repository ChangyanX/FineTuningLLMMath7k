# Audit of downloaded mandatory runs

Verified from downloaded run artifacts and scheduler records, 4 November 2026.

## Verdict

All four selected training directories and all eight full benchmark evaluations are present and marked complete. Prediction counts, answer extraction, correctness, settings, adapter hashes and checkpoint manifests pass the integrity checks. **The experiments are structurally complete, but M3 and M4 are not healthy fine-tuning results.** Preserve them as observed results; investigate and rerun those models under a separate, validated profile before treating the table as a successful comparison of PEFT methods.

| Selected run | Train job | AQuA job | AddSub job | Training integrity | Training health |
|---|---:|---:|---:|---|---|
| M1_e1_retry2 | 46924 | 46927 | 47120 | Pass | Learning evident; one logged nonfinite gradient norm |
| M2_e1_main | 47309 | 47391 | 47749 | Pass | Learning evident; no logged nonfinite gradient norms |
| M3_e1_main | 47780 | 47786 | 47853 | Pass | Numerical/optimization concerns; mostly zero gradients |
| M4_e1_retry1 | 49897 | 49903 | 49916 | Pass | Numerical/optimization concerns; worsening validation loss |

Run selection was fixed independently of test accuracy. Smoke tests, diagnostic subsets and unrelated runs are excluded.

## Checks performed

The repeatable audit is [scripts/audit_mandatory_runs.py](../scripts/audit_mandatory_runs.py). Its detailed output is [MANDATORY_RUN_AUDIT.json](MANDATORY_RUN_AUDIT.json). From the project root:

```bash
python3 scripts/audit_mandatory_runs.py --tensors
```

- All four training runs reached epoch 1.0 and Trainer global step 1,683. This is not proof that every optimizer step was applied; FP16 overflow can skip updates.
- The same split is used throughout: 6,731 training and 120 validation IDs, disjoint and covering all 6,851 records. Split SHA-256: `90338055ef76f8308bb4b3e72fffc88c7835a50df41a4442d7e99dc6e64ac7ef`.
- All eight evaluations are full runs, not subsets: 254 AQuA and 395 AddSub examples per model/method; 2,596 predictions in total. Status, progress and metric counts agree. IDs are complete, ordered and unique; labels and instructions match the local benchmark records.
- Re-extracting answers from every raw generation reproduces all saved predictions, correctness flags, parse-failure flags, accuracy figures and generation-time totals.
- Every file recorded in each evaluation's `adapter_files` matches its downloaded final adapter. Both evaluations use the intended run's adapter, base model and pinned revision.
- All 12 retained training checkpoint manifests pass: **132 file hashes**, including weights, optimizer, scheduler, RNG and scaler state. The selected best checkpoint is available in each run.
- All final adapter tensors are finite and match the selected best checkpoint exactly: M1 104 tensors, M2 52, M3 88, M4 44. M3 has 22 entirely zero tensors; these correspond to its query-projection LoRA B weights in the earlier school diagnostic. Other runs have no entirely zero adapter tensors.
- Saved training/evaluation source identities agree within each selected run and across all four runs. The current local `finetune.py` differs; `git show 5a8e35c:finetune.py` matches the recorded training hash. The original training source is recoverable from that commit. This source difference is documented in the [epoch comparison](BASELINE_EPOCH_RESULTS.md).

## Training-health evidence

`nan` diagnostics are saved as JSON strings by the logger. The audit converts them back to floating-point values for detection; checking only JSON numeric values would miss them.

| Run | Nonfinite gradient logs / 168 | Zero gradient logs / 168 | Validation loss at step 100 | Last validation loss, step 1600 | Selected checkpoint | Best validation loss |
|---|---:|---:|---:|---:|---|---:|
| M1_e1_retry2 | 1 | 0 | 0.835207 | 0.718773 | checkpoint-1600 | 0.718773 |
| M2_e1_main | 0 | 0 | 0.858980 | 0.694462 | checkpoint-1600 | 0.694462 |
| M3_e1_main | 2 | 161 | 1.781734 | 1.782624 | checkpoint-100 | 1.781734 |
| M4_e1_retry1 | 2 | 128 | 1.883506 | 2.076821 | checkpoint-100 | 1.883506 |

These are logging observations every ten Trainer steps, not counts of every individual batch. M1 and M2 show clear validation improvement. M3 changes very little; M4's validation loss worsens. Their best checkpoints occur early, at approximately 0.059 epochs, even though training traversed one epoch.

FP16 scaler evidence supports numerical concerns: at checkpoint 100, M3's scale is `6.103515625e-05` and its scheduler has advanced only 70 steps; M4's scale is `0.0001220703125` and its scheduler has advanced 71 steps. At checkpoint 1600 their scheduler counts are 1570 and 1571. M1's scheduler is one step behind its Trainer count; M2's counts agree. These are consistent with skipped FP16 updates, but do not by themselves identify the full cause of poor generations. The adapters do contain some changed tensors, so it would be inaccurate to claim no learning occurred at all.

Generation quality is also problematic. M3 produces 15 empty AQuA and 37 empty AddSub outputs. M4 produces 11 empty AQuA and 1 empty AddSub outputs. Repeated questions/prompt markers occur in representative TinyLlama generations. AQuA parsing fails on 253/254 M3 and 254/254 M4 outputs. M1 and M2 also have high AQuA parse-failure rates (148/254 and 219/254), so even their AQuA accuracy reflects both reasoning and answer-format compliance.

The earlier [tiny-audit log](../logs/tiny-audit_48400.out) reports 4,387/6,851 TinyLlama training sequences truncated at cutoff 256 (64.03%); at 512 this falls to 193 (2.82%). This covers the whole source dataset, not just the held-out training split. It is additional evidence of a setup limitation, not proof that truncation caused the optimization failures. Its tokenizer-boundary diagnostic should not be treated as an exact count of lost answer tokens.

## Verified completion and report scope

[logs/mandatory_sacct.txt](../logs/mandatory_sacct.txt) confirms `COMPLETED` with exit code `0:0` for all 12 original one-epoch allocations and their batch steps. The [baseline/epoch scheduler records](../logs/baseline_epoch_sacct.txt) confirm the same for all 12 additional allocations. Scheduler completion is verified for all selected experiments; it does not establish healthy optimization.

The [main report](MANDATORY_EXPERIMENT_RESULTS.md) covers 12 training runs and 28 full evaluations (9,086 predictions). Across the mandatory and [baseline/epoch audits](BASELINE_EPOCH_RESULTS.md), 396 checkpoint-file hashes passed. All expected evaluations, final adapters and selected checkpoints are present.

The report must retain the TinyLlama numerical/optimization findings, high AQuA parse-failure rates, single-seed limitation and hardware differences. Mandatory M1-M3 used A40, while M4 retry1 used L40S; these runs do not support hardware-controlled speed comparisons. Any future repaired profile should be evaluated as a separate controlled experiment rather than substituted into these observed results.
