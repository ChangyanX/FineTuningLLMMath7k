# Baseline and Task 4 Option 1 results and audit

Verified from downloaded artifacts and scheduler records, 4 November 2026. All figures use the saved strict scorer.

## Completion verdict

**Both untouched-base models and all eight two-/three-epoch runs are structurally complete.** Both baseline directories contain full AQuA and AddSub evaluations: four baseline evaluations in total. The epoch runs add eight trainings and 16 evaluations. This update verifies 20 full evaluations and **6,490 predictions**, plus **264 file hashes across 24 retained training checkpoints**. No expected baseline/Option 1 evaluation, final adapter or selected checkpoint is missing.

All new final adapters contain finite tensors and exactly match their recorded best checkpoint. However, **TinyLlama training is still unhealthy**: more epochs retain very early checkpoints, mostly zero logged gradients and poor generations. Treat the results as observed outcomes of this configuration, not as a clean comparison of successfully optimized adapters.

The [main report](MANDATORY_EXPERIMENT_RESULTS.md) combines these results with the mandatory one-epoch runs. Across that complete selected scope there are 12 trainings, 28 evaluations, 9,086 predictions and 396 checked checkpoint-file hashes. Unrelated training-data experiments and smoke/diagnostic runs are excluded.

## Full untouched-base results

| Run / model | AQuA job | AQuA correct / accuracy | AddSub job | AddSub correct / accuracy | AQuA / AddSub parse failures |
|---|---:|---:|---:|---:|---:|
| M1_baseline_main / OpenLLaMA 3B v2 | 50173 | 0/254 (0.00%) | 50174 | 16/395 (4.05%) | 254 / 3 |
| M3_baseline_main / TinyLlama v1.1 | 50375 | 1/254 (0.39%) | 50377 | 10/395 (2.53%) | 253 / 31 |

The baseline evaluations have `lora_weights=null` and `adapter_files=null`: no adapter was loaded. Their `adapter=LoRA` display field does not mean they used a LoRA adapter. One baseline per base model is reused for both methods. The model SHA, benchmark SHA, seed, four-beam decoding, 256-token limit, cache setting and strict scoring match the fine-tuned evaluations.

At one epoch, OpenLLaMA LoRA gains +9.06/+34.68 percentage points over base (AQuA/AddSub), and AdapterP gains +2.36/+34.94. TinyLlama LoRA gains 0/+0.25 and AdapterP changes -0.39/0. AQuA gains include changes in answer-format compliance: neither the OpenLLaMA base's 0% nor TinyLlama's near-zero score isolates mathematical reasoning ability from generation/parsing failures.

## Epoch-run results and job provenance

Each combined job ID appears in the training and both evaluation event files. Every training has one recorded fresh attempt (`resume=false`). Total Trainer steps are 3,366 at two epochs and 5,049 at three epochs; this does not imply every optimizer update was applied.

| Run | Combined job | Training GPU | AQuA correct / accuracy | AddSub correct / accuracy |
|---|---:|---|---:|---:|
| M1_e2_epochs | 50414 | L40S | 26/254 (10.24%) | 162/395 (41.01%) |
| M2_e2_epochs | 50416 | A40 | 25/254 (9.84%) | 157/395 (39.75%) |
| M3_e2_epochs | 50485 | A40 | 1/254 (0.39%) | 11/395 (2.78%) |
| M4_e2_epochs | 50529 | A40 | 0/254 (0.00%) | 10/395 (2.53%) |
| M1_e3_epochs | 50530 | A40 | 24/254 (9.45%) | 150/395 (37.97%) |
| M2_e3_epochs | 50571 | A40 | 10/254 (3.94%) | 134/395 (33.92%) |
| M3_e3_epochs | 50655 | A40 | 1/254 (0.39%) | 11/395 (2.78%) |
| M4_e3_epochs | 50743 | A40 | 0/254 (0.00%) | 10/395 (2.53%) |

**OpenLLaMA:** two epochs have the best recorded scores for both methods. LoRA reaches 10.24% AQuA and 41.01% AddSub, compared with 9.06%/38.73% at one epoch and 9.45%/37.97% at three. AdapterP reaches 9.84%/39.75% at two epochs, compared with 2.36%/38.99% and 3.94%/33.92%. The one-to-two-epoch LoRA improvement is only three AQuA and nine AddSub questions; no repeated-seed significance claim is made. The highest observed test score is descriptive, not a basis for choosing settings repeatedly on the test set.

**TinyLlama:** additional epochs do not improve recorded accuracy. M3's complete raw AQuA and AddSub generations match exactly across one, two and three epochs. M4's two-/three-epoch generations match each other exactly; its one-epoch retry1 has some different text but the same accuracy. Do not describe this as stable successful learning: all these runs export checkpoint 100.

## Training-health and checkpoint evidence

Gradient counts refer to logged norms every ten Trainer steps, not every batch. `nan` strings in JSON are converted to nonfinite floats for detection.

| Run | Nonfinite gradient logs | Zero gradient logs | First validation loss | Last validation loss | Best validation loss | Exported checkpoint |
|---|---:|---:|---:|---:|---:|---|
| M1_e2_epochs | 0/336 | 0/336 | 0.834692 | 0.703496 | 0.703496 | checkpoint-3300 |
| M2_e2_epochs | 0/336 | 0/336 | 0.858980 | 0.713636 | 0.713636 | checkpoint-3300 |
| M3_e2_epochs | 2/336 | 275/336 | 1.781734 | 1.782483 | 1.781734 | checkpoint-100 |
| M4_e2_epochs | 3/336 | 302/336 | 1.881618 | 2.438171 | 1.881618 | checkpoint-100 |
| M1_e3_epochs | 0/504 | 0/504 | 0.835207 | 0.703885 | 0.702938 | checkpoint-3300 |
| M2_e3_epochs | 0/504 | 0/504 | 0.858980 | 0.835812 | 0.725680 | checkpoint-3300 |
| M3_e3_epochs | 2/504 | 444/504 | 1.781734 | 1.853911 | 1.781734 | checkpoint-100 |
| M4_e3_epochs | 3/504 | 466/504 | 1.881618 | 2.914382 | 1.881618 | checkpoint-100 |

All four additional OpenLLaMA adapters export checkpoint 3300 (approximately 1.96 epochs), including the runs that traverse three epochs. These are **best-validation** comparisons, not exact two-/three-epoch-end weight comparisons. A different epoch target changes the learning-rate schedule, so equal step numbers across independent runs do not imply equal adapters. M2's three-epoch validation deterioration from best 0.725680 to last 0.835812 is consistent with overfitting or optimization deterioration; the evidence does not establish a unique cause.

Both TinyLlama methods export checkpoint 100 (approximately 0.059 epochs). M3 final adapters contain 22 entirely zero tensors out of 88. At checkpoint 100, M3 FP16 scale is `6.103515625e-05` and scheduler count 70; M4 scale is `0.0001220703125` and scheduler count 71. These small scales and lagging counts support numerical/overflow concerns. Later weights do change and all final tensors are finite, so claiming that no learning occurred at all would be too strong. M4's validation loss worsens substantially as training continues.

## Answer-format and generation evidence

| Run | AQuA parse failures | AddSub parse failures | Empty outputs, AQuA / AddSub |
|---|---:|---:|---:|
| M1_e2_epochs | 143/254 | 0/395 | 0 / 0 |
| M2_e2_epochs | 153/254 | 3/395 | 0 / 0 |
| M3_e2_epochs | 253/254 | 37/395 | 15 / 37 |
| M4_e2_epochs | 254/254 | 2/395 | 11 / 1 |
| M1_e3_epochs | 147/254 | 4/395 | 0 / 0 |
| M2_e3_epochs | 207/254 | 2/395 | 0 / 0 |
| M3_e3_epochs | 253/254 | 37/395 | 15 / 37 |
| M4_e3_epochs | 254/254 | 2/395 | 11 / 1 |

AQuA remains dominated by missing parseable option answers, particularly for TinyLlama. Numeric extraction on AddSub may still return the last number in copied question text or repetitive generations. Reproducing the scorer verifies numerical consistency of the saved scores, not semantic correctness of every extracted final answer.

## Audit procedure and comparability

The [machine-readable audit](BASELINE_EPOCH_RUN_AUDIT.json) is reproducible from the project root with the existing audit script:

```bash
python3 scripts/audit_mandatory_runs.py --extensions --tensors
```

Checks cover full status/progress/count agreement; sequential unique IDs; benchmark hash, label and instruction identity; independent raw-answer extraction/correctness/parse flags; time/accuracy totals; adapter fingerprints; model revisions; complete disjoint train/validation IDs; retained checkpoint-file hashes; and exact final-versus-best-checkpoint tensor equality. The shared split SHA-256 remains `90338055ef76f8308bb4b3e72fffc88c7835a50df41a4442d7e99dc6e64ac7ef`.

The recorded training settings match the one-epoch controls apart from total epochs, output paths and an explicit FP16 setting in the newer source. M1/M2 at two epochs use training-source SHA `58d17af09d2ddbb1a26369031884b6daa1c7000fffa826fc557dc50c28e3336e`, matching the current local `finetune.py`. The other epoch runs use `9956d8419cd0f20d1e1f58d6b5d6ff686293f631dc98fb2ea8d2dada2475417c`, matching the repository's committed original training file. The inspected code difference adds precision selection; `precision=fp16` retains the original FP16 model dtype/Trainer setting. All other recorded source-file hashes match, including evaluator, scorer and custom PEFT. This is not a byte-identical-source experiment, and the different hardware (M1 two epochs on L40S, the other additional runs on A40) prevents clean runtime comparisons.

## Scheduler verification

[logs/baseline_epoch_sacct.txt](../logs/baseline_epoch_sacct.txt) confirms `COMPLETED` with exit code `0:0` for all four baseline and eight combined epoch allocations, including their batch steps. All completed within the six-hour limit; allocation elapsed times range from 00:23:35 to 03:28:44. Combined-job elapsed times include training and both evaluations, not training alone. `MaxRSS` describes host memory, not GPU memory.

[logs/mandatory_sacct.txt](../logs/mandatory_sacct.txt) confirms the same successful states and exit codes for all 12 original one-epoch allocations and their batch steps. Scheduler completion is verified for every selected run. The TinyLlama training-health limitations above remain relevant despite successful job exits.
