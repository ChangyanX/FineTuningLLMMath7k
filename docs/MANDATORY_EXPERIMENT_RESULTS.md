# AI6130 Assignment 2: mandatory, baseline and epoch-comparison results

## Scope and status

This report covers the selected one-epoch runs (`M1_e1_retry2`, `M2_e1_main`, `M3_e1_main`, `M4_e1_retry1`), both untouched-base models and all eight two-/three-epoch runs. All 12 training runs and 28 full evaluations are complete. Raw-generation rescoring and adapter/checkpoint checks passed; [mandatory](../logs/mandatory_sacct.txt) and [baseline/epoch](../logs/baseline_epoch_sacct.txt) scheduler records show `COMPLETED`/`0:0` for all 24 allocations and their batch steps. TinyLlama optimization problems persist across epoch counts. Supporting evidence is in the [mandatory audit](MANDATORY_RUN_AUDIT.md) and [baseline/epoch audit](BASELINE_EPOCH_RESULTS.md).

## Experimental settings

Both required base models were fine-tuned separately with LoRA and AdapterP on `math_7k.json`. The dataset contains 6,851 examples, with an identical seeded split of 6,731 training and 120 validation examples for all four runs. Mandatory runs traversed one epoch (1,683 Trainer steps); Option 1 runs traversed two or three epochs (3,366 or 5,049 Trainer steps). Each started independently from the pinned base. Settings were effective/micro batch size 4, learning rate `3e-4`, 100 warmup steps, cutoff 256 tokens, seed 42, FP16 training, and loss on both prompt and response tokens. Validation and checkpoint saving occurred every 100 steps. Final adapters were selected by lowest validation loss, rather than by test accuracy or the final training step.

LoRA used rank 8, alpha 16, dropout 0.05 and `q_proj`/`v_proj` targets. AdapterP used bottleneck size 256, tanh activation, dropout 0, scaling 1.0 and `down_proj` targets. The methods consequently have different trainable parameter counts; this is not a parameter-budget-matched comparison.

| Model | Immutable revision |
|---|---|
| `openlm-research/open_llama_3b_v2` | `bce5d60d3b0c68318862270ec4e794d83308d80a` |
| `TinyLlama/TinyLlama_v1.1` | `ff3c701f2424c7625fdefb9dd470f45ef18b02d6` |

Training used PyTorch `2.2.2+cu121`, Transformers `4.38.0`, Accelerate `0.28.0` and bundled custom PEFT. The mandatory M1-M3 runs trained on NVIDIA A40; M4 retry1 trained on L40S. For Option 1, M1 at two epochs used L40S, while the other seven runs used A40. M1/M2 at two epochs used a newer training-source hash exposing an explicit `precision=fp16`; the other runs used the original source with FP16 hard-coded. Other recorded training settings match apart from total epochs/output paths. This source/hardware variation limits strict causal and timing comparisons. Evaluation used the same instruction prompt format, deterministic four-beam decoding, maximum 256 generated tokens, seed 42 and cache disabled. No evaluation used 8-bit loading.

Strict scoring requires a parseable A-E answer for AQuA. AddSub uses numeric extraction and tolerance 0.001. Unparseable outputs count as incorrect; accuracy always uses the full test-set denominator. The saved scorer is reproduced exactly in the audit; its numeric fallback can extract the last number from an unfinished/repetitive answer, so a parseable answer does not guarantee a coherent final solution.

## Full benchmark results

| Run | Model | Method | AQuA correct / 254 | AQuA accuracy | AddSub correct / 395 | AddSub accuracy |
|---|---|---|---:|---:|---:|---:|
| M1_e1_retry2 | OpenLLaMA 3B v2 | LoRA | 23 | **9.06%** | 153 | 38.73% |
| M2_e1_main | OpenLLaMA 3B v2 | AdapterP | 6 | 2.36% | 154 | **38.99%** |
| M3_e1_main | TinyLlama v1.1 | LoRA | 1 | 0.39% | 11 | 2.78% |
| M4_e1_retry1 | TinyLlama v1.1 | AdapterP | 0 | 0.00% | 10 | 2.53% |

All eight evaluations cover their complete test sets: 2,596 predictions in total. Smoke tests and five-example diagnostic runs are excluded.

| Run | AQuA parse failures | AddSub parse failures | Trainable parameters | Selected checkpoint | Best validation loss |
|---|---:|---:|---:|---|---:|
| M1_e1_retry2 | 148/254 (58.27%) | 1/395 (0.25%) | 2,662,400 | 1600 | 0.718773 |
| M2_e1_main | 219/254 (86.22%) | 0/395 (0.00%) | 42,598,400 | 1600 | 0.694462 |
| M3_e1_main | 253/254 (99.61%) | 37/395 (9.37%) | 1,126,400 | 100 | 1.781734 |
| M4_e1_retry1 | 254/254 (100.00%) | 2/395 (0.51%) | 23,068,672 | 100 | 1.883506 |

## Untouched-base baselines

Each untouched-base model is evaluated on both benchmarks and shared by both adapter comparisons.

| Untouched model | AQuA correct / 254 | AQuA accuracy | AddSub correct / 395 | AddSub accuracy | AQuA / AddSub parse failures |
|---|---:|---:|---:|---:|---:|
| OpenLLaMA 3B v2 (`M1_baseline_main`) | 0 | 0.00% | 16 | 4.05% | 254 / 3 |
| TinyLlama v1.1 (`M3_baseline_main`) | 1 | 0.39% | 10 | 2.53% | 253 / 31 |

Using the same model revisions, test data, prompt/decoding settings and strict scorer, mandatory OpenLLaMA LoRA improves by **9.06 percentage points on AQuA and 34.68 on AddSub** over its base; AdapterP improves by **2.36 and 34.94 points**. TinyLlama LoRA matches its base on AQuA and adds one correct AddSub answer (+0.25 points); AdapterP is one AQuA answer worse (-0.39 points) and ties on AddSub. These are changes in measured strict accuracy under this protocol. In particular, the OpenLLaMA base's 0% AQuA score has 254 parse failures; it is not evidence of zero underlying reasoning ability.

## Task 4, Option 1: epoch comparison

Each cell is **AQuA accuracy / AddSub accuracy (%)**. All settings use seed 42 and best-validation adapters, not necessarily the last epoch's weights.

| Model | Method | 1 epoch | 2 epochs | 3 epochs |
|---|---|---:|---:|---:|
| OpenLLaMA 3B v2 | LoRA | 9.06 / 38.73 | **10.24 / 41.01** | 9.45 / 37.97 |
| OpenLLaMA 3B v2 | AdapterP | 2.36 / 38.99 | **9.84 / 39.75** | 3.94 / 33.92 |
| TinyLlama v1.1 | LoRA | 0.39 / 2.78 | 0.39 / 2.78 | 0.39 / 2.78 |
| TinyLlama v1.1 | AdapterP | 0.00 / 2.53 | 0.00 / 2.53 | 0.00 / 2.53 |

Two epochs give the highest observed scores for both OpenLLaMA methods. Relative to one epoch, LoRA gains three AQuA and nine AddSub answers (+1.18/+2.28 percentage points); AdapterP gains 19 and three (+7.48/+0.76 points). Three epochs decrease both scores relative to two. These observations are descriptive, not a test-set-based recommendation to select an epoch count.

All additional OpenLLaMA adapters were selected from checkpoint 3300, approximately **1.96 epochs** into training. Even a three-epoch run therefore evaluates an earlier checkpoint. Different total epochs also change the learning-rate schedule, so checkpoint 3300 in a two-epoch run is not the same trained state as checkpoint 3300 in a three-epoch run. For AdapterP at three epochs, validation loss rises from its best 0.725680 to 0.835812 by step 5000, consistent with deterioration later in training rather than a guaranteed benefit from more passes.

TinyLlama still selects checkpoint 100 (approximately 0.059 epochs) in every run. M3's raw generations are identical across all three epoch settings. M4's two-/three-epoch raw generations are identical to each other, although not all match retry1; their scores tie. This repetition reflects the exported early checkpoints and does not demonstrate stable, successful multi-epoch learning. Detailed gradient and validation-loss evidence is in the [extension audit](BASELINE_EPOCH_RESULTS.md).

## Discussion

**One-epoch OpenLLaMA results.** LoRA scores 6.69 percentage points higher than AdapterP on AQuA under strict scoring. Much of this comparison involves answer-format compliance: only 106 LoRA outputs and 35 AdapterP outputs are parseable. It is therefore insufficient to interpret the score difference as a pure measure of mathematical reasoning. On AddSub, AdapterP exceeds LoRA by one question (0.25 percentage points); a single seed and this small difference do not establish a robust advantage. AdapterP's lower validation loss also does not imply better AQuA performance.

**One-epoch TinyLlama results.** Both methods perform poorly and frequently repeat question text or prompt markers. This accompanies numerical/optimization concerns: M3 and M4 have two logged nonfinite gradient norms each, and respectively 161/168 and 128/168 gradient log entries are exactly zero. M3 validation loss remains near 1.782, while M4 worsens from 1.884 at step 100 to 2.077 at step 1600. Both final adapters are selected from checkpoint 100, approximately 0.059 epochs into training. Although all saved final-adapter tensors are finite and some weights changed, these are not reliable demonstrations of successful one-epoch learning. Precision, data truncation and output formatting need controlled investigation before attributing the low scores to model capacity or adapter design.

**Generation examples.** M1 correctly answers AddSub example 0 with `7 - 3 = 4` and a final answer of 4. On the same question, M2 first obtains 4, then adds 7 again and concludes 11, illustrating a reasoning error despite parseable output. M3 repeatedly copies the statement about Mary taking 3 crayons and is scored as 3.0; M4 shows similar repetition. This illustrates why low parse-failure counts on AddSub do not imply good reasoning. On AQuA example 0, M1 repeats invalid arithmetic about the recorded property value and produces no parseable option, showing both reasoning and termination/formatting problems.

**Training context and limitations.** The TinyLlama diagnostic finds that cutoff 256 truncates 4,387/6,851 source examples (64.03%). Increasing cutoff to 512 reduces that to 193 (2.82%), but no improvement in test accuracy can be claimed without a new controlled run. The original experiments also train on prompt tokens, and the 256-token generation budget may end responses before a final answer. Answer-format compliance affects before/after accuracy comparisons. These are single-seed observations, and hardware differences preclude controlled timing comparisons. Task 4's epoch option is complete; other datasets, models and benchmarks are outside this report.

## Findings

OpenLLaMA improves substantially over its untouched base under strict scoring, with the highest observed scores at two epochs and no consistent benefit from three. TinyLlama's numerical/optimization problems prevent a clean inference about model capacity or adapter quality. A future validated numerical profile would require fresh controlled comparisons. These single-seed results do not establish an intrinsic model or PEFT ranking.

This Markdown is source material for the required report of at most four pages excluding the cover; the final page layout still needs checking.
