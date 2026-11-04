# Run Assignment 2 on the NTU school GPU

This guide uses the existing `finetune.py` and `evaluate.py`, with focused persistence/correctness fixes. It follows the supplied [school GPU kit](school_gpu_kit/README.md). **No school login, allocation or GPU test has been performed by this implementation.** Local validation is recorded at the end. School limits in the kit are historical, not a live entitlement check.

## 1. Layout and rules

| Item | Value / action |
|---|---|
| Access | NTU VPN off campus (`vpngate.ntu.edu.sg`), then SSH/SFTP `changyan003@10.96.189.12` |
| Scheduler | SLURM; partition `MGPU-TC2`, QoS `normal` |
| Request | 1 GPU, 4 CPUs, 24 GB **host RAM**, maximum 6 hours per job |
| Account queue limit | **2 total jobs: at most 1 running + 1 queued**; further submissions are rejected |
| GPU | Expect A40 48 GB from supplied evidence; confirm actual allocation |
| Persistent project | `~/ai6130-ass2` (verify quota and compute-node accessibility) |
| Frozen source | `~/ai6130-ass2/source-v1`; never upload changed code over an active experiment |
| Environment | Project-specific Conda prefix `~/ai6130-ass2/env` |
| Runs/cache | `~/ai6130-ass2/runs`, `~/ai6130-ass2/cache/huggingface` |

**Head node:** login, file transfer, directory creation, `sbatch`, queue/accounting and log inspection only. **Allocated compute node:** installation, downloads, Python, tests, training, evaluation, summary and export. The launcher checks SLURM variables **and actual hostname membership** before application work. Do not use `bash job.sbatch`, `nohup` or `tmux` as a substitute for an allocation.

## 2. Transfer source from your Mac

Connect VPN. Verify a new SSH host fingerprint through a trusted school channel; use interactive authentication, never put credentials in files.

On the Mac:

```bash
ssh changyan003@10.96.189.12
```

Inside that SSH shell (file management only):

```bash
mkdir -p ~/ai6130-ass2/source-v1 ~/ai6130-ass2/logs
exit
```

Back on the Mac:

```bash
cd '/Users/changyanxu/Documents/NTU/AI6130 - LLM/Assignment2/FineTuningLLMMath7k'
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.env*' --exclude='runs/' --exclude='trained_models/' \
  --exclude='cache/' --exclude='exports/' --exclude='summary/' \
  --exclude='logs/' --exclude='*.egg-info/' \
  ./ changyan003@10.96.189.12:~/ai6130-ass2/source-v1/
ssh changyan003@10.96.189.12
```

FileZilla alternative: SFTP, port 22, same host/account; upload reviewed source to `source-v1`. Do not transfer your Mac environment. Use a new `source-v2` and new run IDs for later scientific/code changes; preserve earlier source and environment for resume.

## 3. Create environment, check, and download models

Inside the SSH/head-node shell:

```bash
cd ~/ai6130-ass2
sbatch source-v1/scripts/tc2/job.sbatch setup
# Submitted batch job 46893
```

Record its returned job ID. Inspect it using the monitoring commands in section 7; proceed only after successful setup. Setup runs on compute, loads `anaconda`, creates Python 3.10, installs PyTorch **2.2.2 CUDA 12.1**, the versions in `requirements-a40.txt`, and editable **bundled custom PEFT** with `--no-deps`. It saves `environment.freeze.txt`. Do not also run the original notebook's pip cells or install standard PEFT.

```bash
# Monitor job 46893
squeue -j 46893
tail -f logs/math7k_46893.out
# Once it finishes, check its exit status and logs:
sacct -j 46893 --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 80 logs/math7k_46893.out
tail -n 80 logs/math7k_46893.err
# Also confirm the package snapshot exists:
ls -lh environment.freeze.txt
```

Then, sequentially after each preceding job succeeds:

```bash
sbatch source-v1/scripts/tc2/job.sbatch check
# Submitted batch job 46898
sbatch source-v1/scripts/tc2/job.sbatch prepare
# Submitted batch job 46899

# After prepare succeeds:
sbatch source-v1/scripts/tc2/job.sbatch audit
# Submitted batch job 46900
```

`check` prints CUDA/GPU/package/import evidence and runs tests. `prepare` resolves immutable Hugging Face commit SHAs, saves `models.json` and `revisions.env`, and downloads one available weight format plus configuration/tokenizer files for both models. Rerunning prepare reuses the recorded SHAs. All training/evaluation jobs read these pins automatically. `audit` validates dataset schemas/counts and saves token-length/truncation statistics for both tokenizers to `data_audit.json`; review this before choosing cutoff 256.

Check at least 30 GB free, preferably 50 GB. If the `anaconda` module, CUDA wheel, network access, RAM request or partition is unavailable, stop and use the approved school equivalent. Do not move package installation or model downloads to the head node. If setup fails partway, preserve logs and inspect the incomplete environment on compute; the setup action refuses to overwrite an existing environment. Use a new project environment rather than modifying one used by active runs.

## 4. Smoke tests before expensive runs

Run one job at a time initially; account GPU limits may permit only one GPU anyway. Substitute M1 → M2 → M3 → M4 below, **waiting for training success before evaluation**:

```bash
sbatch source-v1/scripts/tc2/job.sbatch train-smoke M1 1 smoke01
# Submitted batch job 46903
sbatch source-v1/scripts/tc2/job.sbatch eval-smoke M1 1 smoke01 AQuA
# Submitted batch job 46904
sbatch source-v1/scripts/tc2/job.sbatch eval-smoke M1 1 smoke01 AddSub
# Submitted batch job 46905
```

Training smoke uses 4 optimizer steps, 8 validation records, checkpoints/evaluation every 2 updates, and no warmup. Evaluation smoke uses 5 test examples, up to 32 generated tokens. It checks save/load/inference; its scores are **not report results**. The `smoke01` tag keeps these separate from full runs.

Inspect: finite train/validation loss, nonzero intended trainable-parameter count, correct adapter config, plausible generations, parsed answers and `status.json`. The checkpoint contains optimizer/scheduler/RNG state plus adapter weights. Try both models/methods; merely importing PEFT does not validate AdapterP inference.

For memory failures: start a fresh tagged run with smaller micro-batch and compensating accumulation after editing a new frozen source profile. For effective batch 4, micro-batch 1 means accumulation 4. On A40, start with micro-batch 4. Keep shared scientific settings consistent across methods.

## 5. Mandatory runs

Each training row is followed by **two separate evaluation jobs**. The epochs argument is total epochs for a fresh run, not additional epochs on a resumed run. Resume retains the same scientific settings.

| Run | Model | Method | Training / evaluation counts |
|---|---|---|---|
| M1 | OpenLLaMA 3B v2 | LoRA | 1 / 2 |
| M2 | OpenLLaMA 3B v2 | AdapterP | 1 / 2 |
| M3 | TinyLlama v1.1 | LoRA | 1 / 2 |
| M4 | TinyLlama v1.1 | AdapterP | 1 / 2 |
| Total | | | **4 training + 8 evaluation jobs** |

Initial settings: 1 epoch, learning rate `3e-4`, cutoff 256, effective/micro batch 4, validation 120, seed 42, save/validation every 100 updates. Dataset has **6,851 total records: 6,731 training + 120 validation**. W&B disabled; local logging is always retained. Evaluation uses deterministic 4-beam search, maximum 256 new tokens, cache disabled, strict scoring. Keep settings fixed for comparable results.

On the SSH/head-node shell, submit **each next command only after its dependency succeeded**:

```bash
cd ~/ai6130-ass2

sbatch source-v1/scripts/tc2/job.sbatch train M1 1 main
# Submitted batch job 46920
sbatch source-v1/scripts/tc2/job.sbatch train M1 1 retry2
# Submitted batch job 46924:
# $ tail -n 100 runs/M1_e1_retry2/train_46924.log
sbatch source-v1/scripts/tc2/job.sbatch eval M1 1 retry2 AQuA
# Submitted batch job 46927
# $ tail -f runs/M1_e1_retry2/eval/AQuA/evaluate_46927.log

sbatch source-v1/scripts/tc2/job.sbatch eval M1 1 retry2 AddSub
# Submitted batch job 47120
# runs/M1_e1_retry2/final_adapter
# $ tail -f runs/M1_e1_retry2/eval/AddSub/evaluate_47120.log
# $ sacct -j 47120 --format=JobID,JobName,State,ExitCode,Elapsed

sbatch source-v1/scripts/tc2/job.sbatch train M2 1 main
# Submitted batch job 47309
# $ tail -f runs/M2_e1_main/train_47309.log
sbatch source-v1/scripts/tc2/job.sbatch eval M2 1 main AQuA
# Submitted batch job 47391
# $ tail -f runs/M2_e1_main/eval/AQuA/evaluate_47391.log
sbatch source-v1/scripts/tc2/job.sbatch eval M2 1 main AddSub
# Submitted batch job 47749
# $ tail -f runs/M2_e1_main/eval/AddSub/evaluate_47749.log

sbatch source-v1/scripts/tc2/job.sbatch train M3 1 main
# Submitted batch job 47780
# $ tail -f runs/M3_e1_main/train_47780.log
sbatch source-v1/scripts/tc2/job.sbatch eval M3 1 main AQuA
# Submitted batch job 47786
# $ tail -f runs/M3_e1_main/eval/AQuA/evaluate_47786.log
sbatch source-v1/scripts/tc2/job.sbatch eval M3 1 main AddSub
# Submitted batch job 47853
# $ tail -f runs/M3_e1_main/eval/AddSub/evaluate_47853.log

sbatch source-v1/scripts/tc2/job.sbatch train M4 1 main
# Submitted batch job 47965
# $ tail -f runs/M4_e1_main/train_47965.log
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 main AQuA
# Submitted batch job 47970
# $ tail -f runs/M4_e1_main/eval/AQuA/evaluate_47970.log
# $ sacct -j 47970 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 main AddSub


---
sbatch source-v1/scripts/tc2/job.sbatch train M4 1 retry1
# Submitted batch job 49897
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 retry1 AQuA
# Submitted batch job 49903
# sacct -j 49903 --format=JobID,JobName,State,ExitCode,Elapsed
# $ tail -f runs/M4_e1_retry1/eval/AQuA/evaluate_49903.log
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 retry1 AddSub
# Submitted batch job 49916
# sacct -j 49916 --format=JobID,JobName,State,ExitCode,Elapsed
# $ tail -f runs/M4_e1_retry1/eval/AddSub/evaluate_49916.log

```

Do **not** paste the entire block as an unattended pipeline: plain `sbatch` returns immediately; it does not wait for training. An evaluation submitted too early fails if no final adapter exists. A dependent job still occupies one of the two allowed slots; do not queue training plus two dependent evaluations. Use the combined actions below for new experiments. Do not automatically chain retries on failure.

### Optional baselines and epochs

Untouched-base baselines (4 evaluations shared by both adapters):

**Account limit: at most two jobs total (running + queued), with only one running at a time. Do not paste all four commands together.** Submit only when `squeue -u "$USER"` shows a free slot. If the queue is empty, you may submit the first two, then wait for both to finish successfully before the next pair. Dependent jobs count toward the same limit. The IDs below record earlier submissions; do not rerun completed evaluations. Resume an interrupted baseline with `sbatch --export=ALL,RESUME=1 ... baseline ...` using its original source/settings.

These baselines support Task 3's before/after discussion; they are additional comparisons, not an explicitly listed Task 4 option.

```bash
cd ~/ai6130-ass2
export SOURCE_DIR="$HOME/ai6130-ass2/source-v1"
sbatch source-v1/scripts/tc2/job.sbatch baseline M1 1 main AQuA
# Submitted batch job 50173
# tail -f runs/M1_baseline_main/eval/AQuA/evaluate_50173.log
# sacct -j 50173 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-v1/scripts/tc2/job.sbatch baseline M1 1 main AddSub
# Submitted batch job 50174
# tail -f runs/M1_baseline_main/eval/AddSub/evaluate_50174.log
# sacct -j 50174 --format=JobID,JobName,State,ExitCode,Elapsed
```

Wait for the first pair to finish successfully before submitting the next pair:

```bash
sbatch source-v1/scripts/tc2/job.sbatch baseline M3 1 main AQuA
# Submitted batch job 50375
# tail -f runs/M3_baseline_main/eval/AQuA/evaluate_50375.log
# sacct -j 50375 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-v1/scripts/tc2/job.sbatch baseline M3 1 main AddSub
# Submitted batch job 50377
# tail -f runs/M3_baseline_main/eval/AddSub/evaluate_50377.log
# sacct -j 50377 --format=JobID,JobName,State,ExitCode,Elapsed
```

### Task 4: simple submissions within the two-job limit

Use the existing **`scripts/tc2/job.sbatch`** for every option. The separate Task 4 launcher and preparation helper have been removed. New actions merge work **sequentially inside one GPU allocation**, without submitting child jobs:

| Action | Work inside one job |
|---|---|
| `train-eval` | Train, then AQuA, then AddSub. An explicit fifth argument selects only that benchmark. |
| `train-eval-smoke` | Four training steps, then five examples each on AQuA/AddSub. Use a separate smoke tag. |
| `eval-pair` | AQuA, then AddSub on an existing adapter. |
| `baseline-pair` | AQuA, then AddSub on one untouched base model. |

For `math_10k`, a default pair is AQuA/SVAMP instead. The original `train`, `eval`, `baseline`, `summary`, `export` and resume commands still work. A combined job stops if any stage fails or is interrupted. All stages share the **same six-hour wall limit**, not six hours each. Keep larger benchmarks separate. Grouped jobs create the same per-run/per-benchmark files as individual jobs and share a SLURM job ID.

**Submission rule for every block below:** first run `squeue -u "$USER"`. Paste a two-command block only when your queue is empty. Wait for **both jobs to leave the queue and verify their successful completion** before the next block. If one job is already running/queued, submit at most one more. Do not paste multiple blocks together or add dependency submissions.

[The mandatory-run audit](MANDATORY_RUN_AUDIT.md) found numerical problems in M3/M4. These commands retain the original FP16/256-token training settings; they do not repair those runs. Validate a repaired profile before expanding unhealthy experiments, and use fresh control runs/tags if settings change.

#### Upload the updated launcher once

On the Mac, upload a new frozen source rather than overwriting active `source-v1` experiments:

```bash
cd '/Users/changyanxu/Documents/NTU/AI6130 - LLM/Assignment2/FineTuningLLMMath7k'
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.env*' --exclude='runs/' --exclude='trained_models/' \
  --exclude='cache/' --exclude='exports/' --exclude='summary/' \
  --exclude='logs/' --exclude='*.egg-info/' \
  ./ changyan003@10.96.189.12:~/ai6130-ass2/source-options-v1/
ssh changyan003@10.96.189.12
```

On the school head node, once per SSH session:

```bash
cd ~/ai6130-ass2
export SOURCE_DIR="$HOME/ai6130-ass2/source-options-v1"
unset RESUME
squeue -u "$USER"
```

Keep this source unchanged after submission. These jobs reuse the existing environment/cache/model pins; do not repeat setup. The optional sixth argument selects `math_7k` (default) or `math_10k`.

#### Optional: merge two baseline evaluations into one job

Only for baselines **not already submitted/completed**; the baseline IDs above already record all four submissions. Do not duplicate them. If starting fresh with no other queued jobs:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch baseline-pair M1 1 main
sbatch source-options-v1/scripts/tc2/job.sbatch baseline-pair M3 1 main
```

This is two allocations for four evaluations. To resume previously started baselines, use their original source and individual baseline-resume commands instead.

#### Option 1: more epochs

Each command combines one training run and two evaluations. Two epochs: **4 jobs**, submitted in two pairs.

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M1 2 epochs
# Submitted batch job 50414
# tail -n 100 runs/M1_e2_epochs/train_50414.log
# tail -f runs/M1_e2_epochs/eval/AQuA/evaluate_50414.log
# tail -f runs/M1_e2_epochs/eval/AddSub/evaluate_50414.log
# sacct -j 50414 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M2 2 epochs
# Submitted batch job 50416
# tail -n 100 runs/M2_e2_epochs/train_50416.log
# tail -f runs/M2_e2_epochs/eval/AQuA/evaluate_50416.log
# tail -f runs/M2_e2_epochs/eval/AddSub/evaluate_50416.log
# sacct -j 50416 --format=JobID,JobName,State,ExitCode,Elapsed
```

Wait for both to finish successfully, then:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M3 2 epochs
# Submitted batch job 50485
# tail -n 100 runs/M3_e2_epochs/train_50485.log
# tail -f runs/M3_e2_epochs/eval/AQuA/evaluate_50485.log
# tail -f runs/M3_e2_epochs/eval/AddSub/evaluate_50485.log
# sacct -j 50485 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M4 2 epochs
# Submitted batch job 50529
# tail -n 100 runs/M4_e2_epochs/train_50529.log
# tail -f runs/M4_e2_epochs/eval/AQuA/evaluate_50529.log
# tail -f runs/M4_e2_epochs/eval/AddSub/evaluate_50529.log
# sacct -j 50529 --format=JobID,JobName,State,ExitCode,Elapsed
```

Optional three-epoch comparison, again one pair at a time:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M1 3 epochs
# Submitted batch job 50530
# tail -n 100 runs/M1_e3_epochs/train_50530.log
# tail -f runs/M1_e3_epochs/eval/AQuA/evaluate_50530.log
# tail -f runs/M1_e3_epochs/eval/AddSub/evaluate_50530.log
# sacct -j 50530 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M2 3 epochs
# Submitted batch job 50571
# tail -n 100 runs/M2_e3_epochs/train_50571.log
# tail -f runs/M2_e3_epochs/eval/AQuA/evaluate_50571.log
# tail -f runs/M2_e3_epochs/eval/AddSub/evaluate_50571.log
# sacct -j 50571 --format=JobID,JobName,State,ExitCode,Elapsed
```

Wait for both, then:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M3 3 epochs
# Submitted batch job 50655
# tail -n 100 runs/M3_e3_epochs/train_50655.log
# tail -f runs/M3_e3_epochs/eval/AQuA/evaluate_50655.log
# tail -f runs/M3_e3_epochs/eval/AddSub/evaluate_50655.log
# sacct -j 50655 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M4 3 epochs
# Submitted batch job 50743
# Submitted batch job 50743
# tail -n 100 runs/M4_e3_epochs/train_50743.log
# tail -f runs/M4_e3_epochs/eval/AQuA/evaluate_50743.log
# tail -f runs/M4_e3_epochs/eval/AddSub/evaluate_50743.log
# sacct -j 50743 --format=JobID,JobName,State,ExitCode,Elapsed
```

These train independently from the pinned base, not from the one-epoch adapter. Output paths include epoch and tag, e.g. `runs/M1_e2_epochs`. Final adapters still come from the best validation checkpoint; report its selected step rather than assuming an epoch-end snapshot.

#### Option 2: other training data, `math_10k.json`

`math_10k` contains 9,919 examples (9,799 train + 120 validation). It overlaps AddSub/MultiArith/SingleEq; do not use those benchmarks for this setting. Use AQuA and SVAMP. Local normalized-question checks found no exact duplicates for these two, which does not guarantee complete semantic separation.

Merge training and **AQuA only**; keep the larger SVAMP evaluation separate. This needs **8 jobs** total, submitted two at a time:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M1 1 math10k AQuA math_10k
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M2 1 math10k AQuA math_10k
```

Wait for both, then:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M3 1 math10k AQuA math_10k
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M4 1 math10k AQuA math_10k
```

Wait for both, then:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M1 1 math10k SVAMP math_10k
sbatch source-options-v1/scripts/tc2/job.sbatch eval M2 1 math10k SVAMP math_10k
```

Wait for both, then:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M3 1 math10k SVAMP math_10k
sbatch source-options-v1/scripts/tc2/job.sbatch eval M4 1 math10k SVAMP math_10k
```

For a dataset comparison, evaluate the original `math_7k` adapters on SVAMP too (Option 4).

#### Option 3: another LLM

M5 = LoRA and M6 = AdapterP on **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**. This is a distinct chat checkpoint with the same architecture/size family. The [official model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) describes Llama compatibility. The experiment retains the assignment's prompt format rather than its native chat template; report that limitation. GPU/custom AdapterP compatibility remains to be checked.

First submit **one preparation job**, then wait for success:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch prepare-extra
```

It pins/downloads the extra model on compute, saving `models-extra.json` and `revisions-extra.env`; existing mandatory model pins remain intact. Then smoke-test both methods in **two jobs**:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval-smoke M5 1 chat_smoke01
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval-smoke M6 1 chat_smoke01
```

Wait for both and inspect gradients, adapter reloads and generations. Only after successful checks, submit **two full jobs**:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M5 1 chat
sbatch source-options-v1/scripts/tc2/job.sbatch train-eval M6 1 chat
```

#### Option 4: more benchmarks on the selected mandatory adapters

These reuse M1 retry2, M2 main, M3 main and M4 retry1. No retraining is needed. Keep each larger benchmark in its own job; the full 16 evaluations can exceed six hours if grouped. **Wait for both jobs after every block**:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M1 1 retry2 MultiArith
sbatch source-options-v1/scripts/tc2/job.sbatch eval M2 1 main MultiArith
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M3 1 main MultiArith
sbatch source-options-v1/scripts/tc2/job.sbatch eval M4 1 retry1 MultiArith
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M1 1 retry2 SingleEq
sbatch source-options-v1/scripts/tc2/job.sbatch eval M2 1 main SingleEq
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M3 1 main SingleEq
sbatch source-options-v1/scripts/tc2/job.sbatch eval M4 1 retry1 SingleEq
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M1 1 retry2 gsm8k
sbatch source-options-v1/scripts/tc2/job.sbatch eval M2 1 main gsm8k
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M3 1 main gsm8k
sbatch source-options-v1/scripts/tc2/job.sbatch eval M4 1 retry1 gsm8k
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M1 1 retry2 SVAMP
sbatch source-options-v1/scripts/tc2/job.sbatch eval M2 1 main SVAMP
```

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch eval M3 1 main SVAMP
sbatch source-options-v1/scripts/tc2/job.sbatch eval M4 1 retry1 SVAMP
```

Expected counts: MultiArith 600, SingleEq 508, gsm8k 1,319, SVAMP 1,000. If a single evaluation exceeds six hours, resume that evaluation using its original source/settings.

#### Resume a combined job and summarize

After an interruption, inspect statuses/logs. With the **same source, model/data/settings/tag**, resubmit the same combined command with `RESUME=1`:

```bash
sbatch --export=ALL,RESUME=1 source-options-v1/scripts/tc2/job.sbatch train-eval M1 2 epochs
```

A completed training stage is skipped. Existing evaluation stages use evaluator resume; stages without saved settings start fresh. A failed stage stops the combined job. Do not queue a resume while the original job is still active, and do not resume through a settings mismatch/corruption error.

When all intended evaluations finish and a slot is free:

```bash
sbatch source-options-v1/scripts/tc2/job.sbatch summary
```

This writes `summary/results.csv` for all complete full benchmarks. Filter exact run IDs for the report. The existing `export` action also accepts M5/M6 in the updated launcher. Download `models-extra.json` and `revisions-extra.env` separately with any extra-model exports. No school jobs were submitted by this edit.

Local checks passed for grouped stage order, stopping after failure, per-stage resume, signal forwarding, dataset-overlap rejection, baseline pairing and extra-model smoke argument routing. All command blocks parse as Bash and contain at most two submissions. These are offline checks, not a validation of GPU runtime or scientific quality.

## 6. Saved outputs and recovery

```text
~/ai6130-ass2/
  logs/math7k_JOBID.out, .err             scheduler attempt logs
  models.json, revisions.env             immutable model SHAs
  environment.freeze.txt                 installed dependency versions
  runs/M1_e1_main/
    settings.json                        args + source/data hashes + runtime
    resolved_model.json                  revision, targets, trainable count, split sizes
    split.json                           ordered training/validation example IDs
    events.jsonl                         UTC/job/host audit, checkpoint events
    status.json                          running/interrupted/failed/complete
    train_JOBID.log                       training stdout/stderr per attempt
    environment_JOBID.json               GPU/driver/package evidence
    metrics.jsonl                        step/epoch/loss/LR logs
    checkpoint-N/                        model + optimizer + scheduler + RNG + Trainer state
      checkpoint_manifest.json           file hashes / completeness evidence
    epochs/epoch-N/                      adapter-only full-epoch snapshots
    final_adapter/                       best-validation adapter + tokenizer
    eval/AQuA/                           AddSub has a separate directory
      settings.json, resolved_model.json, events.jsonl, status.json
      evaluate_JOBID.log, environment_JOBID.json
      predictions.jsonl                  stable IDs, raw/parsed answer, label, correctness, latency
      progress.json, metrics.json        completed/total, accuracy, parse failures
```

Each run is locked to one writer. Source, dataset, runtime or scientific-setting mismatch refuses resume. Checkpoints are saved every 100 optimizer steps, retaining up to 3 under Trainer's best/latest policy, plus separate completed-epoch adapter snapshots. Checkpoint rotation is not a backup. Final adapters contain only adapted weights; keep the base model ID + immutable revision for inference. Final export is based on best validation checkpoint, which may predate the last epoch.

SLURM sends `USR1` 180 seconds before the wall limit. The program requests a stop, finishes the current training step/question, saves committed progress, and exits **75**. This may appear as `FAILED` in scheduler accounting even though the application says `interrupted`. SIGKILL, OOM or node failure can lose work since the last checkpoint; the warning margin is not a guarantee. Check logs/status before deciding whether to resume.

### Training resume, same settings/source/tag

```bash
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch train M1 1 main
```

It verifies checkpoint hashes, selects the latest valid checkpoint, restores adapter/optimizer/scheduler/RNG/Trainer progress, and continues toward the original epoch target. Invalid/incomplete checkpoints are skipped with an audit event; if no valid checkpoint exists, it refuses. Preserve the old directory and use a new tag to restart from scratch. A complete training run refuses retraining in place.

### Evaluation resume

```bash
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch eval M1 1 main AQuA
```

It validates saved settings, reads committed ordered prediction records and continues at the next example. A truncated final JSONL record is preserved in a `.damaged-*` copy before its incomplete tail is removed; interior corruption/duplicate IDs are rejected. Completed evaluation resume does not generate more answers. Predictions are durable after each question; progress/metrics are regenerated from records.

### Intentionally verify stop/resume on school

Use a **separate pilot tag** and a sufficiently long actual run so it is still running when signalled:

```bash
sbatch source-v1/scripts/tc2/job.sbatch train M1 1 resumecheck
# After it starts and passes at least one optimizer update:
scancel --signal=USR1 --batch JOB_ID
# Wait for exit; check interrupted status and a valid saved checkpoint, then:
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch train M1 1 resumecheck
```

Repeat with an evaluation on that pilot adapter, using the matching `resumecheck` tag. Compare saved global steps and prediction IDs across attempts; inspect optimizer/scheduler state. Test both adapter types. CPU tests already compare resumed vs uninterrupted tiny-model weights, but this does not verify SLURM delivery, CUDA FP16 or the shared filesystem.

## 7. Monitoring, failures and time budget

On the head node (replace `JOB_ID`):

```bash
squeue -j JOB_ID -o '%.18i %.12j %.10T %.10M %.20N %.30R'
scontrol show job JOB_ID
tail -f ~/ai6130-ass2/logs/math7k_JOB_ID.out
tail -f ~/ai6130-ass2/runs/M1_e1_main/train_JOB_ID.log
# Evaluation example:
tail -f ~/ai6130-ass2/runs/M1_e1_main/eval/AQuA/evaluate_JOB_ID.log
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

`PENDING` means not running. Ctrl+C stops `tail`, not the job. Empty queue is not success: require scheduler accounting plus application `complete` and full benchmark coverage. Use `scancel JOB_ID` only for that unwanted job; it may interrupt saving. Never submit two writers to the same output path. Preserve all failed-attempt logs.

| Failure | Next action |
|---|---|
| Compute guard refuses | Submit using `sbatch` from project root; never remove guard |
| PEFT/module import error | Inspect `environment_JOBID.json`; use custom PEFT and pinned environment |
| CUDA OOM | New config/tag with lower micro-batch; retain effective batch; retest |
| Host-memory OOM | Inspect `MaxRSS`; school limit may allow up to 30 GB historically, verify entitlement before adjusting |
| Resume mismatch | Restore original immutable source/env/config, or start new experiment; do not bypass identity checks |
| No recovery checkpoint | Keep evidence; restart under a new tag |
| Output directory exists | Explicit resume only for identical settings, otherwise a fresh tag |
| Partial evaluation metrics | Do not report as full benchmark; resume first |

Budget for all mandatory work on A40: **roughly 7–20 hours total across jobs**, not one allocation. Each job requests up to 6 hours. See [requirements timing table](../ASSIGNMENT_REQUIREMENTS.md) for per-run ranges. Measure 50–100 training steps and 20 representative examples before trusting estimates. The first download/setup can be slower; answer length strongly affects inference. Cache is deliberately unchanged from the reference until compatibility is tested.

Audit training-token lengths under both tokenizers before committing to 256. If truncation removes important solutions, use a consistent larger cutoff in a new source/config version and revise timing. Use held-out validation to choose checkpoints; do not tune against test results.

## 8. Summarize, verify, export and download

After full evaluation jobs finish:

```bash
sbatch source-v1/scripts/tc2/job.sbatch summary
```

This independently re-extracts answers from raw generations and recomputes correctness, requires sequential unique IDs and exactly 254 AQuA / 395 AddSub predictions, rejects mismatched counts, excludes smoke/partial results and writes `summary/results.csv`. Inspect representative raw answers and parse failures as well; structural checks cannot prove that an answer parser is semantically correct. Strict scoring prefers explicit AQuA final-answer text and records unsupported/unparseable numeric forms as failures; `--scoring legacy` is available for a separate reference comparison.

Export each stopped run (M1 shown; repeat M2/M3/M4) in a compute allocation:

```bash
sbatch source-v1/scripts/tc2/job.sbatch export M1 1 main
```

Export locks the run and its evaluations to reject active writers, creates a timestamped archive with the run, frozen source, environment freeze, and relevant scheduler logs, plus internal SHA-256 manifest and archive checksum. It excludes base-model cache and credentials. Capture final `sacct` output separately after jobs end and keep it with downloaded evidence; application status does not replace accounting. The CSV can be downloaded separately.

From your Mac, replace exact filenames printed by export:

```bash
mkdir -p school-results
cd school-results
scp changyan003@10.96.189.12:~/ai6130-ass2/exports/EXACT_ARCHIVE.tar.gz ./
scp changyan003@10.96.189.12:~/ai6130-ass2/exports/EXACT_ARCHIVE.tar.gz.sha256 ./
scp changyan003@10.96.189.12:~/ai6130-ass2/summary/results.csv ./
shasum -a 256 EXACT_ARCHIVE.tar.gz
cat EXACT_ARCHIVE.tar.gz.sha256
tar -tzf EXACT_ARCHIVE.tar.gz
mkdir restored
tar -xzf EXACT_ARCHIVE.tar.gz -C restored
```

Compare the printed digest before extraction; inspect archive paths. Verify the internal manifest on your Mac (ordinary Python, no GPU needed):

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('restored')
manifest = json.loads((root/'ARTIFACT_SHA256.json').read_text())
for name, expected in manifest.items():
    h = hashlib.sha256()
    with (root/name).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == expected, name
print('All archived files verified')
PY
```

Keep remote originals until download/restore checks pass; maintain a second independent backup. Do not put model weights into Git. The report remains ≤4 pages excluding cover and should explain settings, accuracy, findings and limitations.

## 9. Direct commands inside an allocated, activated compute shell

The batch launcher already constructs these; they are useful for understanding or a school-approved interactive allocation. **Do not run them directly in an ordinary SSH shell.** Run from frozen source root, activate `~/ai6130-ass2/env`, set `HF_HOME`, and load `revisions.env` first.

```bash
export WANDB_MODE=disabled
export HF_HOME="$HOME/ai6130-ass2/cache/huggingface"
source "$HOME/ai6130-ass2/revisions.env"

python finetune.py \
  --base_model=openlm-research/open_llama_3b_v2 \
  --model_revision="$OPENLLAMA_REVISION" \
  --data_path=./ft-training_set/math_7k.json \
  --output_dir="$HOME/ai6130-ass2/runs/M1_e1_manual" \
  --adapter_name=lora --num_epochs=1 --batch_size=4 --micro_batch_size=4 \
  --learning_rate=3e-4 --cutoff_len=256 --val_set_size=120 \
  --eval_step=100 --save_step=100 --seed=42

python evaluate.py \
  --base_model openlm-research/open_llama_3b_v2 \
  --model_revision "$OPENLLAMA_REVISION" \
  --adapter LoRA \
  --adapter_weights "$HOME/ai6130-ass2/runs/M1_e1_manual/final_adapter" \
  --dataset AQuA --output_dir "$HOME/ai6130-ass2/runs/M1_e1_manual/eval/AQuA" \
  --num_beams 4 --max_new_tokens 256 --scoring strict
```

AdapterP training replaces `--adapter_name=lora` with `--adapter_name=bottleneck --use_adapterp=True`. TinyLlama changes the model ID and corresponding pinned revision. Baseline evaluation omits `--adapter_weights`. Train resume adds `--resume_from_checkpoint=latest`; evaluation resume adds `--resume`. New scientific settings require a fresh output path. Direct commands print stdout; redirect it to a run-specific log if bypassing the batch launcher.

## 10. Validation and limits

Local tests use a separate temporary dependency environment, synthetic tiny models and no downloaded LLM weights. They check both adapter training/save/reload paths, exact CPU resumed-vs-uninterrupted adapter weights, evaluator interruption/continuation/coverage, answer parsing, settings identity, writer locking, damaged JSONL and checkpoint corruption rejection. Shell syntax and compute-guard refusal are checked locally.

**Still to verify on school:** editable custom PEFT with Linux bitsandbytes, A40 CUDA FP16, actual model/tokenizer revisions, memory/timing, SLURM warning delivery, compute filesystem locks/durability and recovery across two allocations. Do the smoke/resume sequence before launching the full matrix. No full-GPU experiment has been submitted automatically.

Validation record for this implementation (24 October 2026): **10 local tests passed** with PyTorch 2.2.2, Transformers 4.38.0, Accelerate 0.28.0 and editable bundled PEFT. The training integration test uses tiny random LLaMA models on CPU, checks both adapter methods and exercises scaler persistence with a disabled CPU test scaler. It does not establish actual CUDA FP16 recovery. Both script CLIs import successfully; shell syntax and rejection of an unallocated/head-node launch pass. Full model weights were not downloaded locally.
