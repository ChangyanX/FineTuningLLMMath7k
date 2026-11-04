#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=math7k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
# SLURM copies this script; SOURCE_DIR must refer to the uploaded immutable tree.
project=${PROJECT_ROOT:-"$HOME/ai6130-ass2"}
source_dir=${SOURCE_DIR:-"$project/source-v1"}
source "$source_dir/scripts/tc2/assert_compute.sh"
cd "$source_dir"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HOME="$project/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
action=${1:?Usage: sbatch job.sbatch ACTION [M1..M6] [EPOCHS] [TAG] [DATASET] [math_7k|math_10k]}
run=${2:-M1}
epochs=${3:-1}
tag=${4:-main}
dataset=${5:-AQuA}
case "$action" in train-eval|train-eval-smoke|eval-pair|baseline-pair) dataset=${5:-both} ;; esac
train_set=${6:-math_7k}
[[ "$run" =~ ^M[1-6]$ && "$epochs" =~ ^[1-9][0-9]*$ && "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
case "$dataset" in
 AQuA|AddSub|MultiArith|SingleEq|gsm8k|SVAMP) ;;
 both) case "$action" in train-eval|train-eval-smoke|eval-pair|baseline-pair) ;; *) exit 2 ;; esac ;;
 *) exit 2 ;;
esac
case "$train_set" in math_7k|math_10k) ;; *) exit 2 ;; esac
mkdir -p "$project/cache" "$project/runs" "$project/summary" "$project/exports"
module load anaconda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
if [[ "$action" == setup ]]; then
    [[ ! -e "$project/env" ]] || { echo 'Environment exists; do not mutate it while runs use it.' >&2; exit 2; }
    conda create -y -p "$project/env" python=3.10 pip
fi
set +u
conda activate "$project/env"
set -u
if [[ "$action" == setup ]]; then
    python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements-a40.txt
    python -m pip install --no-deps -e ./peft
    python -m pip freeze > "$project/environment.freeze.txt"
    exec python scripts/tc2/preflight.py
fi
if [[ "$action" == prepare ]]; then
    exec python scripts/tc2/prepare_models.py "$project"
fi
if [[ "$action" == prepare-extra ]]; then
    exec python scripts/tc2/prepare_models.py "$project" extra
fi
if [[ "$action" == check ]]; then
    python scripts/tc2/preflight.py
    exec python -m unittest discover -s tests -v
fi
if [[ "$action" == summary ]]; then
    exec python scripts/summarize.py "$project/runs" "$project/summary/results.csv"
fi
if [[ "$action" == export ]]; then
    # Only after all application writers have stopped; select the experiment explicitly.
    directory="$project/runs/${run}_e${epochs}_${tag}"
    [[ -d "$directory" ]] || exit 2
    exec python scripts/tc2/export_run.py "$project" "$source_dir" "$directory"
fi
[[ ! -f "$project/revisions.env" ]] || source "$project/revisions.env"
[[ ! -f "$project/revisions-extra.env" ]] || source "$project/revisions-extra.env"
if [[ "$action" == audit ]]; then
    exec python scripts/tc2/audit_data.py "$project"
fi
case "$run" in
 M1|M2) model=openlm-research/open_llama_3b_v2; revision=${OPENLLAMA_REVISION:?Set immutable model commit SHA after preflight} ;;
 M3|M4) model=TinyLlama/TinyLlama_v1.1; revision=${TINYLLAMA_REVISION:?Set immutable model commit SHA after preflight} ;;
 M5|M6) model=TinyLlama/TinyLlama-1.1B-Chat-v1.0; revision=${TINYLLAMA_CHAT_REVISION:?Run prepare-extra first} ;;
esac
case "$run" in
 M1|M3|M5) method=LoRA; adapter=(--adapter_name lora) ;;
 M2|M4|M6) method=AdapterP; adapter=(--adapter_name bottleneck --use_adapterp=True) ;;
esac
out="$project/runs/${run}_e${epochs}_${tag}"
resume_train=(); resume_eval=()
if [[ "${RESUME:-0}" == 1 ]]; then
    resume_train=(--resume_from_checkpoint=latest)
    resume_eval=(--resume)
fi
# Sequential stages share this one allocation; never submit child sbatch jobs.
case "$action" in
 train-eval|train-eval-smoke|eval-pair|baseline-pair)
    run_stage() {
        local stage=$1 benchmark=$2 directory stage_resume=0 child code stop_requested=0
        directory="$out"
        [[ "$stage" != baseline ]] || directory="$project/runs/${run}_baseline_${tag}"
        case "$stage" in eval|eval-smoke|baseline) directory="$directory/eval/$benchmark" ;; esac
        if [[ "${RESUME:-0}" == 1 && -f "$directory/settings.json" ]]; then
            stage_resume=1
            if [[ "$stage" == train || "$stage" == train-smoke ]] &&
                python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["state"] != "complete")' "$directory/status.json"; then
                echo "Already complete: $directory; continuing to evaluations"
                return 0
            fi
        fi
        RESUME=$stage_resume bash "$source_dir/scripts/tc2/job.sbatch" "$stage" "$run" "$epochs" "$tag" "$benchmark" "$train_set" &
        child=$!
        trap 'stop_requested=1; kill -USR1 "$child" 2>/dev/null || true' USR1 TERM INT
        set +e
        wait "$child"
        code=$?
        if kill -0 "$child" 2>/dev/null; then wait "$child"; code=$?; fi
        set -e
        trap - USR1 TERM INT
        if [[ "$stop_requested" == 1 && "$code" == 0 ]]; then code=75; fi
        return "$code"
    }
    eval_stage=eval
    case "$action" in
        train-eval) run_stage train AQuA ;;
        train-eval-smoke) run_stage train-smoke AQuA; eval_stage=eval-smoke ;;
        baseline-pair) eval_stage=baseline ;;
    esac
    if [[ "$dataset" != both ]]; then
        run_stage "$eval_stage" "$dataset"
        exit 0
    fi
    second=AddSub
    [[ "$train_set" != math_10k ]] || second=SVAMP
    run_stage "$eval_stage" AQuA
    run_stage "$eval_stage" "$second"
    exit 0
    ;;
esac
case "$action" in
 train|train-smoke)
    extra=()
    if [[ "$action" == train-smoke ]]; then
        [[ "$tag" != main ]] || { echo 'Use a separate smoke TAG'; exit 2; }
        extra=(--max_steps=4 --warmup_steps=0 --val_set_size=8 --eval_step=2 --save_step=2)
    fi
    mkdir -p "$out"
    python scripts/tc2/preflight.py > "$out/environment_${SLURM_JOB_ID}.json"
    exec python -u finetune.py --base_model="$model" --model_revision="$revision" \
        --data_path="./ft-training_set/${train_set}.json" --output_dir="$out" \
        --batch_size=4 --micro_batch_size=4 --num_epochs="$epochs" \
        --learning_rate=3e-4 --cutoff_len=256 --val_set_size=120 \
        --seed=42 --eval_step=100 --save_step=100 \
        "${adapter[@]}" ${extra[@]+"${extra[@]}"} ${resume_train[@]+"${resume_train[@]}"} \
        >> "$out/train_${SLURM_JOB_ID}.log" 2>&1
    ;;
 eval|eval-smoke|baseline)
    weights=(--adapter_weights "$out/final_adapter")
    extra=()
    if [[ "$action" == baseline ]]; then
        weights=()
        # M1/M3 only: one shared baseline per model, not one per adapter.
        [[ "$run" == M1 || "$run" == M3 ]] || exit 2
        out="$project/runs/${run}_baseline_${tag}"
    elif [[ "$action" == eval-smoke ]]; then
        [[ "$tag" != main ]] || exit 2
        extra=(--max_samples 5 --max_new_tokens 32)
    fi
    if [[ "$action" != baseline ]]; then
        python - "$out/settings.json" "$model" "$revision" "$train_set" "$dataset" <<'PY'
import json, pathlib, sys
s = json.load(open(sys.argv[1]))
assert s['base_model'] == sys.argv[2] and s['model_revision'] == sys.argv[3], 'Training model/revision mismatch'
assert pathlib.Path(s['data_path']).stem == sys.argv[4], 'Training dataset mismatch'
assert not (sys.argv[4] == 'math_10k' and sys.argv[5] in ('AddSub', 'MultiArith', 'SingleEq')), 'MAWPS overlap: use AQuA/SVAMP'
assert json.load(open(pathlib.Path(sys.argv[1]).with_name('status.json')))['state'] == 'complete', 'Training is not complete'
PY
    fi
    result="$out/eval/$dataset"
    mkdir -p "$result"
    python scripts/tc2/preflight.py > "$result/environment_${SLURM_JOB_ID}.json"
    exec python -u evaluate.py --base_model "$model" --model_revision "$revision" \
        --adapter "$method" ${weights[@]+"${weights[@]}"} --dataset "$dataset" \
        --output_dir "$result" --scoring strict --num_beams 4 \
        ${extra[@]+"${extra[@]}"} ${resume_eval[@]+"${resume_eval[@]}"} >> "$result/evaluate_${SLURM_JOB_ID}.log" 2>&1
    ;;
 *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
