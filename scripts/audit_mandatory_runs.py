"""Audit downloaded mandatory or baseline/epoch runs without modifying run files."""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import extract_answer, correct

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ['M1_e1_retry2', 'M2_e1_main', 'M3_e1_main', 'M4_e1_retry1']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_evaluations(run, settings, baseline=False):
    name = run.name
    checks = {}
    evaluations = []
    for dataset, expected in [('AQuA', 254), ('AddSub', 395)]:
        directory = run / 'eval' / dataset
        config = read(directory / 'settings.json')
        metrics = read(directory / 'metrics.json')
        progress = read(directory / 'progress.json')
        es = read(directory / 'status.json')
        records = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        test = read(ROOT / 'dataset' / dataset / 'test.json')
        checks[f'{dataset}_coverage'] = (metrics['complete'] and not metrics['subset'] and config['max_samples'] is None
            and es['state'] == 'complete' and metrics['completed'] == metrics['total'] == es['completed'] == es['total'] == progress['completed'] == progress['total'] == expected
            and len(records) == len(test) == expected and [r['id'] for r in records] == list(range(expected)))
        checks[f'{dataset}_test_identity'] = (sha(ROOT / 'dataset' / dataset / 'test.json') == config['data_sha256']
            and all(r['label'] == t['answer'] and r['instruction'] == t['instruction'] for r, t in zip(records, test)))
        count = sum(correct(extract_answer(r['output_pred'], dataset, config['scoring']), r['label'], dataset) for r in records)
        failures = sum(r['prediction'] is None for r in records)
        checks[f'{dataset}_scoring'] = (all(extract_answer(r['output_pred'], dataset, config['scoring']) == r['prediction']
            and correct(r['prediction'], r['label'], dataset) == r['correct'] and r['parse_failed'] == (r['prediction'] is None) for r in records)
            and count == metrics['correct'] and failures == metrics['parse_failures']
            and math.isclose(100 * count / expected, metrics['accuracy_percent'])
            and math.isclose(sum(r['elapsed_seconds'] for r in records), metrics['generation_seconds']))
        checks[f'{dataset}_adapter_hashes'] = (config['lora_weights'] is None and config['adapter_files'] is None) if baseline else all((run / 'final_adapter' / f).is_file() and sha(run / 'final_adapter' / f) == h for f, h in config['adapter_files'].items())
        checks[f'{dataset}_settings'] = ((config['lora_weights'] is None if baseline else Path(config['lora_weights']).parent.name == name)
            and config['base_model'] == settings['base_model'] and config['model_revision'] == settings['model_revision']
            and read(directory / 'resolved_model.json')['revision'] == settings['model_revision']
            and config['source'] == settings['source'] and config['scoring'] == 'strict'
            and config['num_beams'] == 4 and config['max_new_tokens'] == 256 and not config['use_cache'])
        evaluations.append(dict(dataset=dataset, **metrics, empty_outputs=sum(not r['output_pred'].strip() for r in records)))
    return checks, evaluations


def audit(selected=SELECTED):
    results = []
    for name in selected:
        run = ROOT / 'runs' / name
        settings = read(run / 'settings.json')
        status = read(run / 'status.json')
        resolved = read(run / 'resolved_model.json')
        state = read(run / 'trainer_state.json')
        split = read(run / 'split.json')
        rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        grads = [float(row['grad_norm']) for row in rows if 'grad_norm' in row]
        vals = [row['eval_loss'] for row in rows if 'eval_loss' in row]
        checks = {}
        checks['training_complete'] = (status['state'] == 'complete' and state['epoch'] == settings['num_epochs']
            and state['global_step'] == status['step'] == 1683 * settings['num_epochs'])
        ids = split['train_ids'] + split['validation_ids']
        checks['split_complete_disjoint'] = len(split['train_ids']) == 6731 and len(split['validation_ids']) == 120 and sorted(ids) == list(range(6851))
        checks['training_data_hash'] = sha(ROOT / settings['data_path']) == settings['data_sha256']
        checks['train_revision'] = settings['model_revision'] == resolved['revision']
        checkpoints = list(run.glob('checkpoint-*'))
        checked_files = 0
        for directory in checkpoints:
            manifest = read(directory / 'checkpoint_manifest.json')
            checks[f'{directory.name}_manifest'] = all((directory / f).is_file() and sha(directory / f) == h for f, h in manifest.items())
            checked_files += len(manifest)
        checks['best_checkpoint_present'] = (run / Path(status['best_checkpoint']).name / 'trainer_state.json').is_file()
        eval_checks, evaluations = audit_evaluations(run, settings)
        checks.update(eval_checks)
        results.append(dict(run=name, checks=checks, all_integrity_checks_pass=all(checks.values()),
            manifest_files_checked=checked_files, split_sha256=sha(run / 'split.json'),
            source_mismatches=[f for f, h in settings['source'].items() if not (ROOT / f).exists() or sha(ROOT / f) != h],
            nonfinite_grad_logs=sum(not math.isfinite(v) for v in grads), zero_grad_logs=sum(v == 0 for v in grads), grad_logs=len(grads),
            first_validation_loss=vals[0], last_validation_loss=vals[-1], best_validation_loss=state['best_metric'],
            epochs=settings['num_epochs'], best_checkpoint=Path(status['best_checkpoint']).name, training_seconds=status['elapsed_seconds'],
            peak_gpu_gib=status['peak_gpu_bytes'] / 2**30, trainable_parameters=resolved['trainable_parameters'],
            training_gpu=read(next(run.glob('environment_*.json')))['gpu'], evaluations=evaluations))
    return results


if __name__ == '__main__':
    extensions = '--extensions' in sys.argv
    selected = [f'M{i}_e{epoch}_epochs' for epoch in (2, 3) for i in range(1, 5)] if extensions else SELECTED
    results = audit(selected)
    if extensions:
        for name, model, variable in [('M1_baseline_main', 'openlm-research/open_llama_3b_v2', 'OPENLLAMA_REVISION'),
                                      ('M3_baseline_main', 'TinyLlama/TinyLlama_v1.1', 'TINYLLAMA_REVISION')]:
            run = ROOT / 'runs' / name
            settings = read(run / 'eval/AQuA/settings.json')
            checks, evaluations = audit_evaluations(run, settings, baseline=True)
            checks['base_model_pin'] = settings['base_model'] == model and settings['model_revision'] == read(ROOT / 'models.json')[variable]
            results.append(dict(run=name, baseline=True, checks=checks, all_integrity_checks_pass=all(checks.values()), evaluations=evaluations))
    if '--tensors' in sys.argv:
        import torch
        for result in results:
            if result.get('baseline'):
                continue
            run = ROOT / 'runs' / result['run']
            final = torch.load(run / 'final_adapter/adapter_model.bin', map_location='cpu', weights_only=True)
            epoch = torch.load(run / f'epochs/epoch-{result["epochs"]}/adapter_model.bin', map_location='cpu', weights_only=True)
            best = torch.load(run / result['best_checkpoint'] / 'pytorch_model.bin', map_location='cpu', weights_only=True)
            result['checks']['final_adapter_matches_best_checkpoint'] = (final.keys() == best.keys()
                and all(torch.equal(v, best[k]) for k, v in final.items()))
            result['all_integrity_checks_pass'] = all(result['checks'].values())
            result['tensors'] = dict(count=len(final),
                nonfinite=sum(not bool(torch.isfinite(v).all()) for v in final.values()),
                zero=sum(not bool(torch.count_nonzero(v)) for v in final.values()),
                changed_final_vs_last_epoch=sum(not torch.equal(v, epoch[k]) for k, v in final.items()))
            checkpoints = sorted(run.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            result['checkpoint_diagnostics'] = []
            for directory in (checkpoints[0], checkpoints[-1]):
                scheduler = torch.load(directory / 'scheduler.pt', map_location='cpu', weights_only=True)
                scaler = torch.load(directory / 'scaler.pt', map_location='cpu', weights_only=True)
                result['checkpoint_diagnostics'].append(dict(checkpoint=directory.name,
                    scheduler_last_epoch=scheduler['last_epoch'], fp16_scale=scaler['scale']))
    output = ROOT / 'docs' / ('BASELINE_EPOCH_RUN_AUDIT.json' if extensions else 'MANDATORY_RUN_AUDIT.json')
    output.write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        print(result['run'], 'integrity:', result['all_integrity_checks_pass'], 'nonfinite/zero gradient logs:', result.get('nonfinite_grad_logs', 'N/A'), result.get('zero_grad_logs', 'N/A'))
        for check, passed in result['checks'].items():
            if not passed:
                print('  FAILED:', check)
    print('Wrote', output)
    if not all(result['all_integrity_checks_pass'] for result in results):
        raise SystemExit(1)
