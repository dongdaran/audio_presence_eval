#!/usr/bin/env python3
"""Run queued full evaluations sequentially under Supervisor."""
import csv
import fcntl
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/random3000/logs'
CONFIG = ROOT / 'config/random3000.yaml'
JOBS = [
    ('cnn14', ['--task', 'cnn14'], 'cnn14_scores.tsv'),
    ('qwen', ['--task', 'llm', '--model', 'qwen3_omni_instruct'], 'llm_qwen3_omni_instruct.tsv'),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'started': now(), 'status': 'running', 'jobs': []}

        def save():
            temporary = RUN / 'status.json.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(RUN / 'status.json')

        save()
        for name, arguments, filename in JOBS:
            job = {'model': name, 'status': 'running', 'started': now(),
                   'output': str(ROOT / 'results/random3000' / filename), 'log': str(RUN / f'{name}.log')}
            state['jobs'].append(job)
            save()
            command = [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/run_experiment.py'), '--config', str(CONFIG), *arguments]
            print(f'{now()} START {name}: {command}', flush=True)
            with Path(job['log']).open('a') as log:
                log.write(f'\n{now()} START {command}\n')
                log.flush()
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job.update(returncode=completed.returncode, finished=now())
            output = Path(job['output'])
            if output.exists():
                with output.open(newline='') as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                job['rows'] = len(rows)
                job['statuses'] = dict(Counter(row.get('status', '') for row in rows))
                job['unique_samples'] = len({row['sample_id'] for row in rows})
            job['status'] = ('completed' if completed.returncode == 0
                             and job.get('rows') == 3000
                             and job.get('unique_samples') == 3000
                             and job.get('statuses', {}).get('ok', 0) > 0 else 'failed')
            save()
            print(f'{now()} FINISH {name}: {job}', flush=True)
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run_experiment.py'), '--config', str(CONFIG), '--task', 'report'], cwd=ROOT, check=True)
        state.update(status='completed' if all(j['status'] == 'completed' for j in state['jobs']) else 'failed', finished=now())
        save()
        if state['status'] == 'failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
