"""Budgets and immutable attempts shared by independent and full workflows."""
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import time
import uuid


class BudgetExceeded(RuntimeError):
    """A calculation stopped at a safe boundary; completed data remain reusable."""


class RunBudget:
    def __init__(self, settings):
        self.settings = settings
        self.started = time.monotonic()
        self.solves = 0

    def check(self, reserve=0):
        if ((self.settings.max_solves is not None and self.solves+reserve > self.settings.max_solves)
                or (self.settings.max_seconds is not None and
                    time.monotonic()-self.started >= self.settings.max_seconds)):
            raise BudgetExceeded('Solve/time budget exhausted between batches; rerun with resume enabled.')

    def consume(self, count):
        self.solves += int(count)


def resolved_cache(config):
    import os
    directory = os.environ.get('VLAC_CACHE_DIR')
    if directory:
        return Path(directory).expanduser().resolve()/Path(config.output.cache_sqlite).name
    return config.resolved_path(config.output.cache_sqlite)


def checkpoint_directory(config):
    return resolved_cache(config).parent/'checkpoints'


def resolve_published(directory):
    path = Path(directory).expanduser().resolve()
    pointer = path/'latest.json'
    if path.is_dir() and pointer.is_file():
        value = json.loads(pointer.read_text(encoding='utf-8'))
        target = (path/value['attempt']).resolve()
        if path not in target.parents or not target.is_dir():
            raise ValueError('Invalid published attempt path')
        return target
    return path


@contextmanager
def output_attempt(directory, command):
    """Publish latest.json only after success; root files are compatibility copies.

    Failed attempts never replace a prior successful result. The immutable
    attempt selected by latest.json is authoritative, including after a crash
    during compatibility-copy publication.
    """
    from .block_io import write_json
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = root/'.attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    previous = resolve_published(root)
    if previous != root:
        for source in previous.iterdir():
            if source.is_file() and source.name != 'attempt.json':
                shutil.copy2(source, attempt/source.name)
    write_json(attempt/'attempt.json', {'command': command, 'status': 'running'})
    try:
        yield attempt
    except BaseException as exc:
        write_json(attempt/'attempt.json', {'command': command, 'status': 'incomplete' if isinstance(exc, BudgetExceeded) else 'failed',
                                           'error': f'{type(exc).__name__}: {exc}'})
        write_json(root/'last_attempt.json', {'attempt': attempt.relative_to(root).as_posix(), 'status': 'incomplete' if isinstance(exc, BudgetExceeded) else 'failed',
                                            'error': str(exc)})
        # Failure diagnostics are additive; successful result files stay intact.
        for source in attempt.glob('*failure.json'):
            shutil.copy2(source, root/source.name)
        raise
    else:
        write_json(attempt/'attempt.json', {'command': command, 'status': 'complete',
                                           'completed_utc': datetime.now(timezone.utc).isoformat()})
        for source in attempt.iterdir():
            if source.is_file():
                temporary = root/(source.name+'.publishing')
                shutil.copy2(source, temporary)
                temporary.replace(root/source.name)
        pointer = {'attempt': attempt.relative_to(root).as_posix(), 'status': 'complete'}
        write_json(root/'latest.json', pointer)
        write_json(root/'last_attempt.json', pointer)


def overall_status(report):
    if not report:
        return 'review'
    own = report.get('status', 'review')
    upstream = report.get('upstream_validation')
    if own in {'failed', 'incomplete'}:
        return own
    if own != 'passed' or (upstream and overall_status(upstream) != 'passed'):
        return 'review'
    return 'passed'


def resume_source(config, output):
    """Find the furthest compatible completed stage, including interrupted attempts."""
    from .block_io import load_block
    from .provenance import artifact_identity
    root = Path(output)
    if root.parent.name == '.attempts': root = root.parent.parent
    folders = [p for p in (root/'.attempts').glob('*') if p.is_dir()]
    folders.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    folders.append(root)
    current = config.as_dict()
    for block in (3, 2, 1):
        for folder in folders:
            path = folder/f'block{block}_result.json'
            if not path.is_file(): continue
            try:
                *_, artifact = load_block(path, block, config)
                if artifact.get('provenance') != artifact_identity(config): continue
                keys = ['endpoint_search']+(['device', 'optimizer'] if block>=2 else [])+(['characterization'] if block>=3 else [])
                if any(artifact['config'].get(key) != current.get(key) for key in keys): continue
                return block+1, path
            except (ValueError, KeyError, OSError, TypeError):
                continue
    return None
