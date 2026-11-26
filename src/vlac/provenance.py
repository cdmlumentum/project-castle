"""Computational identities, independent of output locations and timestamps."""
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
import os


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=32)
def _installation_file(path, size, modified):
    return file_digest(path)


def installation_identity(config):
    raw = config.solver.api_path or os.environ.get('LUMERICAL_PYTHON_API')
    if not raw:
        from importlib.util import find_spec
        try:
            spec = find_spec('lumapi')
            raw = spec.origin if spec is not None else None
        except (ValueError, ImportError):
            raw = None
    api = Path(raw or '')
    if api.is_file():
        api = api.parent
    def signature(path):
        if not path.is_file():
            return None
        stat = path.stat()
        return _installation_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    return {'api_sha256': signature(api/'lumapi.py'),
            'executable_sha256': signature(api.parent.parent/'bin/mode-solutions.exe')}


@lru_cache(maxsize=16)
def implementation_identity(family='optical'):
    from .block_io import digest
    names = ['geometry.py', 'materials.py', 'modes.py', 'solver.py', 'native_backend.py',
             'native/vlac_core.lsf']
    if family == 'workflow':
        names += ['profiles.py', 'optimize.py', 'profile_planning.py', 'adaptive_endpoint.py', 'characterize.py', 'propagate.py']
    return digest({name: sha256(files('vlac').joinpath(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                   for name in names})


def optical_identity(config):
    return {'implementation': implementation_identity(), 'installation': installation_identity(config)}


def checkpoint_identity(config, block, extra=None):
    from .block_io import digest, physics_signature
    return digest({'physics': physics_signature(config, block), 'optical': optical_identity(config),
                   'workflow': implementation_identity('workflow'), 'extra': extra})


def artifact_identity(config):
    return dict(optical_identity(config), workflow=implementation_identity('workflow'))
