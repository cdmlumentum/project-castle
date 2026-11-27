"""Checksummed numeric checkpoints. Loading never executes pickle content."""
from dataclasses import fields, is_dataclass
from io import BytesIO
import json
import numpy as np


def encode(value):
    arrays = {}
    def pack(obj):
        if isinstance(obj, np.ndarray):
            key = f'a{len(arrays)}'
            if obj.dtype.hasobject:
                raise TypeError('Object arrays are not portable checkpoints')
            arrays[key] = obj
            return {'array': key}
        if isinstance(obj, np.generic):
            return pack(obj.item())
        if isinstance(obj, complex):
            return {'complex': [obj.real, obj.imag]}
        if is_dataclass(obj):
            return {'class': type(obj).__name__, 'fields': {f.name: pack(getattr(obj, f.name)) for f in fields(obj)}}
        if isinstance(obj, dict):
            return {'dict': [[pack(k), pack(v)] for k, v in obj.items()]}
        if isinstance(obj, tuple):
            return {'tuple': [pack(v) for v in obj]}
        if isinstance(obj, list):
            return [pack(v) for v in obj]
        return obj
    metadata = json.dumps(pack(value), separators=(',', ':'), allow_nan=True).encode()
    arrays['metadata'] = np.frombuffer(metadata, dtype=np.uint8)
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def decode(payload):
    from .modes import ModeSnapshot, ModeReference, BareModeReference, Coefficients
    from .solver import BareModes
    from .profiles import ProfileSolution
    allowed = {cls.__name__: cls for cls in (ModeSnapshot, ModeReference, BareModeReference, Coefficients, BareModes, ProfileSolution)}
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        def unpack(obj):
            if isinstance(obj, list):
                return [unpack(v) for v in obj]
            if not isinstance(obj, dict):
                return obj
            if 'array' in obj:
                return arrays[obj['array']].copy()
            if 'complex' in obj:
                return complex(*obj['complex'])
            if 'tuple' in obj:
                return tuple(unpack(v) for v in obj['tuple'])
            if 'dict' in obj:
                return {unpack(k): unpack(v) for k, v in obj['dict']}
            if 'class' in obj:
                return allowed[obj['class']](**{k: unpack(v) for k, v in obj['fields'].items()})
            raise ValueError('Unsupported checkpoint encoding')
        return unpack(json.loads(arrays['metadata'].tobytes().decode()))


class Checkpoint:
    def __init__(self, directory, name, identity):
        from pathlib import Path
        self.path = Path(directory)/f'{name}-{identity}.checkpoint'

    def load(self):
        from hashlib import sha256
        if not self.path.is_file():
            return None
        checksum, payload = self.path.read_bytes().split(b'\n', 1)
        if sha256(payload).hexdigest().encode() != checksum:
            raise ValueError(f'Checkpoint checksum mismatch: {self.path}')
        return decode(payload)

    def save(self, value):
        from hashlib import sha256
        from pathlib import Path
        from tempfile import NamedTemporaryFile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = encode(value)
        temp = None
        try:
            # A short, unique sibling keeps atomic replacement without extending
            # the identity filename beyond Windows' legacy path-length limit.
            with NamedTemporaryFile(dir=self.path.parent, prefix='.checkpoint-',
                                    suffix='.tmp', delete=False) as stream:
                temp = Path(stream.name)
                stream.write(sha256(payload).hexdigest().encode()+b'\n'+payload)
            temp.replace(self.path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
