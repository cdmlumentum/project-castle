"""Read the numeric structs/cells written by MODE, without a solver session."""
from pathlib import Path
import numpy as np


def load_mat(path):
    import h5py
    from scipy.io import loadmat
    path = Path(path)
    if not h5py.is_hdf5(path):
        return loadmat(path, simplify_cells=True)
    with h5py.File(path, 'r') as file:
        visiting = set()
        def read(node):
            if node.name in visiting: raise ValueError('Cyclic MAT references are unsupported')
            visiting.add(node.name)
            try:
                kind = node.attrs.get('MATLAB_class', b'').decode('ascii')
                if isinstance(node, h5py.Group):
                    if kind != 'struct': raise ValueError('Unsupported native MAT group: '+kind)
                    return {name: read(value) for name,value in node.items()}
                if node.attrs.get('MATLAB_empty', False):
                    return '' if kind == 'char' else np.empty(tuple(int(v) for v in node[()]))
                data = node[()]
                if h5py.check_dtype(ref=node.dtype) is not None:
                    values = [read(file[ref]) if ref else None for ref in np.asarray(data).T.flatten(order='F')]
                    return values if kind == 'cell' or len(values)!=1 else values[0]
                if kind == 'char':
                    return np.asarray(data, dtype='<u2').T.flatten(order='F').tobytes().decode('utf-16-le')
                if data.dtype.names == ('real','imag'):
                    data = data['real']+1j*data['imag']
                if data.dtype.kind not in 'biufc': raise ValueError('Unsupported native MAT numeric type')
                data = np.asarray(data).T.copy()  # MATLAB and HDF5 dimension order differ.
                if kind == 'logical': data = data.astype(bool)
                return data.item() if data.size == 1 else data
            finally:
                visiting.remove(node.name)
        return {name:read(value) for name,value in file.items() if not name.startswith('#')}
