"""Native integration contracts; optical fixtures are synthetic, not device data."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import loadmat

from test_efficiency import geometry, fixture_modes
from test_python_pso import ROOT
from vlac import cli
from vlac.cache import SQLiteCache
from vlac.config import load_config
from vlac.materials import build_material_bank, material_samples
from vlac.modes import ModeReference, extract_coupling
from vlac.native_backend import NativeLumapiBackend, native_script, pack_references, snapshot_from_native
from vlac.native_mode import prepare_bundle
from vlac.profiles import evaluate_design_profiles


class NativeBackendTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/python_joint.json")

    def adapter(self, folder=None):
        backend = object.__new__(NativeLumapiBackend)
        backend.config = self.config
        backend.bank = build_material_bank(self.config)
        backend.mesh_accuracy = self.config.solver.mesh_accuracy
        backend.script_hash = "test_native_core"
        backend.cache = SQLiteCache(Path(folder)/"native.sqlite") if folder else None
        backend._build_domain()
        return backend

    def test_prepare_is_offline_and_embeds_exact_materials(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(NativeLumapiBackend, "__init__", side_effect=AssertionError("MODE must not launch")):
            bundle = prepare_bundle(self.config, folder, width_samples=7)
            values = loadmat(bundle/"native_config.mat", simplify_cells=True)["vlac_cfg"]
            expected = material_samples(build_material_bank(self.config).sin, np.linspace(1.26e-6, 1.34e-6, 81))
            np.testing.assert_array_equal(values["sin_table"], expected)
            self.assertEqual(len(values["endpoint_widths_m"]), 7)
            self.assertEqual(values["geometry"]["h_gap_m"], 1e-6)
            for number in (1, 2, 3):
                self.assertIn(f"vlac_block{number}", (bundle/f"block{number}.lsf").read_text())
            entry = (bundle/'block1.lsf').read_text()
            self.assertLess(entry.index('vlac_block1_inspection'),entry.rindex('save("block1.lms")'))
            self.assertNotIn('rm(', entry)
            self.assertLess(entry.index('incomplete_native_attempt'),entry.index('block1_native=vlac_block1'))
            self.assertIn('result.mode_profiles=profiles', (bundle/'vlac_blocks.lsf').read_text())
            manifest = json.loads((bundle/"manifest.json").read_text())
            self.assertEqual(manifest["standalone_algorithms"]["block2"], "coordinate_pattern")
            with self.assertRaises(FileExistsError):
                prepare_bundle(self.config, folder)

    def test_native_isolated_fresh_verification_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.adapter(folder)
            payload = {"mode": asdict(fixture_modes()[0]), "overlap": 1.}
            raw = {'ok': 1., 'error': '', 'n': 1., 'attempts': 1., 'items': [asdict(fixture_modes()[0])]}
            def dispatch(function, args, **kwargs):
                return {'ok': 1., 'n': 1., 'items': [raw]} if function == 'vlac_raw_batch' else payload
            with patch.object(backend, "_call_native", side_effect=dispatch) as call:
                for _ in range(2):
                    backend.solve_isolated_mode(geometry(self.config), 1.3e-6, guide="upper")
                self.assertEqual(call.call_count, 2)
                backend.solve_isolated_mode(geometry(self.config), 1.3e-6, guide="upper", use_cache=False)
                self.assertEqual(call.call_count, 4)

    def test_partial_raw_batch_preserves_success_and_distinguishes_field_failure(self):
        raw={'ok':1.,'error':'','attempts':1.,'n':2.,'items':[asdict(m) for m in fixture_modes()]}
        with tempfile.TemporaryDirectory() as folder:
            backend=self.adapter(folder)
            requests=[backend._raw_request(geometry(self.config,width=w),1.3e-6,True,False) for w in (1e-6,1.1e-6)]
            response={'n':2.,'items':[raw,{'ok':0.,'error':'unusable_fields'}]}
            with patch.object(backend,'_call_native',return_value=response):
                with self.assertRaisesRegex(RuntimeError,'unusable_fields'):backend._raw_packets(requests)
            with patch.object(backend,'_call_native',side_effect=AssertionError('successful sample repeated')):
                backend._native_memory.clear()
                self.assertEqual(backend._raw_packets(requests[:1])[0]['n'],2.)
            self.assertEqual(backend.cache.count(),1)

    def test_tracking_references_encode_complex_scalars_for_v221(self):
        from vlac.native_backend import reference_to_native
        from vlac.modes import ModeReference
        modes=fixture_modes()
        packet=reference_to_native(ModeReference(modes[0],modes[1],tuple(modes)))
        for value in packet.values():
            self.assertIsInstance(value['neff'],np.ndarray)
            self.assertEqual(value['neff'].shape,(1,1))

    def test_v73_numeric_cell_reader_preserves_orientation_and_complex_data(self):
        import h5py
        from vlac.mat_io import load_mat
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.mat'
            expected=np.arange(6).reshape(2,3)+2j
            with h5py.File(path,'w') as file:
                group=file.create_group('packet');group.attrs['MATLAB_class']=np.bytes_('struct')
                dtype=np.dtype([('real','<f8'),('imag','<f8')])
                data=np.empty((3,2),dtype=dtype);data['real']=expected.real.T;data['imag']=expected.imag.T
                field=group.create_dataset('field',data=data);field.attrs['MATLAB_class']=np.bytes_('double')
                text=group.create_dataset('label',data=np.array([ord(c) for c in 'SYNTHETIC'],dtype='u2').reshape(-1,1))
                text.attrs['MATLAB_class']=np.bytes_('char')
                cells=group.create_dataset('cells',(1,1),dtype=h5py.ref_dtype);cells.attrs['MATLAB_class']=np.bytes_('cell')
                cells[0,0]=field.ref
            packet=load_mat(path)['packet']
            np.testing.assert_array_equal(packet['field'],expected)
            np.testing.assert_array_equal(packet['cells'][0],expected)
            self.assertEqual(packet['label'],'SYNTHETIC')

    def test_native_reference_round_trip_preserves_complex_modes(self):
        # Synthetic lossy/complex fields exercise every guide and supermode slot.
        base = fixture_modes()[0]
        modes = [replace(base, neff=neff, ex=base.ex*(1+2j), ey=base.ex*(3-4j),
                         identity=f"synthetic-{index}")
                 for index, neff in enumerate((1.9+2e-7j, 1.8-3e-7j, 1.95+0j, 1.75))]
        reference = ModeReference(modes[0], modes[1], (modes[2], modes[3]))
        packed = pack_references((reference, reference))
        self.assertEqual(packed["n"], 2.)
        for item in packed["items"]:
            for name, original in zip(("upper", "lower", "plus", "minus"), modes):
                with self.subTest(slot=name):
                    encoded = item[name]
                    self.assertEqual(encoded["neff"].shape, (1, 1))
                    self.assertTrue(np.iscomplexobj(encoded["neff"]))
                    decoded = snapshot_from_native(encoded)
                    self.assertEqual(decoded.neff, original.neff)
                    self.assertEqual(decoded.identity, original.identity)
                    self.assertEqual(decoded.te_fraction, original.te_fraction)
                    self.assertEqual(decoded.core_fraction, original.core_fraction)
                    np.testing.assert_array_equal(decoded.ex, original.ex)
                    np.testing.assert_array_equal(decoded.ey, original.ey)
        self.assertEqual(pack_references(()), {"n": 0., "items": [0.]})

    def test_wavelength_tracking_crosses_the_real_native_serialization_boundary(self):
        # Unlike the cache test, exercise _call_native and each session.putv.
        # This stub enforces v221's rejection of Python complex scalar leaves.
        def supported(value):
            if isinstance(value, dict):
                for child in value.values():
                    supported(child)
            elif isinstance(value, list):
                for child in value:
                    supported(child)
            elif isinstance(value, complex):
                raise TypeError("Unsupported data type: complex scalar")

        backend = self.adapter()
        backend.progress = None
        mode = replace(fixture_modes()[0], neff=1.9+2e-7j)
        backend.session = Mock()
        backend.session.putv.side_effect = lambda name, value: supported(value)
        backend._native_loaded = True
        from vlac.native_backend import snapshot_to_native
        raw = {'ok':1.,'error':'','n':1.,'attempts':1.,'items':[snapshot_to_native(mode)]}
        def result(name):
            script = backend.session.eval.call_args.args[0]
            if 'vlac_raw_batch(' in script:
                return {'ok':1.,'n':1.,'solves':1.,'items':[raw]}
            return {"ok":1.,"mode":asdict(mode),"overlap":1.}
        backend.session.getv.side_effect = result
        first, _ = backend.solve_isolated_mode(geometry(self.config), 1.26e-6, guide="lower")
        tracked, overlap = backend.solve_isolated_mode(geometry(self.config), 1.34e-6,
                                                       guide="lower", reference=first)
        self.assertEqual(tracked.neff, mode.neff)
        self.assertEqual(overlap, 1.)
        self.assertEqual(sum('vlac_raw_batch(' in call.args[0] for call in backend.session.eval.call_args_list),2)
        self.assertEqual(sum('vlac_select_isolated(' in call.args[0] for call in backend.session.eval.call_args_list),2)

    def test_sub_float_display_changes_do_not_collide_in_native_cache(self):
        backend = self.adapter()
        first = backend._cache_identity(geometry(self.config, width=1e-6), 1.3e-6)
        second = backend._cache_identity(geometry(self.config, width=1e-6+1e-17), 1.3e-6)
        # Existing cache serialization rounded arrays; native keys include exact values.
        self.assertNotEqual(first, second)
        before = backend._cache_identity(geometry(self.config), 1.3e-6)
        backend.script_hash = "revised_native_core"
        self.assertNotEqual(before, backend._cache_identity(geometry(self.config), 1.3e-6))

    def test_profile_routes_whole_grid_to_native_backend(self):
        backend = self.adapter()
        marker = [[extract_coupling(4., 2., 5., 1.)]]
        with patch.object(backend, "evaluate_profile_grid", return_value=marker) as run:
            actual = evaluate_design_profiles(backend, self.config, backend.bank, None,
                                               np.arange(4), np.arange(4), np.arange(3), np.arange(3), state="bar")
        self.assertIs(actual, marker)
        self.assertEqual(run.call_args.kwargs["state"], "bar")

    def test_engine_selection_keeps_python_as_default(self):
        self.assertEqual(self.config.solver.engine, "python")
        with patch("vlac.cli.LumapiBackend", return_value="original"):
            self.assertEqual(cli._backend(self.config), "original")
        config = replace(self.config, solver=replace(self.config.solver, engine="native"))
        with patch("vlac.native_backend.NativeLumapiBackend", return_value="native"):
            self.assertEqual(cli._backend(config), "native")
        with self.assertRaisesRegex(ValueError, "solver.engine"):
            replace(self.config, solver=replace(self.config.solver, engine="invalid")).validate()

    def test_standalone_optimizer_rejects_unsupported_fabrication_constraints(self):
        config = replace(self.config, optimizer=replace(self.config.optimizer, maximum_slope_m_per_m=.01))
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "slope/curvature"):
            prepare_bundle(config, folder)

    def test_packaged_native_sources_define_all_three_blocks(self):
        self.assertIn("function vlac_profile(", native_script("vlac_core.lsf"))
        text = native_script("vlac_blocks.lsf")
        for number in (1, 2, 3):
            self.assertIn(f"function vlac_block{number}(", text)


if __name__ == "__main__":
    unittest.main()
