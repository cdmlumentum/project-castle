"""Common-background contracts; all optical values/fields here are SYNTHETIC."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_efficiency import backend_without_mode, geometry
from test_python_pso import ROOT, SyntheticBackend
from vlac.adaptive_endpoint import WidthEvaluator, search_endpoints_adaptive
from vlac.block_io import assert_compatible, physics_signature, write_artifact, load_block
from vlac.config import config_from_dict, load_config
from vlac.endpoint_diagnostics import export_mode_profiles
from vlac.materials import build_material_bank
from vlac.native_backend import NativeLumapiBackend
from vlac.native_mode import config_payload, prepare_bundle
from vlac.python_pso import EndpointEvaluator


class LiquidSensitiveLower(SyntheticBackend):
    def solve_isolated_mode(self, g, wavelength, **kwargs):
        mode, overlap = super().solve_isolated_mode(g, wavelength, **kwargs)
        if kwargs['guide'] == 'lower' and g.state == 'bar':
            mode.neff += .002  # Deliberate SYNTHETIC sensitivity, not film data.
        return mode, overlap


class SlabReferenceTests(unittest.TestCase):
    def setUp(self):
        self.legacy = load_config(ROOT/'config/python_joint.json')
        self.config = replace(self.legacy,
            solver=replace(self.legacy.solver, isolated_reference='retained_slabs_v1'),
            endpoint_search=replace(self.legacy.endpoint_search, pso_lower_mode_state_independent=False))
        self.bank = build_material_bank(self.config)

    def test_legacy_loading_and_corrected_default_configs(self):
        self.assertEqual(config_from_dict({}).solver.isolated_reference, 'removed_guide_v23')
        for name in ('default', 'native', 'native_sweep'):
            config = load_config(ROOT/f'config/{name}.json')
            self.assertEqual(config.solver.isolated_reference, 'retained_slabs_v1')
            self.assertFalse(config.endpoint_search.pso_lower_mode_state_independent)
            self.assertEqual(config.optimizer.method, 'sweep_anchored_v1')

    def test_unsafe_cross_state_reuse_and_unknown_reference_rejected(self):
        bad = replace(self.config, endpoint_search=replace(self.config.endpoint_search,
                                                          pso_lower_mode_state_independent=True))
        with self.assertRaisesRegex(ValueError, 'retained_slabs_v1 requires'):
            bad.validate()
        bad = replace(self.config, solver=replace(self.config.solver, isolated_reference='unknown'))
        with self.assertRaisesRegex(ValueError, 'isolated_reference'):
            bad.validate()

    def test_only_ridge_is_removed_and_full_geometry_is_unchanged(self):
        legacy = backend_without_mode(self.legacy, self.bank)
        fixed = backend_without_mode(self.config, self.bank)
        for enabled in ((True, False), (False, True), (True, True)):
            actual = {r.name for r in fixed._active_rectangles(geometry(self.config), *enabled)}
            self.assertTrue({'upper_slab', 'lower_slab'} <= actual)
            self.assertEqual('upper_ridge' in actual, enabled[0])
            self.assertEqual('lower_ridge' in actual, enabled[1])
        self.assertEqual(fixed._active_rectangles(geometry(self.config), True, True),
                         legacy._active_rectangles(geometry(self.config), True, True))
        self.assertNotIn('lower_slab', {r.name for r in legacy._active_rectangles(geometry(self.config), True, False)})
        strip = replace(geometry(self.config), h_rib_m=0.)
        self.assertFalse(any(r.name.endswith('_slab') for r in fixed._active_rectangles(strip, True, False)))

    def test_python_scene_updates_slab_enable_flags(self):
        class SceneCaptured(Exception):
            pass
        backend = backend_without_mode(self.config, self.bank)
        scripts = []
        def capture(script):
            scripts.append(script)
            if 'addfde;' in script:
                raise SceneCaptured()
        backend.session = SimpleNamespace(eval=capture)
        with patch.object(backend, '_install_materials', return_value='Liquid'):
            with self.assertRaises(SceneCaptured):
                backend._mode_candidates(geometry(self.config), 1.26e-6, include_upper=True, include_lower=False)
        scene = scripts[-1]
        self.assertIn('setnamed("lower_slab","enabled",1);', scene)
        self.assertIn('setnamed("upper_slab","enabled",1);', scene)
        self.assertIn('setnamed("lower_ridge","enabled",0);', scene)
        self.assertIn('setnamed("upper_ridge","enabled",1);', scene)

    def test_native_payload_and_all_cache_layers_distinguish_reference(self):
        old, new = config_payload(self.legacy), config_payload(self.config)
        self.assertEqual((old['retain_both_slabs'], new['retain_both_slabs']), (0., 1.))
        self.assertEqual(new['endpoint_lower_state_independent'], 0.)
        def adapter(config):
            backend = object.__new__(NativeLumapiBackend)
            backend.config, backend.bank = config, self.bank
            backend.mesh_accuracy, backend.script_hash = config.solver.mesh_accuracy, 'SYNTHETIC cache identity'
            backend._build_domain()
            return backend
        a, b = adapter(self.legacy), adapter(self.config)
        request = a._raw_request(geometry(self.config), 1.26e-6, True, False)
        self.assertNotEqual(a._raw_key(request), b._raw_key(request))
        self.assertNotEqual(a._cache_identity(geometry(self.config), 1.26e-6),
                            b._cache_identity(geometry(self.config), 1.26e-6))

    def test_versioned_handoffs_reject_reference_mixing(self):
        import test_sequential_profiles
        fixture = test_sequential_profiles.SequentialTests()
        fixture.setUp()
        legacy = fixture.config
        fixed = replace(legacy, solver=replace(legacy.solver, isolated_reference='retained_slabs_v1'),
                        endpoint_search=replace(legacy.endpoint_search, pso_lower_mode_state_independent=False))
        self.assertNotIn('isolated_reference', physics_signature(legacy, 1))
        for source, target in ((legacy, fixed), (fixed, legacy)):
            with tempfile.TemporaryDirectory() as folder:
                write_artifact(folder, 1, source, fixture.endpoint)
                self.assertEqual(load_block(folder, 1, source)[0].w1_input_m, fixture.endpoint.w1_input_m)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    load_block(folder, 1, target)

    def test_lower_modes_are_reused_by_state_and_not_by_width(self):
        backend = LiquidSensitiveLower(self.config)
        evaluator = WidthEvaluator(backend, self.config, self.bank)
        evaluator.precompute_lower()
        first = evaluator.evaluate_width(.7e-6)
        evaluator.evaluate_width(1.6e-6)
        lower_calls = [c for c in backend.calls if c[0] == 'lower']
        self.assertEqual(len(lower_calls), 4)
        self.assertEqual({c[1].state for c in lower_calls}, {'cross', 'bar'})
        rows = {(r['state'], r['wavelength_m']): r for r in first['records']}
        for wavelength in evaluator.wavelengths:
            difference = rows['bar', wavelength]['beta_lower_real_rad_per_m']-rows['cross', wavelength]['beta_lower_real_rad_per_m']
            self.assertAlmostEqual(difference, 2*np.pi*.002/wavelength, places=6)

    def test_fresh_verification_retains_separate_state_fields(self):
        backend = LiquidSensitiveLower(self.config)
        evaluator = EndpointEvaluator(backend, self.config, self.bank)
        evaluator.precompute_lower(use_cache=False)
        _, rows = evaluator.evaluate((.7e-6, 1.6e-6), use_cache=False)
        self.assertEqual(len(backend.calls), 12)
        self.assertEqual(len(evaluator.mode_profiles), 12)
        self.assertTrue(all(not p['shared_states'] for p in evaluator.mode_profiles if p['guide'] == 'lower'))
        with tempfile.TemporaryDirectory() as folder:
            index = export_mode_profiles(folder, evaluator.mode_profiles, SimpleNamespace(endpoint_records=rows))
            self.assertEqual(len(index), 12)

    def test_adaptive_endpoint_supports_state_specific_lower_reference(self):
        config = replace(self.config, endpoint_search=replace(self.config.endpoint_search, adaptive_max_widths=13))
        backend = LiquidSensitiveLower(config)
        with tempfile.TemporaryDirectory() as folder:
            result = search_endpoints_adaptive(backend, config, self.bank, validation_only=True, output_dir=folder)
            report = json.loads(Path(folder, 'endpoint_adaptive_verification.json').read_text())
        self.assertFalse(result.search_summary['lower_mode_state_independent'])
        self.assertEqual(report['fresh_mode_solves'], 12)
        self.assertEqual(len([c for c in backend.calls if not c[3]]), 12)

    def test_standalone_rejects_unsupported_lower_reference_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'standalone Block 1'):
                prepare_bundle(self.config, folder)


if __name__ == '__main__':
    unittest.main()
