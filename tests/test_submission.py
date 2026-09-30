import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from amp_pilot.submission import main, publish, check_hash, verify_assets

ROOT = Path(__file__).resolve().parents[1]

class SubmissionTests(unittest.TestCase):
    def test_historical_methods_complete(self):
        recipes=json.loads((ROOT/'assets/sampling_recipes.json').read_text())
        recorded=json.loads((ROOT/'provenance/runs_production_v1_pool_base_complete.json').read_text())['prior_pools']
        serialized={r['kind']+'/'+r['variant']+'/eligible.fasta':r['files']['eligible.fasta'] for r in recipes}
        self.assertEqual(serialized,{p:r['hash'] for p,r in recorded.items()})
        self.assertEqual(sum(r['kind']=='flow' for r in recipes),5)
        self.assertEqual(sum(r['kind']=='ar' for r in recipes),4)
        self.assertTrue(any(r['kind']=='ar' and not r['settings'].get('generator_revision') for r in recipes))

    def test_stale_outputs_removed_on_prerequisite_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'generate';out.mkdir()
            for name in ['library.fasta','top.fasta','success.json']:(out/name).write_text('stale')
            with patch('amp_pilot.submission.verify_assets',side_effect=RuntimeError('bad checkpoint')):
                with self.assertRaisesRegex(RuntimeError,'bad checkpoint'):
                    main(['--assets-root',str(ROOT),'--output-dir',str(out)])
            self.assertFalse(any(out.iterdir()))

    def test_mismatch_does_not_publish_one_valid_file(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);lib=out/'candidate-library';top=out/'candidate-top'
            lib.write_text('new library');top.write_text('new top')
            expected={'library':hashlib.sha256(lib.read_bytes()).hexdigest(),'top':'0'*64}
            with patch('amp_pilot.submission.validate_outputs'):
                with self.assertRaisesRegex(RuntimeError,'approved top ranking'):
                    publish(lib,top,out,expected,ROOT/'data/antibacterial.fasta')
            self.assertFalse((out/'library.fasta').exists());self.assertFalse((out/'top.fasta').exists())

    def test_assets_are_pinned(self):
        verify_assets(ROOT)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weight';p.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'reproduction mismatch'):
                check_hash(p,'0'*64,'weight')

    def test_bootstrap_rejects_wrong_installer_without_execution(self):
        from amp_pilot.bootstrap import ensure_conda
        import platform
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();downloads=root/'external/downloads';downloads.mkdir(parents=True)
            pin={'platform':platform.system(),'architecture':platform.machine(),'version':'test','url':'https://example.invalid/installer','bytes':5,'sha256':'0'*64}
            (root/'assets/conda_bootstrap.json').write_text(json.dumps(pin))
            (downloads/'Miniforge3-test-Linux-x86_64.sh').write_bytes(b'wrong')
            with patch('amp_pilot.bootstrap.subprocess.run') as execute:
                with self.assertRaisesRegex(RuntimeError,'bootstrap installer reproduction mismatch'):
                    ensure_conda(root)
                execute.assert_not_called()

    def test_private_bootstrap_reuse_does_not_require_system_conda(self):
        from amp_pilot.bootstrap import ensure_conda
        import platform
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();prefix=root/'external/bootstrap/miniforge';(prefix/'bin').mkdir(parents=True)
            pin={'platform':platform.system(),'architecture':platform.machine(),'version':'test'}
            (root/'assets/conda_bootstrap.json').write_text(json.dumps(pin));(prefix/'amp-bootstrap.json').write_text(json.dumps(pin));(prefix/'bin/conda').write_text('private');(prefix/'bin/python').write_text('private-python')
            with patch('amp_pilot.bootstrap.shutil.which',side_effect=AssertionError('system Conda lookup')):
                self.assertEqual(ensure_conda(root),prefix/'bin/conda')

    def test_conda_launcher_uses_own_python_with_conflicting_path(self):
        from amp_pilot.bootstrap import prepare_launcher
        import os,subprocess
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/('long-prefix-'+'x'*140);(prefix/'bin').mkdir(parents=True)
            own=prefix/'bin/python';own.write_text('#!/bin/sh\necho private-python "$@"\n');own.chmod(0o755)
            launcher=prefix/'bin/conda';launcher.write_text('#!/usr/bin/env python\n');launcher.chmod(0o755)
            foreign=Path(d)/'uv-bin';foreign.mkdir();wrong=foreign/'python';wrong.write_text('#!/bin/sh\nexit 91\n');wrong.chmod(0o755)
            prepare_launcher(prefix)
            output=subprocess.check_output([str(launcher),'--version'],env={'PATH':str(foreign)+':/usr/bin:/bin'},text=True)
            self.assertEqual(output.strip(),'private-python -I -m conda --version')
            self.assertEqual((prefix/'bin/conda.amp-original').read_text(),'#!/usr/bin/env python\n')

    def test_model_helper_is_in_installed_package(self):
        import amp_pilot.production as production
        self.assertTrue(Path(production.__file__).with_name('run_battle.py').is_file())


class OfficialTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location('official_validator', ROOT/'scripts/verify_submission.py')
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)

    def fasta(self, directory, sequences, headers=None):
        path = Path(directory)/'fixture.fasta'
        headers = headers or [str(i) for i in range(len(sequences))]
        path.write_text(''.join('>'+h+'\n'+s+'\n' for h,s in zip(headers,sequences)))
        return path

    def test_library_all_template_constraints(self):
        v=self.validator
        with tempfile.TemporaryDirectory() as d, patch.object(v,'LIBRARY_SIZE',2):
            good=['A'*8,'C'*50]
            self.assertEqual(v._verify_sequences(self.fasta(d,good)),set(good))
            cases=[([],None),(['A'*8],None),(['A'*8,'A'*8],None),(['A'*7,'C'*50],None),(['A'*8,'C'*51],None),(['A'*7+'X','C'*50],None),(['','C'*50],None),(good,['','ok'])]
            for sequences,headers in cases:
                with self.subTest(sequences=sequences,headers=headers):
                    with self.assertRaises(ValueError):
                        v._verify_sequences(self.fasta(d,sequences,headers))

    def test_top_count_membership_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as d:
            library={'A'*8,'C'*8}
            self.validator._verify_top(self.fasta(d,list(library)),library,2)
            for top in [['A'*8],['A'*8,'D'*8],['A'*8,'A'*8]]:
                with self.subTest(top=top), self.assertRaises(ValueError):
                    self.validator._verify_top(self.fasta(d,top),library,2)

    def test_exact_reference_overlap(self):
        self.validator._verify_no_overlap({'A'*8},{'C'*8})
        with self.assertRaises(ValueError):
            self.validator._verify_no_overlap({'A'*8},{'A'*8})

    def test_levenshtein_reference_boundary_and_runtime_equivalence(self):
        import Levenshtein
        from rapidfuzz.distance import Indel
        reference='A'*10
        self.assertEqual(Levenshtein.ratio(reference,'A'*8+'CC'),.8)
        self.validator._veritfy_max_simularity({'A'*8+'CC'},{reference})
        with self.assertRaises(ValueError):
            self.validator._veritfy_max_simularity({'A'*9+'C'},{reference})
        for candidate in ['A'*8+'CC','A'*9+'C','A'*8,'C'*50,'ACDEFGHIK']:
            self.assertAlmostEqual(Levenshtein.ratio(candidate,reference),Indel.normalized_similarity(candidate,reference))

    def test_runtime_preserves_levenshtein_reference_rule(self):
        from amp_pilot.submission import validate_outputs
        for candidate,rejected in [('A'*8+'CC',False),('A'*9+'C',True)]:
            def read(path,*args):
                return {'library':[candidate]+['C'*10]*49999,'top':[candidate]*100,'reference':['A'*10]}[path]
            with self.subTest(candidate=candidate), patch('amp_pilot.common.read_fasta',side_effect=read), patch('amp_pilot.production.valid'):
                if rejected:
                    with self.assertRaisesRegex(ValueError,'Levenshtein ratio'):
                        validate_outputs('library','top','reference')
                else:
                    validate_outputs('library','top','reference')

    def test_approved_saved_outputs_against_complete_template_rules(self):
        v=self.validator
        saved=ROOT/'generate' if (ROOT/'generate/library.fasta').is_file() else ROOT/'reference'
        library=v._verify_sequences(saved/'library.fasta')
        v._verify_top(saved/'top.fasta',library,v.TOP_SIZE)
        _,refs=v._read_fasta(ROOT/'data/antibacterial.fasta')
        v._verify_no_overlap(library,set(refs))
        _,top=v._read_fasta(saved/'top.fasta')
        v._veritfy_max_simularity(set(top),set(refs))

    def test_validator_runs_generation_twice_and_checks_both_file_bytes(self):
        v=self.validator
        for changed in [None,'library.fasta','top.fasta']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as d:
                root=Path(d);out=root/'generate';out.mkdir()
                (out/'library.fasta').write_bytes(b'library')
                (out/'top.fasta').write_bytes(b'top')
                calls=[]
                def run(*args):
                    calls.append(1)
                    if len(calls)==2 and changed:
                        (out/changed).write_bytes(b'different')
                with patch.object(v,'_check_tool'),patch.object(v,'_clone_git_repository'),patch.object(v,'_sync_uv'),patch.object(v,'_verify_sequences',return_value=set()),patch.object(v,'_verify_top'),patch.object(v,'_uv_run',side_effect=run):
                    if changed:
                        with self.assertRaisesRegex(ValueError,'Reproducibility check failed'):
                            v.verify_setup(root,'https://example.invalid/repo')
                    else:
                        v.verify_setup(root,'https://example.invalid/repo')
                self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
