"""Fresh inference replay of the approved seed-42 method; fail closed on mismatch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected, label):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f'{label} reproduction mismatch: {actual} != {expected}')


def verify_assets(root):
    for name, expected in json.loads((root/'assets/hashes.json').read_text()).items():
        check_hash(root/name, expected, name)


def verify_battle(repo, pins):
    for name, pin in pins.items():
        folder = repo if name == 'battle' else repo/'models'/name
        head = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if head != pin['commit']:
            raise RuntimeError(f'{name} commit mismatch: {head}')
        for name, expected in pin['files'].items():
            check_hash(folder/name, expected, f'BATTLE {folder.name}/{name}')


def validate_outputs(library, top, reference):
    from .common import read_fasta
    from .production import valid
    from rapidfuzz.distance import Indel
    lib = read_fasta(library, False)
    ranked = read_fasta(top, False)
    refs = set(read_fasta(reference, False))
    valid(lib, 'library'); valid(ranked, 'top')
    if len(lib) != 50000 or len(ranked) != 100 or not set(ranked) <= set(lib):
        raise ValueError('Expected 50,000 unique peptides and 100 ranked library members')
    if set(lib) & refs:
        raise ValueError('Library overlaps official antibacterial reference')
    # Levenshtein.ratio is normalized Indel similarity. Strict >0.8 only.
    for seq in ranked:
        for ref in refs:
            if Indel.normalized_similarity(seq, ref) > .8:
                raise ValueError('Top peptide exceeds official Levenshtein ratio: '+seq)


def publish(library, top, output, expected, reference):
    validate_outputs(library, top, reference)
    check_hash(library, expected['library'], 'approved library')
    check_hash(top, expected['top'], 'approved top ranking')
    # Publish only files produced by this invocation, after both checks pass.
    shutil.copyfile(library, output/'library.fasta.tmp')
    shutil.copyfile(top, output/'top.fasta.tmp')
    (output/'library.fasta.tmp').replace(output/'library.fasta')
    (output/'top.fasta.tmp').replace(output/'top.fasta')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=Path.cwd())
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('generate'))
    parser.add_argument('--battle-repo', type=Path, default=None)
    parser.add_argument('--snakemake', default=None)
    parser.add_argument('--seed', type=int, default=42, choices=[42])
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    parser.add_argument('--workers', default=4, type=int, choices=[4])
    parser.add_argument('--stage', default='all', choices=['all','generate','score','select'])
    a = parser.parse_args(argv)
    root = a.assets_root.resolve(); output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous successful outputs BEFORE any prerequisite can fail.
    for name in ['library.fasta','top.fasta','library.fasta.tmp','top.fasta.tmp','success.json']:
        (output/name).unlink(missing_ok=True)
    verify_assets(root)
    expected = json.loads((root/'assets/expected.json').read_text())
    work = a.work_dir.resolve() if a.work_dir else Path(tempfile.mkdtemp(prefix='replay-', dir=root))
    if a.stage in ['all','generate']:
        if work.exists() and any(work.iterdir()):
            raise RuntimeError('Generation requires an empty work directory; resume is not a fresh replay')
        work.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root/'assets/pilot', work/'pilot')
        for kind in ['flow','ar']:
            shutil.copyfile(root/'checkpoint'/kind/'best.pt',work/'pilot'/kind/'best.pt')
    elif not (work/'pilot/manifest.json').exists():
        raise RuntimeError('Staged replay requires --work-dir containing a completed generation stage')
    pilot = work/'pilot'; prod = work/'production'; selection = work/'hybrid'; exchange = work/'exchange'
    from .common import seed_all, json_write, read_fasta, write_fasta
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(a.workers); seed_all(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for faithful replay; run on Linux/WSL with GPU access')
    if a.stage in ['all','score'] and a.battle_repo is None and a.snakemake is None:
        if not (root/'external/setup_verified.json').exists():
            from .setup_battle import main as setup
            setup(['--root',str(root)])
        from .bootstrap import ensure_conda, scorer_environment
        conda=ensure_conda(root)
        scorer_env=scorer_environment(root,conda)
        for key in ['CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_SHLVL','CONDA_EXE','CONDA_PYTHON_EXE','PYTHONHOME','PYTHONPATH']:
            os.environ.pop(key,None)
        # production.score_candidates inherits these values for every model batch.
        for key in ['PATH','CONDA_PKGS_DIRS','PIP_CACHE_DIR','XDG_CACHE_HOME']:
            os.environ[key]=scorer_env[key]
    # Put the checksum-pinned ESM files in a per-replay torch cache.
    hub = work/'torch/hub'; (hub/'checkpoints').mkdir(parents=True, exist_ok=True)
    for p in (root/'assets/esm').glob('*.pt'):
        shutil.copyfile(p, hub/'checkpoints'/p.name)
    torch.hub.set_dir(str(hub))
    from . import production, selector_lab, hybrid_refine
    from . import generate, legacy_generate
    from types import SimpleNamespace
    if a.stage in ['all','generate']:
        recipes = json.loads((root/'assets/sampling_recipes.json').read_text())
        for recipe in recipes:
            settings = recipe['settings']
            args = SimpleNamespace(out=str(pilot), samples=settings['samples'], sample_batch=settings['batch'], seed=settings['seed']-1000, device=a.device)
            runner = generate if settings.get('generator_revision') == 2 else legacy_generate
            dest = runner.generate(args, recipe['kind'], settings['variant'])
            for name, digest in recipe['files'].items():
                check_hash(dest/name, digest, 'historical '+recipe['kind']+'/'+settings['variant']+'/'+name)
        production.main(['--pilot',str(pilot),'--out',str(prod),'--stage','generate'])
        for recipe in json.loads((root/'assets/production_recipes.json').read_text()):
            dest = prod/'sampling'/recipe['kind']/recipe['variant']
            actual_settings = json.loads((dest/'generation.json').read_text())['settings']
            if actual_settings != recipe['settings']:
                raise RuntimeError('Production sampling settings changed: '+recipe['variant'])
            for name,digest in recipe['files'].items():
                check_hash(dest/name,digest,'production '+recipe['variant']+'/'+name)
        check_hash(prod/'candidates.fasta',expected['pool'],'candidate pool')
        json_write(work/'generation_verified.json', {'pool': sha(prod/'candidates.fasta')})
        if a.stage == 'generate': return
    if a.stage in ['all','score']:
        if not (work/'generation_verified.json').exists():
            raise RuntimeError('Scoring requires a verified inference generation stage')
        check_hash(prod/'candidates.fasta', expected['pool'], 'candidate pool')
        repo = (a.battle_repo or root/'external/battle').resolve()
        snake = a.snakemake or str(root/'external/envs/pipeline/bin/snakemake')
        if not Path(snake).is_file() and a.battle_repo is None and a.snakemake is None:
            from .setup_battle import main as setup
            setup(['--root',str(root)])
        if not Path(snake).is_file():
            raise RuntimeError('Pinned Snakemake executable missing: '+snake)
        verify_battle(repo, json.loads((root/'assets/battle_pins.json').read_text()))
        env = os.environ.copy(); env['PATH'] = str(Path(snake).resolve().parent)+os.pathsep+env.get('PATH','')
        # Re-infer the historical pilot input in its original batch, then let the
        # original production routine infer the remaining 20,000-sequence batches.
        historical = pilot/'experiments_v3'; historical.mkdir(exist_ok=True)
        scoring_recipe = json.loads((root/'assets/scoring/recipe.json').read_text())
        historical_sequences = sorted({s for pool in scoring_recipe['pools'] for s in read_fasta(pilot/pool/'eligible.fasta',False)})
        scoring_input = historical/'scoring_input.fasta'
        write_fasta(scoring_input,historical_sequences)
        check_hash(scoring_input,scoring_recipe['sha256'],'historical scoring batch')
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_battle.py')),'--repo',str(repo),'--fasta',str(scoring_input),'--output',str(historical/'battle_scores.csv'),'--models','amplify,mbc-attention','--cores','4'],env=env,check=True)
        production.main(['--pilot',str(pilot),'--out',str(prod),'--stage','score','--battle-repo',str(repo),'--snakemake',snake])
        check_hash(prod/'scores.csv',expected['scores'],'fresh predictions')
        json_write(work/'scoring_verified.json', {'scores': sha(prod/'scores.csv')})
        if a.stage == 'score': return
    if not (work/'scoring_verified.json').exists():
        raise RuntimeError('Selection requires a verified fresh scoring stage')
    if (prod/'selection_complete.json').exists() or selection.exists() or exchange.exists():
        raise RuntimeError('Selection must be fresh; existing selector state cannot prove reproduction')
    check_hash(prod/'scores.csv',expected['scores'],'fresh predictions')
    production.main(['--pilot',str(pilot),'--out',str(prod),'--stage','select'])
    check_hash(prod/'baseline.fasta',expected['baseline'],'reference-balanced baseline')
    selector_lab.main(['--pilot',str(pilot),'--production',str(prod),'--out',str(selection),'--policies','hybrid','--target','0.80','--repair-budget','3000'])
    check_hash(selection/'hybrid/library.fasta',expected['hybrid'],'original hybrid')
    hybrid_refine.main(['--pilot',str(pilot),'--production',str(prod),'--source',str(selection),'--out',str(exchange),'--modes','exchange','--potency-floor','0.88','--epochs','60'])
    lib = exchange/'exchange/library.fasta'
    check_hash(lib,expected['library'],'exchange library')
    # Preserve the approved original top_candidates ranking and tie breaking.
    from .production_select import max_similarity
    seqs = read_fasta(lib,False)
    known = sorted(set(read_fasta(pilot/'all_positive.fasta')) | set(read_fasta(pilot/'forbidden.fasta',False)))
    known = [s for s in known if s and set(s)<=set('ACDEFGHIKLMNPQRSTVWY')]
    with threadpool_limits(limits=4):
        top = production.top_candidates(seqs, production.csv_rows(prod/'scores.csv'), max_similarity(seqs,known,4),100)
    write_fasta(work/'top.fasta',top)
    publish(lib,work/'top.fasta',output,expected,root/'data/antibacterial.fasta')
    json_write(output/'success.json',{'fresh_replay':True,'work':str(work),'library':sha(output/'library.fasta'),'top':sha(output/'top.fasta')})

if __name__ == '__main__':
    main()
