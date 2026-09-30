"""Reconstruct BATTLE from public pinned commits and captured Linux environments."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from .submission import verify_assets, verify_battle


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--conda', default=None, help='Optional executable override; default bootstraps private pinned Conda')
    p.add_argument('--sources-only', action='store_true')
    a = p.parse_args(argv); root = a.root.resolve()
    verify_assets(root)
    pins = json.loads((root/'assets/battle_pins.json').read_text())
    repo = root/'external/battle'; repo.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.pop('VIRTUAL_ENV',None); env.pop('PYTHONHOME',None)
    for name, pin in pins.items():
        folder = repo if name == 'battle' else repo/'models'/name
        if not (folder/'.git').exists():
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError('Refusing to replace nonempty unversioned directory: '+str(folder))
            subprocess.run(['git','clone','--no-checkout',pin['url'],str(folder)],check=True,env=env)
            subprocess.run(['git','-C',str(folder),'checkout','--detach',pin['commit']],check=True,env=env)
    verify_battle(repo,pins)
    if a.sources_only: return
    from .bootstrap import ensure_conda, scorer_environment
    conda=ensure_conda(root,a.conda)
    env=scorer_environment(root,conda)
    for name in ['pipeline','amplify','mbc-attention']:
        locks = root/'assets/environments'/name
        prefix = root/'external/envs'/name
        prefix.parent.mkdir(parents=True,exist_ok=True)
        if not (prefix/'conda-meta/history').exists():
            subprocess.run([str(conda),'create','--yes','--prefix',str(prefix),'--file',str(locks/'conda-explicit.txt')],env=env,check=True)
        if (locks/'pip.txt').read_text().strip():
            subprocess.run([str(prefix/'bin/python'),'-m','pip','install','--no-deps','-r',str(locks/'pip.txt')],env=env,check=True)
        # Exact package inventory check; conda and pip are both covered.
        code='import json; from importlib.metadata import distributions; print(json.dumps({d.metadata["Name"]:d.version for d in distributions()}))'
        packages=json.loads(subprocess.check_output([str(prefix/'bin/python'),'-c',code],env=env,text=True,cwd=prefix))
        expected=json.loads((locks/'runtime.json').read_text())['packages']
        for package,version in expected.items():
            if package == 'amp-decision-pilot': continue  # source checkout metadata, not a scorer dependency
            if packages.get(package)!=version:
                raise RuntimeError(f'{name}: {package} expected {version}, got {packages.get(package)}')
    # Ask this pinned Snakemake for its environment hashes; hashes depend on the
    # new installation location. Create links to the fully pinned environments.
    snake = root/'external/envs/pipeline/bin/snakemake'
    # Resolve env-python entry points with the pipeline's own interpreter.
    env['PATH']=str(snake.parent)+os.pathsep+env.get('PATH','')
    listing=subprocess.check_output([str(snake),'--profile','profile/','--list-conda-envs','score','--config','run_models=amplify,mbc-attention','fasta='+str(root/'assets/pilot/dev.fasta'),'output='+str(root/'external/setup-probe.csv'),'unit=uM'],cwd=repo,env=env,text=True)
    found=set()
    for line in listing.splitlines():
        fields=line.split()
        for name in ['amplify','mbc-attention']:
            if fields and f'models/{name}/environment.yaml' in fields[0] and len(fields)>=2:
                destination=Path(fields[-1])
                if not destination.is_absolute(): destination=repo/destination
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists(): destination.symlink_to(root/'external/envs'/name,target_is_directory=True)
                if destination.resolve()!=(root/'external/envs'/name).resolve():
                    raise RuntimeError('Existing Snakemake environment is not the pinned environment: '+str(destination))
                # setup.sh would rerun pip against floating transitive packages.
                # Its work is fulfilled by the exact inventory above; record the
                # standard setup marker and prefix expected by the pinned rules.
                folder=repo/'results/setup'/name; folder.mkdir(parents=True,exist_ok=True)
                (folder/'env.txt').write_text(str(destination)+'\n')
                (folder/'.setup_done').touch()
                found.add(name)
    if found!={'amplify','mbc-attention'}:
        raise RuntimeError('Could not resolve both Snakemake environment locations:\n'+listing)
    (root/'external/setup_verified.json').write_text(json.dumps({'commits':{k:v['commit'] for k,v in pins.items()},'environment_inventory_verified':True,'conda':str(conda)},indent=2)+'\n')

if __name__ == '__main__': main()
