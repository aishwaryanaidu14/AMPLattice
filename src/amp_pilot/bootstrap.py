"""Private, checksum-pinned Conda bootstrap; no system Conda prerequisite."""
import json
import os
import platform
import shutil
import shlex
import subprocess
import urllib.request
from pathlib import Path
from .submission import check_hash


def ensure_conda(root, explicit=None):
    root=Path(root).resolve()
    if explicit:
        executable=shutil.which(explicit) or str(Path(explicit).resolve())
        if not Path(executable).is_file():
            raise RuntimeError('Explicit Conda executable not found: '+explicit)
        return Path(executable)
    pin=json.loads((root/'assets/conda_bootstrap.json').read_text())
    if platform.system()!=pin['platform'] or platform.machine()!=pin['architecture']:
        raise RuntimeError('The faithful scorer bootstrap supports Linux x86_64, including WSL')
    prefix=root/'external/bootstrap/miniforge'
    executable=prefix/'bin/conda'
    marker=prefix/'amp-bootstrap.json'
    if marker.exists():
        if json.loads(marker.read_text())!=pin or not executable.is_file():
            raise RuntimeError('Private Conda bootstrap is incomplete or its pin changed')
        prepare_launcher(prefix)
        return executable
    downloads=root/'external/downloads';downloads.mkdir(parents=True,exist_ok=True)
    installer=downloads/('Miniforge3-'+pin['version']+'-Linux-x86_64.sh')
    if not installer.exists():
        temporary=installer.with_suffix('.part')
        print('Downloading pinned private Conda bootstrap:',pin['version'],flush=True)
        try:
            request=urllib.request.Request(pin['url'],headers={'User-Agent':'amp-challenge-submission'})
            with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as target:
                shutil.copyfileobj(response,target)
            if temporary.stat().st_size!=pin['bytes']:
                raise RuntimeError('Conda bootstrap installer size mismatch')
            check_hash(temporary,pin['sha256'],'Conda bootstrap installer')
            temporary.replace(installer)
        finally:
            temporary.unlink(missing_ok=True)
    check_hash(installer,pin['sha256'],'Conda bootstrap installer')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['VIRTUAL_ENV','PYTHONHOME','PYTHONPATH','CONDA_EXE','CONDA_PYTHON_EXE']:
        env.pop(key,None)
    if not prefix.exists():
        subprocess.run(['bash',str(installer),'-b','-p',str(prefix)],env=env,check=True)
    elif not (prefix/'conda-meta/history').is_file():
        raise RuntimeError('Incomplete private Conda install at '+str(prefix))
    # Recover a completed installer whose CLI validation failed under uv PATH.
    # Verify with its own interpreter before writing the completion marker.
    python=prefix/'bin/python'
    if not python.is_file() or not executable.is_file():
        raise RuntimeError('Private Conda installation is incomplete')
    subprocess.run([str(python),'-I','-m','conda','--version'],env=env,check=True)
    prepare_launcher(prefix)
    subprocess.run([str(executable),'--version'],env=env,check=True)
    marker.write_text(json.dumps(pin,indent=2)+'\n')
    return executable


def prepare_launcher(prefix):
    """Avoid long-prefix env-python shebangs and uv/Snakemake PATH collisions."""
    prefix=Path(prefix);python=prefix/'bin/python';executable=prefix/'bin/conda'
    if not python.is_file() or not executable.is_file():
        raise RuntimeError('Private Conda installation is incomplete')
    content='#!/bin/sh\nexec '+shlex.quote(str(python))+' -I -m conda "$@"\n'
    if executable.read_text()==content:
        return
    backup=prefix/'bin/conda.amp-original'
    if not backup.exists():
        shutil.copy2(executable,backup)
    temporary=prefix/'bin/conda.amp-tmp'
    temporary.write_text(content);temporary.chmod(0o755);temporary.replace(executable)


def scorer_environment(root, conda):
    """Give all scorer children private Conda and writable project-local caches."""
    root=Path(root).resolve();env=os.environ.copy()
    for key in ['VIRTUAL_ENV','PYTHONHOME','PYTHONPATH','CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_SHLVL','CONDA_EXE','CONDA_PYTHON_EXE']:
        env.pop(key,None)
    env['PATH']=str(Path(conda).resolve().parent)+os.pathsep+env.get('PATH','')
    for key,relative in [('CONDA_PKGS_DIRS','external/cache/conda-pkgs'),('PIP_CACHE_DIR','external/cache/pip'),('XDG_CACHE_HOME','external/cache/xdg')]:
        path=root/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    return env
