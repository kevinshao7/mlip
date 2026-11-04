#!/usr/bin/env python3
"""Build GitHub MACE/Polar dependency and install without pip.

Run with isolated setuptools, wheel and installer tools. The target interpreter
is explicit; PyTorch and unrelated dependencies are never resolved/upgraded.
"""
from __future__ import annotations
import argparse
import email
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(source):
    return subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()


def target_inventory(python):
    code="""import sys,sysconfig,json,importlib.metadata as md
rows={}
for name in ['mace-torch','graph_longrange']:
 d=md.distribution(name); roots=set(); paths=sysconfig.get_paths()
 sites={str(__import__('pathlib').Path(paths[k]).resolve()) for k in ['purelib','platlib']}
 for f in d.files or []:
  p=__import__('pathlib').Path(d.locate_file(f)).resolve()
  for site in sites:
   base=__import__('pathlib').Path(site)
   if p.is_relative_to(base):
    rel=p.relative_to(base)
    p=base/rel.parts[0] if rel.parts[0] not in {'__pycache__','tests','scripts'} else p
    break
  if p.exists(): roots.add(str(p))
 rows[name]={'version':d.version,'roots':sorted(roots)}
print(json.dumps({'prefix':sys.prefix,'paths':sysconfig.get_paths(),'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions() if d.metadata.get('Name')},'old':rows}))
"""
    return json.loads(subprocess.check_output([str(python),'-c',code],text=True,cwd='/tmp'))


def build(source,output,editable):
    from setuptools import build_meta
    output.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(source)
        # Compat PEP660 mode exposes the real checkout on sys.path and avoids
        # a namespace collision with this project's enclosing mace/ directory.
        filename=(build_meta.build_editable(str(output),config_settings={'editable_mode':'compat'})
                  if editable else build_meta.build_wheel(str(output)))
    finally:
        os.chdir(previous)
    path=output/filename
    return {'wheel':str(path),'sha256':digest(path),'source':str(source),
            'commit':git_revision(source),'editable':editable}


def check_dependencies(entries,inventory):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    versions={canonicalize_name(k):v for k,v in inventory['packages'].items()}
    metadata=[]
    for entry in entries:
        with zipfile.ZipFile(entry['wheel']) as archive:
            name=next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            msg=email.message_from_bytes(archive.read(name))
        versions[canonicalize_name(msg['Name'])]=msg['Version']; metadata.append(msg)
    errors=[]
    for msg in metadata:
        for raw in msg.get_all('Requires-Dist',[]):
            req=Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra':'polar'}): continue
            version=versions.get(canonicalize_name(req.name))
            if version is None or not req.specifier.contains(version,prereleases=True):
                errors.append(f'{msg["Name"]}: {req}, installed/planned={version}')
    if errors: raise RuntimeError('Dependency mismatch; target unchanged:\n'+'\n'.join(errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-python',type=Path,required=True)
    parser.add_argument('--mace-source',type=Path,required=True)
    parser.add_argument('--graph-source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    python=args.target_python.absolute(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'build_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text()); entries=manifest['entries']
        for entry in entries:
            if digest(entry['wheel'])!=entry['sha256'] or git_revision(entry['source'])!=entry['commit']:
                raise ValueError('Source/built-wheel changed; use a new build directory')
    else:
        entries=[build(args.graph_source.resolve(),out/'wheels',False),build(args.mace_source.resolve(),out/'wheels',True)]
        manifest={'entries':entries,'target_python':str(python),'method':'setuptools PEP517/660 + PyPA installer; no pip'}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    inventory=target_inventory(python); check_dependencies(entries,inventory)
    if not args.install:
        print('Build and dependency validation passed; target environment unchanged.'); return
    from installer import install
    from installer.destinations import SchemeDictionaryDestination
    from installer.sources import WheelFile
    stage=out/'staged_install'; stage.mkdir(exist_ok=False)
    paths=inventory['paths']; scheme={**paths,'headers':paths['include']}
    for entry in entries:
        source=Path(entry['source'])
        origin=({'url':source.as_uri(),'dir_info':{'editable':True}} if entry['editable'] else
                {'url':subprocess.check_output(['git','-C',str(source),'remote','get-url','origin'],text=True).strip(),
                 'vcs_info':{'vcs':'git','commit_id':entry['commit'],'requested_revision':'v0.4.4'}})
        destination=SchemeDictionaryDestination(scheme,interpreter=str(python),script_kind='posix',destdir=str(stage))
        with WheelFile.open(entry['wheel']) as wheel:
            install(wheel,destination,additional_metadata={'INSTALLER':b'installer',
                    'direct_url.json':json.dumps(origin).encode(),'source_build.json':json.dumps(entry).encode()})
    prefix=Path(inventory['prefix']).resolve()
    old={Path(p).resolve() for record in inventory['old'].values() for p in record['roots']}
    old={p for p in old if not any(q!=p and q in p.parents for q in old)}
    if any(not p.is_relative_to(prefix) for p in old): raise ValueError('Old package record escapes target environment')
    new=[]
    for folder in {paths['purelib'],paths['platlib'],paths['scripts']}:
        staged=stage/Path(folder).relative_to('/')
        if staged.exists():
            new.extend((p,(Path(folder)/p.name).resolve()) for p in staged.iterdir())
    for _,target in new:
        if target.exists() and not any(target==p or p in target.parents for p in old):
            raise FileExistsError(f'Unmanaged target collision: {target}')
    backup=out/'rollback'; moved=[]; published=[]
    try:
        for source in sorted(old):
            target=backup/source.relative_to(prefix); target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),target); moved.append((source,target))
        for source,target in new:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),target); published.append((source,target))
    except BaseException:
        for source,target in reversed(published):
            source.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(target),source)
        for source,target in reversed(moved):
            source.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(target),source)
        raise
    record={**manifest,'installed_utc':datetime.now(timezone.utc).isoformat(),
            'prefix':str(prefix),'rollback':str(backup),
            'replaced':{name:row['version'] for name,row in inventory['old'].items()},
            'rollback_paths':[(str(a),str(b)) for a,b in moved]}
    (out/'install_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Source-built packages installed:',json.dumps(record,indent=2))


if __name__=='__main__': main()
