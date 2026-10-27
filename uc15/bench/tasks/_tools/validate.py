#!/usr/bin/env python3
"""Validate a reverse-commit task: base+gold tests must FAIL, gold must PASS."""
import json, os, subprocess, sys, shutil
ROOT='/srv/work/w2/uc15'
T=f'{ROOT}/tools'
ENV=dict(os.environ, PATH=f'{T}/venv/bin:{T}/dart-sdk/bin:{T}/node/bin:/usr/local/go/bin:/usr/bin:/bin',
         GOPATH=f'{T}/gopath', GOCACHE=f'{T}/gocache', GOFLAGS='-mod=mod', PUB_CACHE=f'{T}/pub-cache',
         PYTHONDONTWRITEBYTECODE='1', CGO_ENABLED=os.environ.get('CGO_ENABLED','1'))
def sh(cmd, cwd, timeout=900):
    try:
        p=subprocess.run(cmd, shell=True, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout+p.stderr)
    except subprocess.TimeoutExpired as e:
        return 124, 'TIMEOUT'
def git(args, cwd): return subprocess.run(['git']+args, cwd=cwd, capture_output=True, text=True, check=True).stdout
def prepare(t, tdir, mode):
    repo=f"{ROOT}/repos/{t['repo'].split('/')[1]}"
    w=f"{ROOT}/val/{t['id']}"
    shutil.rmtree(w, ignore_errors=True)
    subprocess.run(['git','clone','-q','--shared',repo,w],check=True)
    if mode=='base':
        git(['checkout','-q',t['base_commit']],w)
        bp=os.path.join(tdir,'base.patch')
        if os.path.exists(bp): git(['apply',bp],w)
        for f in t['test_files']:
            os.makedirs(os.path.dirname(os.path.join(w,f)) or w, exist_ok=True)
            data=subprocess.run(['git','show',f"{t['gold_commit']}:{f}"],cwd=w,capture_output=True,check=True).stdout
            open(os.path.join(w,f),'wb').write(data)
    else:
        git(['checkout','-q',t['gold_commit']],w)
    return os.path.join(w,t.get('workdir','.'))
def run(t,tdir,mode):
    wd=prepare(t,tdir,mode)
    if t.get('setup_cmd'):
        rc,out=sh(t['setup_cmd'],wd)
        if rc: return rc,'SETUP FAIL\n'+out[-1500:]
    return sh(t['test_cmd'],wd,t.get('timeout_s',900))
if __name__=='__main__':
    tdir=os.path.abspath(sys.argv[1]); t=json.load(open(os.path.join(tdir,'task.json')))
    brc,bout=run(t,tdir,'base'); grc,gout=run(t,tdir,'gold')
    ok = brc!=0 and grc==0
    skip = ('skipped' in gout.lower()) or (' SKIP' in gout)
    print(json.dumps({'id':t['id'],'ok':ok,'base_rc':brc,'gold_rc':grc,'gold_skip_mention':skip}))
    if '-v' in sys.argv or not ok:
        print('--- BASE tail\n'+bout[-1200:]); print('--- GOLD tail\n'+gout[-1200:])
    shutil.rmtree(f"{ROOT}/val/{t['id']}", ignore_errors=True)
