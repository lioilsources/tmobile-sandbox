#!/usr/bin/env python3
"""gen.py id repo sha lang workdir [difficulty] -> staging/<id>/task.json"""
import json, os, re, subprocess, sys
ROOT='/srv/work/w2/uc15'
tid,repo,sha,lang,wd=sys.argv[1:6]; diff=sys.argv[6] if len(sys.argv)>6 else 'medium'
R=f'{ROOT}/repos/{repo}'
g=lambda *a: subprocess.run(['git','-C',R]+list(a),capture_output=True,text=True,check=True).stdout
gold=g('rev-parse',sha).strip(); base=g('rev-parse',sha+'^').strip()
TEST=re.compile(r'(_test\.go$|(^|/)test_[^/]*\.py$|_test\.dart$|\.test\.ts$)')
ns=[l.split('\t') for l in g('show','--numstat','--format=',gold).strip().split('\n') if l]
tests=[f for a,d,f in ns if TEST.search(f) and a!='-']
# keep only test files that exist in gold
tests=[f for f in tests if subprocess.run(['git','-C',R,'cat-file','-e',f'{gold}:{f}']).returncode==0]
supp=[f for a,d,f in ns if lang=='dart' and f.startswith(os.path.join(wd,'test/').lstrip('./')) and not TEST.search(f)]
tests+=supp
prod=[(a,d,f) for a,d,f in ns if f not in tests and a!='-' and re.search(r'\.(go|py|dart|ts|tsx|sql|svelte|js)$',f)]
add=sum(int(a) for a,d,f in prod); dele=sum(int(d) for a,d,f in prod)
patch=g('show',gold,'--',*[f for f in tests if TEST.search(f)])
if lang=='go': f2p=re.findall(r'^\+func (Test\w+)\(',patch,re.M)
elif lang=='python': f2p=re.findall(r'^\+\s*def (test_\w+)\(',patch,re.M)
else: f2p=re.findall(r'^\+\s*test\(\s*[\'"](.+?)[\'"]',patch,re.M)
rel=lambda f: os.path.relpath(f, wd)
if lang=='go':
    pk=sorted({'./'+os.path.dirname(rel(f)) if os.path.dirname(rel(f)) else '.' for f in tests})
    cmd=f"go test -count=1 {' '.join(pk)}"; setup=None
elif lang=='dart': cmd='dart test'; setup='dart pub get'
elif lang=='python': cmd='python -m pytest -q -p no:cacheprovider '+' '.join(rel(f) for f in tests if TEST.search(f)); setup=None
subj=g('log','-1','--format=%s',gold).strip()
t=dict(id=tid,repo=f'lioilsources/{repo}',lang=lang,kind='reverse-commit',base_commit=base,gold_commit=gold,workdir=wd,
       test_files=tests,setup_cmd=setup,test_cmd=cmd,fail_to_pass=f2p,pass_to_pass=[f'all other tests run by test_cmd (same packages/files)'],
       timeout_s=900,difficulty=diff,diff_stats=f'+{add}/-{dele} in {len(prod)} files',prod_files=[f for a,d,f in prod],source=subj)
os.makedirs(f'{ROOT}/staging/{tid}',exist_ok=True)
json.dump(t,open(f'{ROOT}/staging/{tid}/task.json','w'),indent=1,ensure_ascii=False)
print(tid, cmd, f2p[:8], t['diff_stats'])
