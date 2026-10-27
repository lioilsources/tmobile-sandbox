#!/usr/bin/env python3
"""genrm.py id repo lang workdir difficulty 'test_cmd' f2p_testfile[:regex] ... -> removal task (HEAD + base.patch)"""
import json, os, re, subprocess, sys, shutil
ROOT='/srv/work/w2/uc15'
tid,repo,lang,wd,diff,cmd=sys.argv[1:7]; specs=sys.argv[7:]
R=f'{ROOT}/repos/{repo}'
head=subprocess.run(['git','-C',R,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
tests=[];f2p=[]
for s in specs:
    f,_,rx=s.partition(':'); tests.append(f)
    src=open(os.path.join(R,f)).read()
    names=re.findall(r'^func (Test\w+)\(',src,re.M) if lang=='go' else re.findall(r'^\s*test\(\s*[\'"](.+?)[\'"]',src,re.M)
    f2p+=[n for n in names if not rx or re.search(rx,n)]
p=f'{ROOT}/staging-patches/{tid}.patch'
pt=open(p).read()
prod=re.findall(r'^\+\+\+ b/(.+)$',pt,re.M)
add=len(re.findall(r'^-(?!--)',pt,re.M))
os.makedirs(f'{ROOT}/staging/{tid}',exist_ok=True); shutil.copy(p,f'{ROOT}/staging/{tid}/base.patch')
t=dict(id=tid,repo=f'lioilsources/{repo}',lang=lang,kind='removal',base_commit=head,gold_commit=head,base_patch='base.patch',
  workdir=wd,test_files=tests,setup_cmd='dart pub get' if lang=='dart' else None,test_cmd=cmd,fail_to_pass=f2p,
  pass_to_pass=['all other tests run by test_cmd'],timeout_s=900,difficulty=diff,
  diff_stats=f'gold restores ~{add} removed lines in {len(prod)} files',prod_files=prod,
  source='synthetic removal: function bodies stubbed out at HEAD, original implementation is the gold solution')
json.dump(t,open(f'{ROOT}/staging/{tid}/task.json','w'),indent=1,ensure_ascii=False)
print(tid,len(f2p),f2p[:5])
