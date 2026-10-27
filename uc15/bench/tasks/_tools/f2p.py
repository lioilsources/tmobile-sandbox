import json, os, re, sys, shutil
sys.path.insert(0,'/srv/work/w2/uc15/tools'); import validate as V
for tid in sys.argv[1:]:
    tdir=f'/srv/work/w2/uc15/staging/{tid}'; t=json.load(open(tdir+'/task.json'))
    wd=V.prepare(t,tdir,'base'); V.sh(t['setup_cmd'],wd)
    rc,out=V.sh('dart test -r expanded',wd)
    failed=sorted({re.sub(r'^\S+ \+\d+.*?: ','',l).rsplit(' [E]',1)[0].split(': ',1)[-1] for l in out.splitlines() if l.rstrip().endswith('[E]')})
    t['fail_to_pass']=failed; json.dump(t,open(tdir+'/task.json','w'),indent=1,ensure_ascii=False)
    print(tid,len(failed),failed[:4]); shutil.rmtree(f"/srv/work/w2/uc15/val/{tid}",ignore_errors=True)
