import subprocess, re, sys, os
TEST=re.compile(r'(_test\.go$|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)tests?/.*\.py$|_test\.dart$|\.test\.ts$|\.spec\.ts$)')
CODE=re.compile(r'\.(go|py|dart|ts|tsx)$')
root='/srv/work/w2/uc15/repos'
for r in sorted(os.listdir(root)):
    p=os.path.join(root,r)
    out=subprocess.run(['git','-C',p,'log','--no-merges','--format=@@%H|%s','--numstat'],capture_output=True,text=True).stdout
    for blk in out.split('@@')[1:]:
        lines=blk.strip().split('\n'); h,s=lines[0].split('|',1)
        tst=[];prod=0;pf=0;other=0
        for l in lines[1:]:
            if not l.strip(): continue
            a,d,f=l.split('\t',2)
            if a=='-': continue
            if TEST.search(f): tst.append(f)
            elif CODE.search(f) and not f.endswith(('.g.dart','.freezed.dart','.pb.go')) and '_gen' not in f: prod+=int(a)+int(d); pf+=1
        if tst and 15<=prod<=350 and pf<=6:
            print(f"{r}\t{h[:10]}\t+{prod}/{pf}f\t{len(tst)}t\t{s[:90]}")
