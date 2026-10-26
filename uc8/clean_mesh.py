#!/usr/bin/env python3
"""UC8: vyčištění surových meshů z Hunyuan3D a decimace na limit Roblox UGC (4 000 trojúhelníků), vzorek každý 5. model.

Ponechá komponenty s aspoň 5 % trojúhelníků hlavní části (zbytek je šum ze surface net), pak quadric decimace.
Výstup clean_sample.json a náhledy preview/NNN_dec.png pro report. Spouštět v /srv/work/uc8 venvem UC8
(trimesh, fast-simplification, scipy, networkx, matplotlib).
"""
import trimesh, json, time, numpy as np, matplotlib, statistics as st
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R={json.loads(l)["i"]:json.loads(l) for l in open("models.jsonl")}
out=[]
for i in range(0,200,5):
    m=trimesh.load(f"glb/{i:03d}.glb",force="mesh"); t=time.time()
    c=sorted(m.split(only_watertight=False),key=lambda x:-len(x.faces))
    big=[x for x in c if len(x.faces)>=0.05*len(c[0].faces)]
    k=trimesh.util.concatenate(big); kept=len(k.faces)
    d=k.simplify_quadric_decimation(face_count=4000) if kept>4000 else k
    dt=time.time()-t
    out.append({"i":i,"predmet":R[i]["predmet"],"raw":len(m.faces),"komponent":len(c),"hlavni":kept,"casti":len(big),"dec":len(d.faces),"wt_hlavni":bool(c[0].is_watertight),"wt_dec":bool(d.is_watertight),"sec":round(dt,1)})
    if i in (0,30,60,110,150,190):
        v=d.vertices-d.vertices.mean(0); v/=np.abs(v).max()
        fig=plt.figure(figsize=(3,3),dpi=100); ax=fig.add_subplot(projection="3d")
        tri=v[d.faces]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
        sh=0.3+0.7*np.abs(n@np.array([0.45,0.75,0.5]))
        pc=Poly3DCollection(tri[:,:,[0,2,1]]*[1,-1,1],facecolors=np.c_[sh*0.36,sh*0.62,sh*0.66],edgecolor="none")
        ax.add_collection3d(pc); ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1); ax.view_init(15,-70); ax.axis("off"); ax.set_box_aspect((1,1,1))
        fig.savefig(f"preview/{i:03d}_dec.png",transparent=True,bbox_inches="tight",pad_inches=0); plt.close(fig)
json.dump(out,open("clean_sample.json","w"),ensure_ascii=False)
f=lambda k: st.median(o[k] for o in out)
print(len(out),"komponent med",f("komponent"),"hlavni med",f("hlavni"),"casti",f("casti"),"dec<=4100", sum(o["dec"]<=4100 for o in out),"wt hlavni",sum(o["wt_hlavni"] for o in out),"wt dec",sum(o["wt_dec"] for o in out),"sec med",f("sec"))
print("hlavni podil raw %", round(100*st.median(o["hlavni"]/o["raw"] for o in out),1))
