#!/usr/bin/env python3
"""Seznam stažených modelů + revize + sha256 z HF (modely nezálohujeme, jen tenhle manifest)."""
import json, os, pathlib
from huggingface_hub import HfApi, scan_cache_dir

api, out = HfApi(), []
for repo in scan_cache_dir(os.environ.get("HF_HOME", "/srv/hf") + "/hub").repos:
    for rev in repo.revisions:
        out.append({"repo": repo.repo_id, "revision": rev.commit_hash, "where": "hf_cache",
                    "files": {f.file_name: None for f in rev.files}})
for d in pathlib.Path("/srv/comfy/dl").glob("*/*"):
    files = [p.relative_to(d).as_posix() for p in d.rglob("*.safetensors")]
    out.append({"repo": f"{d.parent.name}/{d.name}", "where": "comfy", "files": dict.fromkeys(files)})

for m in out:  # doplnit sha256 z LFS metadat (bez lokálního hashování 200 GB)
    try:
        info = api.model_info(m["repo"], revision=m.get("revision"), files_metadata=True)
        m.setdefault("revision", info.sha)
        meta = {s.rfilename: s.lfs.sha256 if s.lfs else None for s in info.siblings}
        m["files"] = {f: meta.get(f) for f in m["files"]}
    except Exception as e:
        m["error"] = str(e)
print(json.dumps(out, indent=1))
