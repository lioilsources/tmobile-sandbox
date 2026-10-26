#!/usr/bin/env python3
"""UC7 krok 1: z FFHQ (gaunernst/ffhq-1024-wds, tar po 1000 webp 1024×1024) vyřízne pár očí i s obočím.

Body obličeje = MediaPipe FaceLandmarker (CPU, 478 bodů + blendshapes). Výřez je srovnaný podle spojnice středů
očí, šířka 2,3× vzdálenost očí, poměr 2:1, uloží se jako 1024×512 JPEG. Vyřazuje: mrknutí (eyeBlink > 0,35),
hlavu natočenou o víc než 20°, malé oči (výřez užší než 380 px ze zdroje). Ostrost (rozptyl Laplaciánu) se jen
zapisuje, práh se volí až ve výběru (select_caption.py) podle rozdělení.

  crop_eyes.py --tars /srv/work/uc7/ffhq/*.tar --out /srv/work/uc7/crops --model /srv/work/uc7/face_landmarker.task
Výstup: <out>/img/<id>.jpg, <out>/meta.jsonl (id, d, blink, yaw, pitch, sharp, src_w).
"""
import argparse, io, json, math, pathlib, tarfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

L_EYE, R_EYE = [33, 133, 159, 145], [362, 263, 386, 374]  # koutky + víčka
W, H = 1024, 512
_lm = None


def _init(model):
    global _lm
    from mediapipe.tasks.python import BaseOptions, vision
    _lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model), output_face_blendshapes=True,
        output_facial_transformation_matrixes=True, num_faces=1))


def _pose(m):
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def process(job):
    import mediapipe as mp
    tar, out = job
    rows = []
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.name.endswith(".webp"):
                continue
            iid = pathlib.Path(m.name).stem
            rgb = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
            res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                continue
            pts = np.array([[p.x * rgb.shape[1], p.y * rgb.shape[0]] for p in res.face_landmarks[0]])
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            yaw, pitch = _pose(res.facial_transformation_matrixes[0])
            le, re = pts[L_EYE].mean(0), pts[R_EYE].mean(0)
            d = float(np.linalg.norm(re - le)); ang = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
            blink = max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))
            src_w = 2.3 * d
            row = {"id": iid, "d": round(d, 1), "blink": round(blink, 3), "yaw": round(yaw, 1), "pitch": round(pitch, 1),
                   "src_w": round(src_w)}
            if blink > 0.35 or abs(yaw) > 20 or src_w < 380:
                row["vyrazeno"] = "mrk" if blink > 0.35 else "natoceni" if abs(yaw) > 20 else "male"
                rows.append(row); continue
            # srovnat podle očí, střed mírně nahoru k obočí
            c = (le + re) / 2 + np.array([0, -0.12 * d])
            s = W / src_w
            M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, s)
            M[:, 2] += [W / 2 - c[0], H / 2 - c[1]]
            crop = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            g = cv2.cvtColor(cv2.resize(crop, (512, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            row["sharp"] = round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)
            Image.fromarray(crop).save(out / "img" / f"{iid}.jpg", quality=95)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    (a.out / "img").mkdir(parents=True, exist_ok=True)
    meta = a.out / "meta.jsonl"
    done = {json.loads(l)["id"][:2] for l in open(meta)} if meta.exists() else set()  # tar = prefix id (00000.tar → 00xxx)
    jobs = [(t, a.out) for t in sorted(a.tars) if pathlib.Path(t).stem[:2] not in done]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model,)) as ex, open(meta, "a") as f:
        for t, rows in zip(jobs, ex.map(process, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            ok = sum("vyrazeno" not in r for r in rows)
            print(f"{pathlib.Path(t[0]).name}: {len(rows)} tváří, {ok} výřezů", flush=True)


if __name__ == "__main__":
    main()
