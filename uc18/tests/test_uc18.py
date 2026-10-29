#!/usr/bin/env python3
"""UC18/UC19 unit tests — pure CPU, seconds.  python3 uc18/tests/test_uc18.py  (or pytest)"""
import json, os, random, sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
sys.path.insert(0, os.path.join(H, "..", "..", "uc19"))
import comfy, common, models  # noqa: E402

ROOT = os.path.join(H, "..")


def test_prompt_set():
    import prompts_src
    recs = prompts_src.records()
    assert len(recs) == 300 and len({r["id"] for r in recs}) == 300
    by = {}
    for r in recs:
        by[r["style"]] = by.get(r["style"], 0) + 1
        assert not common.sfw_hits(r["prose"] + " , " + r["tags"]), r["id"]
    assert set(by.values()) == {30} and len(by) == 10


def test_cz_set():
    import prompts_cz
    ev, fs = prompts_cz.eval_records(), prompts_cz.fewshot_records()
    assert len(ev) == 200 and len(fs) == 15
    assert sum(r["category"] != "plain" for r in ev) == 50
    assert not {f["cz"] for f in fs} & {r["cz"] for r in ev}
    for r in ev:
        assert not common.sfw_hits(r["cz"] + " " + r["ref_en"] + " " + ",".join(r["must_tags"])), r["cid"]
        if r["category"] == "negation":
            assert r["must_not"], r["cid"]


def test_sfw_regex():
    assert common.sfw_hits("1girl, solo") == []
    assert common.sfw_hits("a girl") == ["girl"]
    assert common.sfw_hits("nahá žena") and common.sfw_hits("malé dítě")
    assert common.sfw_hits("bloody") and not common.sfw_hits("blue sky")


def test_workflows_validate():
    oi = json.load(open(os.path.join(ROOT, "object_info.json")))
    for ck in models.CKPTS:
        g = comfy.build(ck, "a fox", models.CKPTS[ck]["negative"], 123, "uc18/x")
        assert comfy.validate(g, oi) == [], ck
    g = comfy.build("pony", "a", "b", 1, "p")
    g["ks"]["inputs"]["sampler_name"] = "nope"
    g["ks"]["inputs"]["steps"] = 0
    g["dec"]["inputs"]["vae"] = ["ckpt", 1]
    errs = comfy.validate(g, oi)
    assert any("sampler_name" in e for e in errs) and any("steps" in e for e in errs) and any("CLIP" in e for e in errs)


def test_parse_history():
    h = {"status": {"status_str": "success", "completed": True, "messages": [["execution_start", {"timestamp": 1000}],
                                                                              ["execution_success", {"timestamp": 5500}]]},
         "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "uc18", "type": "output"}]}}}
    p = comfy.parse_history(h)
    assert p["status"] == "success" and p["exec_s"] == 4.5 and p["images"][0]["filename"] == "a.png"


def test_bt_trueskill_recover_order():
    import rank
    rng = random.Random(1)
    true = {"a": 1.0, "b": 0.3, "c": -0.5, "d": -1.0}
    pairs = []
    for i in range(400):
        x, y = rng.sample(list(true), 2)
        p = 1 / (1 + pow(2.718281828, true[y] - true[x]))
        pairs.append((f"p{i % 50}", "s", x, y, float(rng.random() < p)))
    bt = rank.bradley_terry(pairs, sorted(true))
    assert sorted(bt, key=bt.get, reverse=True) == ["a", "b", "c", "d"], bt
    ts = rank.trueskill(pairs, sorted(true))
    assert max(ts, key=lambda m: ts[m]["mu"]) == "a" and min(ts, key=lambda m: ts[m]["mu"]) == "d"


def test_judge_schemas():
    import vlm_judge
    vlm_judge.schema_check()


def test_dialect_post():
    import dialects as D
    r = D.post_tags("illustrious", ["1girl", "masterpiece", "score_9", "long_hair, red hair", "source_anime", "nude", "zzz"],
                    {"1girl", "long hair", "red hair"})
    assert r["tags"][:3] == ["1girl", "long hair", "red hair"]
    assert "masterpiece" in r["quality_removed"] and "nude" in r["unsafe_removed"] and r["oov"] == ["zzz"]
    assert r["tags"][-1] == "zzz"
    assert "zzz" in D.post_tags("illustrious", ["zzz", "1girl"], {"1girl"}, "filter")["dropped"]
    assert "source_anime" in r["dropped"]
    assert D.post_tags("pony", ["source_anime", "1girl"], {"1girl"})["tags"] == ["source_anime", "1girl"]
    s, bad = D._clean_weights("a (fox:1.2) in (snow:3.0)")
    assert s == "a (fox:1.2) in snow" and bad == 1
    from jsonschema import Draft202012Validator
    for sch in D.SCHEMA.values():
        Draft202012Validator.check_schema(sch)


def test_seed_shared_across_models():
    import gen_matrix
    assert gen_matrix.seed_for("x", 1) == gen_matrix.seed_for("x", 1) != gen_matrix.seed_for("x", 2)


if __name__ == "__main__":
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            try:
                f()
                print("ok  ", k)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", k, repr(e)[:300])
    sys.exit(1 if fails else 0)
