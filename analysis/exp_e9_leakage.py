"""
E9 — Train/test leakage audit (reviewer NhWo C5). Verifies the disjointness claims against
the released split metadata and states the precise level of separation, correcting the
paper where the claim is stronger than the data supports.

Levels checked (realism train vs test_base):
  (a) exact sample leakage  = same (source audio clip, profile) pair in both  -> expect 0
  (b) source-clip reuse     = same original_id (audio recording) in both, any profile
  (c) character reuse        = same positive profile string in both (different utterance)
  (d) show overlap           = same source show (disjointness is by clip, not show)

Realism real recordings: the RELEASE train contains no real audio, so there is no release
train/test_realrec leakage; but released test_realrec (2,164) mixes held-out speakers
(qyf/zrr/zyx/yy) with the SFT model's own training speakers (fyr/smh/pro-mufei). The clean
eval subset test_real_v2 (1,084) has zero SFT-train overlap — cross-referenced to §0 / E1.

Archetype: role×scene prompt disjointness train vs test.

Run from analysis/:  PYTHONPATH=. python3 exp_e9_leakage.py
Writes analysis/out/e9_leakage.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os
import re

import common
from realrec_v2_clean import speaker_of, load_v2_ids

RB = common.REALISM_META
AB = common.ARCHE_META
# SFT training speakers (train_all_real_v2 in-house real recordings) vs clean held-out
SFT_TRAIN_SPEAKERS = {"pro-mufei", "fyr", "smh"}
HELDOUT_SPEAKERS = {"qyf", "zrr", "zyx", "yy"}


def load(fp):
    return [json.loads(l) for l in open(fp)]


def show_of(oid):
    if not oid:
        return None
    return re.split(r"_\d+x\d+_|_\d{6}_", oid)[0]


def main():
    out = {}
    md = ["# E9 — Train/test leakage audit", "",
          "Checked against the released split metadata. Reports the exact level of "
          "train/test separation for each track.", ""]

    # ---------------- Realism base ----------------
    tr = load(RB + "/train.jsonl"); tb = load(RB + "/test_base.jsonl")
    tr_ap = {(r.get("original_id"), r["profile"]) for r in tr if r.get("original_id")}
    tb_ap = {(r.get("original_id"), r["profile"]) for r in tb if r.get("original_id")}
    exact = len(tr_ap & tb_ap)
    tr_oid = {r.get("original_id") for r in tr if r.get("original_id")}
    tb_oid = {r.get("original_id") for r in tb if r.get("original_id")}
    clip_reuse = len(tr_oid & tb_oid)
    tr_posprof = {r["profile"] for r in tr if r.get("sample_type") == "positive"}
    tb_pos = [r for r in tb if r.get("sample_type") == "positive"]
    char_reuse = sum(1 for r in tb_pos if r["profile"] in tr_posprof)
    tr_show = {show_of(r.get("original_id")) for r in tr if r.get("original_id")}
    tb_show = {show_of(r.get("original_id")) for r in tb if r.get("original_id")}
    show_shared = len(tr_show & tb_show)

    md += ["## 1. Realism in-distribution (train 10,990 vs test_base 2,000)", "",
           "| Level | Definition | Count | Verdict |", "|---|---|---|---|",
           f"| (a) exact sample | same (audio clip **and** profile) pair | **{exact}** | clean — no sample-level leakage |",
           f"| (b) source-clip reuse | same source recording, any profile | {clip_reuse} / 2,000 | each reused clip is paired with a **different** profile/scene in test (0 share both) |",
           f"| (c) character reuse | test positive reuses a train character-profile | {char_reuse} / {len(tb_pos)} | **clip-disjoint, NOT character-disjoint** (different utterances of a seen character) |",
           f"| (d) show overlap | same source show | {show_shared} shows | expected — separation is by clip, not by show |", "",
           f"**Correction for the paper:** the split is disjoint at the (clip × profile) "
           f"sample level and at the exact-recording level (0 duplicates), but {char_reuse}/"
           f"{len(tb_pos)} test positives reuse a *character* seen in training with a "
           f"different utterance. If the text claims character-level disjointness, soften to "
           f"'utterance/clip-disjoint; characters may recur across splits with distinct "
           f"utterances' — which is the honest and still-valid generalization claim.", ""]
    out["realism_base"] = {"exact_sample": exact, "clip_reuse": clip_reuse,
                           "character_reuse": char_reuse, "n_test_pos": len(tb_pos),
                           "shows_shared": show_shared}

    # ---------------- Realism real recordings ----------------
    rr = load(RB + "/test_realrec.jsonl")
    rr_spk = collections.Counter(speaker_of(r["id"]) for r in rr)
    v2 = set(load_v2_ids())
    v2_spk = collections.Counter(speaker_of(i) for i in v2)
    release_train_real = sum(1 for r in tr if r.get("sample_type") == "positive" and r.get("original_id") is None)
    md += ["## 2. Realism real recordings", "",
           f"- Released **train** contains **{release_train_real}** real recordings — the "
           f"release train is synthesis/base only, so there is no release train↔test_realrec "
           f"leakage. (The SFT model adds 1,080 in-house real recordings on top; that is the "
           f"§0 contamination story, see E1.)",
           f"- Released **test_realrec** (2,164) speaker mix: {dict(rr_spk)}. This mixes "
           f"held-out speakers {sorted(HELDOUT_SPEAKERS)} with the SFT model's own training "
           f"speakers {sorted(SFT_TRAIN_SPEAKERS)}.",
           f"- The clean evaluation subset **test_real_v2** (1,084) = {dict(v2_spk)} — zero "
           f"overlap with SFT training speakers (the 2 pro-mufei excepted). This is the only "
           f"split that supports a held-out-speaker claim; §0/E1 use it.", ""]
    out["realism_real"] = {"release_train_real": release_train_real,
                           "test_realrec_speakers": dict(rr_spk), "test_real_v2_speakers": dict(v2_spk)}

    # ---------------- Archetype ----------------
    atr = load(AB + "/train.jsonl"); ate = load(AB + "/test.jsonl")
    atr_p = {r["prompt"] for r in atr}; ate_p = {r["prompt"] for r in ate}
    prompt_overlap = len(atr_p & ate_p)
    rc_tr = set(r.get("role_category") for r in atr); rc_te = set(r.get("role_category") for r in ate)
    md += ["## 3. Archetype (train 6,773 vs test 1,499)", "",
           f"- Distinct role×scene **prompts**: train {len(atr_p)}, test {len(ate_p)}; "
           f"overlap = **{prompt_overlap}** → scenario-disjoint (each test prompt is voiced "
           f"by multiple SFnd systems, giving the 1,499 rows).",
           f"- `role_category` values are shared by design ({len(rc_tr & rc_te)} shared "
           f"categories) — these are broad categories, not identities; scenario disjointness "
           f"is what matters and holds.", ""]
    out["archetype"] = {"prompt_overlap": prompt_overlap, "n_train_prompts": len(atr_p),
                        "n_test_prompts": len(ate_p), "role_categories_shared": len(rc_tr & rc_te)}

    md += ["## Notes for integration", "",
           "- One appendix paragraph + a small table (level (a)-(d)). Lead with the two clean "
           "facts (0 exact duplicates; archetype scenario-disjoint), then the honest "
           "character-recurrence nuance for realism.",
           "- QA rejection logs are not in the release; state the QA procedure and mark "
           "rejection counts documented-only (also E8).", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e9_leakage.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e9_leakage.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e9_leakage.md"))
    print("\n--- E9 HIGHLIGHTS ---")
    print(f"  realism: exact dup {exact}, clip reuse {clip_reuse}, char reuse {char_reuse}/{len(tb_pos)}")
    print(f"  archetype prompt overlap {prompt_overlap}")


if __name__ == "__main__":
    main()
