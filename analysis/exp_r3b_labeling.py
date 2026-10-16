"""
R3-B — Bottom-up labeling reliability + definition of "media context" (reviewer ZB3A RC2:
"how reliable is the LLM at reconstructing scene/profile from the spoken response and media
context? what does media context include?"). Overlaps NhWo C5 (QA rejection).

Two deliverables:
  (1) A precise definition of "media context" recovered from the construction pipeline
      (documented below with source paths; the models and exact inputs the labeling LLMs
      received).
  (2) A measured reliability signal from the released + raw annotations: humans rated the
      bottom-up-generated profile/scene as fitting the real audio (Local Scene Fit / Global
      Coherence / Semantic Match on positives), and explicitly judged prompt sufficiency
      during annotation.

Run from analysis/:  PYTHONPATH=. python3 exp_r3b_labeling.py
Writes analysis/out/r3b_labeling.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common

FIT_DIMS = ["local_scene_fit", "global_story_coherence", "semantic_match"]
RAW_TEST_BASE = os.path.join(common.PROJECT, "preprocess/overall_process/evalset_v3/annotation_result.jsonl")

# --- (1) media-context definition, from source inspection (see report paths) ---
MEDIA_CONTEXT_DOC = """\
## 1. What "media context" is (recovered from the construction pipeline)

The bottom-up labels are generated in two stages; the transcript is NEVER modified.

**Profile** — `preprocess/global_process/scripts/guess_global_story.py`, model **Gemini-2.0-flash**.
Inputs given to the LLM:
  - the full scene dialogue, turn-by-turn as `[Speaker X]: text` (all participants, in order);
  - the target speaker's emotion caption and audio caption (from a prior Gemini audio-understanding pass);
  - instruction to focus the profile on the target speaker, forbid accent mention, forbid
    hedging, and keep it internally consistent (<100 words).
  It does NOT receive the episode name, subtitles, or any external character database — the
  profile is inferred from dialogue + audio-derived captions only.

**Scene (local context)** — `preprocess/local_process/scripts/guess_local_story_gen.py`, model **DeepSeek-V3-0324**.
Inputs: the character profile (from stage 1), the full scene transcript, a focal span of
three consecutive target-speaker sentences, and optional episode info when available. Output
is a structured JSON (Immediate_Situation, Character_Intention, Emotional_Subtext,
Emotion_Subtext_List, Character_Dynamics, Overall_Summary).

**Negatives** — `preprocess/negative_global/` (role) and `preprocess/negative_local/` (context),
three graded severity levels (L0 surface → L1 tone → L2 fundamental contradiction). Only the
profile/scene text is perturbed; the audio and transcript stay fixed (see E8/E9).

So "media context" = turn-by-turn scene dialogue + audio-derived emotion/audio captions
(profile), and transcript + profile + focal span (scene). We will state this verbatim in
the §realism-data paragraph (RC2).
"""


def load(fp):
    return [json.loads(l) for l in open(fp)]


def validation_block():
    lines = ["## 2. Human validation of the bottom-up labels", "",
             "Positives are real clips whose profile/scene were LLM-reconstructed; if the "
             "reconstruction fits the audio, humans should rate Local/Global/Semantic high.", "",
             "| Split | Local Scene Fit | Global Coherence | Semantic Match | conf | n |",
             "|---|---|---|---|---|---|"]
    res = {}
    for split, label, filt in [("realism_real", "test_realrec (2,164 bottom-up positives)", None),
                               ("realism_base", "test_base positives (210)", "positive")]:
        meta = common.load_metadata(split)
        recs = [r for r in meta.values() if (filt is None or r.get("sample_type") == filt)]
        cells = {}
        for d in FIT_DIMS:
            vals = [common.human_mean(r, d, "mask") for r in recs]
            vals = [v for v in vals if v is not None]
            cells[d] = (float(np.mean(vals)), float(np.mean([v >= 3 for v in vals])))
        conf = [common.human_mean(r, "confidence", "mask") for r in recs]
        conf = [v for v in conf if v is not None]
        lines.append(f"| {label} | {cells['local_scene_fit'][0]:.2f} "
                     f"({100*cells['local_scene_fit'][1]:.0f}% ≥3) | "
                     f"{cells['global_story_coherence'][0]:.2f} "
                     f"({100*cells['global_story_coherence'][1]:.0f}% ≥3) | "
                     f"{cells['semantic_match'][0]:.2f} "
                     f"({100*cells['semantic_match'][1]:.0f}% ≥3) | "
                     f"{np.mean(conf):.2f} | {len(recs)} |")
        res[split] = {d: {"mean": cells[d][0], "frac_ge3": cells[d][1]} for d in FIT_DIMS}
        res[split]["confidence"] = float(np.mean(conf))
    lines += ["",
              "_On the real recordings, ~98% of positives receive Local/Global/Semantic ≥ 3 "
              "with mean ≈ 3.6-3.7 and annotator confidence 4.2/5 — humans confirm the "
              "LLM-reconstructed profile/scene fits the actual audio. The released splits are "
              "also uniformly QA-passed (`prompt_quality_pass`=True, `contradictory`=False, "
              "`sufficient`=True for all)._", ""]
    return lines, res


def sufficiency_block():
    """Vote-level prompt-sufficiency from the raw test_base annotations."""
    recs = load(RAW_TEST_BASE)
    val = collections.Counter()
    for r in recs:
        for k in r:
            if k.startswith("annotation") and k.endswith("_prmopt_quality"):
                v = r[k]
                v = (v[0] if isinstance(v, list) and v else v)
                if not v:
                    continue
                lv = v.lower()
                is_suff = ("insufficient" not in lv) and ("sufficient" in lv or "valid" in lv)
                val["sufficient" if is_suff else "insufficient"] += 1
    tot = sum(val.values())
    suff_rate = val["sufficient"] / tot if tot else float("nan")
    lines = ["## 3. Prompt-sufficiency: the human verdict on the LLM reconstructions", "",
             "During annotation each rater judged whether the reconstructed profile/scene was "
             "*sufficient to evaluate the voice sample* (`prmopt_quality`). Aggregated across "
             "all raters × all 2,000 test_base samples "
             f"(`{os.path.relpath(RAW_TEST_BASE, common.PROJECT)}`):", "",
             f"- **{val['sufficient']:,} / {tot:,} = {100*suff_rate:.1f}%** of rater "
             f"judgements = *sufficient*; {val['insufficient']:,} ({100*(1-suff_rate):.1f}%) "
             f"= insufficient.",
             "- The released dataset retains only samples that passed this check "
             "(released `prompt_quality_pass` is uniformly True), so the released bottom-up "
             "labels are the human-validated subset. Full pre-filter rejection logs were not "
             "retained in the release (documented-only; see E8/E9).", ""]
    return lines, {"vote_sufficient": val["sufficient"], "vote_insufficient": val["insufficient"],
                   "sufficiency_rate": suff_rate}


def main():
    out = {}
    md = ["# R3-B — Bottom-up labeling reliability & 'media context' definition", "",
          "Reviewer ZB3A RC2. Establishes what the labeling LLM saw and that humans validated "
          "the reconstructions.", "", MEDIA_CONTEXT_DOC, ""]
    l, r = validation_block(); md += l; out["validation"] = r
    l, r = sufficiency_block(); md += l; out["sufficiency"] = r
    md += ["## Integration", "",
           "- **W-media-context (RC2):** one paragraph in §realism-data stating the section 1 "
           "definition (models + exact inputs; transcript unmodified).",
           "- **W-bottom-up-reliability (RC2):** short paragraph + appendix pointer with the "
           "section 2/3 numbers (positives' fit scores + 97% sufficiency verdict). Frame: the "
           "bottom-up labels are LLM-*proposed* but human-*validated*; humans are the ground "
           "truth, the LLM is a proposal mechanism whose output they accept ~98% of the time.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "r3b_labeling.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "r3b_labeling.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "r3b_labeling.md"))
    print("\n--- R3-B HIGHLIGHTS ---")
    v = out["validation"]["realism_real"]
    print(f"  test_realrec positives Local {v['local_scene_fit']['mean']:.2f} "
          f"Global {v['global_story_coherence']['mean']:.2f} Semantic {v['semantic_match']['mean']:.2f}")
    print(f"  prompt sufficiency rate {100*out['sufficiency']['sufficiency_rate']:.1f}%")


if __name__ == "__main__":
    main()
