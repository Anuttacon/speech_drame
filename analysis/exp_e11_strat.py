"""
E11 (stratification only) — Semantic-match text-visibility strata for the negatives
(reviewer gEEq G1.2). The full E11 text-only / audio-ablated baselines need GPU/cluster
inference; this offline piece quantifies how much of the negative signal is even
text-detectable in principle.

Premise (verified, PLAN_review2): DRAME-Eval's input is [audio] + profile/scene/traits +
rubric; the transcript is NEVER provided, so a "text shortcut" already requires ASR from the
audio. Independently, a negative is only text-detectable if its content is inconsistent with
the (perturbed) profile/scene — measured by the annotators' semantic_match score:
  - sm ≥ 4  : content fully consistent with the perturbed prompt → mismatch carried ONLY by
              prosody/delivery → TEXT-INVISIBLE (an oracle-transcript judge has ~0 signal).
  - 3 ≤ sm < 4 : partial.
  - sm < 3  : content itself is off → text-detectable in principle (humans see it too).

Run from analysis/:  PYTHONPATH=. python3 exp_e11_strat.py
Writes analysis/out/e11_strat.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common


def sm(rec):
    v = rec.get("annotations", {}).get("semantic_match")
    return float(v) if v is not None else None


def strat(recs):
    vals = [sm(r) for r in recs]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    ge4 = np.mean([v >= 4 for v in vals])
    mid = np.mean([3 <= v < 4 for v in vals])
    lt3 = np.mean([v < 3 for v in vals])
    return {"n": n, "sm_ge4": float(ge4), "sm_mid": float(mid), "sm_lt3": float(lt3),
            "mean": float(np.mean(vals))}


def main():
    meta = common.load_metadata("realism_base")
    recs = list(meta.values())
    groups = {"role-negative": [r for r in recs if r.get("negative_type") == "role"],
              "context-negative": [r for r in recs if r.get("negative_type") == "context"],
              "positive": [r for r in recs if r.get("sample_type") == "positive"]}
    md = ["# E11 (stratification) — semantic-match text-visibility of negatives (test_base)", "",
          "How much of each negative type is even detectable from content alone. sm≥4 = "
          "content consistent with the perturbed prompt → mismatch is prosody-only → invisible "
          "to a text/oracle-transcript judge. sm<3 = content itself off → text-detectable.", "",
          "| Group | n | sm≥4 (text-invisible) | 3≤sm<4 (partial) | sm<3 (text-detectable) | mean sm |",
          "|---|---|---|---|---|---|"]
    out = {}
    for g, gr in groups.items():
        s = strat(gr)
        if not s:
            continue
        md.append(f"| {g} | {s['n']} | {100*s['sm_ge4']:.1f}% | {100*s['sm_mid']:.1f}% | "
                  f"{100*s['sm_lt3']:.1f}% | {s['mean']:.2f} |")
        out[g] = s
    md += ["",
           f"_**{100*out['role-negative']['sm_ge4']:.1f}% of role negatives** and "
           f"**{100*out['context-negative']['sm_ge4']:.1f}% of context negatives** are "
           f"text-invisible (sm≥4): their mismatch is carried purely by delivery/prosody, "
           f"where an oracle-transcript text judge has no signal by construction. Roughly "
           f"half have sm<3 (content-detectable — those were designed to be catchable given "
           f"the context swap; humans use the same text). This is the offline half of E11; "
           f"the text-only and audio-ablated DRAME baselines (which quantify the shortcut "
           f"inside our own model) require cluster inference and are deferred._", "",
           "## Integration", "",
           "- Directly answers gEEq G1.2's premise: a large fraction of the benchmark's "
           "discrimination is prosody-only and cannot be a text shortcut. Report these three "
           "strata; commit to publishing the audio-ablated/text-only baseline numbers when "
           "the cluster run lands.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e11_strat.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e11_strat.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e11_strat.md"))
    print("\n--- E11-strat HIGHLIGHTS ---")
    for g in ("role-negative", "context-negative"):
        s = out[g]
        print(f"  {g:16s} sm>=4 (text-invisible) {100*s['sm_ge4']:.1f}%  sm<3 (detectable) {100*s['sm_lt3']:.1f}%")


if __name__ == "__main__":
    main()
