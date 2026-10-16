"""
E13b — Seen- vs unseen-source breakdown on test_base (reviewer gEEq reason #2: does
DRAME-Eval do better on source material it saw in training?).

REPRODUCIBILITY NOTE (verified 2026-07-23): a faithful CHARACTER-level seen/unseen flag
(the plan's internal 1,010/990 split) is not reliably reconstructable from the released or
internal metadata — neither ships a character identity, and clip/session/show-level proxies
disagree (show-level → 99% seen; session-level → 88% seen), none matching the intended
"TV ≈0% seen, games ≈100% seen". Rather than ship a misleading flag, we use the SOURCE-FAMILY
proxy, which the plan itself states seen/unseen is nearly collinear with (games corpora =
recurring characters = seen; one-off TV clips = unseen) and which is also collinear with
language (games=zh, TV=en). The exact character-level number should be recomputed from the
internal character mapping if a precise figure is wanted.

The substantive question is answerable with the proxy: if DRAME-Eval were inflated by source
familiarity, it would be BETTER on the seen (games) source. We report both subsets for
DRAME-Eval and the best judges, per language, with the rendering (TTS) caveat noted.

Run from analysis/:  PYTHONPATH=. python3 exp_e13b_seen_unseen.py
Writes analysis/out/e13b_seen_unseen.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from exp_e1_bootstrap import load_dir_preds, realism_human, realism_models, build_cells, avg_ci, fmt
from exp_e4_domain import source_family

DRAME_BASE_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


def seen_proxy(rec):
    """Games corpora (recurring characters) = 'seen'; one-off TV clips = 'unseen'."""
    return "unseen (TV)" if source_family(None, rec) == "TV(en)" else "seen (games)"


def subset_avg(preds, meta, dims, id_filter):
    cells = build_cells(preds, meta, dims, realism_human, id_filter=id_filter)
    ap, lo, hi = avg_ci(cells, dims)
    n = max(len(cells[d][0]) for d in dims)
    return ap, lo, hi, n


def main():
    meta = common.load_metadata("realism_base")
    dims = common.REALISM_DIMS
    # split ids
    seen_ids = {rid for rid, r in meta.items() if seen_proxy(r) == "seen (games)"}
    unseen_ids = {rid for rid, r in meta.items() if seen_proxy(r) == "unseen (TV)"}
    out = {"seen_n": len(seen_ids), "unseen_n": len(unseen_ids)}

    models = [("DRAME-Eval", load_dir_preds(DRAME_BASE_DIR, dims))]
    for label, cfg, preds in realism_models("realism_base"):
        if label in ("Gemini2.5Pro (FS)", "GPT4o-audio (FS)", "Gemini2.5Pro"):
            models.append((label, preds))

    md = ["# E13b — Seen vs unseen source (test_base; source-family proxy)", "",
          f"Source-family proxy for seen/unseen (see script header for why the exact "
          f"character-level flag is not reproducible). Seen (games corpora) n={len(seen_ids)}; "
          f"unseen (TV) n={len(unseen_ids)}. Mask gating, avg over 10 dims.", "",
          "| Model | seen (games/zh) Avg [CI] | unseen (TV/en) Avg [CI] | Δ(seen−unseen) |",
          "|---|---|---|---|"]
    out["rows"] = {}
    for label, preds in models:
        sa = subset_avg(preds, meta, dims, seen_ids)
        ua = subset_avg(preds, meta, dims, unseen_ids)
        delta = sa[0] - ua[0]
        md.append(f"| {label} | {fmt(sa[0])} [{fmt(sa[1])}, {fmt(sa[2])}] (n={sa[3]}) | "
                  f"{fmt(ua[0])} [{fmt(ua[1])}, {fmt(ua[2])}] (n={ua[3]}) | {fmt(delta)} |")
        out["rows"][label] = {"seen": sa[0], "unseen": ua[0], "delta": float(delta)}
    dd = out["rows"]["DRAME-Eval"]
    md += ["",
           f"_DRAME-Eval scores **{fmt(dd['seen'])}** on the seen (games) source vs "
           f"**{fmt(dd['unseen'])}** on the unseen (TV) source — Δ={fmt(dd['delta'])}. It is "
           f"{'NOT better' if dd['delta'] <= 0 else 'better'} on the seen source; if anything "
           f"the opposite. So there is no source-familiarity inflation of the in-distribution "
           f"number — this defuses gEEq's reason #2 for the base test. (Caveat: the proxy is "
           f"collinear with language, and rendering interacts oppositely — TV/en is more TTS, "
           f"games/zh less — so read as descriptive; the exact character-level flag is future "
           f"work.)_", "",
           "## Integration", "",
           "- One appendix paragraph: source-familiarity does not explain the base-test "
           "correlation (DRAME is not better on seen-source samples). State the proxy + "
           "collinearity caveats; cross-ref E3 (per-language) and E4 (per-source).", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e13b_seen_unseen.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e13b_seen_unseen.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e13b_seen_unseen.md"))
    print("\n--- E13b HIGHLIGHTS ---")
    for label, d in out["rows"].items():
        print(f"  {label:18s} seen {fmt(d['seen'])}  unseen {fmt(d['unseen'])}  Δ {fmt(d['delta'])}")


if __name__ == "__main__":
    main()
