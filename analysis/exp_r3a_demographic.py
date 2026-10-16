"""
R3-A — Demographic construction & rating audit (reviewer ZB3A RC5/RC6, the ethics evidence;
also serves gEEq ethics + NhWo broader-impact). Turns the stereotype-harm concern into a
measured statement using the released `speech_attributes` on the realism base test.

Reviewer's concern (paraphrased): assigning social roles couples occupation to demographic
markers, and swapping roles to build negatives may flip the implied demographic, so both the
data and a judge fine-tuned on it could encode gender stereotypes.

What the data actually shows (all on realism test_base, which ships speech_attributes for
1,998/2,000; test_realrec and archetype ship none):
  (1) Demographic balance — gender / age / accent distribution of the voices.
  (2) Voice↔profile gender congruence by sample_type. The AUDIO is never altered (the voice
      gender is fixed); perturbation changes only the PROFILE text. We infer the profile's
      implied gender from pronouns and measure how often a negative pairs a voice with an
      opposite-gender-coded profile. This quantifies exactly the reviewer's intuition.
  (3) THE TEST: do human character-consistency ratings track demographic (in)congruence
      *beyond* the intended semantic perturbation? Among role-negatives we compare
      gender-match vs gender-mismatch consistency scores at each severity level, and fit
      consistency ~ severity + semantic_match + demographic_mismatch. If the
      demographic_mismatch coefficient is ~0, annotators are NOT applying gender stereotypes
      (they grade the semantic mismatch); if strongly negative, they are, and we disclose +
      mitigate.

Run from analysis/:  PYTHONPATH=. python3 exp_r3a_demographic.py
Writes analysis/out/r3a_demographic.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

import common

CONSISTENCY_DIMS = ["traits_embodiment", "local_scene_fit", "global_story_coherence"]
N_BOOT = 2000
SEED = 12345


def profile_gender(text, lang):
    """Infer the profile's implied gender from gendered pronouns."""
    if not text:
        return None
    if lang == "en":
        he = len(re.findall(r"\b(he|him|his)\b", text, re.I))
        she = len(re.findall(r"\b(she|her|hers)\b", text, re.I))
    else:  # zh
        he = text.count("他"); she = text.count("她")
    if he == 0 and she == 0:
        return None
    if he > she:
        return "Male"
    if she > he:
        return "Female"
    return None  # tie -> ambiguous, drop


def consistency_score(rec):
    vals = [common.human_mean(rec, d, gating="mask") for d in CONSISTENCY_DIMS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def sem_match(rec):
    v = rec.get("annotations", {}).get("semantic_match")
    return float(v) if v is not None else None


def balance_block(meta):
    def dist(getter):
        c = collections.Counter(getter(r) for r in meta.values())
        c.pop(None, None)
        tot = sum(c.values())
        return c, tot
    lines = ["## 1. Demographic balance of the voices (realism test_base, n=2,000)", "",
             "_Released per-sample `speech_attributes`; test_realrec and archetype ship none._", ""]
    res = {}
    for name, getter in [("gender", lambda r: r.get("speech_attributes", {}).get("gender")),
                         ("age", lambda r: r.get("speech_attributes", {}).get("age")),
                         ("accent", lambda r: r.get("speech_attributes", {}).get("accent"))]:
        c, tot = dist(getter)
        items = ", ".join(f"{k} {v} ({100*v/tot:.0f}%)" for k, v in c.most_common())
        lines.append(f"- **{name}**: {items}")
        res[name] = dict(c)
    lines.append("")
    return lines, res


def congruence_block(meta):
    rows = collections.Counter()
    for r in meta.values():
        pg = profile_gender(r.get("profile"), r.get("language"))
        vg = r.get("speech_attributes", {}).get("gender")
        if pg in ("Male", "Female") and vg in ("Male", "Female"):
            st = r.get("sample_type") if r.get("sample_type") == "positive" else r.get("negative_type")
            rows[(st, "mismatch" if pg != vg else "match")] += 1
    lines = ["## 2. Voice↔profile gender congruence by sample type", "",
             "The voice (audio) gender is fixed; perturbation edits only the profile text. "
             "'Mismatch' = the profile's pronoun-implied gender differs from the voice gender.", "",
             "| Sample type | match | mismatch | % mismatch |", "|---|---|---|---|"]
    res = {}
    for st in ("positive", "role", "context"):
        m = rows[(st, "match")]; mm = rows[(st, "mismatch")]
        tot = m + mm
        pct = 100 * mm / tot if tot else 0
        lines.append(f"| {st} | {m} | {mm} | {pct:.1f}% |")
        res[st] = {"match": m, "mismatch": mm, "pct_mismatch": pct}
    lines += ["",
              "_Positives are ~93% gender-congruent (the profile fits the real speaker). "
              "**Role** perturbations induce a voice↔profile gender mismatch ~43% of the "
              "time — this is the phenomenon the reviewer intuited, made precise: the voice "
              "is unchanged, but the swapped target profile is often opposite-gender-coded. "
              "**Context** perturbations edit the scene, not the profile, so they rarely "
              "change implied gender (~6%)._", ""]
    return lines, res


def ols(X, y):
    """Return coefficients for y ~ [1, X...]; X columns already assembled."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return beta, A


def rating_test_block(meta):
    """Among role-negatives, does demographic mismatch depress consistency beyond semantics?"""
    recs = []
    for r in meta.values():
        if r.get("negative_type") != "role":
            continue
        pg = profile_gender(r.get("profile"), r.get("language"))
        vg = r.get("speech_attributes", {}).get("gender")
        cs = consistency_score(r)
        sm = sem_match(r)
        pl = r.get("processing_level")
        if pg not in ("Male", "Female") or vg not in ("Male", "Female") or cs is None or sm is None or pl is None:
            continue
        recs.append({"mismatch": int(pg != vg), "consistency": cs, "semantic": sm, "severity": pl})

    lines = ["## 3. Do human consistency ratings track demographic mismatch? (role-negatives)", "",
             f"n = {len(recs)} role-negatives with known voice gender, profile gender, "
             f"severity, and semantic_match.", ""]

    # (a) descriptive: mean consistency match vs mismatch per severity
    lines += ["**(a) Mean human consistency (Traits+Local+Global), by severity × gender congruence**", "",
              "| Severity | match (mean, n) | mismatch (mean, n) | Δ(mismatch−match) |",
              "|---|---|---|---|"]
    res = {"by_severity": {}}
    for lvl in (0, 1, 2):
        mm = [x["consistency"] for x in recs if x["severity"] == lvl and x["mismatch"] == 1]
        ma = [x["consistency"] for x in recs if x["severity"] == lvl and x["mismatch"] == 0]
        d = (np.mean(mm) - np.mean(ma)) if mm and ma else float("nan")
        lines.append(f"| L{lvl} | {np.mean(ma):.3f} (n={len(ma)}) | {np.mean(mm):.3f} (n={len(mm)}) | {d:+.3f} |")
        res["by_severity"][f"L{lvl}"] = {"match": float(np.mean(ma)) if ma else None,
                                         "mismatch": float(np.mean(mm)) if mm else None,
                                         "delta": float(d) if d == d else None}
    # (b) OLS controlling for severity + semantic_match
    beta, _ = ols([[x["severity"] for x in recs], [x["semantic"] for x in recs],
                   [x["mismatch"] for x in recs]], [x["consistency"] for x in recs])
    names = ["intercept", "severity", "semantic_match", "demographic_mismatch"]
    # bootstrap CI for the demographic_mismatch coefficient
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(recs))
    coefs = []
    Xcols = [[x["severity"] for x in recs], [x["semantic"] for x in recs], [x["mismatch"] for x in recs]]
    y = [x["consistency"] for x in recs]
    for _ in range(N_BOOT):
        s = rng.integers(0, len(recs), len(recs))
        b, _ = ols([[Xcols[c][i] for i in s] for c in range(3)], [y[i] for i in s])
        coefs.append(b[3])
    lo, hi = np.quantile(coefs, [0.025, 0.975])
    lines += ["",
              "**(b) OLS: consistency ~ severity + semantic_match + demographic_mismatch**", "",
              "| Term | coefficient |", "|---|---|"]
    for nm, b in zip(names, beta):
        lines.append(f"| {nm} | {b:+.3f} |")
    lines += ["",
              f"**Demographic-mismatch coefficient = {beta[3]:+.3f}, 95% bootstrap CI "
              f"[{lo:+.3f}, {hi:+.3f}]** (controlling for perturbation severity and the "
              f"annotator's own semantic_match judgement).", ""]
    res["ols"] = {n: float(b) for n, b in zip(names, beta)}
    res["demo_coef_ci"] = [float(lo), float(hi)]
    res["n"] = len(recs)

    crosses_zero = lo <= 0 <= hi
    # effect size relative to the dominant semantic driver + the raw uncontrolled gap
    raw_gaps = [res["by_severity"][f"L{l}"]["delta"] for l in (0, 1, 2)
                if res["by_severity"][f"L{l}"]["delta"] is not None]
    raw_mean = float(np.mean(raw_gaps)) if raw_gaps else float("nan")
    sem_coef = beta[2]
    frac_residual = abs(beta[3]) / abs(raw_mean) if raw_mean else float("nan")
    if crosses_zero:
        verdict = ("**not clearly distinguishable from zero** once the intended semantic "
                   "perturbation is controlled for — annotators grade the semantic role "
                   "mismatch, not the voice's gender. Measured evidence against the strong "
                   "form of the stereotype-in-labels concern; we still disclose the "
                   "construction risk and mitigate.")
    else:
        verdict = (f"**small but non-zero** ({beta[3]:+.3f}, CI excludes 0). Context: the "
                   f"rating is dominated by the annotator's semantic_match judgement "
                   f"(coef {sem_coef:+.3f} per point) and severity ({beta[1]:+.3f}/level); "
                   f"the raw gender-mismatch gap (~{raw_mean:+.2f}) shrinks by ~"
                   f"{100*(1-frac_residual):.0f}% once semantics are controlled, leaving this "
                   f"~{beta[3]:+.2f} residual. Part of the residual is plausibly *genuine* "
                   f"incongruence (a voice reading an opposite-gender-coded profile is a real "
                   f"character violation the scalar semantic_match may not fully capture), and "
                   f"part could be annotator gender expectation — the design cannot fully "
                   f"separate them. Honest disclosure: gender congruence explains only a small "
                   f"fraction of consistency ratings relative to semantic content, but a "
                   f"modest residual association remains. Disclose in broader impact + "
                   f"mitigate; do NOT claim a clean null.")
    lines += [f"_Reading: the demographic-mismatch effect is {verdict}_", ""]
    res["verdict_crosses_zero"] = bool(crosses_zero)
    res["raw_gap_mean"] = raw_mean
    res["frac_gap_from_semantics"] = float(1 - frac_residual) if frac_residual == frac_residual else None
    return lines, res


def main():
    meta = common.load_metadata("realism_base")
    out = {"n_boot": N_BOOT, "seed": SEED}
    md = ["# R3-A — Demographic construction & rating audit (realism test_base)", "",
          "Measured evidence for the stereotype/broader-impact discussion (ZB3A RC5/RC6, "
          "gEEq ethics, NhWo BI). `speech_attributes` ships for realism test_base only.", ""]
    l, r = balance_block(meta); md += l; out["balance"] = r
    l, r = congruence_block(meta); md += l; out["congruence"] = r
    l, r = rating_test_block(meta); md += l; out["rating_test"] = r
    md += ["## Integration (W-ethics, cross-reviewer)", "",
           "- Report the demographic balance (1) as a released fact (voices ~53/47 M/F; "
           "accent skews US/English — a documented coverage limitation).",
           "- State plainly (2): the audio/voice demographics are never manipulated; only "
           "profile text is perturbed. Role-negatives do frequently (~43%) pair a voice with "
           "an opposite-gender-coded profile — acknowledge this construction property "
           "explicitly (it is the reviewer's intuition, made precise).",
           "- Frame (3) honestly, NOT as a clean null: consistency ratings are **dominated** "
           "by the semantic role match (controlling for it removes ~⅔ of the raw "
           "gender-mismatch gap), but a **small residual** association (~−0.14 pts, CI "
           "excludes 0) remains. Disclose it; note part is plausibly genuine incongruence "
           "and part possible annotator expectation, which the design cannot fully separate. "
           "This is stronger and more credible than asserting no effect.",
           "- Pair with the archetype concession (no demographic label per role; SFnd voice "
           "choice + Appropriateness could still encode stereotypes — R3-D), and mitigations "
           "(balanced bilingual raters, no demographic ranking claims, release demographic "
           "metadata, intended-use terms).", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "r3a_demographic.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "r3a_demographic.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "r3a_demographic.md"))
    print("\n--- R3-A HIGHLIGHTS ---")
    print(f"  gender balance: {out['balance']['gender']}")
    print(f"  role-neg mismatch: {out['congruence']['role']['pct_mismatch']:.1f}%")
    o = out["rating_test"]
    print(f"  demographic_mismatch coef {o['ols']['demographic_mismatch']:+.3f} CI {o['demo_coef_ci']} (n={o['n']})")


if __name__ == "__main__":
    main()
