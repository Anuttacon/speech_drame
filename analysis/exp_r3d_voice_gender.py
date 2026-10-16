"""
R3-D — Archetype role↔gender analysis (reviewer ZB3A RC5/RC6, the archetype side of the
stereotype concern). Optional/heavier per PLAN_review3 decision #2.

The reviewer asks whether assigning a social role (e.g. "roboticist") pushes the system
toward a gendered voice. Two separable questions:
  (A) Does the BENCHMARK encode a role↔gender mapping in its prompts? — offline-measurable.
  (B) Does the SFnd SYSTEM choose a gendered voice for a gender-neutral role prompt? —
      requires a gender classifier over the 1,499 synthesized clips (GPU; archetype ships
      NO speech_attributes, unlike realism). Documented + offered, not run here.

This script answers (A) and characterises the role set, which is the part we can measure and
the strongest evidence for the concession framing.

Run from analysis/:  PYTHONPATH=. python3 exp_r3d_voice_gender.py
Writes analysis/out/r3d_voice_gender.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os
import re

import common

EN_MALE = r"\b(he|him|his|man|male|father|husband|son|brother|boy|mr)\b"
EN_FEMALE = r"\b(she|her|hers|woman|female|mother|wife|daughter|sister|girl|mrs|ms)\b"
ZH_MALE = r"他|男|父|夫|哥|弟|爸|儿子|先生"
ZH_FEMALE = r"她|女|母|妻|姐|妹|妈|女儿|小姐|太太"


def prompt_gender(text, lang):
    if not text:
        return None
    if lang == "en":
        he = len(re.findall(EN_MALE, text, re.I)); she = len(re.findall(EN_FEMALE, text, re.I))
    else:
        he = len(re.findall(ZH_MALE, text)); she = len(re.findall(ZH_FEMALE, text))
    if he == 0 and she == 0:
        return None
    return "Male" if he > she else ("Female" if she > he else None)


def main():
    meta = common.load_metadata("archetype")
    recs = list(meta.values())
    # distinct prompts (each voiced by ~8 SFnd systems)
    distinct = {}
    for r in recs:
        distinct.setdefault(r.get("prompt"), r)
    out = {}
    md = ["# R3-D — Archetype role↔gender (benchmark side)", "",
          "Reviewer ZB3A RC5/RC6, archetype track. Archetype ships no voice attributes, so "
          "the voice-choice question (B) needs a GPU classifier; here we measure the "
          "benchmark-side question (A).", ""]

    # (A) prompt gender neutrality
    cov = collections.Counter()
    for r in distinct.values():
        cov[prompt_gender(r.get("prompt"), r.get("language")) is not None] += 1
    n = len(distinct)
    neutral_pct = 100 * cov[False] / n
    md += ["## (A) Do role prompts encode gender? (measured)", "",
           f"- Distinct role×scene prompts: **{n}**.",
           f"- Prompts with any gender marker (pronouns/kinship/honorifics): **{cov[True]}** "
           f"({100*cov[True]/n:.1f}%); **gender-neutral: {cov[False]} ({neutral_pct:.1f}%)**.",
           "",
           f"_The benchmark does **not** prescribe a role↔gender mapping: {neutral_pct:.0f}% "
           f"of role prompts carry no gender marker at all (roles like 'gardening enthusiast', "
           f"'firefighter' are stated without gender). This directly rebuts the premise that "
           f"the dataset assigns genders to occupations — it does not._", ""]
    out["prompt_gender"] = {"n_distinct": n, "with_marker": cov[True], "neutral": cov[False],
                            "neutral_pct": neutral_pct}

    # role_category balance
    rc = collections.Counter(r.get("role_category") for r in recs)
    md += ["## Role-category composition (test)", "",
           "| role_category | n |", "|---|---|"]
    for k, v in rc.most_common():
        md.append(f"| {k} | {v} |")
    md += ["", "_Categories are role types (social identity, occupation, personality trait, "
           "named character, …), not demographic labels; none carries a prescribed gender._", ""]
    out["role_category"] = dict(rc)

    # (B) documented — the voice-choice risk we cannot measure offline
    n_clips = len(recs)
    n_systems = len(set(r.get("sfnd", {}).get("system") for r in recs))
    md += ["## (B) Does the SFnd system pick a gendered voice for a neutral role? (not run)", "",
           f"This is the residual, legitimate risk: given a gender-neutral prompt, a "
           f"text/speech system may default to a gendered voice, and the human "
           f"*Appropriateness* rating could reward stereotype-congruent choices. Measuring it "
           f"requires running a voice-gender classifier over the {n_clips} synthesized clips "
           f"({n_systems} SFnd systems) and cross-tabulating detected voice gender against "
           f"role_category — archetype ships no `speech_attributes`, so this is a GPU task, "
           f"not an offline recompute.",
           "",
           "**Recommendation (PLAN_review3 decision #2):** document this as a limitation and "
           "acknowledged risk in the broader-impact section (concede for the archetype "
           "track), and offer to run the voice-gender-by-role measurement if the reviewer "
           "wants the number. Do NOT assert the systems are unbiased — we have not measured "
           "their voice choices.", "",
           "## Integration (W-ethics, archetype concession)", "",
           "- Use (A) to rebut the 'dataset assigns gender to occupations' reading: role "
           f"prompts are {neutral_pct:.0f}% gender-neutral.",
           "- Concede (B): synthesis systems + Appropriateness scoring could still encode/"
           "reward voice-demographic stereotypes; the benchmark does not currently audit the "
           "systems' voice choices. Mitigations: no demographic ranking claims, intended-use "
           "terms, release of prompts so others can audit, future voice-gender audit.", ""]
    out["voice_choice_measured"] = False
    out["n_clips"] = n_clips
    out["n_systems"] = n_systems

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "r3d_voice_gender.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "r3d_voice_gender.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "r3d_voice_gender.md"))
    print("\n--- R3-D HIGHLIGHTS ---")
    print(f"  role prompts gender-neutral: {neutral_pct:.1f}% ({cov[False]}/{n} distinct)")
    print(f"  voice-choice audit (B): NOT run (needs GPU gender classifier on {n_clips} clips)")


if __name__ == "__main__":
    main()
