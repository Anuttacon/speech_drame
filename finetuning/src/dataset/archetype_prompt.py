# =========================
# Omni Audio Gen Eval (v2)
# Last Updated: 2025-07-17
# =========================

system_prompt = "You are an expert evaluator of speech delivery and storytelling."

# Shared header for single-dimension scoring
base_header = (
    "Based on the given information, rate the following SINGLE dimension on a 1–5 scale (1 = poor, 5 = excellent).\n"
    "Return ONLY a number between 1 and 5.\n"
)

# Optional pre-check gate (returns "PASS" or "REJECT"). Use before scoring if desired.
rejection_gate_prompt = """You are a gatekeeper for evaluation.
Question: Is the audio a relevant response to the prompt and suitable for rubric scoring?
Return ONLY one of: PASS or REJECT."""

# ------- Detailed single-dimension prompts -------
detailed_archetype_dimension_prompts = {
    "content_pass": base_header +
        "Dimension: ContentPass — Context-free assessment of whether the audio is a relevant response to the prompt and suitable for rubric scoring.\n"
        "Return ONLY one of: 0 (reject) or 1 (pass).\n"
        "Example output: 1",
    # AUDIO QUALITY — context-free, *only* artifacts; to be ignored for other dimensions once recorded.
    "audio_quality": base_header +
        "Dimension: AudioQuality — Context-free assessment of digital/synthetic artifacts only.\n"
        "IMPORTANT: After scoring this dimension, pretend these issues DO NOT exist when scoring other dimensions.\n\n"
        "5 (Excellent): Clean/clear audio; no digital noise/artifacts.\n"
        "4 (Good): Minor artifacts on close listen (slight sharpness/hiss) but overall clean.\n"
        "3 (Fair): Noticeable artifacts (buzz/echo/CG texture) that impact clarity.\n"
        "2 (Poor): Frequent/consistent glitches or severe isolated issues (clipping, metallic resonance, harsh transitions).\n"
        "1 (Very Poor): Heavily degraded throughout (static/crackling, synthesis failure, severe clipping/pitch collapse).\n"
        "Example output: 4",

    # HUMAN LIKENESS — context-free naturalness; do NOT consider audio quality or contextual fit here.
    "human_likeness": base_header +
        "Dimension: HumanLikeness — Context-free naturalness/believability of the voice performance.\n"
        "Ignore audio quality defects and contextual fit (those are separate dimensions).\n\n"
        "5 (Definitely Human): Indistinguishable from a real person; rich intonation, real-time pacing, consistent identity; organic breaths/pauses.\n"
        "4 (Most Likely Human): Generally convincing; small hints of stiffness, overly even timing, or minor awkward pauses.\n"
        "3 (Could be Human or AI): Mixed natural/unnatural traits; some mechanical pauses/inflection but other parts human-like.\n"
        "2 (Mostly Likely AI): Clearly artificial speech patterns; choppy timing, poor modulation, flat/illogical inflections.\n"
        "1 (Definitely AI): Rigid, read-aloud monotony; no natural rhythm/pauses; entirely machine-like delivery.\n"
        "Example output: 3",

    # APPROPRIATENESS — context-dependent fit to role & scene; ignore audio quality and generic naturalness here.
    "appropriateness": base_header +
        "Dimension: Appropriateness — Context-dependent tonal fit to the prompt’s role AND scene.\n"
        "Judge stereotype-fit harshly; if any portion before 30s is ill-fitting, prefer the lower score. Ignore audio quality and generic naturalness.\n\n"
        "5 (Completely Appropriate): Fully aligns with role + scene (emotion, social dynamics, status, events); immersive and consistent.\n"
        "4 (Mostly Appropriate): Strong match with minor flaws (e.g., slight intensity/timing issues) that don’t break alignment.\n"
        "3 (Adequately Appropriate): Broadly fits both role and scene but lacks depth or has mild inconsistencies in key elements.\n"
        "2 (Slightly Appropriate): Only role or scene fit is acceptable; overall superficial/partial alignment; missing/mismatched cues.\n"
        "1 (Completely Inappropriate): Fails role and scene expectations; tone/status/urgency clearly mismatched.\n"
        "Example output: 2",
}

# ------- Simple single-dimension prompts (backward-compat) -------
detailed_archetype_dimension_prompts = {
    "content_pass": base_header +
        "Dimension: ContentPass — Context-free assessment of whether the audio is a relevant response to the prompt and suitable for rubric scoring.\n"
        "Return ONLY one of: 0 (reject) or 1 (pass).\n"
        "Example output: 1",
    "audio_quality": base_header +
        "Dimension: AudioQuality — Rate ONLY digital/synthetic artifacts (ignore content/performance).\n"
        "Example output: 4",
    "human_likeness": base_header +
        "Dimension: HumanLikeness — Rate ONLY human-like naturalness (ignore quality & context).\n"
        "Example output: 3",
    "appropriateness": base_header +
        "Dimension: Appropriateness — Rate ONLY tonal fit to the prompt’s role & scene (ignore quality & generic naturalness).\n"
        "Example output: 2",
}

# ------- General (multi-dimension) prompts -------
general_archetype_prompt_detailed = f"""

### 1) Content Pass (Context-Free)
Focus: whether the audio is a relevant response to the prompt and suitable for rubric scoring.

**Scale**
- 0 (Reject): Not a relevant response to the prompt or not suitable for rubric scoring.
- 1 (Pass): A relevant response to the prompt and suitable for rubric scoring.

---

#### 2) Audio Quality (Context-Free)
Focus: presence/severity of digital distortions or synthetic artifacts.  
Ignore context, ignore human-likeness, ignore appropriateness.  
After this dimension, **pretend these issues no longer exist** for subsequent scoring.

**Scale**
- 5 (Excellent): Clean/clear; no noticeable artifacts.  
- 4 (Good): Small imperfections (slight hiss/sharpness) but overall clean.  
- 3 (Fair): Noticeable artifacts (buzz, echo, synthetic texture) that affect clarity.  
- 2 (Poor): Frequent glitches, clipping, metallic resonance, or harsh transitions.  
- 1 (Very Poor): Dominated by artifacts; static/crackling, synthesis collapse, severe clipping/pitch collapse.  

---

#### 3) Human Likeness (Context-Free)
Focus: how human-like the delivery sounds.  
Ignore audio quality defects (already captured) and ignore contextual fit.  

**Scale**
- 5 (Definitely Human): Indistinguishable from a person; rich intonation, natural pacing, consistent identity, organic pauses/breaths.  
- 4 (Most Likely Human): Generally convincing; minor stiffness or overly even timing/pauses.  
- 3 (Could be Human or AI): Mix of natural and unnatural traits; mechanical in places, human-like in others.  
- 2 (Mostly Likely AI): Clearly artificial speech; choppy rhythm, flat or illogical inflections, awkward breaths.  
- 1 (Definitely AI): Entirely rigid, monotone, read-aloud style; no natural rhythm or pauses.  

---

#### 4) Appropriateness (Context-Dependent)
Focus: tonal fit to the role/scene described in the prompt.  
Ignore audio quality and generic naturalness. Judge harshly against stereotypical expectations. If ANY portion before 30s is off, score lower.

**Scale**
- 5 (Completely Appropriate): Fits both role and scene perfectly; fully aligned with emotion, social dynamics, status, events. Immersive.  
- 4 (Mostly Appropriate): Strong fit overall; minor flaws in timing/intensity/subtlety that don’t break alignment.  
- 3 (Adequately Appropriate): Broad fit; role/scene generally reflected but lacks depth or has small inconsistencies.  
- 2 (Slightly Appropriate): Only role OR scene fit is acceptable; superficial or partial attempt with missing/mismatched cues.  
- 1 (Completely Inappropriate): Fails role and scene; tone/urgency/status heavily mismatched.  


**Present results in a fixed order:** content_pass, audio_quality, human_likeness, appropriateness.

OUTPUT SCHEMA
<answer>[content_pass]0-1[audio_quality]1-5[human_likeness]1-5[appropriateness]1-5</answer>
"""

general_archetype_prompt = """Follow the rubric exactly and return ONLY the schema — no extra text.

DIMENSIONS (1–5, in order)
• content_pass — context-free assessment of whether the audio is a relevant response to the prompt and suitable for rubric scoring.
• audio_quality — artifacts/glitches only.
• HumanLikeness — naturalness only.
• Appropriateness — tonal fit to role & scene only.

OUTPUT SCHEMA
<answer>[content_pass]0-1[audio_quality]1-5[human_likeness]1-5[appropriateness]1-5</answer>

"""

# Optional: keep these addons if you still want think-aloud captions before the score.
archetype_caption_prompt = """In addition to the above rubric, provide a brief audio caption in <think> </think> BEFORE the answer."""
archetype_emotion_prompt = """In addition to the above rubric, provide a brief emotion caption in <think> </think> BEFORE the answer."""
