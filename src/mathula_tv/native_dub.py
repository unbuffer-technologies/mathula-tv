"""Two-pass natural isiZulu dubbing experiment.

Design contract
---------------
1. Pass 1 reads the whole English transcript and corrects only STT errors.
   The authoritative production transcript is never overwritten.
2. Pass 2 reads the whole restored transcript and emits one maximally natural
   isiZulu rendering per source block with a SOFT duration intent: raw 0%-rate TTS
   should land modestly long (default +6%) so final sync is achieved by a small
   pitch-preserving speed-up, never by slowing speech.
3. Azure TTS first speaks at rate_percent=0. Source video frames are never modified.
4. Native timing is source-anchored: a phrase starts at its source start. When enabled,
   a bounded part of an existing following source gap may absorb residual overflow only
   while preserving a protected pause before the next onset; later blocks never move.
5. Measured raw TTS is linguistically repaired only when it ends materially early.
   Overlong speech is always landed on the source mouth-close with pitch-preserving speed-up;
   +12% is a console quality-warning threshold, never a pipeline ceiling.
6. The canonical Pass-2 translation remains immutable once Pass 2 finishes; recasts and
   speed factors are separate auditable timing artifacts and never rewrite it. An optional,
   independent referent-fidelity audit/repair (see `referent_audit.py`) may still correct the
   canonical text as the last step *inside* Pass 2 itself, before anything downstream treats it
   as immutable -- this catches referent substitutions (e.g. "the minister" rendered as "God")
   that Pass 2's own inline self-QA cannot reliably see, because that self-QA shares the same
   reasoning trace as the mistake.
7. Web research is never automatic. Humans may install a reviewed override artifact.
"""
from __future__ import annotations

import difflib
import array
import functools
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema

from .atomic_io import atomic_write_json, read_json
from .azure_stt import AzureFastTranscriptionBackend, normalize as normalize_stt_response
from .azure_tts import (
    SILENCE_AMPLITUDE_THRESHOLD,
    AzureTTSBackend,
    AzureTTSIdempotencyError,
    AzureTTSRequest,
    AzureTTSResult,
)
from .azure_tts_sdk import (
    GROUP_WAV_SUFFIX,
    AzureSpeechSDKBoundary,
    synthesize_group_with_bookmarks,
    write_group_slices,
)
from .foundry_grok import (
    FOUNDRY_GROK_PROVIDER_VERSION,
    FoundryGrokProvider,
    GrokOutputTruncated,
)
from .media import checksum, probe, render_video
from .alignment import atempo_chain, wav_duration_ms
from .contextual_speaker_identity import resolve_contextual_speaker_identities
from .direct_azure_persona import (
    DIRECT_AZURE_PERSONA_VERSION,
    apply_direct_azure_personas,
    collect_source_features,
)
from .direct_voice_analysis import (
    ensure_canonical_speaker_reels,
    ensure_canonical_voice_family_analysis,
)
from .entity_registry import EntityRegistry
from .known_speakers import identify_known_speaker
from .speaker_mapping import build_speaker_id_mapping, canonicalize_voice_family_evidence
from .speech_islands import (
    ENGLISH_TITLE_ABBREVIATIONS,
    ZULU_TITLE_ABBREVIATIONS,
    build_island_phrase_groups,
    derive_english_islands,
    load_raw_word_index,
    slice_wav_by_bookmarks,
    split_into_sentences,
    stitch_islands_to_group_wav,
)
from .autocorrect_research import _OUTPUT_SCHEMA as _NAME_RESEARCH_OUTPUT_SCHEMA
from .autocorrect_research import validate_researched_corrections
from .azure_web_research import AzureResponsesWebResearchProvider
from .mixing import render_dialogue_tracks
from .models import JobManifest, utcnow
from .pronunciation import (
    _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS,
    PronunciationDictionary,
    with_default_organisation_initialisms,
    with_web_researched_organisation_pronunciations,
)
from .referent_audit import (
    DEFAULT_MAX_REFERENT_REPAIR_ROUNDS,
    DEFAULT_REFERENT_AUDIT_BATCH_SIZE,
    DEFAULT_REFERENT_AUDIT_WORKERS,
    REFERENT_AUDIT_SCHEMA_VERSION,
    audit_and_repair_pass2,
)
from .tts_ssml import BreakPart, TextPart
from .transcript_merge import reconcile
from .voice_prosody import bounded_prosody_int
from . import master_case_context
from . import zulu_lexicon
from . import zulu_morphology


NATIVE_DUB_SCHEMA_VERSION = "mathula-native-dub-v3.8-four-block-semantic-window-rush-nonfatal"
STT_RESTORATION_SCHEMA_VERSION = "mathula-native-stt-restoration-v2-cross-segment-duplication-guard"
NATURAL_TRANSLATION_SCHEMA_VERSION = "mathula-native-natural-translation-v2-soft-duration-intent"
ROLLING_NATURAL_TRANSLATION_SCHEMA_VERSION = "mathula-native-natural-translation-v3-rolling-syllable-budget"
TEMPORAL_MASK_TRANSLATION_SCHEMA_VERSION = "mathula-native-natural-translation-v12-phase-a-no-self-check"
TEMPORAL_MASK_PROMPT_VERSION = "native-temporal-mask-window-v10-phase-a-no-self-check"
CONTEXT_LEDGER_SCHEMA_VERSION = "mathula-native-context-ledger-v1"
ZULU_GLOSSARY_SCHEMA_VERSION = "mathula-native-zulu-glossary-v1"
ZULU_GLOSSARY_PROMPT_VERSION = "native-zulu-glossary-v2-formal-register-allows-code-switch"
CANDIDATE_POOL_SCHEMA_VERSION = "mathula-native-candidate-pool-v3-turn-blocks"
CANDIDATE_TRANSLATE_PROMPT_VERSION = "native-candidate-translate-v7-code-switch-shortened-candidate"
TURN_BLOCK_TRANSLATE_PROMPT_VERSION = "native-turn-block-translate-v31-droppable-parallel-list-items"
MANUAL_WEB_OVERRIDE_SCHEMA_VERSION = "mathula-native-manual-web-overrides-v1"
TIMING_REPAIR_SCHEMA_VERSION = "mathula-native-natural-timing-recast-v6-rhetorical-controller"
SPEECH_ISLANDS_SCHEMA_VERSION = "mathula-native-speech-islands-v1"
QA_BACK_TRANSLATION_SCHEMA_VERSION = "mathula-native-qa-back-translation-v3-mqm-severity-direct-adequacy"
TIMING_PLAN_SCHEMA_VERSION = "mathula-native-protected-gap-timing-plan-v3.4"
NATIVE_VOICE_RESOLUTION_SCHEMA_VERSION = "mathula-native-voice-resolution-v2-legacy-classifier-cache"

# Production timing defaults.  Raw TTS deliberately aims slightly long so the final
# operation is a small pitch-preserving acceleration rather than a slow-down.
DEFAULT_PREFERRED_RAW_SPEED_PERCENT = 6
# Real user direction, 2026-09-08: "The rush should always be less than 10%"
# -- lowered from 12. This single shared constant already drives every
# rush-related decision in this file (turn-envelope "fits" checks,
# _trim_by_fact_priority's trigger, render-time rush warnings), so this one
# change tightens the target everywhere consistently, without needing a
# separate edit per mechanism. _PROTECTED_CONTENT_LAST_RESORT_TRIGGER_PERCENT
# (the separate, deliberately-higher bar gating the more destructive
# protected-content-compression fallback) is NOT lowered to match -- it was
# specifically calibrated from a real, confirmed bug where a much smaller gap
# let normal turn-rebalancing residual (12-16%) wrongly trigger that
# destructive mechanism; revisit only with the same kind of real evidence.
DEFAULT_MAX_NATURAL_SPEED_PERCENT = 10
MOUTH_CLOSE_TARGET_TOLERANCE_MS = 40
MOUTH_CLOSE_HARD_TOLERANCE_MS = 80
# A residual "too short" gap this small (roughly a fraction of one syllable) cannot be
# closed reliably by asking a model to reword toward an exact millisecond target -- any
# real edit adds or removes at least a full syllable. Below this bound, and only when it
# fits inside existing free room before the next block, close it with trailing silence
# instead of failing the pipeline over a rounding-level residual.
MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS = 150
DEFAULT_QA_BACK_TRANSLATION_BATCH_SIZE = 15
DEFAULT_QA_MAX_REPAIR_ROUNDS = 1
# Matches DEFAULT_REFERENT_AUDIT_WORKERS's own reasoning: QA batches are fully
# independent of each other (each is its own back-translation/direct-adequacy
# call over a distinct slice of sentences), so there is nothing to serialize.
DEFAULT_QA_AUDIT_WORKERS = 4
# Same independent-batches reasoning as DEFAULT_QA_AUDIT_WORKERS, applied to
# candidate-pool translation/QA fan-out.
DEFAULT_CANDIDATE_POOL_WORKERS = 4
DEFAULT_CANDIDATE_POOL_TTS_WORKERS = 6
# Speaker-turn grouping constant (Phase 13). This gap threshold is a genuine
# "is this still one continuous utterance" signal (mirrors the ASR/diarization
# layer's own max_pause concept in transcript_merge.py), not an artificial
# budget cap -- a real speaker turn is otherwise UNBOUNDED: earlier versions of
# this grouping (Phase 9's "discourse segments") also capped combined duration
# at 30s and sentence count at 8, which fragmented genuine same-speaker turns
# for reasons having nothing to do with speaker or timing discontinuity.
# Confirmed by direct read: nothing about how sentences are chunked from
# ASR/diarization forces those caps -- they were a deliberate, now-removed
# choice inside the grouping function itself. This grouping is the PRIMARY,
# unconditional timing unit (see Phase 9/13) -- it is no longer an opt-in
# rebalancing fallback on top of per-sentence candidates.
SPEAKER_TURN_MAX_GAP_MS = 5000
# A "bridge" sentence (e.g. "All right.", "Paragraph four.") is a short interjection
# that carries no real discourse content -- confirmed in production that these are
# exactly the units that get forced into absurd required-speed numbers (114-158%)
# when judged against their own tiny native ASR window, and that region/segment
# redistribution correctly finds nothing to redistribute for them (a 2-word phrase
# has no content to move to a neighbor). Both conditions required (word count AND
# span) -- starting values calibrated against the two real confirmed examples; see
# the plan's Verification section for the dry-run validation step before relying on
# these in production.
_BRIDGE_MAX_WORD_COUNT = 4
_BRIDGE_MAX_SOURCE_SPAN_MS = 1500
# A bridge unit is judged only on whether it collides with its neighbors, not on
# fitting its own native window -- implemented by handing _effective_protected_late_
# allowance a much larger overflow budget than the flat DEFAULT_MAX_PROTECTED_GAP_
# OVERFLOW_MS. Safe by construction: that function's result is always clamped to
# the real free gap (min(free_gap_ms, ...)), so an oversized override can never
# overshoot into a neighbor's own speech regardless of its value.
BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS = 10_000
# Matches the schema's own per-segment cap on shortened_candidates -- see
# TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT and _turn_block_translate_schema.
_MAX_SHORTENED_CANDIDATES_PER_SEGMENT = 4
# Phase 16: a real, bounded A/B experiment against job 33cd7b46...'s first
# speaker turn confirmed translating a whole turn as ONE merged block (Grok
# free to fold/reorder content across original sentence boundaries) beats
# per-sentence-only editing decisively (35.1s vs a 38.64s window, comfortably
# fits, vs 48.0s -- barely moved -- for the strict 1:1 approach) on the exact
# case that motivated this: a genuinely redundant sentence that only becomes
# visible with the whole turn's context. This constant bounds how many
# original sentences get offered to ONE such merge-reasoning call -- a
# translation-BATCHING cap only, never a timing cap (a turn longer than this
# is still ONE timing unit, translated across multiple windows). Reuses the
# deleted DISCOURSE_SEGMENT_MAX_SENTENCES value for an unrelated reason.
_TURN_BLOCK_TRANSLATE_MAX_MEMBERS = 8
# Windows (not individual sentences) per provider call -- keeps one batch's
# prompt/response size bounded regardless of how many turns a job has.
DEFAULT_TURN_BLOCK_TRANSLATE_BATCH_SIZE = 5
# How many preceding groups' real English feed a window's recent_turns_digest
# (clause-ranking discourse context) -- bounded so a job-wide digest never
# grows unbounded; 3 covers the real confirmed case (a fact stated, then
# asked about, then redundantly restated a turn later).
_RECENT_TURNS_DIGEST_MAX_LOOKBACK = 3
# MQM-lite triage: each flagged sentence carries a category and a severity rather than
# a bare pass/fail. Only severities at or above this floor are auto-repaired -- a
# "minor" finding (e.g. a subtle register nuance a reviewer would let stand) is reported
# for visibility but does not force a Grok repair round, mirroring MQM's own
# minor/major/critical triage instead of treating every flag as equally urgent.
QA_REPAIR_SEVERITY_FLOOR = frozenset({"major", "critical"})
_QA_SEVERITY_RANK = {"none": -1, "minor": 0, "major": 1, "critical": 2}
_QA_SEVERITY_ENUM = ("none", "minor", "major", "critical")
_QA_CATEGORY_ENUM = (
    "none", "accuracy", "negation", "omission", "addition", "attribution", "structure", "fluency", "other",
)
# isiZulu marks negation through verb-prefix infixes (e.g. "-nga-") that are also used
# for potential/conditional mood -- exactly the ambiguity that produced a real, only
# partially-resolved flag in production (native_phrase_0119: "it could be viewed this
# way" vs. a back-translation reading "cannot be seen this way"). A negation-bearing
# English source sentence is never allowed to settle at "minor" severity even if a
# check under-calls it, since a missed negation flip silently reverses a claim.
_QA_NEGATION_CUE_PATTERN = re.compile(r"\b(?:not|never|no|none|nobody|nothing|neither|nor)\b|n't\b", re.IGNORECASE)
# 2026-08-29: a genuine rule system, not another incident report. CONFIRMED_TRANSLATION_PITFALLS
# below only ever encodes what has ALREADY gone wrong -- a real ceiling for spontaneous human
# speech, which keeps producing constructions no prior incident anticipated. This block instead
# states the actual isiZulu mood/concord/extension system as rules to APPLY to a novel sentence,
# not a pattern to match against a memorized example. Grounded in Wikipedia's "Zulu grammar"
# article (itself sourced from Doke's canonical reference grammar) and cross-validated against
# Pretorius/Bosch's formal CFG paper "Grammar rules for the isiZulu complex verb" (arXiv 1612.06581,
# read in full) -- that paper's own grammar independently defines separate "s" (positive subject
# concord: ngi/u/si/ba/...) and "ns" (negative subject concord: angi/awu/aka/...) terminal sets,
# confirming the primary/secondary concord distinction below is real, not a Wikipedia-only claim.
ZULU_GRAMMAR_FOUNDATION = r"""ISIZULU GRAMMATICAL FOUNDATION -- apply these as genuine rules to
derive the correct form for whatever sentence you are given, including constructions you have not
seen a specific worked example for. Do not default to whichever surface form is most familiar if
it does not match this sentence's actual mood, polarity, and semantic relation.

CONCORD TYPES -- isiZulu has FOUR distinct subject/object concord sets. Using the wrong TYPE for
the sentence's actual mood/polarity is a structural error, not a vocabulary choice:
- PRIMARY subject concord: positive indicative only (e.g. "ngi-", "u-", "ba-"). "Ngiyahamba" (I am
  going).
- SECONDARY subject concord: negative AND subjunctive mood -- a DIFFERENT set from primary (e.g.
  negative "angi-" vs. positive "ngi-"). "Angihambi" (I am not going); "ngihambe" (that I go).
- PARTICIPIAL subject concord: participial mood (subordinate/simultaneous clauses) -- identical to
  secondary except final "-a" becomes "-e". "Ekhuluma" (while speaking, class 1).
- OBJECT concord: infixed immediately before the verb stem; marks the object, never the subject.

MOOD/TENSE/NEGATION PARADIGM -- the mood category and its negation pattern are a matched PAIR;
never negate a form by copying a different mood's negation pattern onto it:
- Present: affirmative "[primary](ya)...a" / negative "a[secondary]...i".
- Recent past: affirmative "[primary]...ile/-e" / negative "a[secondary]...anga".
- Remote past: affirmative "[primary]a...a" / negative "a[secondary]...anga".
- Future: affirmative "[primary]zo(ku)...a" / negative "a[secondary]zu(ku)...a".
- Subjunctive (present): affirmative "[secondary]...e" / negative "[secondary]nga...i".
- Imperative: "(yi)...a" (bare) or "[obj]...e" (with an object); negate via the subjunctive
  negative pattern, not a bespoke imperative negative.
- Stative (a state resulting from a completed process, e.g. "Ufile" = he is dead): affirmative
  "[primary]...ile" / negative "a[primary]...ile" -- note stative negation keeps the PRIMARY
  concord, unlike every other mood above.
- Counterfactual ("ngabe"): "ngabe" + subject concord + verb stem + perfective "-e". "ngabe" is
  inherently a dedicated marker, not a modal to combine with the "-kwazi" ability stem -- see the
  confirmed pitfall on this exact failure below.

VERBAL EXTENSIONS -- each conveys a SPECIFIC semantic relation; never substitute one for another
as a stylistic paraphrase:
- Passive "-(i)wa": the grammatical subject undergoes the action; the doer may be marked with the
  "ngu-" agentive. Required whenever English marks "was/were [verb]-ed by X" -- use this, not an
  active-voice rewording, whenever preserving which party is agent vs. patient matters.
- Neuter-passive "-eka": a state or capability arising from the action, with NO agent named or
  implied (e.g. "-fundeka" = "to be readable") -- not interchangeable with true "-(i)wa" passive
  when an agent is present or implied by the English.
- Applicative "-ela": the action is performed FOR/TO/AT a beneficiary or location -- required
  whenever English marks a "for"/"to"/"at" argument as a benefactive or locative target of the
  verb, not merely its direct object.
- Causative "-isa": the subject CAUSES another party to perform the action -- required for English
  "made X do Y" / "had X do Y"; never used merely to intensify an action.
- Reciprocal "-ana": mutual action between parties ("-thandana" = to love each other).
- Intensive "-isisa": thorough/careful performance of the action, distinct from "very much"
  (an adverb of degree), which does not require this suffix at all."""

# Hand-maintained, not an automated accumulation database: each entry is a real
# mistranslation confirmed against a live job, fed into Pass 2's wire payload as a
# standing reminder. Add to this list only after confirming a new case the same way,
# never speculatively.
CONFIRMED_TRANSLATION_PITFALLS: tuple[str, ...] = (
    "English \"colleague\" does not mean isiZulu \"umlimi\" (a farmer) -- use \"umlingani\" "
    "or another word meaning a co-worker/associate.",
    "For an English clause like \"X, who [someone] told me was Y\" (a reporting verb nested "
    "inside a relative clause), do not let the isiZulu relative concord or a bare subject concord "
    "on a following clause attach to the person being described instead of the person doing the "
    "telling (e.g. \"Mogotsi, who HE told me was Y\" must not read as \"Mogotsi, who told me he "
    "was Y\"). Fix: REPEAT THE TELLER'S PROPER NOUN itself as the explicit subject of the telling "
    "clause/sentence (e.g. \"ULincoln wangitshela ukuthi...\") rather than relying on a relative "
    "construction or bare subject concord that can default to whichever name was most recently "
    "mentioned.",
    "For an affirmative English \"could have\" (e.g. \"I could have worded it better\"), use "
    "isiZulu's dedicated counterfactual marker \"ngabe\" + subject concord + verb stem + "
    "perfective \"-e\" (e.g. \"Ngabe ngikubeke kangcono\"). Do NOT combine \"ngabe\" with the "
    "\"-kwazi\" ability-modal stem in any form (with or without \"-nga-\", with \"-ile\" or "
    "otherwise) -- \"-kwazi\" constructions reliably back-translate as either \"was NOT able to\" "
    "or a plain past-tense \"WAS able to\", both of which lose the affirmative counterfactual "
    "meaning. Use \"ngabe\" with a plain content verb stem ending in perfective \"-e\" instead "
    "(e.g. \"ngikwenze\", \"ngikubeke\").",
    "For an English clause reporting who told/asked whom to do something, especially with an "
    "anonymized or partially-identified party (e.g. \"Witness G testifies that HE told HIM to "
    "meet HIM at Orlando SAPS\" -- three pronouns for at most two or three people), do not let "
    "isiZulu's subject/object concords default to the nearest or most salient referent "
    "(\"wamtshela\"-style constructions repeatedly leave who told whom ambiguous). Fix: when a "
    "clause has three or more pronouns/concords that could plausibly share a referent, REPEAT "
    "THE PROPER NOUN itself (e.g. \"Witness G\") in place of the reporting verb's subject "
    "concord/pronoun -- clarity of attribution outranks stylistic economy for testimony content. "
    "Only fall back to a concord/pronoun once the referent is unambiguous.",
    "For a short English denial/negation responding to a prior question or claim (e.g. \"No, you "
    "were not.\"), the denial's SCOPE is whatever the prior question/claim actually asserted "
    "(e.g. denying \"you were discussing X\"), not whatever noun is nearest in the isolated "
    "sentence -- and the MAIN VERB of the isiZulu denial must be a NEGATED form of the disputed "
    "ACTIVITY itself (e.g. a negative form of \"-xoxa\", to discuss), never a negated "
    "existential/locative verb (\"-khona\"/\"-yikho\", to be there/present) with the activity "
    "merely tacked on as a modifier (that denies presence, not the activity). E.g. \"Cha, "
    "anizange nixoxe ngalezo zindaba\" (\"No, you did not discuss those matters\"), with no "
    "existential/locative verb at all. When the denied content isn't explicit in the sentence, "
    "infer it from context_ledger/discourse rather than the most generic literal reading.",
    "For English \"[I/you] was at [someone]'s house/place\" (a locative predicate), the locative "
    "noun must keep its full \"e-...-ini\" shape as ONE word (\"indlu\" [house] -> \"endlini\" "
    "[at the house], never a bare \"ndlini\" with the \"e-\" augment dropped), with the possessor "
    "attached directly as \"ka-\" + name, no word break (\"endlini kaLincoln\", not \"endlini ka "
    "Lincoln\"). If a clean copulative fusion feels uncertain, prefer the existential "
    "construction with \"khona\" instead (\"[subject] ngangikhona endlini kaLincoln\" = "
    "\"[I/you] was there, at Lincoln's house\") -- a safe alternative avoiding the fusion "
    "boundary entirely.",
    "isiZulu's 2nd- and 3rd-person-singular (class 1) subject concords are IDENTICAL in several "
    "core tenses (\"u-\" present/recent-past, \"wa-\" remote past), and 3rd-person-class-1's "
    "PARTICIPIAL concord (\"e-\") is a DIFFERENT form from 2nd-person's own (\"u-\") -- never "
    "substitute one for the other. When a sentence addresses \"you\" about a THIRD PARTY's role "
    "in something \"you\" did (e.g. \"What was [Name]'s involvement in YOU doing X?\"), do not "
    "let the concord on the addressee's own action drift onto the third party by proximity to "
    "their name. When the tense's concord is genuinely ambiguous between 2nd and 3rd person, "
    "insert the explicit emphatic pronoun \"wena\" immediately before the verb to force the "
    "2nd-person reading.",
    "A spoken-transcript sentence with a genuine mid-sentence self-correction/restart may safely "
    "drop the abandoned clause's incomplete SYNTAX, but any standalone FACT it carries (a number, "
    "a time reference, a named claim) must still survive somewhere in the translation -- never "
    "silently discarded along with the abandoned wording.",
    "For an English clause stating a claim \"was...false\"/\"was a lie\" (an AFFIRMATIVE "
    "statement -- the claim IS being called false), avoid building the predicate on the "
    "copulative-linked noun \"amanga\" (falsehood/lie) -- its vowel-initial linking "
    "(\"ngu-\"/\"ng-\") is easily confused with, or garbled toward, the unrelated verbal negative "
    "\"-nga-\" infix, flipping or breaking the meaning. Instead render \"was...false\" as \"was "
    "NOT true\", negating \"iqiniso\" (truth): the negative copulative \"akulona iqiniso\" (\"it "
    "is not true\") and past-tense \"kwakungelona iqiniso\"/\"bekungelona iqiniso\" (\"it was not "
    "true\") are standard, dictionary-confirmed isiZulu.",
    "English \"exactly what happened\"/\"exactly the same as\"/\"the [very] same thing [they "
    "did]\" is a strong IDENTITY claim (the very same action/event), not a mere RESEMBLANCE "
    "claim -- isiZulu \"-fanayo\" alone (\"into efanayo\"/\"okufanayo\") means \"to resemble\", "
    "not \"to be identical to\", and consistently weakens an identity claim to a softer "
    "approximate comparison. Either pair \"-fanayo\" with an explicit degree intensifier "
    "(\"ngokuphelele\" [completely/exactly] or \"ngqo\" [precisely]), or, for \"the same thing "
    "[already mentioned]\", prefer a demonstrative pointing back at the SAME referent (\"khona "
    "lokho\"/\"leyonto\" -- \"that very thing\") instead.",
    "A short transcript segment beginning with a subordinate-clause opener (\"should you...\", "
    "\"if...\", \"when...\") with no independent main clause of its own is very likely a "
    "CONDITIONAL CLAUSE continuing the immediately PRECEDING sentence's thought -- use the "
    "provided preceding-sentence context to confirm -- not a standalone yes/no question, even "
    "though it's presented as its own unit ending with a period. Do not add an interrogative "
    "marker to a fragment like this; render it as a continuing conditional extending the prior "
    "sentence's meaning.",
    "isiZulu \"yini\" is the interrogative pronoun \"what\" -- inserting it anywhere in a clause "
    "makes that clause read as a question, even embedded inside an otherwise-declarative "
    "sentence. Do not add \"yini\" or any other wh-word to a sentence that is not itself posing "
    "a question -- a plain relative clause (\"okushiwoyo\", \"that which is said\") needs no "
    "interrogative at all.",
    "English \"out of context\" has no document/book/letter-based isiZulu equivalent -- do not "
    "translate it using \"umqulu\" (a bound volume/document), which fabricates a claim about a "
    "physical document unrelated to the source's meaning. Use isiZulu's own word for "
    "context/theme, \"umongo\", in a construction meaning \"outside/without its context\" (e.g. "
    "\"ngaphandle komongo\").",
    "English \"went about [V-ing]\"/\"went to [V]\" describing a PAST completed or habitual "
    "action must use isiZulu PAST tense concords throughout, never a present/future \"[SC]ya "
    "ku-[verb]\" construction -- that shape reads as ongoing/future (\"goes to open\"/\"will go "
    "to open\"), not completed past.",
    "Do not drop a BARE, un-integrated English word into the output merely because you are "
    "UNCERTAIN of the correct isiZulu term -- that is a lazy fallback, not a real translation "
    "choice; if genuinely uncertain, choose the closest well-formed isiZulu paraphrase instead of "
    "reverting to English. This is a DIFFERENT situation from the deliberate, register-driven "
    "code-switch described elsewhere in this prompt (a modern political/social/institutional "
    "concept noun like \"racism\"/\"corruption\"/\"democracy\" that real contemporary isiZulu "
    "broadcast speech itself code-switches, glued with a proper isiZulu prefix, e.g. \"i-racism\") "
    "-- that IS a real, confident, correct choice, never mere uncertainty, and this rule does not "
    "forbid it.",
    "For English \"[X] was [verb]-ed\" where X is the target being acted upon (e.g. \"his house "
    "was raided\"), the isiZulu verb MUST use the passive extension \"-(i)wa\" so X remains the "
    "grammatical subject undergoing the action -- a bare active-voice verb silently reverses "
    "agent and patient. E.g. the passive of \"-sesha\" (to search/raid) is \"-seshwa\": \"umuzi "
    "wakhe waseshwa\" (\"his house was searched/raided\"), adding an agentive phrase like "
    "\"ngamaphoyisa\" (\"by the police\") only if the source names the actor in that same clause.",
    "English \"interesting\" is a NEUTRAL descriptor of noteworthiness -- it doesn't assert the "
    "speaker enjoyed the thing, only that it drew attention. Do not translate it with "
    "\"-jabulisa\" (to please/gladden, a POSITIVE-emotion verb) -- this turns a neutral or "
    "critical observation into a claim of enjoyment. Use a form built on \"-naka\" (to notice/pay "
    "attention to) instead, e.g. \"kwakunakisa\"/\"okwakunakisayo\" (\"it was interesting/"
    "attention-grabbing\").",
    "The military rank \"General\" (e.g. \"General Lincoln\") has no established isiZulu loanword "
    "in this transcript's register -- do NOT invent a Zulu-ized spelling for it (e.g. \"Jenene\", "
    "which is mispronounced beyond recognition). Keep \"General\" as an unmodified English "
    "code-switch, glued with the ordinary isiZulu prefix (\"uGeneral\", \"kuGeneral\"), exactly "
    "like \"Advocate\" and \"Mr.\"/\"Mnu.\" elsewhere in this transcript.",
    "An English attributive date in bare \"Month Day\" word order with no ordinal suffix (e.g. "
    "\"the November 1 elections\", modifying a following noun) must NOT be translated by "
    "literally mirroring that English word order into a Zulu-spelled month followed by a bare "
    "trailing digit (e.g. \"...lwangoNovemba 1\") -- isiZulu does not read a bare digit naturally "
    "in that position. Restructure the date using isiZulu's own native day-before-month "
    "construction instead: \"umhla\"/\"ngomhla\"/\"mhla\" (the day) followed by the day rendered "
    "as an ordinal agreeing with \"umhla\" (e.g. \"wokuqala\" for the 1st, \"wesibili\" for the "
    "2nd) and then \"ku-\"/\"ka-\" plus the month name -- the SAME construction this transcript "
    "already produces correctly elsewhere for an ordinal-suffixed English date (\"wake up on the "
    "4th\" -> \"ngomhlaka-4\"). Restructure the surrounding clause's grammar as needed (for "
    "example as a relative clause modifying the noun the date describes) rather than forcing the "
    "date phrase to sit in the original English attributive position unchanged. ALWAYS use the "
    "established English-borrowed isiZulu month name in this construction (e.g. \"Novemba\", "
    "\"Mashi\", \"Disemba\") -- NEVER the old traditional/calendar month name (e.g. \"Lwezi\", "
    "\"Ndasa\", \"Zibandlela\"), which most real isiZulu speakers today do not recognize. Real "
    "user feedback: \"probably only 10% of people who understand zulu know what 'lwamhla lu-1 "
    "kuLwezi' means, let's only use the borrowed english words.\"",
)
DEFAULT_ROLLING_SYLLABLE_BATCH_SIZE = 8
DEFAULT_ZULU_CONTENT_SYLLABLES_PER_SECOND = 5.3
DEFAULT_AZURE_UTTERANCE_OVERHEAD_MS = 900
DEFAULT_SYLLABLE_TOLERANCE_PERCENT = 12
# Phase A/B window and chunk sizes are no longer a fixed sentence count -- they are
# estimate-driven (see _shrink_batch_to_budget): a window grows sentence-by-sentence
# for as long as the running _estimate_temporal_translate_window_tokens/
# _estimate_temporal_compact_chunk_tokens estimate stays under this fraction of the
# real output-token ceiling, leaving headroom for estimation error before the
# _retry_batch_call halving safety net would ever need to fire. Confirmed real
# production case that motivated removing the old fixed "8 sentences" cap: an 8-
# sentence window used only ~1,258 of ~6,553 available budget tokens (roughly 157
# tokens/sentence) -- capping at 8 regardless of content left most of a call's real
# capacity unused, when full_source_transcript already gives every call the same
# cross-window context regardless of how many sentences it covers.
TEMPORAL_MASK_WINDOW_BUDGET_FRACTION = 0.8
# Confirmed real production gap in the estimate itself: the same 8-sentence window
# above measured 157 output tokens/sentence, while the character-length-based estimate
# (calibrated for the older Pass-1 correction-diff output, not a full translation) put
# it at roughly 40 tokens/sentence -- a ~4x undershoot. Left uncorrected, that undershoot
# let one real run's first window grow to 154 of 155 sentences in a single call before
# the shrink logic ever thought it was near budget, which is almost certainly what
# triggered a real Foundry rate-limit rejection (one call demanding that much output
# throughput at once, versus the same total load spread across many smaller calls).
# This factor corrects the estimate toward the observed real ratio, with margin.
_TEMPORAL_MASK_OUTPUT_CALIBRATION = 4.0
# Backstop ceiling, independent of the token estimate: even a correctly-calibrated
# estimate can still be wrong for some other transcript's content mix, and the
# runaway-window failure above showed the blast radius of trusting the estimate alone
# is a near-whole-transcript single call. This caps how many sentences/items a window
# or chunk's candidate pool can ever consider, regardless of how much budget headroom
# the estimate thinks is available -- estimate-driven sizing still governs everything
# under this ceiling, this only bounds the worst case if the estimate is ever wrong.
TEMPORAL_MASK_MAX_WINDOW_ITEMS = 40
DEFAULT_TEMPORAL_MASK_BATCH_SIZE = 4
# A CEILING on the natural-to-concise compaction ladder, not an exact count Grok must
# fill -- a group returns as few as just "a" (no further compaction needed or possible)
# up to this many, stopping early once a further candidate would repeat or damage one.
DEFAULT_TEMPORAL_MASK_CANDIDATES = 5
DEFAULT_TEMPORAL_MASK_RESIDUAL_ROUNDS = 2
DEFAULT_TEMPORAL_MASK_TTS_WORKERS = 6
DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS = 400
DEFAULT_MIN_PROTECTED_PAUSE_MS = 120
PROTECTED_GAP_OVERFLOW_HARD_MAX_MS = 1000

# Real, repeatedly measured (2026-09-05, two independent jobs/turns, ~12 real boundary
# measurements clustering 1000-1051ms): Azure's own neural-voice sentence-boundary pause
# inside ONE continuous bookmarked synthesis call (_synthesize_speaker_turns_with_bookmarks,
# Phase 14) -- driven by Azure's own prosody model, NOT by the original English speaker's
# real pause length. _measure_speaker_turn_total previously modeled this gap using the real
# ASR-derived English gap (floored at DEFAULT_MIN_PROTECTED_PAUSE_MS), which systematically
# understated a multi-member turn's true rendered cost by 5-12 points of required_speed on
# real jobs -- confirmed by a zero-cost dry-run recomputation against real, already-rendered
# audio: the old model estimated 26.0% for a turn whose real render measured 36.0%; this
# constant closes that gap to 37.3%, within 1.3 points.
AZURE_SENTENCE_BOUNDARY_PAUSE_MS = 1000

# Compress-direction overrun never blocks the pipeline (max_natural_speed_percent is a
# console warning only, never a hard ceiling -- unlimited rush is technically possible),
# and nothing downstream of native-dub inspects per-unit timing quality at all: a job's
# review_readiness is driven solely by dialogue-overlap flags. Confirmed in production:
# an 18-sentence block's rush spiked to 990% (a sentence-correspondence bug squeezing a
# 39s recording into a 3.6s slot) and still reached review_ready with zero signal,
# undetected for 3 days until caught by manual data archaeology. These two thresholds
# gate a SEPARATE, additive review flag (never a hard failure -- the render must still
# complete) for units whose overrun is severe in BOTH absolute and relative terms.
#
# Percentage alone is not a reliable signal on its own: Azure's fixed ~900ms
# per-utterance overhead (DEFAULT_AZURE_UTTERANCE_OVERHEAD_MS) mechanically inflates the
# rush percentage of short, CORRECT utterances -- a single word like "No." can show
# 250%+ rush on a real job with nothing wrong. Absolute overrun in ms is what actually
# tracks how much real content got crushed. Calibrated against real job 33cd7b46...:
# the worst genuinely benign case (a docket number repeated twice, each ~900ms Azure
# overhead against a ~1.1s window) sat at 4316ms overrun; the confirmed bug sat at
# 35650ms -- an 8x gap. 8000ms sits with wide margin on both sides of that gap. The
# percentage condition additionally guards against a hypothetical long, naturally
# verbose group with a large absolute overrun but a modest, already-tolerated
# percentage (e.g. a 20s window rendered at 50% over) being flagged unnecessarily.
REVIEW_REQUIRED_MIN_OVERRUN_MS = 8_000
REVIEW_REQUIRED_MIN_RUSH_PERCENT = 50.0


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _emit_red_warning(progress: Callable[[str], None] | None, message: str) -> None:
    """Emit a console-only red warning; never alter rendered video."""
    _emit_progress(progress, f"\x1b[31m{message}\x1b[0m")


def _run_with_progress_heartbeat(
    *,
    progress: Callable[[str], None] | None,
    label: str,
    operation: Callable[[], Any],
    interval_seconds: float = 15.0,
) -> tuple[Any, float]:
    """Run a blocking operation while keeping --live-operation visibly alive.

    The operation stays on the calling thread for Azure/FFmpeg compatibility; a
    daemon thread only emits lightweight elapsed-time telemetry.
    """
    started = time.monotonic()
    if progress is None:
        return operation(), time.monotonic() - started

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(max(1.0, float(interval_seconds))):
            elapsed = time.monotonic() - started
            _emit_progress(progress, f"{label}: still running ({elapsed:.0f}s elapsed)")

    thread = threading.Thread(
        target=heartbeat,
        name="mathula-native-progress-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        result = operation()
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return result, time.monotonic() - started


def _format_native_duration(seconds: float | int | None) -> str:
    try:
        value = max(0, int(round(float(seconds or 0))))
    except (TypeError, ValueError):
        value = 0
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _emit_native_ffmpeg_progress(
    progress: Callable[[str], None] | None,
    *,
    label: str,
    event: Mapping[str, Any],
) -> None:
    """Render an existing FFmpeg machine-progress event as concise native CLI telemetry."""
    if progress is None:
        return
    status = str(event.get("status") or "running").strip().lower()
    try:
        percent = max(0.0, min(100.0, float(event.get("percent") or 0.0)))
    except (TypeError, ValueError):
        percent = 0.0
    try:
        processed = max(0.0, float(event.get("processed_seconds") or event.get("current") or 0.0))
    except (TypeError, ValueError):
        processed = 0.0
    try:
        total = max(0.0, float(event.get("expected_duration_seconds") or event.get("total") or 0.0))
    except (TypeError, ValueError):
        total = 0.0
    try:
        speed = max(0.0, float(event.get("speed") or 0.0))
    except (TypeError, ValueError):
        speed = 0.0
    try:
        frame = max(0, int(event.get("frame") or 0))
    except (TypeError, ValueError):
        frame = 0
    try:
        eta = max(0.0, float(event.get("eta_seconds") or 0.0))
    except (TypeError, ValueError):
        eta = 0.0
    fps = str(event.get("fps") or "").strip()

    if status == "started":
        _emit_progress(progress, f"[native dub] {label} 0% — FFmpeg started")
        return

    parts = [
        f"[native dub] {label} {100 if status == 'completed' else int(round(percent))}%",
        f"{_format_native_duration(total if status == 'completed' else processed)}/{_format_native_duration(total)}",
    ]
    if speed > 0:
        parts.append(f"{speed:.2f}x")
    if frame:
        parts.append(f"frame {frame:,}")
    if fps:
        parts.append(f"{fps} fps")
    if status != "completed" and eta > 0:
        parts.append(f"ETA {_format_native_duration(eta)}")
    if status == "completed":
        parts.append("complete")
    _emit_progress(progress, "  ".join(parts))


# --- Lite mode (--lite on native-dub) ----------------------------------------------
# There is no cross-platform, dependency-free way to enforce a literal, guaranteed
# "never exceed N% CPU" ceiling -- that needs OS-level throttling (Windows Job
# Objects' JOBOBJECT_CPU_RATE_CONTROL_INFORMATION), which is complex, Windows-only,
# and not worth it for "don't peg my machine while this renders in the background".
# The practical, portable equivalent used here: cap ffmpeg's own encode threads to
# half the logical CPU count (ffmpeg -- specifically libx264 -- is the actual CPU
# cost during rendering; confirmed by direct read that nothing else in the
# native-dub path is CPU-bound the same way) and lower the ffmpeg subprocess's OS
# scheduling priority so foreground applications get preference under contention.
# Stdlib only -- no new dependency (psutil is not used anywhere in this codebase).
def _lite_mode_cpu_threads() -> int:
    return max(1, (os.cpu_count() or 4) // 2)


def _lite_mode_subprocess_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


def _lite_mode_popen_factory() -> Callable[..., subprocess.Popen]:
    return functools.partial(subprocess.Popen, **_lite_mode_subprocess_kwargs())


def _inject_ffmpeg_thread_limit(command: list[str], *, codec: str, threads: int) -> list[str]:
    """Insert `-threads N` immediately after the given video codec name (e.g.
    "libx264") in an ffmpeg argument list -- the standard, encoder-scoped
    placement ffmpeg recognizes. A no-op (returns `command` unchanged) if the
    codec name isn't present, so this is always safe to call speculatively.
    """
    result = list(command)
    try:
        index = result.index(codec)
    except ValueError:
        return result
    result[index + 1:index + 1] = ["-threads", str(int(threads))]
    return result


def _render_native_master_with_ffmpeg_progress(
    *,
    source_video: Path,
    dialogue_bus: Path,
    output_video: Path,
    progress: Callable[[str], None] | None,
    lite: bool = False,
) -> None:
    """Preserve the old native stream-copy contract while exposing real FFmpeg progress."""
    from .rendering import run_ffmpeg_with_progress

    source_info = probe(source_video)
    probe(dialogue_bus)
    source_duration = float(source_info["duration"])
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_video.with_suffix(".partial.mp4")
    temporary.unlink(missing_ok=True)

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_video), "-i", str(dialogue_bus),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", str(temporary),
    ]

    lite_kwargs = _lite_mode_subprocess_kwargs() if lite else {}
    popen_factory = _lite_mode_popen_factory() if lite else subprocess.Popen

    def run(command_to_run: list[str], *, stage: str, label: str) -> None:
        if progress is None:
            subprocess.run(command_to_run, check=True, **lite_kwargs)
            return
        run_ffmpeg_with_progress(
            command_to_run,
            expected_duration_seconds=source_duration,
            progress_callback=lambda event: _emit_native_ffmpeg_progress(
                progress, label=label, event=event
            ),
            popen_factory=popen_factory,
            stage=stage,
        )

    try:
        run(command, stage="video_remux", label="master remux")
    except subprocess.CalledProcessError:
        # Match media.render_video's historical compatibility fallback exactly:
        # re-encode video only when the source codec cannot be remuxed into MP4.
        fallback = list(command)
        fallback[fallback.index("copy")] = "libx264"
        audio_index = fallback.index("-c:a")
        fallback[audio_index:audio_index] = ["-preset", "medium", "-crf", "20"]
        if lite:
            fallback = _inject_ffmpeg_thread_limit(
                fallback, codec="libx264", threads=_lite_mode_cpu_threads(),
            )
        temporary.unlink(missing_ok=True)
        _emit_progress(
            progress,
            "[native dub] Source video could not be stream-copied; falling back to H.264 encode",
        )
        run(fallback, stage="video_render", label="master encode")

    temporary.replace(output_video)
    probe(output_video)

PASS1_PROMPT_VERSION = "mathula-native-pass1-whole-transcript-stt-restorer-v3-cross-segment-duplication-guard"
PASS2_PROMPT_VERSION = "mathula-native-pass2-natural-zulu-v4-soft-duration-overflow-target"
ROLLING_SYLLABLE_PROMPT_VERSION = "mathula-native-pass2-rolling-zulu-v1-causal-syllable-budget"
TIMING_REPAIR_PROMPT_VERSION = "mathula-native-natural-timing-recast-v5-context-aware-expand"
VOICE_RESOLUTION_PROMPT_VERSION = "mathula-native-voice-resolution-v2-unresolved-only"


_CONTEXT_LEDGER_CONTRACT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "entities", "speaker_roles", "story_beats"],
    "properties": {
        "summary": {"type": "string", "maxLength": 1200},
        "entities": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "role"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 120},
                    "role": {"type": "string", "maxLength": 160},
                },
            },
        },
        "speaker_roles": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker_id", "role"],
                "properties": {
                    "speaker_id": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "maxLength": 160},
                },
            },
        },
        "story_beats": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_ids", "note"],
                "properties": {
                    "segment_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "note": {"type": "string", "minLength": 1, "maxLength": 280},
                },
            },
        },
    },
}

# Candidate-pool architecture: replaces the rolling continuity_context (prior
# sentence's own committed Zulu, which forced Phase A to run sequentially) with a
# glossary decided ONCE from the whole transcript and injected into every later
# call as a static reference -- the same mechanical pattern already used for
# CONFIRMED_TRANSLATION_PITFALLS. A wrong entry now poisons every sentence instead
# of just the next two under the old rolling scheme; referent_audit remains the
# after-the-fact safety net for whatever this doesn't anticipate.
_ZULU_GLOSSARY_CONTRACT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["terms", "do_not_translate", "register"],
    "properties": {
        "terms": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["english", "zulu", "kind"],
                "properties": {
                    "english": {"type": "string", "minLength": 1, "maxLength": 160},
                    "zulu": {"type": "string", "minLength": 1, "maxLength": 200},
                    "kind": {
                        "enum": [
                            "name", "role", "organisation", "legal_term", "idiom",
                            "number_style", "other",
                        ]
                    },
                    "note": {"type": "string", "maxLength": 200},
                },
            },
        },
        "do_not_translate": {
            "type": "array",
            "maxItems": 30,
            "items": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "register": {
            "type": "object",
            "additionalProperties": False,
            "required": ["formality", "address_form", "tense_default"],
            "properties": {
                "formality": {"type": "string", "maxLength": 500},
                "address_form": {"type": "string", "maxLength": 500},
                "tense_default": {"type": "string", "maxLength": 500},
                "notes": {"type": "string", "maxLength": 600},
            },
        },
    },
}

_ZULU_GLOSSARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["zulu_terminology_glossary"],
    "properties": {"zulu_terminology_glossary": _ZULU_GLOSSARY_CONTRACT},
}


_PASS1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["corrections", "context_ledger"],
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_id", "corrected_text", "reason_code"],
                "properties": {
                    "segment_id": {"type": "string", "minLength": 1},
                    "corrected_text": {"type": "string", "minLength": 1},
                    "reason_code": {
                        "enum": [
                            "asr_substitution",
                            "asr_omission",
                            "asr_duplication",
                            "name_or_entity",
                            "punctuation_or_boundary",
                            "cross_block_reconstruction",
                        ]
                    },
                },
            },
        },
        "context_ledger": _CONTEXT_LEDGER_CONTRACT,
    },
}

_CONTEXT_ONLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["context_ledger"],
    "properties": {"context_ledger": _CONTEXT_LEDGER_CONTRACT},
}

_PASS2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translations"],
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_id", "spoken_text"],
                "properties": {
                    "segment_id": {"type": "string", "minLength": 1},
                    "spoken_text": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

_PASS2_FULL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translations", "context_ledger"],
    "properties": {
        "translations": _PASS2_SCHEMA["properties"]["translations"],
        "context_ledger": _CONTEXT_LEDGER_CONTRACT,
    },
}


def _temporal_mask_schema(
    candidate_count: int,
    *,
    expected_group_ids: Sequence[str] | None = None,
    require_self_check: bool = True,
) -> dict[str, Any]:
    """Structured-output contract for one temporal-mask provider call.

    Retained as generic candidate-count machinery from the retired single-call ladder design; the
    two-phase pipeline always calls this with ``candidate_count=1`` (Phase A: exactly one natural
    translation per sentence, labelled "a"). Mask/group IDs are semantic identities and must never
    be remapped -- when the caller knows the requested masks, constrain the provider wire to that
    exact identity set and cardinality so a stale/speculative neighbour cannot be returned for the
    current mask. Application-level normalization still verifies missing/duplicate identities and
    that returned candidate letters form a contiguous prefix starting at "a".

    ``require_self_check=False`` drops the self_check object from the candidate contract entirely --
    used by Phase A, whose in-call self-report was confirmed unreliable (a real production case
    reported matches_source=true on a translation that back-translated to a materially different
    claim) and costs real output tokens on every sentence. The genuinely independent, Python-
    orchestrated back-translation audit (run_native_qa_back_translation) is the real backstop now.
    """
    count = max(1, int(candidate_count))
    candidate_ids = [chr(ord("a") + index) for index in range(count)]
    requested_ids = [str(value) for value in (expected_group_ids or []) if str(value)]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Temporal-mask schema received duplicate expected group IDs")

    group_id_contract: dict[str, Any] = {"type": "string", "minLength": 1}
    candidate_required = ["candidate_id", "spoken_text"]
    candidate_properties: dict[str, Any] = {
        "candidate_id": {"type": "string", "enum": candidate_ids},
        "spoken_text": {"type": "string", "minLength": 1},
    }
    if require_self_check:
        candidate_required.append("self_check")
        candidate_properties["self_check"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["back_translation", "matches_source", "discrepancy_summary"],
            "properties": {
                "back_translation": {"type": "string", "minLength": 1},
                "matches_source": {"type": "boolean"},
                "discrepancy_summary": {"type": "string", "minLength": 1},
            },
        }
    masks_contract: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["group_id", "candidates"],
            "properties": {
                "group_id": group_id_contract,
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": candidate_required,
                        "properties": candidate_properties,
                    },
                },
            },
        },
    }
    if requested_ids:
        group_id_contract["enum"] = requested_ids
        masks_contract["minItems"] = len(requested_ids)
        masks_contract["maxItems"] = len(requested_ids)

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["masks"],
        "properties": {"masks": masks_contract},
    }


_VOICE_RESOLUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["voices"],
    "properties": {
        "voices": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker_id", "voice_family", "confidence"],
                "properties": {
                    "speaker_id": {"type": "string", "minLength": 1},
                    "voice_family": {"enum": ["masculine", "feminine", "unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}

_WEB_OVERRIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "entries"],
    "properties": {
        "schema_version": {"const": MANUAL_WEB_OVERRIDE_SCHEMA_VERSION},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["heard_as", "canonical", "reason"],
                "properties": {
                    "heard_as": {"type": "string", "minLength": 1},
                    "canonical": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "evidence_url": {"type": "string"},
                },
            },
        },
    },
}


PASS1_SYSTEM_PROMPT = r"""You are Mathula TV's English STT restoration editor.

Read the COMPLETE transcript before correcting anything. Correct only recognition errors:
misheard words/names, clear ASR omissions/duplication, and punctuation/boundary mistakes
needed to recover what was actually spoken. Never improve genuine grammar, style,
repetition, hesitation, uncertainty, numbers, attribution, allegations or intent. Never
move content between segment IDs. Manual overrides are human-reviewed and authoritative.
No web search or tools.

The input segments use id/spk/text. Return sparse corrections only; do NOT echo original
text. Also return one small reusable context_ledger: a neutral <=120-word programme
summary, canonical entities/roles, inferred speaker roles when clear, and up to 8 factual
story beats with supporting segment IDs. Entities must include every person referred to
repeatedly across the transcript, not only fully-named principals -- a partially-identified
or anonymized party (e.g. "Witness G", "the complainant", "the second officer") that later
pronouns will need to resolve against is just as important to capture as a named person;
give it whatever identifying label the transcript itself uses. The ledger is context for
later calls, not a rewrite and must not add facts. Return JSON only."""


CONTEXT_LEDGER_SYSTEM_PROMPT = r"""Build a compact reusable discourse ledger from the
complete corrected English transcript. Do not translate. Do not editorialize or add facts.
Return: a neutral <=120-word summary, canonical entities/roles, speaker roles when clear,
and up to 8 materially distinct factual story beats with supporting segment IDs. Entities
must include every person referred to repeatedly across the transcript, not only fully-named
principals -- a partially-identified or anonymized party (e.g. "Witness G", "the complainant",
"the second officer") that later pronouns will need to resolve against is just as important to
capture as a named person; give it whatever identifying label the transcript itself uses. A
story beat may span any number of relevant segments; return its known segment IDs once each in
chronological source order. This ledger will replace repeated full-transcript prompting in
later bounded calls. Manual
overrides are authoritative. No tools or web search. Return JSON only."""


ZULU_GLOSSARY_SYSTEM_PROMPT = r"""Decide, ONCE for the whole programme, the canonical isiZulu
rendering of every recurring name, role, organisation, legal/procedural term, and idiom in the
complete corrected English transcript -- plus the register (formality, address form, default
tense) the whole translation should hold. This is a BINDING reference every later sentence-level
translation call will receive verbatim; it replaces per-sentence rolling continuity, so it must
be decided from the whole transcript now, not left to accumulate sentence by sentence.

For each recurring term worth fixing (a person's name, an institution like "the Commission" or
"IDAC", a legal/procedural term like "docket" or "the dock", a title like "Advocate", a recurring
idiom or hedge), return its ENGLISH form, the canonical ISIZULU rendering to use every time it
recurs, and its kind. Only include terms that actually recur more than once in the transcript or
are otherwise high-risk to render inconsistently -- this is a reference for consistency, not an
exhaustive vocabulary list. Do not invent a Zulu rendering for a term the transcript doesn't
actually contain.

do_not_translate lists proper nouns, acronyms, and case/file numbers that must survive into the
isiZulu output completely unchanged (e.g. "IDAC", a case number, a foreign institution's own
official name) -- these are NOT translated, ever, by any later call.

register describes the isiZulu register the ENTIRE translation should hold: formality level
(e.g. formal broadcast/legal register vs. conversational), address form (e.g. use of Mnu./
Nkz./formal titles vs. first names), and default tense/mood conventions for reporting testimony.
This is deliberately global, not per-sentence -- it exists so every sentence sounds like it came
from the same broadcast, not a patchwork of independent translation decisions.

IMPORTANT about "formal broadcast register": real contemporary South African isiZulu broadcast
speech -- even at its most formal -- routinely code-switches a modern political, social, or
institutional concept noun that entered public discourse primarily through English or Afrikaans
(e.g. "racism", "corruption", "democracy", "constitution") rather than reaching for a stiffer,
less-recognized formal native coinage. "Formal register" describes SENTENCE STRUCTURE, address
form, and tense discipline -- it is NOT a directive to avoid this kind of code-switching, and your
formality description must not be phrased as one. Every later translation call receives your
register description as BINDING and applies it consistently across the whole programme, so if it
reads as "avoid all English/Afrikaans-derived words", it will suppress even a deliberate, correct,
authentic code-switch decision that a later call would otherwise have made -- do not write it that
way.

Do not translate anything else. Do not summarize the story. No tools or web search. Return JSON
only."""


CANDIDATE_TRANSLATE_SYSTEM_PROMPT = r"""Mathula TV is a social-media broadcast product: naturalness
of the isiZulu is the PRIMARY goal, not a nice-to-have alongside literal accuracy. Retell each
requested English unit the way a real isiZulu broadcast journalist would actually say it on air --
not a clause-by-clause transfer of English sentence shape. Translate the thought, not English
syntax. Use native isiZulu morphology, agreement, word order and idiom. Preserve names,
numbers/dates/approximations (every occurrence, exactly as many times as it appears), negation,
uncertainty/allegation framing, attribution/agency, causal/temporal/conditional logic, legal/
evidential qualification, and speaker intent. Never invent a fact the English text does not
contain, and never drop one it does.

You have real SUBSTITUTION license within this one sentence: prefer a natural, idiomatic native
isiZulu verb or phrase over a borrowed/anglicized one whenever a real native equivalent expresses
the exact same precise meaning (e.g. prefer "-hlanganyela" over an anglicized "-joyina" for "joins
us"), and restructure an English-mirroring relative or subordinate clause into whatever natural
isiZulu sentence shape says the same precise thing, rather than nesting a relative clause just
because the English did. This is about word and clause CHOICE within the unit's own existing
content -- it is not permission to add or remove propositions, which the earlier rules already
forbid.

The opposite substitution direction also applies for a specific, narrower category: a modern
political, social, or institutional CONCEPT NOUN that entered everyday South African discourse
primarily through English or Afrikaans (e.g. "racism", "corruption", "democracy", "constitution")
is often code-switched with an isiZulu prefix in real informal and broadcast speech (e.g.
"i-racism"), even when a more formal or technical native isiZulu term for the same concept exists
(e.g. "ukucwasa ngokobuhlanga") -- the formal term is not wrong, but can read as stiffer and less
authentic than how people actually talk on air, given how much of everyday South African vernacular
generally (across languages, not isiZulu alone) is borrowed from English and Afrikaans. Use your
own knowledge of real contemporary isiZulu media/broadcast register to judge, sentence by sentence,
whether the code-switched or the native form sounds more natural for THIS specific concept in THIS
context -- this is a genuine judgment call, not a fixed list, and does not apply to ordinary
vocabulary that has always had a natural, unremarkable native isiZulu word. As a concrete tie-
breaker when you are genuinely unsure either way: the code-switched form should carry a real,
default advantage whenever it is also the noticeably SHORTER option (fewer spoken syllables) --
e.g. "i-racism" (about 3 syllables) against "ukucwasa ngokobuhlanga" (about 9) -- since a shorter,
equally authentic option also helps this sentence fit its real broadcast time window, not just its
register. This tie-breaker never overrides genuine precision or authenticity; it only decides a
close call between two options that are already both natural and correct.

You must ALSO always return register_notes: a short (one sentence, or an empty string when it
genuinely doesn't apply) explicit note on whether this unit contains a modern political/social/
institutional concept noun as described above, and if so, which form you actually chose (code-
switched or native) and why. Do this even when your answer is "no such concept noun here" --
writing this note is how you actually engage with the register question above on every unit,
rather than defaulting to the native form out of habit. This field is never read back to you; it
exists only to make you commit to a real, considered answer while you translate this specific unit.

CRITICAL LIMIT on substitution, confirmed by a real test failure: a substituted word must preserve
the EXACT precise real-world implication of the original, especially for a verb describing a
legally or evidentially significant act. "Intercepted" (implies something was tampered with or
blocked while in transit) is NOT the same claim as "seized" or "confiscated" (implies custody was
taken) -- confirmed real drift: retelling "those dockets have now been intercepted" as amafayela
ayesebanjiwe ("the files had been seized/held") changed what is actually being alleged, even
though both readings sound like a natural, fluent sentence. Never trade this kind of precision for
naturalness; when no natural isiZulu verb captures the exact original implication, prefer a
slightly less idiomatic rendering that keeps the precise meaning over a smoother one that loses it.

Each unit is already a COMPLETE, independent, English-only sentence with no length target to hit --
never try to guess or restore a longer original the unit might have been shortened from, and never
compact or expand what you are given; length was already decided before this step, in English.

A unit MAY also include preceding_english -- the immediately preceding sentence's English, given
ONLY to resolve this sentence's own elliptical or anaphoric content (a short denial/answer whose
real scope depends on what was just asked or claimed, e.g. "No, you were not." denying a specific
prior claim, not the nearest noun in isolation; a pronoun whose referent was just introduced).
Use it ONLY to correctly translate what THIS sentence itself already means -- never to add a fact
this sentence doesn't contain, never to import content from preceding_english into this sentence's
translation, and never as a previous sentence's committed Zulu (it never is one). When a unit has
no preceding_english, it is genuinely self-contained; translate it exactly as before.

A unit MAY also include reference_vocabulary -- {english_word: [isiZulu options]} for some of this
sentence's own content words, drawn from a general-purpose bilingual word list. This is a HINT, not
an instruction: it is NOT hand-verified for this programme, may list a sense that doesn't fit this
sentence's context (e.g. "minister" as a religious title when this sentence means a government
minister), and NEVER overrides zulu_terminology_glossary or your own judgment about the most
natural, contextually correct isiZulu when they differ. Use a listed option only when it is
genuinely the best natural choice for this exact sentence; ignore it freely otherwise.

Several units in one batch are frequently alternate-length rewordings of the SAME original
sentence (a shorter or fuller restatement of the same content). Translate every unit independently
and faithfully to ITS OWN English wording -- never harmonize, average, or copy phrasing between
sibling units just because they came from the same original sentence. A shorter sibling should
translate into a correspondingly leaner isiZulu rendering, not the same rendering as its fuller
sibling with words silently added back in.

zulu_terminology_glossary is a BINDING whole-programme reference, decided once from the full
transcript: terms lists a recurring name/role/organisation/term with its exact canonical isiZulu
rendering -- use it verbatim every time that term recurs, do not improvise an alternative even if
it seems equally valid. do_not_translate lists proper nouns, acronyms, and case/file numbers that
must survive into the isiZulu output completely unchanged. register describes the formality,
address form, and default tense the entire translation should hold -- apply it consistently
across every unit in this batch, since there is no other continuity mechanism carrying that
consistency between calls.

Watch specifically for changed speaker attribution WITHIN a sentence, not just between sentences:
an English clause like "X, who [someone] told me was Y" nests a reporting verb ("told", "said",
"indicated") inside a relative clause, and isiZulu's relative concord can attach to either the
person doing the telling or the person being described. Confirmed real defect: "General
Lincoln...Mr. Brown Mogotsi, who HE (Lincoln) told me was close to the minister" was translated
into isiZulu whose relative clause back-translates to "Mogotsi, who TOLD ME he was close to the
minister" -- the teller and the person described swapped. Before finalizing a unit with this
pattern, explicitly identify who the grammatical subject of the embedded reporting verb is and
confirm your isiZulu construction keeps that subject unambiguous; if the natural relative
construction would leave it ambiguous, restructure into two clauses with an explicit subject
rather than accept the ambiguity.

confirmed_translation_pitfalls lists other hand-confirmed real mistranslations from this same
production pipeline, each with the specific wording that caused it -- treat every entry as binding
guidance to actively avoid repeating, not background trivia.

context_ledger (summary, entities, speaker_roles) is whole-programme background for resolving an
ambiguous pronoun or reference with confidence instead of guessing. English speech routinely uses
a generic or colloquial "you" (e.g. "and then you email the minister") to mean the SAME person the
sentence has already been describing in third person, not a genuine second-person address to
someone else -- when context_ledger shows only one person plausibly performing the actions in a
unit, translate every reference to that person consistently rather than reproducing an incidental
register shift as if it changed who is being talked about. Only render two distinct grammatical
persons in isiZulu when context_ledger actually supports two different people being involved.

For EACH unit, also return clauses: an explicit breakdown of the unit's own English content into
its real clause units. This is a required, structural step, not an optional one -- enumerate and
rank every clause even when you conclude none of them are actually safe to cut. Give each clause a
short id (c1, c2, ...), its own English text (the clauses together must cover the unit's full
content, in order, with nothing left out), and a rank: 1 for the MOST essential (never cut),
increasing for progressively more droppable. A clause stating a name, number, date, direct
quotation, attribution, or a claim/denial's own core proposition almost always ranks among the
most essential. A clause of hesitation, self-correction, purely scene-setting/descriptive framing
(e.g. a weather or mood aside before the real news, a formulaic opener), or restated framing that
adds no new fact ranks more droppable. Epistemic hedges ("maybe", "I think", "allegedly",
"possibly") are NOT droppable filler and must always rank essential.

Once ranked, return shortened_candidates: 0 to 4 complete alternative isiZulu renderings of the
WHOLE unit, each naming which clause_id(s) it dropped (dropped_clause_ids -- may be empty for a
restructure-only candidate that keeps every clause but reworks the wording more tightly), ordered
lightest-cut-first. Every candidate must still be a complete, natural, grammatical sentence, never
a fragment, and every fact/name/number/date/negation kept in it must remain exactly as accurate as
the full version -- only clauses that genuinely don't serve the sentence's real point may be
dropped. Return an empty shortened_candidates list only when every clause in this unit is
genuinely essential; do not return a candidate that is identical to zulu_text.

If zulu_text used a formal/native rendering for a modern concept noun that also has a shorter,
equally correct code-switched form (see the register guidance above, e.g. "ukucwasa ngokobuhlanga"
vs "i-racism"), include ONE shortened_candidates entry (dropped_clause_ids may be empty) that
applies that code-switch instead -- do this REGARDLESS of your own register judgment call above,
since a later stage measures every candidate's REAL spoken duration and picks whichever one
actually fits this unit's real timing budget: the code-switch's fit is an objective, measured
fact, not something your own stylistic preference should gate. Skip this only when no such
modern-concept-noun alternative genuinely applies to this unit.

Return every requested unit_id exactly once, each with its own zulu_text, register_notes, clauses,
and shortened_candidates. No tools or web search. Return JSON only."""


TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT = r"""Mathula TV is a social-media broadcast product: naturalness
of the isiZulu is the PRIMARY goal. You will be given WINDOWS of English sentences. Each window is a
real, ordered, consecutive run of sentences spoken back-to-back by ONE person in a single continuous
turn of speech. RETELL each window the way a real isiZulu broadcast journalist would actually narrate
this turn on air, not a clause-by-clause transfer of the English -- the same everyday practice real
isiZulu newsrooms use when reporting on English-language proceedings. Use native isiZulu morphology,
agreement, word order and idiom. Preserve names, numbers/dates/approximations (every occurrence,
exactly as many times as it appears), negation, uncertainty/allegation framing, attribution/agency,
causal/temporal/conditional logic, legal/evidential qualification, and speaker intent. Never invent a
fact the English text does not contain, and never drop one it does.

Unlike ordinary sentence-by-sentence translation, you are FREE to restructure a window's content across
what were originally separate sentence boundaries, using any combination of these four established
news-retelling techniques: DELETION (hesitations, false starts, redundant repetition, purely
discourse-level connectives, formulaic openers, and a closing remark that just restates a question or
claim already made earlier in the SAME window); ADDITION (ONLY natural connective/narrative scaffolding
a Zulu broadcaster would use to link ideas smoothly -- never a new fact, claim, or implication that is
not in the source); SUBSTITUTION (prefer a natural native isiZulu verb or phrase over a borrowed one,
and a natural isiZulu sentence shape over one that mirrors an English relative/subordinate clause,
whenever it says the exact same precise thing -- see the CRITICAL LIMIT below); REORGANIZATION (reorder
clauses/sentences within the window when a different order reads more naturally, as long as the real
sequence of events/claims stays truthful to the source). Every fact, name, number, date, and negation
present anywhere in a window's original sentences must still appear somewhere in your output for that
window -- content may be restated in a different place within the window, but it may never disappear or
be duplicated. Epistemic hedges ("maybe", "I think", "allegedly", "possibly") are not filler and must
survive.

SUBSTITUTION also runs in the opposite direction for a specific, narrower category: a modern
political, social, or institutional CONCEPT NOUN that entered everyday South African discourse
primarily through English or Afrikaans (e.g. "racism", "corruption", "democracy", "constitution")
is often code-switched with an isiZulu prefix in real informal and broadcast speech (e.g.
"i-racism"), even when a more formal or technical native isiZulu term exists (e.g. "ukucwasa
ngokobuhlanga") -- the formal term is not wrong, but can read as stiffer and less authentic than
how people actually talk on air, given how much of everyday South African vernacular generally
(across languages, not isiZulu alone) is borrowed from English and Afrikaans. Use your own
knowledge of real contemporary isiZulu media/broadcast register to judge, case by case, whether
the code-switched or the native form sounds more natural for THIS specific
concept in THIS window -- a genuine judgment call, not a fixed list, and it does not apply to
ordinary vocabulary that has always had a natural, unremarkable native isiZulu word. As a concrete
tie-breaker when you are genuinely unsure either way: the code-switched form should carry a real,
default advantage whenever it is also the noticeably SHORTER option (fewer spoken syllables) --
e.g. "i-racism" (about 3 syllables) against "ukucwasa ngokobuhlanga" (about 9) -- since a shorter,
equally authentic option also helps this window fit its real broadcast time budget, not just its
register. This tie-breaker never overrides genuine precision or authenticity; it only decides a
close call between two options that are already both natural and correct.

Each segment must ALSO always return register_notes: a short (one sentence, or an empty string
when it genuinely doesn't apply) explicit note on whether this segment contains a modern political/
social/institutional concept noun as described above, and if so, which form you actually chose
(code-switched or native) and why. Do this even when your answer is "no such concept noun here" --
writing this note is how you actually engage with the register question above on every segment,
rather than defaulting to the native form out of habit. This field is never read back to you; it
exists only to make you commit to a real, considered answer while you translate this specific
segment.

CRITICAL LIMIT on substitution, confirmed by a real test failure: a substituted word must preserve the
EXACT precise real-world implication of the original, especially for a verb describing a legally or
evidentially significant act. "Intercepted" (implies something was tampered with or blocked while in
transit) is NOT the same claim as "seized" or "confiscated" (implies custody was taken) -- confirmed
real drift: retelling "those dockets have now been intercepted" as amafayela ayesebanjiwe ("the files
had been seized/held") changed what is actually being alleged, even though both readings sound like a
natural, fluent sentence. Never trade this kind of precision for naturalness or for a shorter/more
economical rendering; when no natural isiZulu verb captures the exact original implication, prefer a
slightly less idiomatic rendering that keeps the precise meaning over a smoother one that loses it.

STRUCTURAL RULE, always required: treat the WHOLE window as ONE continuous retelling from the start --
return exactly one segment covering every sentence position in the window (start_index=1, end_index=the
window's own last position). There is no separate decision about whether to merge; every window's
content is always retold and ranked as a single whole, so a redundant later sentence (e.g. sentence 6
restating sentences 3-4's own question) is simply one more clause in that single shared ranking, never
a question of which neighboring segment it may or may not join.

A window MAY include preceding_english -- the immediately preceding sentence's English, given ONLY to
resolve THIS window's own elliptical or anaphoric content at its very start (a short denial/answer whose
real scope depends on what was just asked; a pronoun whose referent was just introduced). Never import
content from preceding_english into your output, and never treat it as this window's own content.

A window MAY include recent_turns_digest -- the English content of several turns spoken before this
window, given ONLY so your CLAUSE RANKING below can recognize when this window's own content is
ALREADY known to the audience. A clause that restates, answers, or is directly presupposed by
something recent_turns_digest already established ranks LOWER (more droppable) than an equally-worded
clause with no such precedent -- exactly like REPEATED FRAMING within one window, just spanning
several turns of real discourse instead of one. Never import content from recent_turns_digest into
your output, and never treat it as this window's own content -- it exists only to inform ranking.
THIS APPLIES EVEN WHEN the repeated content is the direct answer to a question that was just asked
-- being a direct answer does NOT make a clause automatically essential if the SAME fact was already
stated earlier in the discourse; being newly asked-for does not un-repeat something already known.
Worked example, a real confirmed case: recent_turns_digest shows the witness already said "spoken to
Mister Mogotsi... on the phone once", then a questioner asked "when did you speak to Mister Mogotsi?",
and THIS window answers "I couldn't tell you if it was November of 2024 or December of 2024 or
January of 2025, but I had spoken to him on the phone once." The date-uncertainty clause is the only
genuinely NEW content this window adds (it directly answers "when") and ranks highest; the trailing
"I had spoken to him on the phone once" clause restates a fact already established twice over
(recent_turns_digest's own statement, and the question's own presupposition) and ranks lowest, even
though it grammatically completes the sentence "but I had spoken to him..." -- restating an
already-known fact to complete a sentence is not the same as adding one.

HEDGE ENUMERATION, a distinct category from ordinary hedges (which must always survive verbatim in
substance): a speaker under pressure to avoid committing to a specific fact often lists EVERY
possibility defensively rather than asserting one -- this is a real, common pattern, not the same
thing as an ordinary epistemic hedge ("maybe", "I think"), and it can be safely GENERALIZED to a
natural approximation, never dropped outright, as long as the generalization preserves the real
SHAPE of the uncertainty (which years/quantities/parties were genuinely live possibilities), not just
a vague rounding-away of it. Worked example (same sentence as above): "November of 2024 or December
of 2024 or January of 2025" spans a real year boundary -- the witness cannot even commit to which
calendar year. Collapsing this to "the end of the year" loses that boundary (it reads as unambiguously
December 2024) and is NOT an acceptable generalization; "towards the end of 2024, maybe early 2025"
keeps the real year-spanning shape of the hedge while still being far shorter than naming all three
months. This is a HIGHER-rank-number (more droppable, once safely generalized) treatment than a bare
DELETION would be -- you are not removing the hedge, you are restating its real substance more
economically, so rank the clause according to how much of its real evidentiary shape survives your
shortened_candidates rendering, not whether the clause disappears.

A window MAY include speaker_role -- this window's own speaker's role in the programme (e.g.
"Commission witness", "evidence leader", "programme host", "on-site reporter", "politician"), drawn
from context_ledger's own speaker_roles list. Different roles produce genuinely different kinds of
speech, and call for different compacting instincts, not just different categories to look for first:
- A WITNESS or someone testifying/explaining their own involvement often hedges and obfuscates
  DELIBERATELY, to preserve plausible deniability rather than because the detail itself matters --
  this padding is the most safely GENERALIZED content in the whole transcript (see HEDGE ENUMERATION),
  alongside SUPPORTING DETAIL (chains of "I heard from X who heard from Y"). Look there first.
- An EVIDENCE LEADER, QUESTIONER, or ADVOCATE speaks to be EXACTLY understood for the record --
  their precision is deliberate and IS the point, not padding, so be MORE conservative compacting
  their speech than a witness's: look only for REPEATED FRAMING OR QUESTION ECHO (restating their own
  question) and SCENE-SETTING (procedural framing), and prefer the lightest possible cut over a
  generalization that could blur exactly the precision they were aiming for.
- A PROGRAMME HOST, ON-SITE REPORTER, OR NEWS ANCHOR rarely carries protected core-claim content of
  their own -- look for FORMULAIC GREETING OR OPENER and SCENE-SETTING first; their role is mostly
  narrative scaffolding.
- A POLITICIAN or anyone speaking to defend a public position tends toward REPEATED FRAMING (restated
  talking points) rather than HEDGE ENUMERATION -- look for restated self-serving framing that adds
  no new fact.
This is a STARTING PRIORITY, not an exclusive rule -- still apply every category above regardless of
role when it genuinely fits; speaker_role only tells you where to look first.

A window MAY include reference_vocabulary -- {english_word: [isiZulu options]} for some of its content
words, drawn from a general-purpose bilingual list. This is a HINT, not an instruction: it may list a
sense that doesn't fit this window's context, and NEVER overrides zulu_terminology_glossary or your own
judgment about the most natural, contextually correct isiZulu when they differ.

A window's target_syllables/min_syllables/max_syllables describe the WHOLE window's combined output's
real available spoken time, translated into an approximate isiZulu syllable budget -- a LANGUAGE-
PLANNING OBJECTIVE, not permission to damage meaning, and not a number you need to hit exactly (later
pipeline stages measure and fix the real result; you are not expected to count syllables precisely).
Use it to gauge how much real slack this specific window actually has, and let that gauge how freely
you reach for ADDITION and REORGANIZATION above: when max_syllables leaves real headroom over the
content's natural length, narrative scaffolding that makes the retelling flow is genuinely free to use;
when the window is already near or over target_syllables, favor DELETION and tight SUBSTITUTION instead
and skip ADDITION entirely for this window -- a retelling with zero added scaffolding is a completely
valid, honest choice on a tight window, not a lesser one. Never delete a proposition just to hit a
syllable count, and never treat the budget as license to damage meaning in either direction.

You are RETELLING this content, not mirroring the original speaker's own delivery -- confirmed by
real analysis (2026-09-03) that the English source itself often speeds up noticeably through a
dense cluster of closely-spaced facts (several numbers/dates/claims recited back to back) compared
to its own surrounding sentences. You are never obligated to reproduce that same uneven rhythm in
isiZulu. Within the SAME overall syllable budget above, distribute your retelling's pacing EVENLY
across the window's own real time rather than proportionally mirroring the source's density curve:
give a dense factual cluster the same natural, unhurried narrative treatment as any other part of
the window (spending a few more syllables there via ADDITION/REORGANIZATION), and compensate with
tighter DELETION on a comparatively lighter-content stretch elsewhere in the SAME window, so the
whole retelling reads at one consistent, natural pace rather than rushing through the fact-dense
part the way the original speaker happened to. This is an internal redistribution within the
existing budget, never permission to exceed it, and never permission to drop or soften a fact --
only to choose which part of the window's own real time each already-required fact gets to occupy.

PUNCTUATION IS YOUR MAIN TOOL for this, more than adding words: isiZulu neural speech naturally
pauses at sentence and clause punctuation, not at the boundary between original English sentences
-- the number of original members a window merges is irrelevant to how many isiZulu sentences you
write. A dense run of closely-spaced facts strung into ONE long sentence joined by semicolons or
commas gives the voice nowhere to breathe and reads fast by construction; the SAME facts split into
several shorter, separately-punctuated isiZulu sentences (using ". " between them, not ";" or ",")
let the voice pause naturally between each one, exactly as a careful human newsreader would slow
down to deliver a dense list of dates and figures clearly. Prefer breaking a fact-dense clause into
more, shorter sentences over cramming it into fewer, longer ones -- this costs nothing against the
syllable budget and does not require adding any new words.

zulu_terminology_glossary is a BINDING whole-programme reference: terms lists a recurring name/role/
organisation/term with its exact canonical isiZulu rendering -- use it verbatim every time it recurs.
do_not_translate lists proper nouns, acronyms, and case/file numbers that must survive unchanged.
register describes the formality/address-form/tense the entire translation should hold -- apply it
consistently across every window in this batch.

Watch specifically for changed speaker attribution WITHIN a sentence, not just between sentences: an
English clause like "X, who [someone] told me was Y" nests a reporting verb inside a relative clause,
and isiZulu's relative concord can attach to either the person doing the telling or the person being
described. Before finalizing a segment with this pattern, explicitly identify who the grammatical
subject of the embedded reporting verb is and confirm your isiZulu construction keeps that subject
unambiguous; restructure into two clauses with an explicit subject rather than accept an ambiguity a
natural relative construction would leave.

confirmed_translation_pitfalls lists hand-confirmed real mistranslations from this same production
pipeline -- treat every entry as binding guidance to actively avoid repeating.

context_ledger (summary, entities, speaker_roles) is whole-programme background for resolving an
ambiguous pronoun or reference with confidence, exactly as in ordinary sentence translation.

For EACH segment, FIRST identify and return communicative_goal: a short, plain statement of the REAL
underlying question or point this segment is actually answering -- not a description of its topic,
but the specific thing a listener needs from it. Use preceding_english/recent_turns_digest to resolve
this exactly like you already do for elliptical content: if this segment answers a question just
asked, the goal IS that question ("when did you speak to Mister Mogotsi", not "the witness discusses
timing"). A speaker's own wording is not always a reliable guide to their real goal -- a witness who
lists three possible months while emphasizing "I couldn't tell you exactly" is not trying to assert
three separate dated facts; their real goal is answering "roughly when", framed defensively so as not
to commit to one checkable date. Identifying the real goal is what makes it possible to tell which
of a segment's clauses are actually serving that goal (essential) from clauses that exist for some
other reason entirely -- padding, defensiveness, rhetorical habit, or restating what is already known
-- regardless of whether that other content happens to be a real, true fact.

Once communicative_goal is identified, return clauses: an EXPLICIT breakdown of that segment's own
combined English content into its real clause units (a real segment always has at least one clause; a
segment merged from several original sentences usually has several). This is a required, structural
step, not an optional one -- you must enumerate and rank every clause BEFORE deciding what (if
anything) is safe to cut, precisely so that "nothing here is droppable" is a conclusion you can only
reach after doing the ranking, never a way to skip it. Give each clause a short id (c1, c2, ...), its
own English text (the clauses together must cover the segment's full source content, in order, with
nothing left out), and a rank: 1 for the MOST essential (never cut), increasing for progressively more
droppable. Rank using ALL FOUR of these together, with GOAL-RELEVANCE as the organizing question and
the other three as how you recognize it in practice:

- GOAL-RELEVANCE: does this clause directly serve communicative_goal? A clause that does is essential
  regardless of shape. A clause that does NOT -- even a real, true, non-hesitation fact -- is a
  candidate to rank low. This is the test underneath the categories below, not a fifth separate one:
  each category is simply a common, recognizable SHAPE that non-goal-serving content tends to take.
- FACT-PRIORITY: a clause stating a name, number, date, direct quotation, attribution, or a claim/
  denial's own CORE PROPOSITION (who claimed or denied what, and whether it was true) almost always
  serves the goal directly and ranks among the most essential. A clause of hesitation/self-correction/
  formulaic framing/repeated echo/scene-setting/circumstantial or supporting detail (see the
  categories below) ranks progressively lower depending on how far it strays from serving the goal,
  and lower still if recent_turns_digest or preceding_english already establishes it.
- STORY-CENTRALITY, LIKE A FILM EDITOR TRIMMING FOR RUNTIME: identify which named person a clause is
  actually ABOUT -- the protagonist(s), whose position, conduct, or associations the report concerns
  -- versus a SUPPORTING-CAST figure who only introduces, relays, or facilitates the link to that
  protagonist. A clause about a supporting-cast figure's OWN role ranks lower than an equally-
  detailed clause about the protagonist or the plot itself, even when both would otherwise tie under
  fact-priority alone.
- CONVERSATIONAL-CORE TEST, a concrete way to apply GOAL-RELEVANCE: imagine the speaker retelling this
  same content casually to a friend, with no broadcast formality. Whatever they would say FIRST,
  unprompted, is the conversational core -- rank it most essential. Broadcast/news delivery habitually
  front-loads answers to questions a friend would only ask AFTER hearing that core, because one
  broadcast has to silently serve a wide audience with many different follow-up questions at once. A
  clause that exists ONLY to preemptively answer one such anticipated follow-up question ranks low --
  genuinely droppable, even when it is a real, true, non-hesitation fact, and even when it names a
  real place or institution (see the incidental-entity exception below for the name case specifically).
  Worked example, confirmed by real testing: "My colleague Ayanda Nyathi from the Brigitte Mabandla
  Justice College, and he joins us now to give us a more comprehensive wrap of what the Commission has
  heard thus far." Retold casually to a friend, the conversational core is "My colleague Ayanda Nyathi
  joins us from the Commission." "from the Brigitte Mabandla Justice College" only answers a friend's
  possible follow-up "from where, exactly?" -- ranks low. "to give us a more comprehensive wrap" only
  answers a friend's possible follow-up "what kind of coverage?" -- ranks low too, and note this one
  carries no name at all, confirming this test catches more than just incidental entities.

Common, safe LOW-RANK SHAPES that non-goal-serving content tends to take -- do not wait for them to be
the only content left:

- REPORTER SIGN-OFF: a reporter's closing self-identification at the end of a package -- their own
  name, outlet, and location (e.g. "Ofentse Setimo, SABC News, eMalahleni.") -- carries zero story
  content and typically duplicates an on-screen graphic the viewer already sees. This is the FIRST
  thing to consider dropping whenever a segment needs to shrink, ranking lower than every other
  LOW-RANK SHAPE below and lower than any content clause, including a generic category-naming one --
  it never competes with story content for droppability, it is simply the cheapest cut available.
- FORMULAIC GREETING OR OPENER: a pleasantry with zero factual content. Worked example: "...usejoyina
  manje ukuze asinikeze umbiko obanzi... Sawubona ekuseni, Ayanda." -- the greeting clause ranks
  lowest; dropping it leaves "...usejoyina manje ukuze asinikeze umbiko obanzi ngalokho iKhomishana
  ekuzwile kuze kube manje." -- complete and natural on its own, nothing stranded.
- REPEATED FRAMING OR QUESTION ECHO: a question, claim, or framing restated later in the SAME
  segment (or already established in recent_turns_digest/preceding_english) purely for rhetorical
  closure, adding no new fact. Worked example: "...umbuzo obusadingidwa ngowokuthi kungani lolu daba
  lwaluphuthuma kangaka... Umbuzo osamile uthi: kungani kwakuphuthuma kangaka?" -- the closing echo
  ranks low, since it restates rather than adds. This applies just as much when the SAME speaker
  ANNOUNCES a question in one segment, then RE-ASKS the identical question with heavier hedging in
  the very next segment -- confirmed real case, using recent_turns_digest exactly as intended: one
  segment states "the question around why the urgency" (announcing the topic); the very next segment
  then asks "Why was it so urgent to not even allow the process to maybe start being prepared, if you
  will, or the investigation to get underway?" -- this is not new content, it is the SAME question
  restated with more hedging words ("if you will", "maybe") than the announcement already implied.
  Once a question has been announced, a later segment re-asking it (rather than answering it) ranks
  low even though it is phrased as a real, grammatically complete question -- the real payload is
  whatever comes AFTER the question is finally answered, not another restatement of the question
  itself.
- HESITATION OR SELF-CORRECTION: a false start, an immediate self-correction, or a filler
  interjection ("sorry", "I mean") that the speaker themselves abandoned or replaced -- once the
  corrected version is captured in its own clause, the abandoned attempt ranks lowest.
- SCENE-SETTING OR TEMPORAL FRAMING: a phrase locating the moment in the broadcast itself ("just
  before the tea break", "at this point now") rather than describing the actual events/claims being
  reported -- ranks low, since removing it changes nothing about what happened or was alleged.
- A descriptive aside, an illustrative example that is not the only place a fact appears, or purely
  narrative colour added under ADDITION above also ranks low.
- SUPPORTING DETAIL ATTACHED TO A CLAIM OR DENIAL, once you have identified that claim/denial's own
  CORE PROPOSITION (see the HARD RULE below for exactly what that core is): a clause that adds
  circumstantial detail WITHOUT changing who claimed/denied what, or whether it was true, ranks low
  even though it is a real, non-hesitation fact. Worked example: "General Lincoln was in contact
  with Mr. Brown Mogotsi, who he told me was someone close to the minister, something I now know to
  be a lie, not from General Lincoln's side. Had met Mister Mogotsi, not met, spoken to Mister
  Mogotsi, sorry, on the phone once." The core proposition is: Lincoln was in contact with Mogotsi;
  Lincoln told the witness Mogotsi was close to the minister; the witness now knows that to be
  false. BOTH the qualifying clause about whose fault the falsehood was ("not from General Lincoln's
  side") AND the entire trailing personal-contact aside ("spoken to Mister Mogotsi... on the phone
  once") rank low -- neither changes what was claimed, by whom, or that it was false; both are
  supporting circumstantial detail, not the claim/denial itself. Applying STORY-CENTRALITY on top:
  the minister is this example's protagonist -- the claim concerns his own associations; Mr. Brown
  Mogotsi is next -- he is the one alleged to be close to the minister, the actual claim under
  scrutiny; General Lincoln is the supporting-cast member of the three -- he only introduces the
  witness to Mogotsi and relays what he heard. The phone-call aside is entirely about Lincoln's own
  supporting-cast role (introducing the witness to Mogotsi), not about the minister or the claim
  against him, so it ranks lowest of all -- lower than the qualifying clause, which at least concerns
  the claim's truth value directly.

GRANULARITY OF AN ENUMERATED LIST: never enumerate an entire parallel list ("health care, jobs and
agriculture", "mines providing clinics, creating jobs, and addressing racism on farms") as ONE atomic
clause -- give EACH item in the list its OWN clause_id, individually ranked. Lumping a whole list
together as one clause means it can only ever be kept whole or cut whole, which throws away exactly the
flexibility ranking exists to provide: a real segment often needs to drop TWO OR THREE lower-priority
list items, not just the single least essential one, to actually fit its time budget. Build the
shortened_candidates ladder accordingly -- it is entirely normal, and often necessary, for a candidate
several rungs deep to have dropped multiple items from the SAME enumerated list, provided each
remaining item still stands as a complete, natural part of the surviving sentence.

VIRALITY WITHIN AN ENUMERATED SET, a supplementary tie-breaker for a list of parallel items (a list of
stated priorities, achievements, allegations, or categories) that GOAL-RELEVANCE/FACT-PRIORITY alone
don't fully order: Mathula TV is a social-media broadcast product (see the opening framing above) --
within ONE enumerated set, rank each item by how attention-grabbing/shareable it would be to that
audience, not by the order the speaker listed them in. A specific, vivid, or charged claim (a concrete
alleged wrongdoing, a striking figure, a named scandal) ranks MORE essential than a generic category
name covering similar territory, even when the generic one came first. Worked example, from real
content this pipeline has translated: a list of stated priorities naming "jobs and agriculture" is a
generic category-naming item; "racism on farms" in the same list is a specific, vivid, charged claim --
under virality, "jobs and agriculture" ranks MORE droppable than "racism on farms", even though it was
listed first. This heuristic informs but never overrides GOAL-RELEVANCE/FACT-PRIORITY: it only orders
otherwise-similar parallel items within one enumerated set, never pulls a clause out of its enumeration
to compete with unrelated clauses, and never demotes a genuine FACT-PRIORITY item (a name, number,
date, quotation, or a claim/denial's own core proposition) below a merely-vivid one that lacks that
status.

SHARED-REFERENT ECONOMY, a wording choice within how you WRITE isizulu_text (not a separate clause to
rank): when the same literal (a year, a name) is repeated across an enumerated list purely because
the English repeats it, state it once where a natural retelling would -- "koNovemba noma uDisemba
2024" ("November or December 2024") rather than repeating "2024" for each month -- exactly as a
careful newsreader would say it aloud. This applies to isizulu_text itself and to every
shortened_candidates entry; it never changes what the clauses list records (the literal still
belongs to whichever clause(s) it came from), only how naturally you phrase stating it.

Be EXHAUSTIVE, not minimal in your ranking: consider the WHOLE segment for everything that could
rank low across the categories above, not just the first or most obvious candidate -- a longer
segment merged from several original sentences often genuinely has more than one. Do not default
every clause to rank 1 out of caution; a real ranking, even a rarely-used one, is what makes the
next step possible at all.

Once every clause is enumerated and ranked, return shortened_candidates: 0-4 ALTERNATIVE, COMPLETE
Zulu restatements built directly from that ranking, each naming exactly which clause_ids it drops
(dropped_clause_ids) and providing a FRESH, natural, grammatically complete isizulu_text for the
clauses that remain -- never assembled by literally deleting words from the full isizulu_text and
leaving the seam unrepaired, and never a candidate identical or near-identical to the full
isizulu_text. To avoid any ambiguity about direction: rank 1 means MOST ESSENTIAL, and a BIGGER rank
NUMBER means MORE droppable (rank 3 is more droppable than rank 2, which is more droppable than rank
1). Build the ladder starting from the clause with the BIGGEST rank number (the single most
droppable one): your first candidate drops ONLY that one clause; if a second candidate is warranted,
it drops that clause AND the clause with the next-biggest rank number; and so on, working DOWN
toward rank 1, which must never be dropped by any candidate at all. A later, purely mechanical step
measures each rung and picks whichever one fits -- never construct a candidate that drops a clause
with a smaller rank number while a clause with a bigger rank number (more droppable, by definition)
still survives in that same candidate. Since real ranking is now required above, an empty
shortened_candidates list is correct ONLY when literally every clause you enumerated is rank 1 (a
short, fact-dense segment with nothing below that tier) -- it must never be the answer just because
ranking felt unnecessary.

RESTRUCTURE-ONLY CANDIDATE, a distinct kind of shortened_candidates entry from every rung of the
ladder above: when a segment's own isizulu_text is CONVOLUTED in construction even though every one
of its clauses ranked 1 (nothing safe to drop) -- a real, confirmed shape: several clauses correctly
kept but strung together into one rambling, hard-to-follow sentence ("Here was somebody that... he
lays... and then... you email...") -- add ONE candidate with dropped_clause_ids left EMPTY and a
FRESH isizulu_text that restates every one of the SAME clauses, in the SAME order, as a tighter,
cleaner, more natural sentence construction. This is a REWRITE, not a cut: every fact, name, number,
date, and negation the original segment carries must still be present -- the only thing allowed to
change is the grammar/sentence-shape carrying them. Offer this whenever restructuring would genuinely
read more naturally, independent of whether the segment fits its timing budget -- unlike the ladder
above (which exists to reclaim time), this exists to reclaim clarity, and a later, purely mechanical
step decides on its own whether the result also happens to measure shorter.

CODE-SWITCH CANDIDATE, another distinct kind of shortened_candidates entry, dropped_clause_ids left
EMPTY: if isizulu_text used a formal/native rendering for a modern concept noun that also has a
shorter, equally correct code-switched form (see the register guidance above, e.g. "ukucwasa
ngokobuhlanga" vs "i-racism"), add ONE candidate applying that code-switch instead -- do this
REGARDLESS of your own register judgment call above, since the later mechanical step measures every
candidate's REAL spoken duration and picks whichever one actually fits: the code-switch's fit is an
objective, measured fact, not something your own stylistic preference should gate. Skip this only
when no such modern-concept-noun alternative genuinely applies to this segment.

HARD RULE, with exactly TWO narrow, deliberate exceptions below: dropped_clause_ids must NEVER
include a clause carrying a number, a date, a direct quotation, an attribution (who said or did
something), or the CORE PROPOSITION of any claim or denial -- who claimed or denied what, and (for
a denial) whether it was true -- these clauses stay permanently ranked among the most essential,
not merely low-priority, regardless of how the rest of this prompt's economy guidance reads. This
protection covers the core proposition itself, not every qualifying or circumstantial clause
grammatically attached to it -- see the SUPPORTING DETAIL category above for how to tell the two
apart. A PERSON'S OWN NAME, and any ORGANISATION THAT IS ITSELF PART OF WHAT IS BEING REPORTED ON
(an accused party, a body under investigation, an organisation a claim is actually about), are
protected the exact same unconditional way.

THE FIRST EXCEPTION: a name that is PURELY INCIDENTAL CREDENTIAL OR AFFILIATION CONTEXT -- identifying
WHERE someone works, studied, or is otherwise affiliated, when that institution itself plays no role
in the claims/allegations being reported and is never itself the subject of the report -- is NOT
automatically protected merely for being a name. Rank it exactly like any other scene-setting or
supporting detail under STORY-CENTRALITY above, and it may be dropped when it does not serve
communicative_goal. Worked example: "...my colleague Ayanda Nyathi from the Brigitte Mabandla
Justice College, and he joins us now to give us a more comprehensive wrap..." -- "Ayanda Nyathi" is
the colleague's own name and stays protected under the unconditional rule above; "Brigitte Mabandla
Justice College" is purely incidental workplace context that plays no role in the Commission's
substance, so it ranks low and may be dropped ("My colleague Ayanda Nyathi is joining us...") once
it fails to serve the segment's communicative_goal. This exception is NARROW and never applies to a
name that is itself a claim's subject, an accused party, a witness, a protagonist, or an organisation
under investigation or discussion (e.g. IDAC, NPA, or any body whose conduct is being reported on
stays fully protected) -- when there is real doubt about whether a name is purely incidental, keep
it protected.

THE SECOND EXCEPTION: a single item within a LONGER PARALLEL LIST of three or more comparable claims,
promises, or achievements (e.g. one bullet in a party's multi-point manifesto, one item in a list of
stated priorities or accomplishments) is NOT automatically protected merely because it names a
specific action or promise -- rank it like any other content under GOAL-RELEVANCE/VIRALITY (see the
GRANULARITY and VIRALITY guidance above), and it may be dropped, ONE OR SEVERAL such items at a time
if genuinely needed to fit, PROVIDED the list's own general point still comes through with what
remains. This is different from a load-bearing, singular claim/denial (e.g. "Lincoln told the witness
Mogotsi was close to the minister, which is a lie") where the ENTIRE point of the segment IS that one
claim -- an item never qualifies for this exception when it is itself the segment's own
communicative_goal rather than one illustrative example among several. When there is real doubt about
whether an item is a genuinely comparable, illustrative list member versus the segment's actual point,
keep it protected.

Return every requested window_id exactly once, with exactly one segment covering its whole sentence
range as described above. No tools or web search. Return JSON only."""


PASS2_SYSTEM_PROMPT = r"""You are Mathula TV's native isiZulu broadcast translator.

Produce ONE maximally natural, fluent, immediately comprehensible spoken isiZulu
translation for every requested segment. Translate the thought, not English syntax. Use
native isiZulu morphology, agreement, word order and idiom. Preserve names,
numbers/dates/approximations, negation, uncertainty/allegation framing, attribution/agency,
causal/temporal/conditional logic, legal/evidential qualification and speaker intent. Do not
move facts between segment IDs.

Each input segment may include src_ms and target_raw_ms. These are a SOFT dubbing intent,
not permission to damage language: target_raw_ms is deliberately a little longer than the
source articulation window because Mathula will make a small pitch-preserving speed-up
after Azure TTS. Prefer wording whose normal 0%-rate spoken duration is near target_raw_ms
and avoid wording likely to finish materially earlier than src_ms. Never sound telegraphic,
never omit information merely to hit time, and never invent filler or new facts. Naturalness
and semantic fidelity remain mandatory.

Input entries use id/spk/text plus optional timing fields. context_ledger is cached
whole-program context. When neighbour_context is supplied it is read-only continuity
context; translate ONLY segments. Return each requested segment exactly once, in order, as
segment_id + spoken_text. Do not echo English source text. If output_contract requests
context_ledger, also create the same small neutral reusable ledger used by later phases. No
tools or web search. Return JSON only."""


ROLLING_SYLLABLE_SYSTEM_PROMPT = r"""You are Mathula TV's causal rolling-window isiZulu broadcast translator.

Process requested segments STRICTLY in chronological order. locked_previous contains up to
two already-approved isiZulu segments and is immutable. For the first requested segment,
use those locked translations plus the English source context. After you write each requested
segment, treat YOUR newly written isiZulu as locked conversational context when writing the
next requested segment in this same response. While writing any segment, use only its two
most recent locked isiZulu predecessors plus the current English and IMMEDIATE next English
segment; ignore later batch items until their turn. next_source is read-only lookahead and
must never be translated early or moved into the current segment.

Translate the thought, not English syntax. Produce natural, fluent spoken isiZulu with native
morphology, agreement, idiom, pronoun/reference continuity and conversational flow. Preserve
every material proposition: names and identities, numbers/dates, negation, uncertainty and
allegation framing, attribution/agency, legal/evidential qualification, causal/temporal logic,
corrections, repetition that is meaningful to the speaker, and question/answer intent. Never
move facts between segment IDs. Never invent filler.

Each requested item carries a deterministic syllable plan derived from its source mouth
window and Mathula's Azure timing model: target_syllables, min_syllables and max_syllables.
The syllable range is a LANGUAGE-PLANNING OBJECTIVE, not permission to damage meaning.
Prefer the shortest natural isiZulu construction that preserves all meaning when the budget
is tight; when the budget is roomy, prefer natural explicitness already licensed by the source.
Avoid English-shaped verbosity. Do not merely count words or characters.

The fixed Azure utterance overhead means very short source windows may be marked
budget_impossible. In those cases produce the shortest semantically faithful natural spoken
isiZulu you can; never delete a proposition just to satisfy an impossible count.

Return every requested segment exactly once as segment_id + spoken_text. Do not echo source
English. No tools or web search. Return JSON only."""


ROLLING_SYLLABLE_REFINEMENT_SYSTEM_PROMPT = r"""You are refining one ordered rolling-window isiZulu translation batch against deterministic
syllable measurements calculated by Mathula. Process the batch in order. locked_previous is
immutable prior context. Each item contains current_spoken_text, measured_syllables and its
target range.

For items with revise=false, return current_spoken_text EXACTLY unchanged. For revise=true,
recast naturally toward min_syllables..max_syllables while preserving every source proposition,
identity, number/date, uncertainty, attribution and discourse relationship. A severe over-budget
item requires structural compression, not synonym swapping. A severe under-budget item may
make implicit source-licensed meaning naturally explicit but may not add facts or padding.
Treat each revised output as the conversational context for the next item. If the budget is
marked impossible, choose semantic fidelity and the shortest natural form rather than dropping
meaning. Return every requested segment exactly once. JSON only."""


QA_DRIFT_JUDGMENT_CRITERIA = r"""SEPARATELY from meaning-fidelity, check the COMMITTED ISIZULU ITSELF for internal repetition: does
it restate the same clause or sentence more than once when the original English does not repeat
it? This is a structural defect (wastes speaking time, sounds broken to a listener) and must be
treated as drift even when the repeated content is otherwise a faithful, meaning-preserving
rendering of the English -- restating a true claim twice is exactly the kind of redundancy that
needs attention. Do not let the lexical-variance guidance below excuse this: that guidance is
about different WORDS for the same fact, not the same isiZulu sentence appearing twice. Confirmed
real production case: "Yes, and so what is in paragraph four is in fact false." (one English
sentence) was committed as two back-to-back, nearly identical isiZulu sentences both making the
same claim -- a real defect independent of whether the claim itself was translated correctly.

THE CENTRAL TEST -- apply this before flagging anything, not just the named cases below: for a phrase
that seems to have changed or vanished, ask "if I deleted this exact phrase from the ORIGINAL English,
would the sentence still be asking the same question, making the same claim, naming the same facts,
and identifying the same agent?" If deleting it changes NONE of that, the phrase was decoration, not
a proposition -- its absence in the back-translation is not drift, no matter what it's called or
whether it's in the list of named examples below. If deleting it WOULD change what's being claimed,
asked, or established as fact, its absence is drift. This is a category to apply by reasoning, not a
fixed list to pattern-match against -- the examples below are confirmed real cases, not the full set.

Several categories that are NOT drift, and must never be flagged as drift on their own:

- isiZulu is a pro-drop language: the subject ("I", "you", "he", "she", "we", "they") is normally
  carried by a verb prefix (a concord), not a separate word. When a back-translation of a Zulu
  sentence with no separate subject word comes out as "Went to the shop" instead of "You went to the
  shop", that is a back-translation completeness gap, not a translation error -- infer the subject
  the concord encodes (from context and the verb's own prefix) before comparing, the same way a
  fluent isiZulu speaker would read it. Only flag a pronoun/agent problem when the SUBJECT ITSELF
  changed (a different person did the action) or the sentence genuinely shifted to an impersonal/
  passive construction (a real isiZulu impersonal prefix, not just an omitted concord) -- never
  merely because the back-translation happened to render it without an explicit subject.
- Discourse filler, hedges, and framing/parenthetical clauses that carry no factual content of their
  own -- apply the central test above; confirmed real examples include "if you understand what I'm
  saying", "so at this point now", "if you will", "you know", "he was somebody that" (a framing
  clause introducing who does what next, not a claim itself), "for lack of a better term" (a hedge
  about word choice), "to be honest with you" (an honesty hedge), and "that's the word she used" (a
  meta-commentary aside about phrasing). Dropping any of these, or others like them, is this
  pipeline's own deliberate, correct translation policy, not an omission.
- A greeting or pleasantry ("a very good morning to you", "hello") rendered as the other language's
  natural shorter greeting is not an omission -- greetings do not carry factual content requiring
  exact preservation, only the fact that a greeting occurred.
- Lexical/register variance: a different word or phrase for the SAME underlying fact, event, or
  claim (a synonym, a more general or more specific way of naming the same thing, a different
  register) is not drift by itself. Only flag a word-choice difference when it changes WHAT is being
  claimed -- a stronger or weaker claim, a different specific meaning, a different implied cause,
  actor, degree, or manner -- never merely for using a different word to say the same thing. Confirmed
  real case where a word-choice difference WAS worth flagging: "the police are doing EXACTLY what
  happened" (asserting identical conduct) rendered as "doing SIMILAR to what happened" (a weaker,
  hedged claim) -- apply the central test: deleting "exactly" and substituting "similar" changes the
  strength of the claim being made, so that one is real drift.

Confirmed real production case this exists to catch: "When did you speak to Mister Mogotsi?" was
rendered as isiZulu that back-translates to "When did you say to speak with Mr. Mogotsi?" -- a
different question (reporting intent vs. reporting action) that reading the isiZulu alone sounds
fluent and plausible enough to approve without noticing. Actually performing the back-translation,
not reasoning about the isiZulu directly, is what exposes this."""


TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT = r"""You are Mathula TV's SENTENCE-LEVEL TEMPORAL MASK isiZulu translator (Phase A of two).

ONE PROVIDER CALL = ONE COMPLETE WINDOW. Python will NOT send you a residual repair request for this
phase. You must do the drafting, back-translation QA, semantic comparison, and all wording revisions
INSIDE THIS SINGLE provider call before returning the final translation for every sentence in the
window.

Your ONLY job in this call is to produce the single most natural, complete, and faithful isiZulu
translation of each sentence -- there is no compaction ladder here and nothing to benchmark against
a length target. Real Azure text-to-speech will measure your translation afterward; if a sentence
measures oversized against its real window, a SEPARATE, later pass will ask for a targeted
compaction of that specific sentence using the real measured gap -- that pass is out of scope here
and you do not need to anticipate it beyond writing naturally, not padding for length.

WINDOW CONTRACT
- Each mutable unit in this window is exactly ONE English sentence -- never more. The current window
  contains the current chronological sentence plus its immediately following sentences, exactly as
  many as sentence_semantic_window.blocks lists for this call -- window size varies call-to-call
  based on real content length, not a fixed count, so always read it from blocks/actual_block_count
  rather than assuming a particular size.
- Near the end of the transcript, the window may contain fewer remaining sentences. Never prepend
  earlier sentences to manufacture a full window. Earlier committed Zulu may appear only in
  continuity_context for discourse understanding.
- full_source_transcript carries the COMPLETE restored English transcript, in order, for the whole
  clip -- use it freely to resolve references, names, and discourse threads that reach beyond the
  current window's own sentences, but do not treat it as license to add content to the current
  window that its own sentences don't call for.
- Treat the entire supplied window as ONE conversation and ONE comprehension unit, while preserving the
  speaker owner of every individual sentence.
- Return exactly ONE final isiZulu wording per requested mutable group_id (each group_id is one
  sentence). Never omit, duplicate, reorder, or substitute a group_id.

MEANING / COMPREHENSION HARD CONTRACT
- The complete Zulu window must preserve the combined meaning and comprehension of the complete English window.
- Preserve facts, names, identities, numbers, dates, negation, uncertainty, allegation framing, attribution/agency,
  pronouns/references, chronology, causality, questions/answers, self-corrections, discourse transitions, and speaker stance.
- Preserve which speaker owns every proposition. Never move a fact across a speaker boundary.
- Never summarize a sentence merely because it is long.
- Do not invent information.
- English outside the current window (including full_source_transcript) is context only and must not
  become new content within the current window.
- continuity_context is read-only and may clarify references, but it is not permission to add facts to the current window.

LENGTH CONTRACT
- Your translation is the single most natural, complete, and faithful isiZulu rendering of the
  source content, full stop. It is NOT benchmarked against a syllable count, a target duration, or
  any other number -- write it exactly as a fluent isiZulu speaker would say it, preserving every
  proposition. Python measures its real synthesized length afterward, against the real recorded
  window, using real Azure synthesis; you never predict or aim for a length while drafting it.
- Do not pad, repeat, or add elaboration merely to lengthen a sentence, and do not compress meaning
  merely to shorten one. Being naturally paced -- neither padded nor needlessly clipped -- is a
  property of good translation, not a target to hit for its own sake.
- Dense, informal, or unscripted English source content (hedges, false starts, discourse filler,
  redundant framing such as "so at this point now", "if you will", "he was somebody that") carries
  no factual content. Tighten or drop it directly in your translation. This genuinely shortens the
  natural Zulu without losing any comprehension, and is simply better, more natural isiZulu -- it is
  not the same thing as compressing meaning, which remains forbidden.
- Each sentence is now its own independently timed and measured unit -- there is no cross-sentence
  merging here (a short standalone acknowledgment like "All right." and a following "Thank you." are
  separate group_ids, each with its own real window). Render each sentence as naturally and
  concisely as it deserves on its own; do not pad a genuinely short sentence merely because it is
  short.
- Each block's temporal_mask carries two independent pace signals, both worth checking:
  - source_pace_ratio_vs_this_jobs_typical (and source_faster_than_this_jobs_typical): how fast
    THIS block's English was spoken compared to the median pace across THIS SAME transcript. A
    true flag means this specific block is unusually fast for this recording -- look especially
    hard there for hedges/false-starts/filler.
  - source_pace_ratio_vs_typical_broadcast_english (and source_faster_than_typical_broadcast_english):
    how fast THIS block runs compared to a fixed calm-broadcast-English benchmark, independent of
    the rest of this transcript. A recording can be uniformly fast throughout -- every block looking
    "typical" relative to each other while the whole thing runs well above normal broadcast pace.
    When this flag is true even though the job-relative one is not, that means MOST blocks in this
    transcript likely need the same filler-tightening treatment, not just isolated outliers.
  A pace ratio near 1.0 on both means no special tightening effort is needed beyond the usual
  filler check.
- Never omit a proposition, name, number, date, negation, uncertainty, allegation framing,
  attribution, or the speaker's own DELIBERATE repetition/emphasis (for example a figure restated
  for a second, distinct claim, or a rhetorical question echoed as a bookend) merely to shorten --
  that is content, not filler, and cutting it changes the meaning.
- If a fully faithful, filler-tightened rendering is still short of the natural source-equivalent
  pace, use natural context-grounded expansion to reach it: fuller grammatical phrasing, useful
  repetition, tautology, pleonasm, periphrasis, or natural discourse markers -- never meaningless
  filler, and never expansion beyond what closing that gap requires.
- WORKED EXAMPLE — the same discipline applies to whole framing clauses, not just single words,
  and it applies even when the clause matches a pattern already named above as filler:
    English: "He was somebody that less than 24 hours after laying those complaints, lays the
    criminal charges, and then less than 24 hours [later] e-mails the minister..."
    Weaker draft (still over budget): "Wayengumuntu owathi emahoreni angaphansi kwama-24 emva
    kokufaka lezo zikhalazo wafaka amacala obugebengu bese emahoreni angaphansi kwama-24
    uthumela i-imeyili kuNgqongqoshe..."
    Tighter, equally faithful: "Emahoreni angaphansi kwama-24 emva kokufaka lezo zikhalazo
    wafaka amacala obugebengu bese emahoreni angaphansi kwama-24 uthumela i-imeyili
    kuNgqongqoshe..."
  "He was somebody that" is already named above as filler -- drop the WHOLE clause, not just a
  word inside it, whenever the subject is recoverable from the verb concord or established
  context; matching a named pattern is not enough, the fix has to actually land. Also: the
  deliberately repeated phrase ("emahoreni angaphansi kwama-24") is rendered IDENTICALLY both
  times, at its shortest natural form. If you already found a tight, correct wording for a
  phrase once in this block, reuse that exact wording for its repeat -- do not let a later
  revision cycle introduce a longer alternate phrasing (for example adding "emva kwa-"/"ngemva
  kwa-" before a number phrase that was already short) purely because it also sounds natural.
  A longer alternative is not a bug by itself, but paying that cost on BOTH occurrences of a
  repeated phrase, in a block that is already over budget, is exactly the kind of hidden
  regression that erases the savings you found elsewhere.

INTERNAL SELF-QA LOOP — DO NOT OUTPUT DRAFTS OR REASONING
You have up to FOUR internal revision cycles. The following happens INSIDE THIS ONE PROVIDER CALL:

1. Read all English blocks together and understand the complete conversation.
2. Draft all mutable isiZulu blocks as one coherent conversational combination.
3. Back-translate the COMPLETE Zulu window into English -- by actually translating the isiZulu
   WORDS you just drafted, not by recalling or restating the English source from memory. The
   single most common way this step fails silently is producing a "back-translation" that is
   really just the original English typed out again, which will always appear to match even when
   the Zulu itself is wrong. isiZulu negation is carried by compact verb prefixes/suffixes (for
   example ngeke, akakwazi, wayengekwazi, angikwazi) that are easy to draft correctly once and
   then invert by a single affix in a later revision cycle without noticing -- deliberately
   re-derive the polarity (positive/negative) and tense of every verb from its actual Zulu form,
   never from what you intended to write.
4. Compare the back-translation against the COMPLETE original English window.
5. Check specifically for:
   - omitted or added propositions;
   - changed speaker attribution -- including WITHIN a sentence, not just between blocks: an
     English clause like "X, who [someone] told me was Y" nests a reporting verb ("told", "said",
     "indicated") inside a relative clause, and isiZulu's relative concord can attach to either
     the person doing the telling or the person being described. Confirmed real defect: "General
     Lincoln...Mr. Brown Mogotsi, who HE (Lincoln) told me was close to the minister" was committed
     as isiZulu whose relative clause back-translates to "Mogotsi, who TOLD ME he was close to the
     minister" -- the teller and the person described swapped. When back-translating this pattern,
     explicitly identify who the grammatical subject of the embedded reporting verb is and confirm
     it is the same person in both languages; if the Zulu relative construction is ambiguous on
     this point, restructure into two clauses with an explicit subject rather than accept the
     ambiguity;
   - changed names, numbers, dates, organisations, or protected literals;
   - changed negation, uncertainty, allegation framing, or evidential status -- this is precise,
     technical content (legal testimony) where a flipped polarity is a serious defect, not a style
     variance: confirm the polarity of every back-translated verb against its actual Zulu affix,
     not against your memory of what you meant to write;
   - changed question/answer intent;
   - changed temporal/aspectual distinctions such as "about to do" versus "has done";
   - lost self-corrections or discourse relationships;
   - pronoun/reference drift.
6. If ANY semantic defect exists, revise the COMPLETE mutable window, not one isolated block. A
   defect found during back-translation QA -- including a negation/polarity flip -- is not
   acceptable to leave unresolved: this is technical, precise-language content, and a correct
   isiZulu wording that preserves the exact claim exists for every English source sentence. Find
   that wording and revise to it before returning your answer.
7. Back-translate the revised COMPLETE Zulu window again and repeat the checks.
8. Stop as soon as the COMPLETE window passes the semantic check. Otherwise continue revising until
   the four-cycle budget is exhausted -- do not stop early because a defect seems minor or a fix
   seems hard to find. For negation/polarity specifically, a correct fix is always available as a
   same-length or near-same-length wording change (a different verb prefix), never a tradeoff
   against timing, so there is no reason to leave one unresolved.
9. Return the BEST semantically complete, natural translation you reached for every sentence. This
   internal revision process never appears in your output -- you return only the final translation
   for each sentence, not a transcript of the drafts, back-translations, or reasoning that produced
   it. A separate, independent verification pass runs later against your final committed text and
   catches anything that still slipped through, so there is no self-report to fill in here -- your
   only job is to make the returned translation as correct as the four-cycle budget allows.

IMPORTANT: The back-translation QA is a semantic QA mechanism, not a request for literal English wording. Compare propositions,
relationships, certainty, attribution, chronology and intent, not surface words.

PROTECTED LITERALS
- Preserve high-risk numeric/acronym occurrences required by the source. If the same literal occurs twice in separate
  source claims, preserve both distinct claims, not merely the token count.
- Do not use a second occurrence of a literal as a substitute for a missing first claim.

CONFIRMED TRANSLATION PITFALLS
- The payload's confirmed_translation_pitfalls list names specific word/phrase mistranslations this
  pipeline has previously produced and confirmed against a real broadcast. Check every entry against
  your draft before finalizing; each one describes a real recurring mistake, not a hypothetical.

OUTPUT CONTRACT
- Return JSON only, shaped EXACTLY like this worked example for two sentences (a real response
  covers however many group_ids this window actually requested, in any order):
    {
      "masks": [
        {"group_id": "native_phrase_0001", "candidates": [{"candidate_id": "a", "spoken_text": "..."}]},
        {"group_id": "native_phrase_0002", "candidates": [{"candidate_id": "a", "spoken_text": "..."}]}
      ]
    }
  The top level is ALWAYS an object with exactly one key, "masks", whose value is an array of
  per-sentence objects. Two confirmed WRONG shapes to avoid, both seen in real production output:
  (1) a top-level object keyed directly by group_id instead of the "masks" array -- e.g.
  {"native_phrase_0001": {...}, "native_phrase_0002": {...}} is WRONG; (2) a bare top-level array
  with no "masks" wrapper at all -- e.g. [{"group_id": "native_phrase_0001", ...}, ...] is also
  WRONG, even though the array's own contents would otherwise be correct. Always wrap the array in
  {"masks": [...]} as shown above.
- Return exactly the requested group_ids, each with exactly ONE candidate labelled "a" -- there is
  no ladder in this call.
- Do not return drafts, intermediate back-translations, or reasoning from the internal revision
  cycles -- only the final translated spoken_text for each sentence.
- Do not use tools or web search."""

TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT = r"""You are Mathula TV's SENTENCE-LEVEL TEMPORAL MASK isiZulu compaction editor (Phase B of two).

A separate, earlier pass already produced a complete, natural, fully faithful isiZulu translation for
every sentence in this transcript. Real Azure text-to-speech has since measured each one against its
real recorded window. The sentences listed in this call's `repairs` are the ones that measured
oversized: each item carries the English source, the current isiZulu wording, the REAL
`required_speed_percent` -- how much faster that wording would need to be spoken to fit its real
window if left unchanged -- and a concrete `target_syllable_count` (with `current_syllable_count` for
comparison): the syllable count the current wording would need to reach to close that gap. This
target is a DRAFTING AID, not a pass/fail contract -- use it to gauge roughly how much needs to come
out before you start, instead of guessing; it is never a reason to force an unnatural or unfaithful
rewrite just to land on the exact number.

Your ONLY job is to compact each listed sentence's wording to genuinely close as much of that real
gap as you can, while remaining a fully faithful translation of the same English content -- never by
guessing at a length, but by applying real compaction techniques until you run out of ones that do
not damage meaning. If you genuinely cannot compact further without damaging fluidity, comprehension,
or meaning, say so explicitly via `declined_further_compaction` rather than forcing a defective
rewrite or a synonym-only relabel.

MEANING / COMPREHENSION HARD CONTRACT
- Your compacted wording must preserve the combined meaning and comprehension of the English source
  for that sentence, exactly as the current isiZulu wording already does.
- Preserve facts, names, identities, numbers, dates, negation, uncertainty, allegation framing,
  attribution/agency, pronouns/references, chronology, causality, questions/answers,
  self-corrections, discourse transitions, and speaker stance.
- Never summarize a sentence merely because it is long, and never drop a proposition, name, number,
  date, negation, uncertainty, attribution, or protected literal that the current wording preserves.
- Do not invent information. full_source_transcript is context only, for resolving references --
  never a license to add content to a sentence that its own source doesn't call for.

COMPACTION TECHNIQUES -- apply in roughly this order, only as far as the real gap requires
- Re-check the English source for filler, hedges, and false starts the earlier pass may not have
  fully tightened ("so at this point now", "if you will", "he was somebody that", and similar
  redundant framing that carries no factual content) and tighten or drop it in the isiZulu.
- Prefer the SYNTHETIC, single-verb isiZulu construction over an equally natural but more
  PERIPHRASTIC (multi-word) one, whenever both say exactly the same thing. IsiZulu verb morphology
  fuses subject, object, negation, tense, aspect, and mood onto the verb itself as concords -- one
  correctly conjugated verb form routinely carries content that an analytic, multi-word rendering
  would spend a separate auxiliary, standalone pronoun, or particle on. This is NOT a fidelity
  trade-off (both forms are equally faithful and equally natural) -- it is a genuine structural
  choice, and the more synthetic form is usually also the more natural spoken register, not a
  compromise. Reach for this before the last-resort techniques below, since it costs nothing: when
  you catch yourself drafting a standalone pronoun/particle next to a verb that could carry the same
  information as a concord instead, prefer the concord.
- Re-read the CURRENT isiZulu wording for self-inserted intensifiers and hedges -- words that cost
  nothing in comprehension, are easy to add unconsciously while producing natural-sounding Zulu, and
  exist only in the Zulu draft, not the English source, so the filler check above will not catch
  them. This is a CATEGORY to hunt for, not a fixed list: any word whose only job is to intensify,
  emphasize completeness/thoroughness/extent, or double a hedge the source already carries --
  without adding a new fact -- is a candidate to drop, whichever specific word was reached for.
  Confirmed examples (not exhaustive; a synonym is just as droppable): "kakhulu" (very/a lot),
  "impela"/"ngempela" (really/indeed), "okuphelele"/"okuningiliziwe" (complete/comprehensive/
  detailed), "mhlawumbe" (maybe/perhaps) when it doubles a hedge the source's own uncertainty
  framing already carries, and "empeleni" (in fact) as pure emphasis.
- As a LAST RESORT, if the gap is still not closed: written abbreviated honorifics ("Mnu.", "Nkk.",
  "Nksz.", "Dkt.") are pronounced at their FULL spoken form for TTS purposes ("Mnumzane",
  "Nkosikazi", "Nkosazana", "Dokotela" -- 4, 5, 5, and 4 syllables respectively), not the short
  written abbreviation, so dropping the honorific (the surname alone) is an available length
  reduction -- confirmed real case: "Ubusekhaya likaMnu. Lincoln." measured +54% rush once the
  honorific's true spoken length was accounted for. Try it only after the other techniques above,
  since it changes how formally a person is addressed.
- Never omit a proposition, name, number, date, negation, uncertainty, allegation framing,
  attribution, or the speaker's own DELIBERATE repetition/emphasis merely to shorten -- that is
  content, not filler, and cutting it changes the meaning. If the real gap cannot be closed without
  crossing this line, close as much of it as genuinely possible and set
  `declined_further_compaction: true` -- do not force a broken rewrite to look complete.
- WORKED EXAMPLE — the same discipline applies to whole framing clauses, not just single words:
    English: "He was somebody that less than 24 hours after laying those complaints, lays the
    criminal charges, and then less than 24 hours [later] e-mails the minister..."
    Weaker draft (still over budget): "Wayengumuntu owathi emahoreni angaphansi kwama-24 emva
    kokufaka lezo zikhalazo wafaka amacala obugebengu bese emahoreni angaphansi kwama-24
    uthumela i-imeyili kuNgqongqoshe..."
    Tighter, equally faithful: "Emahoreni angaphansi kwama-24 emva kokufaka lezo zikhalazo
    wafaka amacala obugebengu bese emahoreni angaphansi kwama-24 uthumela i-imeyili
    kuNgqongqoshe..."
  "He was somebody that" is filler -- drop the WHOLE clause, not just a word inside it, whenever the
  subject is recoverable from the verb concord or established context. Also: a deliberately repeated
  phrase is rendered IDENTICALLY at every occurrence, at its shortest natural form -- do not let a
  compaction pass shorten one occurrence while leaving (or lengthening) another.

INTERNAL SELF-QA -- DO NOT OUTPUT DRAFTS OR REASONING
For each requested sentence, inside this one provider call:
1. Draft the compacted isiZulu wording using the techniques above, only as far as the real
   required_speed_percent for that sentence calls for -- target_syllable_count is your rough
   drafting gauge for how much needs to come out, not an exact number to force.
2. Back-translate the wording you just drafted -- by actually translating the isiZulu WORDS, not by
   recalling or restating the English source from memory. isiZulu negation is carried by compact verb
   prefixes/suffixes (ngeke, akakwazi, wayengekwazi, angikwazi) that are easy to invert by a single
   affix while compacting without noticing -- deliberately re-derive the polarity and tense of every
   verb from its actual Zulu form.
3. Compare the back-translation against the English source for that sentence, checking for omitted
   or added propositions, changed attribution/negation/uncertainty, changed names/numbers/dates,
   changed question/answer intent, lost self-corrections, and pronoun/reference drift.
4. If a defect exists, revise and re-check. A correct fix that preserves both fidelity and the real
   length gain is always the goal; only decline further compaction once no genuine technique above
   remains that would not reintroduce a defect.
5. Report the FINAL back-translation and verdict in self_check, with the same judgment discipline
   described below -- discrepancy_summary is NEVER empty, even when matches_source=true (write a
   short phrase such as "faithful, no meaningful difference").

""" + QA_DRIFT_JUDGMENT_CRITERIA + r"""

PROTECTED LITERALS
- Preserve high-risk numeric/acronym occurrences required by the source, exactly as the current
  wording already does. If the same literal occurs twice in separate source claims, preserve both
  distinct claims, not merely the token count.

CONFIRMED TRANSLATION PITFALLS
- The payload's confirmed_translation_pitfalls list names specific word/phrase mistranslations this
  pipeline has previously produced and confirmed against a real broadcast. Check every entry against
  your compacted wording before finalizing.

OUTPUT CONTRACT
- Return JSON only, shaped EXACTLY like this worked example for one sentence (a real response
  covers however many group_ids `repairs` actually listed, in any order):
    {
      "results": [
        {
          "group_id": "native_phrase_0001",
          "spoken_text": "...",
          "declined_further_compaction": false,
          "self_check": {"back_translation": "...", "matches_source": true, "discrepancy_summary": "..."}
        }
      ]
    }
  The top level is ALWAYS an object with exactly one key, "results", whose value is an array of
  per-sentence objects -- never a top-level object keyed directly by group_id, and never a bare
  top-level array with no "results" wrapper.
- Return exactly one result per requested group_id in `repairs` -- never omit, duplicate, reorder,
  or substitute a group_id, and never return a result for a group_id that was not requested.
- Each result's spoken_text is the single compacted isiZulu wording for that sentence (or, if
  declined_further_compaction is true, the current wording returned unchanged).
- Do not return drafts, intermediate back-translations, or reasoning from the internal QA -- the
  only back-translation-related output is each result's self_check object.
- Do not use tools or web search."""

TEMPORAL_MASK_RESIDUAL_SYSTEM_PROMPT = r"""DEPRECATED. Temporal-mask Pass 2 no longer performs Python-driven residual calls.
All semantic revision, back-translation QA, and syllable fitting occur inside the single initial window provider call."""

VOICE_RESOLUTION_SYSTEM_PROMPT = r"""You are resolving Azure isiZulu TTS voice families
for a broadcast dub. This is NOT identity research. Use only the supplied transcript context
and explicit linguistic/contextual cues. Do not use tools or web search. For each requested
speaker choose masculine, feminine, or unknown, plus confidence 0..1. Do not infer from a
name alone when context is unclear. Return each speaker exactly once and nothing except JSON.
The result selects only a TTS voice family; it is not a claim about gender identity."""


@dataclass(frozen=True)
class NativeDubPaths:
    root: Path
    pass1: Path
    pass1_sentences: Path
    pass2: Path
    context_ledger: Path
    zulu_glossary: Path
    english_variants: Path
    pronunciation_research: Path
    candidate_pool: Path
    web_overrides: Path
    referent_audit: Path
    timing_repairs: Path
    speech_islands: Path
    qa_back_translation: Path
    timing_plan: Path
    audio_dir: Path
    dialogue_bus: Path
    video: Path
    publication_video: Path
    publication_manifest: Path
    publication_selection: Path
    editorial_response_checkpoint: Path
    hook_text: Path
    seo_package: Path
    seo_caption: Path
    seo_hashtags: Path
    seo_search_keywords: Path


def native_dub_paths(job_root: Path) -> NativeDubPaths:
    root = Path(job_root) / "native_dub"
    return NativeDubPaths(
        root=root,
        pass1=root / "pass1_restored_transcript.json",
        pass1_sentences=root / "pass1_sentences.json",
        pass2=root / "pass2_natural_translation.json",
        context_ledger=root / "context_ledger.json",
        zulu_glossary=root / "zulu_terminology_glossary.json",
        english_variants=root / "english_variant_ladders.json",
        pronunciation_research=root / "pronunciation_research.json",
        candidate_pool=root / "pass2_candidate_pool.json",
        web_overrides=root / "manual_web_overrides.json",
        referent_audit=root / "referent_audit.json",
        timing_repairs=root / "timing_repairs.json",
        speech_islands=root / "speech_islands.json",
        qa_back_translation=root / "qa_back_translation.json",
        timing_plan=root / "timing_plan.json",
        audio_dir=root / "audio" / "phrases",
        dialogue_bus=root / "audio" / "dialogue_bus.wav",
        video=root / "native_dubbed.mp4",
        publication_video=root / "native_dubbed_with_hook.mp4",
        publication_manifest=root / "tiktok_edit_manifest.json",
        publication_selection=root / "editorial_selection.json",
        editorial_response_checkpoint=root / "editorial_response_checkpoint.json",
        hook_text=root / "tiktok_hook.txt",
        seo_package=root / "tiktok_seo.json",
        seo_caption=root / "tiktok_caption.txt",
        seo_hashtags=root / "tiktok_hashtags.txt",
        seo_search_keywords=root / "tiktok_search_keywords.txt",
    )


def install_manual_web_overrides(job_root: Path, input_path: Path) -> dict[str, Any]:
    """Install a human-reviewed web-research override. Never performs research."""
    value = read_json(Path(input_path))
    jsonschema.validate(value, _WEB_OVERRIDE_SCHEMA)
    canonical_seen: set[tuple[str, str]] = set()
    for item in value["entries"]:
        key = (item["heard_as"].casefold().strip(), item["canonical"].casefold().strip())
        if key in canonical_seen:
            raise ValueError(f"Duplicate manual web override: {item['heard_as']}")
        canonical_seen.add(key)
    paths = native_dub_paths(job_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    artifact = {
        **value,
        "installed_at": utcnow(),
        "source_file": str(Path(input_path).resolve()),
        "source_sha256": checksum(Path(input_path)),
        "manual_only": True,
        "automatic_web_search": False,
    }
    atomic_write_json(paths.web_overrides, artifact)
    return artifact


def _strip_leading_cross_segment_duplication(
    candidate_text: str, previous_restored_text: str
) -> tuple[str, bool]:
    """Detect and strip a Pass-1 correction that echoed the PRECEDING segment's own
    already-restored content onto the front of this segment's correction.

    Confirmed real production case: Grok's single whole-transcript autocorrection call
    (one call corrects all segments at once) produced a corrected_text for one segment
    that began with the ENTIRE previous segment's own already-correct text verbatim,
    before that segment's real content -- a repetition artifact, not a real correction
    (the raw ASR source for that segment was clean; only Grok's own output duplicated
    it). Left unrepaired, downstream sentence-splitting proportionally allocates a
    near-zero-duration sliver of this segment's real time budget to the phantom
    duplicate sentences (confirmed: 17ms/8ms slivers, physically impossible to render,
    that crashed the hard-sync speed fit asking for a >24,000% acceleration).
    """
    candidate = str(candidate_text or "").strip()
    previous = str(previous_restored_text or "").strip()
    if not previous or not candidate:
        return candidate, False
    if not candidate.casefold().startswith(previous.casefold()):
        return candidate, False
    remainder = candidate[len(previous):].strip()
    return remainder, True


# A model can't reliably self-detect its own truncated numeral -- there's nothing in
# "January of 20" that looks wrong to a translator reading it in isolation, so this is
# hardcoded detection rather than a prompt instruction. Confirmed real production case,
# job 33cd7b46d55647e39cc1e8d26ab962ed, native_phrase_0028: "...or January of 20, but I
# had spoken to him..." -- an ASR/Pass-1 restoration artifact (almost certainly a
# clipped "2025") that was translated and carried into the isiZulu output verbatim as a
# broken year. Requires the literal "of" before the digits (the confirmed pattern) so a
# genuine day-of-month reference ("January 5th", "on May 3") never matches -- those are
# phrased without "of" and/or carry an ordinal suffix this pattern explicitly excludes.
_TRUNCATED_YEAR_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+of\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th)\b)(?!\d)",
    re.IGNORECASE,
)


def _find_suspected_truncated_year(text: str) -> str | None:
    """Return the suspicious digit group (e.g. "20") if the text looks like a clipped
    year reference, else None. Detection only -- there's no reliable way to guess the
    missing digits from text alone, so this flags for human review rather than
    auto-correcting a number it cannot actually recover.
    """
    match = _TRUNCATED_YEAR_PATTERN.search(text or "")
    return match.group(1) if match else None


def run_native_pass1(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    paths = native_dub_paths(job_root)
    _emit_progress(progress, "[native pass 1] Preparing whole-transcript STT restoration...")
    raw_transcript_path = Path(job_root) / "analysis" / "transcript_en_raw.json"
    authoritative_transcript_path = Path(job_root) / "analysis" / "transcript_en.json"
    azure_diarization_path = Path(job_root) / "analysis" / "azure_diarization.json"

    # Native dubbing owns its source reconstruction. For newly transcribed jobs
    # build transcript_en_raw.json directly from Azure STT/diarization so the
    # legacy autocorrect/web-research pipeline is never required. Existing raw
    # artifacts remain immutable/reusable; older jobs may still fall back to an
    # already-authoritative transcript when no Azure artifact exists.
    if not raw_transcript_path.is_file() and azure_diarization_path.is_file():
        azure = read_json(azure_diarization_path)
        words = azure.get("words")
        turns = azure.get("turns")
        if not isinstance(words, list) or not words:
            raise ValueError("Azure diarization has no words for native STT reconstruction")
        if not isinstance(turns, list) or not turns:
            raise ValueError("Azure diarization has no speaker turns for native STT reconstruction")
        reconciled = reconcile(words, turns, source="azure")
        azure_warnings = azure.get("warnings")
        if isinstance(azure_warnings, list):
            reconciled.setdefault("warnings", []).extend(
                str(value) for value in azure_warnings if str(value).strip()
            )
        reconciled["native_source_reconstruction"] = {
            "schema_version": "mathula-native-raw-transcript-source-v1",
            "source": "analysis/azure_diarization.json",
            "automatic_web_search": False,
            "legacy_autocorrect_used": False,
        }
        raw_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(raw_transcript_path, reconciled)
        _emit_progress(
            progress,
            f"[native pass 1] Reconstructed raw transcript locally from Azure diarization: "
            f"{len(reconciled.get('segments') or [])} segments",
        )

    if raw_transcript_path.is_file():
        transcript_path = raw_transcript_path
    elif authoritative_transcript_path.is_file():
        transcript_path = authoritative_transcript_path
    else:
        raise FileNotFoundError(
            "Native pass 1 requires analysis/azure_diarization.json, "
            "analysis/transcript_en_raw.json, or analysis/transcript_en.json"
        )
    transcript = read_json(transcript_path)
    segments = _normalized_segments(transcript)
    _emit_progress(
        progress,
        f"[native pass 1] Loaded {len(segments)} source segments from {transcript_path.name}",
    )
    overrides = _load_manual_overrides(paths.web_overrides)
    input_hash = _hash_payload({
        "transcript_sha256": checksum(transcript_path),
        "overrides": overrides,
        "prompt_version": PASS1_PROMPT_VERSION,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
    })
    if paths.pass1.is_file() and not force:
        existing = read_json(paths.pass1)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native pass 1] Reusing matching completed checkpoint")
            return existing

    _whole_transcript_preflight(segments, stage="native pass 1")
    _emit_progress(progress, "[native pass 1] Sending complete transcript to Foundry Grok...")
    payload = {
        "source_locale": str(transcript.get("language") or "en-ZA"),
        "segments": _compact_wire_segments(segments, text_key="source_text"),
        "manual_overrides": _compact_manual_overrides(overrides),
    }
    response = provider.complete_json(
        operation="native_stt_restoration",
        system_prompt=PASS1_SYSTEM_PROMPT,
        payload=payload,
        schema=_PASS1_SCHEMA,
        max_output_tokens=min(4096, int(getattr(provider.config, "max_output_tokens", 8192))),
    )
    corrections = _normalize_pass1(response.data, segments, progress=progress)
    context_ledger = _normalize_context_ledger(
        response.data.get("context_ledger"),
        segments,
    )
    _emit_progress(
        progress,
        f"[native pass 1] Grok returned {len(corrections)} accepted STT correction(s); building restored transcript...",
    )
    corrected = {item["segment_id"]: item for item in corrections}
    restored_segments: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        correction = corrected.get(segment["segment_id"])
        candidate_text = correction["corrected_text"] if correction else segment["source_text"]
        if correction is not None and restored_segments:
            previous_restored_text = str(restored_segments[-1].get("restored_text") or "")
            cleaned_text, duplicated = _strip_leading_cross_segment_duplication(
                candidate_text, previous_restored_text
            )
            if duplicated:
                candidate_text = cleaned_text or str(segment.get("source_text") or "")
                item["cross_segment_duplication_repaired"] = True
                _emit_red_warning(
                    progress,
                    f"[native pass 1] WARNING {segment['segment_id']}: corrected_text duplicated "
                    f"preceding segment {restored_segments[-1]['segment_id']}'s own already-restored "
                    "content; stripped the duplicate and kept the remainder.",
                )
        item["restored_text"] = candidate_text
        item["changed"] = correction is not None
        if correction:
            item["reason_code"] = correction["reason_code"]
        suspected_year = _find_suspected_truncated_year(candidate_text)
        if suspected_year is not None:
            item["suspected_truncated_year"] = True
            _emit_red_warning(
                progress,
                f"[native pass 1] WARNING {segment['segment_id']}: possible truncated year "
                f"\"...of {suspected_year}\" -- looks like a clipped 4-digit year (ASR/restoration "
                "artifact), not a real day-of-month reference. Not auto-corrected (the missing "
                "digits can't be recovered from text alone); review before this reaches translation.",
            )
        restored_segments.append(item)

    artifact = {
        "schema_version": STT_RESTORATION_SCHEMA_VERSION,
        "prompt_version": PASS1_PROMPT_VERSION,
        "provider": provider.provider,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "model": response.model,
        "input_sha256": input_hash,
        "source_transcript_path": str(transcript_path),
        "source_transcript_kind": (
            "raw_stt" if transcript_path == raw_transcript_path else "authoritative_fallback"
        ),
        "source_transcript_sha256": checksum(transcript_path),
        "manual_web_override_sha256": checksum(paths.web_overrides) if paths.web_overrides.is_file() else None,
        "automatic_web_search": False,
        "corrections": corrections,
        "context_ledger": context_ledger,
        "segments": restored_segments,
        "token_strategy": {
            "compact_segment_wire": True,
            "source_text_echo_disabled": True,
            "full_json_schema_sent_to_model": False,
            "context_ledger_generated_in_same_full_transcript_call": True,
            "automatic_web_search": False,
        },
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "attempts": response.attempts,
        },
        "completed_at": utcnow(),
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        paths.context_ledger,
        {
            "schema_version": CONTEXT_LEDGER_SCHEMA_VERSION,
            "source_pass1_input_sha256": input_hash,
            "context_ledger": context_ledger,
            "usage": artifact["usage"],
            "generated_with_pass1": True,
            "completed_at": artifact["completed_at"],
        },
    )
    atomic_write_json(paths.pass1, artifact)
    _emit_progress(progress, f"[native pass 1] Saved {paths.pass1}")
    return artifact


def _estimate_zulu_syllables(text: str) -> int:
    """Deterministic orthographic isiZulu syllable proxy.

    IsiZulu syllables are vowel-centred, so vowel nuclei are a useful cheap proxy.
    Numeric and acronym tokens need explicit treatment because Azure speaks them even
    though the orthography itself contains few/no vowels.  This is intentionally an
    estimator used for language planning; Azure duration remains the final timing judge.
    """
    total = 0
    for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+|\d+(?:[.,]\d+)*", str(text or "")):
        if re.fullmatch(r"\d+(?:[.,]\d+)*", token):
            digits = re.sub(r"\D", "", token)
            total += max(1, int(round(len(digits) * 1.5)))
            continue
        letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", token)
        if not letters:
            continue
        if letters.isupper() and len(letters) > 1:
            # Initialisms are commonly realized as letter names; one syllable per
            # letter is a safer duration proxy than zero-vowel orthography.
            total += len(letters)
            continue
        nuclei = re.findall(r"[aeiouAEIOU]+", letters)
        total += max(1, len(nuclei))
    return max(1, total)


def _estimate_english_word_syllables(word: str) -> int:
    """Vowel-group syllable count for one English word, with standard English adjustments.

    Unlike isiZulu's mostly-phonetic orthography, English spelling needs a couple of
    corrections on top of plain vowel-group counting to stay reasonable: a trailing
    silent "e" is not its own syllable, but a trailing consonant+"le" (e.g. "table",
    "little") is. This is the same well-known heuristic behind common readability
    formulas -- a cheap proxy, not a phonetic dictionary lookup.
    """

    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 0
    nuclei = re.findall(r"[aeiouy]+", letters)
    count = len(nuclei)
    if letters.endswith("e") and count > 1:
        count -= 1
    if letters.endswith("le") and len(letters) > 2 and letters[-3] not in "aeiouy":
        count += 1
    return max(1, count)


def _estimate_english_syllables(text: str) -> int:
    """Deterministic orthographic English syllable proxy, mirroring _estimate_zulu_syllables.

    Used to compare the English source's actual pace against a fixed calm-broadcast
    benchmark and against this job's own median -- a flat word count treats "a" and
    "Mississippi" as equally long, which understates how much content a genuinely
    dense/technical block packs in versus a plain one of the same word count. This is
    intentionally a planning-time estimator, not a phonetic dictionary lookup.
    """

    total = 0
    for token in re.findall(r"[A-Za-z'-]+|\d+(?:[.,]\d+)*", str(text or "")):
        if re.fullmatch(r"\d+(?:[.,]\d+)*", token):
            digits = re.sub(r"\D", "", token)
            total += max(1, int(round(len(digits) * 1.5)))
            continue
        letters = re.sub(r"[^A-Za-z]", "", token)
        if not letters:
            continue
        if letters.isupper() and len(letters) > 1:
            # Initialisms are commonly realized as letter names; one syllable per
            # letter is a safer duration proxy than zero-vowel orthography.
            total += len(letters)
            continue
        total += _estimate_english_word_syllables(letters)
    return max(1, total)


def _rolling_syllable_budget(
    segment: Mapping[str, Any],
    *,
    preferred_raw_speed_percent: int,
    content_syllables_per_second: float,
    azure_utterance_overhead_ms: int,
    tolerance_percent: int,
) -> dict[str, Any]:
    start_ms = int(segment.get("start_ms") or 0)
    end_ms = int(segment.get("end_ms") or 0)
    source_ms = max(1, end_ms - start_ms)
    target_raw_ms = int(round(source_ms * (1.0 + int(preferred_raw_speed_percent) / 100.0)))
    rate = max(0.5, float(content_syllables_per_second))
    overhead = max(0, int(azure_utterance_overhead_ms))
    usable_ms = max(0, target_raw_ms - overhead)
    target = max(1, int(round((usable_ms / 1000.0) * rate)))
    slack = max(1, int(math.ceil(target * max(0, int(tolerance_percent)) / 100.0)))
    min_syllables = max(1, target - slack)
    max_syllables = max(min_syllables, target + slack)
    one_syllable_ms = int(round(1000.0 / rate))
    impossible = target_raw_ms < overhead + one_syllable_ms
    return {
        "source_ms": source_ms,
        "target_raw_ms": target_raw_ms,
        "target_syllables": target,
        "min_syllables": min_syllables,
        "max_syllables": max_syllables,
        "budget_impossible": bool(impossible),
    }


def _syllable_range_distance(count: int, budget: Mapping[str, Any]) -> int:
    value = max(0, int(count))
    lower = int(budget["min_syllables"])
    upper = int(budget["max_syllables"])
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0


def _rolling_syllable_wire(
    segment: Mapping[str, Any],
    *,
    preferred_raw_speed_percent: int,
    content_syllables_per_second: float,
    azure_utterance_overhead_ms: int,
    tolerance_percent: int,
) -> dict[str, Any]:
    budget = _rolling_syllable_budget(
        segment,
        preferred_raw_speed_percent=preferred_raw_speed_percent,
        content_syllables_per_second=content_syllables_per_second,
        azure_utterance_overhead_ms=azure_utterance_overhead_ms,
        tolerance_percent=tolerance_percent,
    )
    return {
        "id": str(segment["segment_id"]),
        "spk": str(segment["speaker_id"]),
        "text": str(segment["source_text"]),
        "src_ms": int(budget["source_ms"]),
        "target_raw_ms": int(budget["target_raw_ms"]),
        "target_syllables": int(budget["target_syllables"]),
        "min_syllables": int(budget["min_syllables"]),
        "max_syllables": int(budget["max_syllables"]),
        "budget_impossible": bool(budget["budget_impossible"]),
    }


def _build_rolling_syllable_batches(
    segments: Sequence[Mapping[str, Any]],
    *,
    max_segments: int,
    max_output_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Bound latency without breaking causal order or oversized provider calls."""
    limit = max(1, int(max_segments))
    token_target = max(700, int(max_output_tokens * 0.48))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_estimate = 80
    for raw in segments:
        item = dict(raw)
        estimate = _estimate_segment_output_tokens(item)
        if current and (len(current) >= limit or current_estimate + estimate > token_target):
            batches.append(current)
            current = []
            current_estimate = 80
        current.append(item)
        current_estimate += estimate
    if current:
        batches.append(current)
    return batches


def _normalize_pass2_by_id(
    data: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = [str(item["segment_id"]) for item in segments]
    raw = list(data.get("translations") or [])
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    expected_set = set(expected)
    for item in raw:
        segment_id = str(item.get("segment_id") or "")
        if segment_id not in expected_set:
            unknown.append(segment_id or "<empty>")
            continue
        if segment_id in by_id:
            duplicates.append(segment_id)
            continue
        by_id[segment_id] = item
    missing = [segment_id for segment_id in expected if segment_id not in by_id]
    if missing or duplicates or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if duplicates:
            details.append("duplicate=" + ",".join(sorted(set(duplicates))))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("Rolling Pass 2 identity mismatch: " + "; ".join(details))
    source = {str(item["segment_id"]): str(item["source_text"]) for item in segments}
    normalized: list[dict[str, Any]] = []
    for segment_id in expected:
        spoken = str(by_id[segment_id].get("spoken_text") or "").strip()
        if not spoken:
            raise ValueError(f"Rolling Pass 2 returned empty spoken_text for {segment_id}")
        normalized.append({
            "segment_id": segment_id,
            "source_text": source[segment_id],
            "spoken_text": spoken,
        })
    return normalized


def _annotate_rolling_syllables(
    translations: Sequence[Mapping[str, Any]],
    source_segments: Sequence[Mapping[str, Any]],
    *,
    preferred_raw_speed_percent: int,
    content_syllables_per_second: float,
    azure_utterance_overhead_ms: int,
    tolerance_percent: int,
) -> list[dict[str, Any]]:
    source_by_id = {str(item["segment_id"]): item for item in source_segments}
    result: list[dict[str, Any]] = []
    for raw in translations:
        item = dict(raw)
        source = source_by_id[str(item["segment_id"])]
        budget = _rolling_syllable_budget(
            source,
            preferred_raw_speed_percent=preferred_raw_speed_percent,
            content_syllables_per_second=content_syllables_per_second,
            azure_utterance_overhead_ms=azure_utterance_overhead_ms,
            tolerance_percent=tolerance_percent,
        )
        measured = _estimate_zulu_syllables(str(item["spoken_text"]))
        distance = _syllable_range_distance(measured, budget)
        item.update({
            **budget,
            "estimated_syllables": int(measured),
            "syllable_distance": int(distance),
            "syllable_fit": bool(distance == 0),
        })
        result.append(item)
    return result


def _rolling_refinement_payload(
    annotated: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    severe = 0
    for item in annotated:
        target = max(1, int(item["target_syllables"]))
        distance = int(item["syllable_distance"])
        severe_threshold = max(2, int(math.ceil(target * 0.20)))
        revise = bool(distance >= severe_threshold and not item.get("budget_impossible"))
        if revise:
            severe += 1
        rows.append({
            "id": str(item["segment_id"]),
            "source_text": str(item["source_text"]),
            "current_spoken_text": str(item["spoken_text"]),
            "measured_syllables": int(item["estimated_syllables"]),
            "target_syllables": target,
            "min_syllables": int(item["min_syllables"]),
            "max_syllables": int(item["max_syllables"]),
            "budget_impossible": bool(item.get("budget_impossible")),
            "revise": revise,
        })
    return rows, severe


def _run_native_pass2_rolling_syllable(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool,
    progress: Callable[[str], None] | None,
    rolling_batch_size: int,
    preferred_raw_speed_percent: int,
    content_syllables_per_second: float,
    azure_utterance_overhead_ms: int,
    syllable_tolerance_percent: int,
) -> dict[str, Any]:
    paths = native_dub_paths(job_root)
    _emit_progress(
        progress,
        "[native pass 2] Rolling syllable mode: causal n-2/n-1 context + n+1 lookahead; "
        f"Azure model {content_syllables_per_second:.2f} syll/s after ~{azure_utterance_overhead_ms} ms utterance overhead",
    )
    if not paths.pass1.is_file():
        raise ValueError("Run native-translate pass 1 first; pass1_restored_transcript.json is missing")
    pass1 = read_json(paths.pass1)
    segments = list(pass1.get("segments") or [])
    if not segments:
        raise ValueError("Pass-1 artifact has no restored segments")
    overrides = _load_manual_overrides(paths.web_overrides)
    request_segments = [
        {
            "segment_id": item["segment_id"],
            "speaker_id": item["speaker_id"],
            "start_ms": item["start_ms"],
            "end_ms": item["end_ms"],
            "source_text": item["restored_text"],
        }
        for item in segments
    ]
    _whole_transcript_preflight(request_segments, stage="native pass 2 rolling syllable")

    context_ledger, cached_context_usage = _load_existing_context_ledger(
        job_root=job_root,
        pass1=pass1,
        segments=segments,
        progress=progress,
    )
    if context_ledger is None:
        context_ledger, generated_usage = _ensure_context_ledger(
            job_root=job_root,
            provider=provider,
            pass1=pass1,
            segments=segments,
            progress=progress,
        )
        context_usage = generated_usage
    else:
        context_usage = dict(cached_context_usage)

    input_hash = _hash_payload({
        "pass1_sha256": checksum(paths.pass1),
        "overrides": _compact_manual_overrides(overrides),
        "prompt_version": ROLLING_SYLLABLE_PROMPT_VERSION,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "mode": "causal_rolling_syllable_budget",
        "rolling_batch_size": int(rolling_batch_size),
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "content_syllables_per_second": round(float(content_syllables_per_second), 4),
        "azure_utterance_overhead_ms": int(azure_utterance_overhead_ms),
        "syllable_tolerance_percent": int(syllable_tolerance_percent),
    })
    if paths.pass2.is_file() and not force:
        existing = read_json(paths.pass2)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native pass 2] Reusing matching rolling-syllable checkpoint")
            return existing

    output_budget = max(512, int(getattr(provider.config, "max_output_tokens", 8192)))
    batches = _build_rolling_syllable_batches(
        request_segments,
        max_segments=int(rolling_batch_size),
        max_output_tokens=output_budget,
    )
    all_ids = [str(item["segment_id"]) for item in request_segments]
    index_by_id = {segment_id: index for index, segment_id in enumerate(all_ids)}
    usage_input = int(context_usage.get("input_tokens") or 0)
    usage_output = int(context_usage.get("output_tokens") or 0)
    usage_attempts = int(context_usage.get("attempts") or 0)
    model = provider.config.deployment
    translations: list[dict[str, Any]] = []
    locked_previous: list[dict[str, Any]] = []
    refinement_calls = 0

    for batch_number, batch in enumerate(batches, start=1):
        first_pos = index_by_id[str(batch[0]["segment_id"])]
        last_pos = index_by_id[str(batch[-1]["segment_id"])]
        next_source = []
        if last_pos + 1 < len(request_segments):
            nxt = request_segments[last_pos + 1]
            next_source = [{
                "id": str(nxt["segment_id"]),
                "spk": str(nxt["speaker_id"]),
                "text": str(nxt["source_text"]),
            }]
        wire = [
            _rolling_syllable_wire(
                item,
                preferred_raw_speed_percent=preferred_raw_speed_percent,
                content_syllables_per_second=content_syllables_per_second,
                azure_utterance_overhead_ms=azure_utterance_overhead_ms,
                tolerance_percent=syllable_tolerance_percent,
            )
            for item in batch
        ]
        _emit_progress(
            progress,
            f"[native pass 2] Rolling batch {batch_number}/{len(batches)}: "
            f"{batch[0]['segment_id']}..{batch[-1]['segment_id']} ({len(batch)} segment(s)); "
            f"locked context={len(locked_previous)}",
        )
        response = provider.complete_json(
            operation=f"native_rolling_syllable_translation_batch_{batch_number:02d}_of_{len(batches):02d}",
            system_prompt=ROLLING_SYLLABLE_SYSTEM_PROMPT,
            payload={
                "source_locale": "en-ZA",
                "target_locale": "zu-ZA",
                "context_ledger": context_ledger,
                "locked_previous": locked_previous,
                "segments": wire,
                "next_source": next_source,
                "manual_overrides": _compact_manual_overrides(overrides),
                "rolling_contract": {
                    "lookback_segments": 2,
                    "lookahead_segments": 1,
                    "previous_outputs_immutable": True,
                    "process_current_batch_sequentially": True,
                    "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
                    "syllable_tolerance_percent": int(syllable_tolerance_percent),
                    "azure_content_syllables_per_second": round(float(content_syllables_per_second), 3),
                    "azure_utterance_overhead_ms": int(azure_utterance_overhead_ms),
                },
            },
            schema=_PASS2_SCHEMA,
            max_output_tokens=min(
                output_budget,
                max(900, int(_estimate_pass2_output_tokens(batch) * 1.40)),
            ),
        )
        normalized = _normalize_pass2_by_id(response.data, batch)
        annotated = _annotate_rolling_syllables(
            normalized,
            batch,
            preferred_raw_speed_percent=preferred_raw_speed_percent,
            content_syllables_per_second=content_syllables_per_second,
            azure_utterance_overhead_ms=azure_utterance_overhead_ms,
            tolerance_percent=syllable_tolerance_percent,
        )
        usage_input += response.input_tokens
        usage_output += response.output_tokens
        usage_attempts += response.attempts
        model = response.model

        refinement_rows, severe_misses = _rolling_refinement_payload(annotated)
        if severe_misses:
            refinement_calls += 1
            _emit_progress(
                progress,
                f"[native pass 2] Rolling batch {batch_number}: {severe_misses} severe syllable miss(es); "
                "running one deterministic-count refinement",
            )
            refinement = provider.complete_json(
                operation=f"native_rolling_syllable_refine_batch_{batch_number:02d}_of_{len(batches):02d}",
                system_prompt=ROLLING_SYLLABLE_REFINEMENT_SYSTEM_PROMPT,
                payload={
                    "context_ledger": context_ledger,
                    "locked_previous": locked_previous,
                    "segments": refinement_rows,
                    "next_source": next_source,
                    "manual_overrides": _compact_manual_overrides(overrides),
                },
                schema=_PASS2_SCHEMA,
                max_output_tokens=min(
                    output_budget,
                    max(900, int(_estimate_pass2_output_tokens(batch) * 1.35)),
                ),
            )
            revised = _normalize_pass2_by_id(refinement.data, batch)
            revised_annotated = _annotate_rolling_syllables(
                revised,
                batch,
                preferred_raw_speed_percent=preferred_raw_speed_percent,
                content_syllables_per_second=content_syllables_per_second,
                azure_utterance_overhead_ms=azure_utterance_overhead_ms,
                tolerance_percent=syllable_tolerance_percent,
            )
            revise_by_id = {row["id"]: bool(row["revise"]) for row in refinement_rows}
            current_by_id = {str(item["segment_id"]): item for item in annotated}
            merged: list[dict[str, Any]] = []
            for candidate in revised_annotated:
                segment_id = str(candidate["segment_id"])
                current = current_by_id[segment_id]
                if not revise_by_id[segment_id]:
                    # Refiner was explicitly told to keep this text byte-for-byte.
                    if str(candidate["spoken_text"]) != str(current["spoken_text"]):
                        candidate = current
                elif int(candidate["syllable_distance"]) > int(current["syllable_distance"]):
                    # Syllable refinement is never allowed to make the deterministic
                    # timing proxy worse; keep the first natural candidate instead.
                    candidate = current
                merged.append(dict(candidate))
            annotated = merged
            usage_input += refinement.input_tokens
            usage_output += refinement.output_tokens
            usage_attempts += refinement.attempts
            model = refinement.model

        fit_count = sum(1 for item in annotated if item.get("syllable_fit"))
        impossible_count = sum(1 for item in annotated if item.get("budget_impossible"))
        _emit_progress(
            progress,
            f"[native pass 2] Rolling batch {batch_number} accepted: {fit_count}/{len(annotated)} "
            f"inside syllable range" + (f"; {impossible_count} physically tiny budget(s)" if impossible_count else ""),
        )
        translations.extend(annotated)
        locked_previous = [
            {
                "id": str(item["segment_id"]),
                "spk": str(batch[index]["speaker_id"]),
                "source_text": str(item["source_text"]),
                "spoken_text": str(item["spoken_text"]),
            }
            for index, item in enumerate(annotated)
        ][-2:]

    if [str(item["segment_id"]) for item in translations] != all_ids:
        raise ValueError("Rolling Pass 2 did not reconstruct exact source order")
    fit_count = sum(1 for item in translations if item.get("syllable_fit"))
    impossible_count = sum(1 for item in translations if item.get("budget_impossible"))
    total_distance = sum(int(item.get("syllable_distance") or 0) for item in translations)
    artifact = {
        "schema_version": ROLLING_NATURAL_TRANSLATION_SCHEMA_VERSION,
        "prompt_version": ROLLING_SYLLABLE_PROMPT_VERSION,
        "provider": provider.provider,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "model": model,
        "input_sha256": input_hash,
        "source_pass1_path": str(paths.pass1),
        "source_pass1_sha256": checksum(paths.pass1),
        "context_ledger_path": str(paths.context_ledger),
        "manual_web_override_sha256": checksum(paths.web_overrides) if paths.web_overrides.is_file() else None,
        "automatic_web_search": False,
        "timing_optimized": True,
        "timing_optimization_mode": "causal_rolling_syllable_budget_then_azure_measurement",
        "rolling_syllable_mode": True,
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "word_count_optimized": False,
        "variant_count": 1,
        "translation_batch_count": len(batches),
        "rolling_context": {
            "lookback_segments": 2,
            "lookahead_segments": 1,
            "previous_outputs_immutable": True,
            "sequential_within_batch": True,
            "max_segments_per_batch": int(rolling_batch_size),
        },
        "syllable_model": {
            "kind": "affine_azure_duration_proxy",
            "content_syllables_per_second": round(float(content_syllables_per_second), 4),
            "azure_utterance_overhead_ms": int(azure_utterance_overhead_ms),
            "tolerance_percent": int(syllable_tolerance_percent),
            "counter": "orthographic_vowel_nuclei_plus_numeric_acronym_proxy_v1",
            "planning_only": True,
            "azure_measured_duration_remains_authoritative": True,
        },
        "syllable_summary": {
            "fit_count": int(fit_count),
            "segment_count": len(translations),
            "fit_percent": round(100.0 * fit_count / max(1, len(translations)), 2),
            "total_distance_from_ranges": int(total_distance),
            "physically_tiny_budget_count": int(impossible_count),
            "refinement_calls": int(refinement_calls),
        },
        "token_strategy": {
            "whole_transcript_repeated_per_batch": False,
            "cached_context_ledger": True,
            "rolling_locked_zulu_context": 2,
            "source_lookahead": 1,
            "sequential_generation_inside_bounded_batches": True,
            "deterministic_syllable_refinement_only_on_severe_misses": True,
        },
        "translations": translations,
        "usage": {
            "input_tokens": int(usage_input),
            "output_tokens": int(usage_output),
            "attempts": int(usage_attempts),
            "context_ledger_usage": context_usage,
        },
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.pass2, artifact)
    _emit_progress(
        progress,
        f"[native pass 2] Saved rolling translation: {fit_count}/{len(translations)} syllable-fit; "
        f"{refinement_calls} refinement call(s) — {paths.pass2}",
    )
    return artifact

def _run_native_pass2_standard(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    paths = native_dub_paths(job_root)
    _emit_progress(progress, "[native pass 2] Preparing maximally natural isiZulu translation...")
    if not paths.pass1.is_file():
        raise ValueError("Run native-translate pass 1 first; pass1_restored_transcript.json is missing")
    pass1 = read_json(paths.pass1)
    segments = list(pass1.get("segments") or [])
    if not segments:
        raise ValueError("Pass-1 artifact has no restored segments")
    overrides = _load_manual_overrides(paths.web_overrides)

    # Prefer a ledger generated as part of Pass 1. For older checkpoints, do not
    # pay another full-transcript input call unless batching actually becomes
    # necessary; a successful full Pass-2 call can return the small ledger in the
    # same response at almost no extra input cost.
    context_ledger, cached_context_usage = _load_existing_context_ledger(
        job_root=job_root,
        pass1=pass1,
        segments=segments,
        progress=progress,
    )
    input_hash = _hash_payload({
        "pass1_sha256": checksum(paths.pass1),
        "overrides": _compact_manual_overrides(overrides),
        "prompt_version": PASS2_PROMPT_VERSION,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "preferred_raw_speed_percent": DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
    })
    if paths.pass2.is_file() and not force:
        existing = read_json(paths.pass2)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native pass 2] Reusing matching completed checkpoint")
            return existing

    request_segments = [
        {
            "segment_id": item["segment_id"],
            "speaker_id": item["speaker_id"],
            "start_ms": item["start_ms"],
            "end_ms": item["end_ms"],
            "source_text": item["restored_text"],
        }
        for item in segments
    ]
    _whole_transcript_preflight(request_segments, stage="native pass 2")

    output_budget = max(512, int(getattr(provider.config, "max_output_tokens", 8192)))
    estimated_output_tokens = _estimate_pass2_output_tokens(request_segments)
    # With source-text echo removed, leave ~20% headroom for isiZulu tokenisation
    # variance and JSON. This job class should usually fit in one grok-4.3 call.
    prebatch = estimated_output_tokens > int(output_budget * 0.80)
    if prebatch:
        _emit_progress(
            progress,
            f"[native pass 2] Estimated output ~{estimated_output_tokens:,} tokens vs "
            f"{output_budget:,} budget; using token-saving context-ledger batches immediately",
        )
    else:
        _emit_progress(
            progress,
            f"[native pass 2] Sending one compact whole-program request for {len(request_segments)} segments...",
        )

    batch_count = 1
    fallback_from_max_tokens = False
    failed_full_usage = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    context_usage = dict(cached_context_usage)
    usage_input = int(context_usage.get("input_tokens") or 0)
    usage_output = int(context_usage.get("output_tokens") or 0)
    usage_attempts = int(context_usage.get("attempts") or 0)
    model = provider.config.deployment
    translations: list[dict[str, Any]] = []

    if not prebatch:
        try:
            full_payload: dict[str, Any] = {
                "source_locale": "en-ZA",
                "target_locale": "zu-ZA",
                "segments": _compact_wire_segments(
                    request_segments,
                    text_key="source_text",
                    soft_timing_speed_percent=DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
                ),
                "manual_overrides": _compact_manual_overrides(overrides),
            }
            if context_ledger is not None:
                full_payload["context_ledger"] = context_ledger
            full_schema = _PASS2_SCHEMA if context_ledger is not None else _PASS2_FULL_SCHEMA
            response = provider.complete_json(
                operation="native_natural_translation",
                system_prompt=PASS2_SYSTEM_PROMPT,
                payload=full_payload,
                schema=full_schema,
                max_output_tokens=min(
                    output_budget,
                    max(1400, int(estimated_output_tokens * 1.45)),
                ),
            )
            translations = _normalize_pass2(response.data, request_segments, progress=progress)
            if context_ledger is None:
                context_ledger = _normalize_context_ledger(
                    response.data.get("context_ledger"),
                    segments,
                )
                _write_context_ledger(
                    paths=paths,
                    pass1=pass1,
                    ledger=context_ledger,
                    usage={"input_tokens": 0, "output_tokens": 0, "attempts": 0},
                    generated_with_pass1=False,
                    generated_with_pass2=True,
                )
            usage_input += response.input_tokens
            usage_output += response.output_tokens
            usage_attempts += response.attempts
            model = response.model
        except GrokOutputTruncated as exc:
            fallback_from_max_tokens = True
            failed_full_usage = {
                "input_tokens": exc.input_tokens,
                "output_tokens": exc.output_tokens,
                "attempts": exc.attempts,
            }
            usage_input += exc.input_tokens
            usage_output += exc.output_tokens
            usage_attempts += exc.attempts
            model = exc.model
            _emit_progress(
                progress,
                "[native pass 2] Grok still hit its output ceiling; switching to compact context-ledger batches...",
            )

    if prebatch or fallback_from_max_tokens:
        if context_ledger is None:
            context_ledger, generated_usage = _ensure_context_ledger(
                job_root=job_root,
                provider=provider,
                pass1=pass1,
                segments=segments,
                progress=progress,
            )
            context_usage = generated_usage
            usage_input += int(generated_usage.get("input_tokens") or 0)
            usage_output += int(generated_usage.get("output_tokens") or 0)
            usage_attempts += int(generated_usage.get("attempts") or 0)
        batches = _build_pass2_batches(
            request_segments,
            max_output_tokens=output_budget,
        )
        batch_count = len(batches)
        translations = []
        all_ids = [item["segment_id"] for item in request_segments]
        index_by_id = {segment_id: i for i, segment_id in enumerate(all_ids)}
        for index, batch in enumerate(batches, start=1):
            _emit_progress(
                progress,
                f"[native pass 2] Translating batch {index}/{batch_count}: {len(batch)} segment(s)...",
            )
            neighbour_context = _batch_neighbour_context(
                request_segments,
                batch,
                index_by_id=index_by_id,
                radius=2,
            )
            batch_response = provider.complete_json(
                operation=f"native_natural_translation_batch_{index:02d}_of_{batch_count:02d}",
                system_prompt=PASS2_SYSTEM_PROMPT,
                payload={
                    "source_locale": "en-ZA",
                    "target_locale": "zu-ZA",
                    "context_ledger": context_ledger,
                    "neighbour_context": neighbour_context,
                    "segments": _compact_wire_segments(
                        batch,
                        text_key="source_text",
                        soft_timing_speed_percent=DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
                    ),
                    "manual_overrides": _compact_manual_overrides(overrides),
                    "batch_index": index,
                    "batch_count": batch_count,
                },
                schema=_PASS2_SCHEMA,
                max_output_tokens=min(
                    output_budget,
                    max(900, int(_estimate_pass2_output_tokens(batch) * 1.35)),
                ),
            )
            translations.extend(
                _normalize_pass2(batch_response.data, batch, progress=progress)
            )
            usage_input += batch_response.input_tokens
            usage_output += batch_response.output_tokens
            usage_attempts += batch_response.attempts
            model = batch_response.model
        if [item["segment_id"] for item in translations] != all_ids:
            raise ValueError("Pass 2 batched translation did not reconstruct exact source order")

    assert context_ledger is not None
    _emit_progress(
        progress,
        f"[native pass 2] Validated one natural isiZulu translation for all {len(translations)} segments",
    )
    artifact = {
        "schema_version": NATURAL_TRANSLATION_SCHEMA_VERSION,
        "prompt_version": PASS2_PROMPT_VERSION,
        "provider": provider.provider,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "model": model,
        "input_sha256": input_hash,
        "source_pass1_path": str(paths.pass1),
        "source_pass1_sha256": checksum(paths.pass1),
        "context_ledger_path": str(paths.context_ledger),
        "manual_web_override_sha256": checksum(paths.web_overrides) if paths.web_overrides.is_file() else None,
        "automatic_web_search": False,
        "timing_optimized": True,
        "timing_optimization_mode": "soft_raw_duration_target_then_measured_speed_fit",
        "preferred_raw_speed_percent": DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
        "word_count_optimized": False,
        "variant_count": 1,
        "translation_batch_count": batch_count,
        "prebatched_for_token_budget": prebatch,
        "estimated_output_tokens": estimated_output_tokens,
        "provider_output_budget_tokens": output_budget,
        "fallback_from_max_tokens": fallback_from_max_tokens,
        "failed_full_request_usage": failed_full_usage if fallback_from_max_tokens else None,
        "token_strategy": {
            "compact_segment_wire": True,
            "source_text_echo_disabled": True,
            "full_json_schema_sent_to_model": False,
            "cached_context_ledger": True,
            "whole_transcript_repeated_per_batch": False,
            "batch_context_radius_segments": 2,
            "preflight_output_budgeting": True,
            "context_ledger_piggybacked_on_full_pass2_when_missing": True,
            "automatic_web_search": False,
            "soft_duration_intent_in_pass2": True,
            "preferred_raw_speed_percent": DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
        },
        "translations": translations,
        "usage": {
            "input_tokens": usage_input,
            "output_tokens": usage_output,
            "attempts": usage_attempts,
            "context_ledger_usage": context_usage,
        },
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.pass2, artifact)
    _emit_progress(progress, f"[native pass 2] Saved {paths.pass2}")
    return artifact


_TERMINAL_SENTENCE_PUNCTUATION = (".", "?", "!", "…")


def _segment_ends_mid_utterance(segment: Mapping[str, Any]) -> bool:
    """True when a segment's own text does not actually end a sentence.

    A trailing "." is not always a real sentence end: a title abbreviation ("Mr.",
    "Dr.", ...) carries its own period without closing the sentence. Confirmed real
    case: "You were at Mr." (one segment) and "Lincoln's house." (the next, 0ms gap,
    same speaker) are one sentence ("You were at Mr. Lincoln's house.") split exactly
    at the abbreviation's period -- naive terminal-punctuation matching would treat
    "You were at Mr." as already complete and never re-join it, same failure shape as
    the plain-fragment case ("More", "And") this function also catches.
    """
    text = str(segment.get("restored_text") or segment.get("source_text") or "").strip()
    if not text:
        return False
    if not text.endswith(_TERMINAL_SENTENCE_PUNCTUATION):
        return True
    if text.endswith(".") and text.split()[-1].strip(".").casefold() in ENGLISH_TITLE_ABBREVIATIONS:
        return True
    return False


def _merge_fragmented_pass1_segments(
    pass1_segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-join adjacent Pass-1 segments the ASR/diarization stage split mid-utterance.

    Confirmed real cases:
    - "More" and "specifically, we know that when Fadiel Adams opened these cases..."
      are the same speaker with a 0ms gap -- "More" carries no terminal punctuation,
      meaning it isn't a complete sentence, just a fragment cut mid-phrase ("More
      specifically,..."). Left unmerged: a 200ms source window with no possible
      natural-paced Zulu rendering (+286% rush).
    - "And" and "there was a bit of an interaction with one of the journalists a
      short while ago." are the same speaker with a 160ms gap -- a real breath pause
      mid-sentence ("And there was..."), not a sentence boundary. A gap-<=0-only
      threshold (this function's first version, mirroring build_phrase_groups' own
      effective behavior) misses this: ordinary speech has small positive gaps
      between words even within one continuous utterance. Left unmerged: a lone
      one-word "And" against an ~120ms window, +287.5% rush, structurally impossible
      regardless of how many pace-variant candidates exist for it -- there is nothing
      left to compact in a single word.

    Sentence splitting can only work within one segment's own text; if the fragment
    boundary itself is wrong, no amount of correct downstream splitting or candidate
    compaction fixes it. The safety net against over-merging is the punctuation
    check, not the gap size: only merge when the EARLIER segment's own text doesn't
    already end in terminal punctuation, so two genuinely separate back-to-back
    complete sentences are never merged back together regardless of how close they
    sit. 900ms (not 0ms) is used for the gap ceiling itself, matching the
    already-established "same articulatory phrase" threshold this codebase uses
    elsewhere (build_phrase_groups' own max_join_gap_ms default).
    """
    merged: list[dict[str, Any]] = []
    for segment in pass1_segments:
        if merged:
            previous = merged[-1]
            gap_ms = int(segment["start_ms"]) - int(previous["end_ms"])
            same_speaker = str(segment["speaker_id"]) == str(previous["speaker_id"])
            if same_speaker and -200 <= gap_ms <= 900 and _segment_ends_mid_utterance(previous):
                previous_text = str(previous.get("restored_text") or previous.get("source_text") or "").strip()
                next_text = str(segment.get("restored_text") or segment.get("source_text") or "").strip()
                previous["restored_text"] = f"{previous_text} {next_text}".strip()
                previous["end_ms"] = int(segment["end_ms"])
                previous["segment_ids"] = list(previous.get("segment_ids") or [str(previous["segment_id"])]) + [
                    str(segment["segment_id"])
                ]
                continue
        item = dict(segment)
        item.setdefault("segment_ids", [str(segment["segment_id"])])
        merged.append(item)
    return merged


def _pass1_segment_to_sentence_pseudo_group(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Shape one (possibly fragment-merged) Pass-1 segment as a "group" for
    derive_english_islands.

    derive_english_islands is otherwise a post-Pass-2 retrofit (speech_islands.py),
    but it has zero dependency on Zulu text -- it only reads source_text/segment_ids/
    group_id/start_ms/source_end_ms. Shaping a Pass-1 segment into that same contract
    lets it run unmodified, before Pass 2, on the English side alone. segment_ids may
    carry more than one original segment id when _merge_fragmented_pass1_segments
    glued adjacent ASR fragments back together -- real word-level alignment needs
    every constituent segment's own source_word_ids, not just the first one's.
    """
    segment_id = str(segment["segment_id"])
    segment_ids = [str(value) for value in segment.get("segment_ids") or [segment_id]]
    return {
        "group_id": segment_id,
        "segment_ids": segment_ids,
        "source_text": str(segment.get("restored_text") or segment.get("source_text") or ""),
        "start_ms": int(segment["start_ms"]),
        "source_end_ms": int(segment["end_ms"]),
    }


def _split_pass1_segment_into_sentences(
    segment: Mapping[str, Any],
    word_index: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Split one Pass-1 restored segment into sentence-level records with real timing.

    Falls back to the segment unchanged (one record) only when no split is needed at
    all (already one sentence). When this job has no raw ASR word timestamps to anchor
    to, derive_english_islands itself still splits by sentence, just using its own
    proportional-by-character-length estimate instead of real word alignment -- passing
    empty word/segment lookups (rather than skipping the call) routes into that same
    fallback instead of silently keeping multi-sentence segments intact.
    """
    segment_id = str(segment["segment_id"])
    speaker_id = str(segment["speaker_id"])
    restored_text = str(segment.get("restored_text") or segment.get("source_text") or "").strip()
    islands: list[dict[str, Any]] | None = None
    if restored_text:
        words_by_id, segments_by_id = word_index if word_index is not None else ({}, {})
        pseudo_group = _pass1_segment_to_sentence_pseudo_group(segment)
        islands = derive_english_islands(pseudo_group, words_by_id, segments_by_id)
    if not islands:
        return [{
            "segment_id": f"{segment_id}__s00",
            "parent_segment_id": segment_id,
            "speaker_id": speaker_id,
            "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]),
            "restored_text": restored_text,
            "alignment_method": "whole_segment",
        }]
    return [
        {
            "segment_id": f"{segment_id}__s{int(island['index']):02d}",
            "parent_segment_id": segment_id,
            "speaker_id": speaker_id,
            "start_ms": int(island["start_ms"]),
            "end_ms": int(island["end_ms"]),
            "restored_text": str(island["source_text"]),
            "alignment_method": str(island["alignment_method"]),
        }
        for island in islands
    ]


def _build_pass1_sentence_records(
    pass1_segments: Sequence[Mapping[str, Any]],
    *,
    job_root: Path,
) -> list[dict[str, Any]]:
    """Split every Pass-1 restored segment into sentence-level records with real timing.

    Runs the same word-timestamp alignment speech_islands.py normally applies AFTER
    Pass 2 (as a post-hoc retrofit), but here -- BEFORE Pass 2 -- so translation and
    per-sentence timing operate on the real atomic unit (one sentence, independently
    voiced) from the start, instead of a multi-sentence block whose aggregate timing
    can mask badly-rushed individual sentences. Deterministic, no AI call.

    First re-joins any adjacent segments the ASR/diarization stage split mid-utterance
    (see _merge_fragmented_pass1_segments) -- otherwise a bare fragment like "More"
    would be treated as its own complete, already-one-sentence unit rather than the
    start of the sentence it actually belongs to.
    """
    word_index = _load_raw_word_index_for_job(job_root)
    merged_segments = _merge_fragmented_pass1_segments(pass1_segments)
    records: list[dict[str, Any]] = []
    for segment in merged_segments:
        records.extend(_split_pass1_segment_into_sentences(segment, word_index))
    return records


def _build_temporal_mask_source_groups(
    sentence_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one temporal-mask group per sentence -- the atomic translation/timing unit.

    Each sentence record (see _build_pass1_sentence_records) already carries real
    (word-aligned or proportionally estimated) start_ms/end_ms for exactly one
    sentence, so no further merging/grouping is needed here.
    """
    groups: list[dict[str, Any]] = []
    for index, record in enumerate(sentence_records, 1):
        segment_id = str(record["segment_id"])
        source_text = str(record["restored_text"])
        groups.append({
            "group_id": f"native_phrase_{index:04d}",
            "speaker_id": str(record["speaker_id"]),
            "segment_ids": [segment_id],
            "start_ms": int(record["start_ms"]),
            "source_end_ms": int(record["end_ms"]),
            "source_span_ms": int(record["end_ms"]) - int(record["start_ms"]),
            "source_segments": [{"segment_id": segment_id, "source_text": source_text}],
            "source_text": source_text,
        })
    return groups


_FAST_SOURCE_PACE_RATIO_THRESHOLD = 1.15


def _english_syllables_per_second(source_text: str, source_span_ms: int) -> float:
    syllables = _estimate_english_syllables(source_text)
    seconds = max(0.001, int(source_span_ms) / 1000.0)
    return syllables / seconds


def _english_source_pace_ratios(groups: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return {group_id: this block's English syllables/sec relative to this job's own median}.

    The syllable budget elsewhere is purely time-based (window duration x an assumed Zulu
    pace) and never looks at how much English content was actually packed into that window
    -- confirmed as the real driver behind large native-dub rushes: a speaker talking
    faster than the job's own typical pace fits more propositions into less time than the
    budget assumes, independent of translation quality. Comparing against THIS job's own
    median pace (not a fixed universal English rate, which would be an arbitrary constant
    unrelated to this recording's actual speakers/register) tells Pass 2 which specific
    blocks are worth extra effort hunting for compressible discourse filler. Syllables, not
    a flat word count, since "a" and "Mississippi" are not equally long to say.
    """

    paces: list[float] = []
    per_group_pace: dict[str, float] = {}
    for group in groups:
        group_id = str(group["group_id"])
        pace = _english_syllables_per_second(group.get("source_text") or "", int(group.get("source_span_ms") or 1))
        per_group_pace[group_id] = pace
        if pace > 0:
            paces.append(pace)
    if not paces:
        return {group_id: 1.0 for group_id in per_group_pace}
    ordered = sorted(paces)
    middle = len(ordered) // 2
    median_pace = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    if median_pace <= 0:
        return {group_id: 1.0 for group_id in per_group_pace}
    return {group_id: round(pace / median_pace, 3) for group_id, pace in per_group_pace.items()}


# ~150-160 words/minute is a commonly cited calm/neutral broadcast news-reading pace
# (~2.6 words/sec); English averages roughly 1.4-1.5 syllables per word, giving ~3.9
# syllables/sec as the equivalent syllable-rate benchmark.
_TYPICAL_BROADCAST_ENGLISH_SYLLABLES_PER_SECOND = 3.9


def _english_absolute_pace_ratio(source_text: str, source_span_ms: int) -> float:
    """This block's English pace relative to a fixed typical-broadcast-English benchmark.

    The job-relative ratio above cannot catch a recording whose speakers run fast
    THROUGHOUT: confirmed in production on a panel-discussion job with a 3.3-3.5
    words/second median across the whole transcript, where every individual block
    looked "typical" relative to the job's own baseline despite the entire recording
    running well above a normal broadcast pace. This absolute benchmark catches that
    case -- a job that is uniformly fast gets uniformly flagged here, even when nothing
    in it is a relative outlier.
    """

    pace = _english_syllables_per_second(source_text, source_span_ms)
    return round(pace / _TYPICAL_BROADCAST_ENGLISH_SYLLABLES_PER_SECOND, 3)


def _temporal_mask_geometry(
    group: Mapping[str, Any],
    *,
    next_source_start_ms: int,
    preferred_raw_speed_percent: int,
    max_natural_speed_percent: int,
    mouth_tolerance_ms: int,
    max_protected_gap_overflow_ms: int,
    min_protected_pause_ms: int,
) -> dict[str, Any]:
    source_ms = max(1, int(group["source_span_ms"]))
    source_end_ms = int(group["source_end_ms"])
    late_ms = _effective_protected_late_allowance(
        int(mouth_tolerance_ms),
        source_end_ms,
        int(next_source_start_ms),
        protected_gap_overflow=True,
        max_protected_gap_overflow_ms=int(max_protected_gap_overflow_ms),
        min_protected_pause_ms=int(min_protected_pause_ms),
    )
    early_ms = int(mouth_tolerance_ms)
    target_raw_ms = int(round(source_ms * (1.0 + int(preferred_raw_speed_percent) / 100.0)))
    minimum_raw_ms = max(1, source_ms - early_ms)
    maximum_raw_ms = int(round((source_ms + late_ms) * (1.0 + int(max_natural_speed_percent) / 100.0)))
    return {
        "preferred_start_ms": int(group["start_ms"]),
        "preferred_end_ms": source_end_ms,
        "hard_earliest_start_ms": int(group["start_ms"]),
        "hard_latest_end_ms": source_end_ms + late_ms,
        "source_window_ms": source_ms,
        "preferred_raw_ms": target_raw_ms,
        "minimum_usable_raw_ms": minimum_raw_ms,
        "maximum_usable_raw_ms": maximum_raw_ms,
        "mouth_close_early_tolerance_ms": early_ms,
        "mouth_close_late_tolerance_ms": late_ms,
        "free_gap_ms": max(0, int(next_source_start_ms) - source_end_ms),
    }


def _temporal_mask_wire(
    groups: Sequence[Mapping[str, Any]],
    index: int,
    *,
    accepted_previous: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    candidate_count: int,
    source_pace_ratio: float = 1.0,
) -> dict[str, Any]:
    """Compact per-sentence mask wire.

    English and previous Zulu are intentionally NOT repeated here. They are sent once
    per provider call in ``sentence_semantic_window`` and ``continuity_context``.
    This keeps window composition token-efficient.
    """
    group = groups[index]
    previous_speaker = str(groups[index - 1]["speaker_id"]) if index else None
    current_source = str(group.get("source_text") or "")
    absolute_pace_ratio = _english_absolute_pace_ratio(current_source, int(group.get("source_span_ms") or 1))
    return {
        "group_id": str(group["group_id"]),
        "speaker_id": str(group["speaker_id"]),
        "segment_ids": [str(value) for value in group["segment_ids"]],
        "semantic_source": "sentence_semantic_window.blocks[group_id]",
        "discourse": {
            "same_speaker_as_previous": bool(previous_speaker and previous_speaker == str(group["speaker_id"])),
            "source_is_unfinished": bool(re.search(r"(?:--|—|\.\.\.)\s*$", current_source.strip())),
        },
        "temporal_mask": {
            "source_start_ms": int(geometry["preferred_start_ms"]),
            "source_end_ms": int(geometry["preferred_end_ms"]),
            "candidate_a_has_no_length_target": True,
            "never_slow_down_but_never_pad_either": True,
            "tighten_discourse_filler_hedges_and_false_starts_not_facts": True,
            "rush_warning_threshold_percent": DEFAULT_MAX_NATURAL_SPEED_PERCENT,
            "source_pace_ratio_vs_this_jobs_typical": round(float(source_pace_ratio), 3),
            "source_faster_than_this_jobs_typical": bool(
                source_pace_ratio >= _FAST_SOURCE_PACE_RATIO_THRESHOLD
            ),
            "source_pace_ratio_vs_typical_broadcast_english": absolute_pace_ratio,
            "source_faster_than_typical_broadcast_english": bool(
                absolute_pace_ratio >= _FAST_SOURCE_PACE_RATIO_THRESHOLD
            ),
        },
        "protected_literals": _protected_timing_literals(group),
        "high_risk_occurrence_requirements": _temporal_mask_source_occurrence_requirements(current_source),
        "max_candidates": int(candidate_count),
    }


# Phase A's own prompt already instructs tightening discourse filler/hedges (see
# TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT's LENGTH CONTRACT), but that's a soft
# instruction the model doesn't reliably apply when a sentence already fits its timing
# window without needing to -- confirmed real cases from a live production job: "so at
# this point now" (native_phrase_0003), "of course" and "quite frankly"
# (native_phrase_0017), and "to be honest with you" (native_phrase_0136) all survived
# into the isiZulu verbatim even though every one of them is a phrase this pipeline's
# own QA_DRIFT_JUDGMENT_CRITERIA already names as safe to drop with no loss of meaning.
# Stripping them from what Phase A even sees removes the dependency on the model
# choosing to comply, for the phrases confirmed hard enough to get reliably right via
# prompting alone. "of course", "quite frankly", "you know", and "I guess" require a
# comma on BOTH sides before stripping -- unlike the others, they can be load-bearing
# outside a parenthetical aside ("Of course." as a standalone answer, "you know him",
# "I guess so." as a standalone answer) -- only their mid-sentence hedge usage is
# targeted. "I mean" requires a TRAILING comma specifically so "I mean it"/"I mean
# this" (a real verb phrase, not a discourse marker) never matches. Deliberately
# excluded after review of a live job's real sentences: "kind of" (embedded inside a
# noun phrase -- "some kind of attempt" -- removal breaks the grammar, not just the
# hedge), "actually" and "essentially" (both can carry real epistemic weight -- "the
# same" vs. "essentially the same" is a genuine precision difference, the same class of
# drift QA_DRIFT_JUDGMENT_CRITERIA already confirms matters for "exactly"/"similar").
_CONFIRMED_DROPPABLE_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r",?\s*\bif you understand what I'?m saying\b\s*,?",
        r"\bso at this point now\b\s*,?",
        r",?\s*\bif you will\b\s*,?",
        r",?\s*\bfor lack of a better term\b\s*,?",
        r",?\s*\bto be honest with you\b\s*,?",
        r",?\s*\bso to speak\b",
        r",\s*\bof course\b\s*,",
        r",\s*\bquite frankly\b\s*,",
        r",\s*\byou know\b\s*,",
        r",\s*\bI guess\b\s*,",
        r",?\s*\bI mean\b\s*,",
    )
)


def _strip_confirmed_filler_phrases(text: str) -> str:
    """Deterministically strip this pipeline's confirmed-safe discourse filler from the
    English handed to Phase A for translation (see _CONFIRMED_DROPPABLE_FILLER_PATTERNS).

    Only affects what Phase A reads as translatable content -- the canonical
    group/segment source_text used for subtitles, QA's original-English comparison,
    pace-ratio timing, and protected-literal extraction is untouched, so this never
    rewrites the record of what was actually said.
    """
    cleaned = str(text or "")
    for pattern in _CONFIRMED_DROPPABLE_FILLER_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[,\s]+", "", cleaned)
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].isalpha() and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _temporal_sentence_semantic_window(
    groups: Sequence[Mapping[str, Any]],
    *,
    mutable_start: int,
    mutable_count: int,
    accepted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the current sentence plus following chronological context, up to
    ``mutable_count`` sentences (the caller's estimate-driven window size for this
    specific call -- see TEMPORAL_MASK_WINDOW_BUDGET_FRACTION -- not a fixed constant).

    Tail windows near the end of the transcript contain only however many sentences
    remain. Earlier accepted Zulu is supplied separately as read-only continuity
    context and is never inserted into the current forward window.
    """
    end = min(len(groups), int(mutable_start) + int(mutable_count))
    start = int(mutable_start)
    accepted_by_id = {str(item.get("group_id")): item for item in accepted}
    blocks: list[dict[str, Any]] = []
    for pos in range(start, end):
        group = groups[pos]
        item = {
            "position": len(blocks),
            "group_id": str(group["group_id"]),
            "speaker_id": str(group["speaker_id"]),
            "english": _strip_confirmed_filler_phrases(str(group.get("source_text") or "")),
            "locked": False,
        }
        blocks.append(item)
    return {
        "window_size": int(mutable_count),
        "actual_block_count": len(blocks),
        "blocks": blocks,
        "hard_contract": (
            "The chronological current+following Zulu window must preserve the combined meaning and comprehension "
            "of the same English sentences. Earlier committed Zulu is continuity context only."
        ),
        "candidate_columns_form_joint_translations": True,
        "locked_blocks_immutable": True,
        "cross_speaker_fact_movement_allowed": False,
    }


def _normalize_temporal_mask_response(
    data: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
) -> dict[str, list[dict[str, Any]]]:
    expected = [str(group["group_id"]) for group in groups]
    expected_set = set(expected)
    returned = list(data.get("masks") or [])
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for item in returned:
        group_id = str(item.get("group_id") or "")
        if group_id not in expected_set:
            unknown.append(group_id or "<empty>")
            continue
        if group_id in by_id:
            duplicates.append(group_id)
            continue
        by_id[group_id] = item
    missing = [group_id for group_id in expected if group_id not in by_id]
    if missing or duplicates or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if duplicates:
            details.append("duplicate=" + ",".join(sorted(set(duplicates))))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("Temporal-mask response identity mismatch: " + "; ".join(details))

    result: dict[str, list[dict[str, Any]]] = {}
    canonical_ids = [chr(ord("a") + index) for index in range(int(candidate_count))]
    canonical_set = set(canonical_ids)
    for group_id in expected:
        candidates = list(by_id[group_id].get("candidates") or [])
        if not (1 <= len(candidates) <= int(candidate_count)):
            raise ValueError(
                f"Temporal mask {group_id} returned {len(candidates)} candidates; expected between "
                f"1 and {candidate_count} (the ladder may stop early once further compaction would "
                "repeat or damage a candidate, but must return at least 'a')"
            )

        raw_ids: list[str] = []
        normalized_candidates: list[dict[str, Any]] = []
        for position, candidate in enumerate(candidates):
            raw_id = str(candidate.get("candidate_id") or "").strip().lower()
            text = str(candidate.get("spoken_text") or "").strip()
            if not text:
                raise ValueError(f"Temporal mask {group_id} returned empty candidate text")
            if not raw_id:
                raw_id = canonical_ids[position]
            raw_ids.append(raw_id)
            raw_self_check = candidate.get("self_check")
            self_check_source = raw_self_check if isinstance(raw_self_check, Mapping) else {}
            self_check = {
                "back_translation": str(self_check_source.get("back_translation") or ""),
                "matches_source": bool(self_check_source.get("matches_source", True)),
                "discrepancy_summary": str(self_check_source.get("discrepancy_summary") or ""),
            }
            normalized_candidates.append({"candidate_id": raw_id, "spoken_text": text, "self_check": self_check})

        if len(set(raw_ids)) != len(raw_ids):
            duplicate_ids = sorted({candidate_id for candidate_id in raw_ids if raw_ids.count(candidate_id) > 1})
            raise ValueError(
                f"Temporal mask {group_id} returned duplicate candidate_id(s): {duplicate_ids}"
            )

        # A group may stop the ladder early -- expect exactly the contiguous canonical
        # prefix matching however many candidates this group actually returned (e.g. just
        # "a", or "a".."d"), not the full canonical set up to the ceiling.
        expected_ids = canonical_ids[: len(candidates)]
        if set(raw_ids) == set(expected_ids):
            by_candidate_id = {item["candidate_id"]: item for item in normalized_candidates}
            result[group_id] = [by_candidate_id[candidate_id] for candidate_id in expected_ids]
        elif not (set(raw_ids) & canonical_set):
            # Residual providers sometimes return fresh labels such as d/e/f even though
            # candidate IDs are only positional wire labels. When every returned label is
            # unique and unrecognized, canonicalize by response position rather than
            # discarding otherwise valid measured wordings.
            result[group_id] = [
                {
                    "candidate_id": expected_ids[position],
                    "spoken_text": item["spoken_text"],
                    "self_check": item["self_check"],
                }
                for position, item in enumerate(normalized_candidates)
            ]
        else:
            missing_candidate_ids = [candidate_id for candidate_id in expected_ids if candidate_id not in raw_ids]
            unknown_candidate_ids = [candidate_id for candidate_id in raw_ids if candidate_id not in canonical_set]
            raise ValueError(
                f"Temporal mask {group_id} candidate identity mismatch: "
                f"missing={missing_candidate_ids}, unknown={unknown_candidate_ids}"
            )
    return result


def _temporal_compact_wire(
    group: Mapping[str, Any],
    a_record: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """One Phase-B repair item: a REAL measured gap to close, not a blind instruction.

    Unlike the retired ladder's masks (which withheld Azure milliseconds from Grok entirely),
    Phase B's whole premise is handing over the real measured evidence so compaction targets an
    actual number instead of guessing when to stop. target_syllable_count goes one step further
    than required_speed_percent: percentage-based reasoning makes Grok do its own arithmetic
    conversion from "% faster needed" to "how many syllables to cut" WHILE also drafting content;
    a concrete pre-computed target removes that arithmetic burden entirely (the "budget allocation
    before writing" pattern that length-controlled-generation research found beats leaving the
    conversion to the model mid-draft). Syllables, not characters, since isiZulu speech duration
    tracks vowel-nuclei count far better than raw orthographic length (_estimate_zulu_syllables is
    the same deterministic proxy this pipeline already uses elsewhere for language planning) --
    honorifics are expanded to their spoken form first so an abbreviated "Mnu." doesn't undercount
    against its real 4-syllable "Mnumzane" pronunciation, the same fix applied earlier this session
    for the analogous syllable-ceiling warning. This is guidance for drafting, not a pass/fail
    contract -- Grok still declines further compaction rather than force an unnatural or unfaithful
    rewrite just to hit the number exactly.
    """
    current_source = str(group.get("source_text") or "")
    current_zulu = str(a_record.get("spoken_text") or "")
    required_speed_percent = float(a_record.get("required_speed_percent") or 0.0)
    current_syllable_count = _estimate_zulu_syllables(_native_dub_tts_ready_text(current_zulu))
    target_syllable_count = max(1, round(current_syllable_count / (1.0 + required_speed_percent / 100.0)))
    return {
        "group_id": str(group["group_id"]),
        "speaker_id": str(group["speaker_id"]),
        "english": current_source,
        "current_zulu": current_zulu,
        "measured_ms": int(a_record.get("measured_ms") or 0),
        "source_window_ms": int(geometry["source_window_ms"]),
        "required_speed_percent": required_speed_percent,
        "current_syllable_count": int(current_syllable_count),
        "target_syllable_count": int(target_syllable_count),
        "protected_literals": _protected_timing_literals(group),
        "high_risk_occurrence_requirements": _temporal_mask_source_occurrence_requirements(current_source),
    }


def _temporal_compact_schema(expected_group_ids: Sequence[str]) -> dict[str, Any]:
    """Structured-output contract for a batched Phase-B targeted compaction call.

    Unlike the retired ladder schema, there is no candidate-letter concept here -- each requested
    group_id returns exactly one compacted wording (or an honest decline), keyed by group_id alone.
    """
    requested_ids = [str(value) for value in expected_group_ids]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Temporal-compact schema received duplicate expected group IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": len(requested_ids),
                "maxItems": len(requested_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["group_id", "spoken_text", "self_check", "declined_further_compaction"],
                    "properties": {
                        "group_id": {"type": "string", "enum": requested_ids},
                        "spoken_text": {"type": "string", "minLength": 1},
                        "declined_further_compaction": {"type": "boolean"},
                        "self_check": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["back_translation", "matches_source", "discrepancy_summary"],
                            "properties": {
                                "back_translation": {"type": "string", "minLength": 1},
                                "matches_source": {"type": "boolean"},
                                "discrepancy_summary": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
    }


def _normalize_temporal_compact_response(
    data: Mapping[str, Any],
    expected_group_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected = [str(value) for value in expected_group_ids]
    expected_set = set(expected)
    returned = list(data.get("results") or [])
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for item in returned:
        group_id = str(item.get("group_id") or "")
        if group_id not in expected_set:
            unknown.append(group_id or "<empty>")
            continue
        if group_id in by_id:
            duplicates.append(group_id)
            continue
        by_id[group_id] = item
    missing = [group_id for group_id in expected if group_id not in by_id]
    if missing or duplicates or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if duplicates:
            details.append("duplicate=" + ",".join(sorted(set(duplicates))))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("Temporal-compact response identity mismatch: " + "; ".join(details))

    result: dict[str, dict[str, Any]] = {}
    for group_id in expected:
        item = by_id[group_id]
        text = str(item.get("spoken_text") or "").strip()
        if not text:
            raise ValueError(f"Temporal compact {group_id} returned empty spoken_text")
        raw_self_check = item.get("self_check")
        self_check_source = raw_self_check if isinstance(raw_self_check, Mapping) else {}
        result[group_id] = {
            "spoken_text": text,
            "declined_further_compaction": bool(item.get("declined_further_compaction")),
            "self_check": {
                "back_translation": str(self_check_source.get("back_translation") or ""),
                "matches_source": bool(self_check_source.get("matches_source", True)),
                "discrepancy_summary": str(self_check_source.get("discrepancy_summary") or ""),
            },
        }
    return result


def _temporal_candidate_group(group: Mapping[str, Any], spoken_text: str) -> dict[str, Any]:
    value = dict(group)
    value["parts"] = [{
        "type": "text",
        "segment_id": str(group["segment_ids"][0]),
        "text": str(spoken_text),
    }]
    return value


_TEMPORAL_NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:R|\$)?\d+(?:(?:,\d{3})+|(?:\.\d+))?(?:%|st|nd|rd|th)?(?!\w)"
)
_TEMPORAL_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z]{2,})*\b")
# English ordinal suffixes ("4th", "1st", "23rd") are an English-specific reading
# convention, not part of the underlying value -- a correct isiZulu ordinal date
# ("ngomhlaka-4") never reproduces "th" verbatim. Real bug found 2026-09-07 (job
# fb3d08b63fed4d90922b08f7e325b906): a genuinely correct, literal-preserving turn-
# block translation containing "wake up on the 4th" -> "...ngomhlaka-4..." was
# spuriously rejected because the checker demanded the literal substring "4th"
# survive in the Zulu output. Stripped ONLY for the count-matching literal (see
# "match_literal" below) -- "literal" itself (with the suffix) is kept unchanged
# for display/blocking-classification, since an ordinal date must stay high-
# stakes even though its bare digits alone would look like an exempt 1-9 count.
_ORDINAL_NUMBER_SUFFIX = re.compile(r"(?:st|nd|rd|th)$", re.IGNORECASE)


def _temporal_literal_context(text: str, start: int, end: int, *, radius: int = 88) -> str:
    """Return a compact source-grounded clause window around one literal occurrence."""
    value = str(text or "")
    left = max(0, int(start) - int(radius))
    right = min(len(value), int(end) + int(radius))
    snippet = value[left:right].strip()
    if left > 0:
        snippet = "…" + snippet
    if right < len(value):
        snippet = snippet + "…"
    return re.sub(r"\s+", " ", snippet)


def _temporal_mask_source_occurrence_requirements(source_text: str) -> list[dict[str, Any]]:
    """Describe high-risk source literals with exact counts and occurrence contexts.

    Repeated literals are not interchangeable bookkeeping: two occurrences of ``24`` may
    encode two distinct temporal claims. The model receives each source occurrence so it
    can preserve both claims instead of inserting a duplicate token in an arbitrary place.
    """
    source = str(source_text or "")
    requirements: list[dict[str, Any]] = []
    for kind, pattern in (("number", _TEMPORAL_NUMBER_PATTERN), ("acronym", _TEMPORAL_ACRONYM_PATTERN)):
        by_literal: dict[str, list[dict[str, Any]]] = {}
        for match in pattern.finditer(source):
            literal = str(match.group(0))
            bucket = by_literal.setdefault(literal, [])
            bucket.append({
                "occurrence": len(bucket) + 1,
                "source_context": _temporal_literal_context(source, match.start(), match.end()),
            })
        for literal, occurrences in by_literal.items():
            match_literal = _ORDINAL_NUMBER_SUFFIX.sub("", literal) if kind == "number" else literal
            requirements.append({
                "kind": kind,
                "literal": literal,
                "match_literal": match_literal,
                "required_count": len(occurrences),
                "source_occurrences": occurrences,
            })
    return requirements


def _is_high_stakes_numeric_literal(kind: str, literal: str) -> bool:
    """Numbers worth a hard stop if missing: everything except a bare 1-9 count.

    A specific reference/section number, a date, a year, a percentage, or a multi-digit
    figure is a precise identifier -- if the count doesn't match, something was likely
    genuinely lost or altered. A bare single digit 1-9 is exactly the range fluent
    isiZulu commonly renders as a natural number word ("owodwa"/"munye") rather than a
    digit -- confirmed in production: "27" (a section number) survived correctly as a
    digit in the same block where "1" (a plain count of "one source") did not, most
    likely because it was naturally worded rather than dropped. Acronyms already have
    their own separate, non-fatal warning path and are never high-stakes here.
    """
    if kind == "acronym":
        return False
    return not re.fullmatch(r"[1-9]", literal)


def _temporal_mask_semantic_guard_details(source_text: str, spoken_text: str) -> list[dict[str, Any]]:
    source_requirements = _temporal_mask_source_occurrence_requirements(source_text)
    target = str(spoken_text or "")
    target_numbers = Counter(match.group(0) for match in _TEMPORAL_NUMBER_PATTERN.finditer(target))
    target_acronyms = Counter(match.group(0) for match in _TEMPORAL_ACRONYM_PATTERN.finditer(target))
    details: list[dict[str, Any]] = []
    for requirement in source_requirements:
        match_literal = str(requirement.get("match_literal", requirement["literal"]))
        observed = int((target_numbers if requirement["kind"] == "number" else target_acronyms)[match_literal])
        required = int(requirement["required_count"])
        if observed >= required:
            continue
        details.append({
            **requirement,
            "observed_count": observed,
            "missing_count": required - observed,
            # The DISPLAY literal (with any ordinal suffix), never match_literal --
            # an ordinal date's bare digits alone would look like an exempt 1-9
            # count, but a date must stay high-stakes regardless.
            "blocking": _is_high_stakes_numeric_literal(str(requirement["kind"]), str(requirement["literal"])),
        })
    return details


def _temporal_mask_high_risk_guard(source_text: str, spoken_text: str) -> tuple[bool, list[str]]:
    """Deterministically protect numeric/acronym values *and occurrence counts*.

    A bare 1-9 count and any acronym are advisory only -- see
    ``_is_high_stakes_numeric_literal`` -- so they never appear in ``missing`` here.
    """
    details = _temporal_mask_semantic_guard_details(source_text, spoken_text)
    missing = [
        f"{item['kind']}:{item['literal']} x{int(item['missing_count'])}"
        for item in details
        if item.get("blocking", True)
    ]
    return not missing, missing


def _measure_temporal_candidate(
    *,
    tts: AzureTTSBackend,
    group: Mapping[str, Any],
    voice_info: Mapping[str, Any],
    candidate: Mapping[str, Any],
    geometry: Mapping[str, Any],
    output_root: Path,
    round_number: int,
    force: bool,
) -> dict[str, Any]:
    text = str(candidate["spoken_text"])
    digest = _hash_payload({
        "group_id": str(group["group_id"]),
        "voice": str(voice_info.get("selected_voice") or ""),
        "text": text,
        "round": int(round_number),
    })[:12]
    candidate_id = str(candidate["candidate_id"])
    output_path = Path(output_root) / str(group["group_id"]) / f"r{int(round_number)}_{candidate_id}_{digest}.wav"
    candidate_group = _temporal_candidate_group(group, text)
    result = _synthesize_group(
        tts=tts,
        group=candidate_group,
        voice_info=voice_info,
        output_path=output_path,
        preferred_ms=int(geometry["preferred_raw_ms"]),
        maximum_ms=max(int(geometry["preferred_raw_ms"]), int(geometry["maximum_usable_raw_ms"])),
        force=force,
    )
    measured_ms = int(result.duration_ms)
    source_for_guard = str(group.get("source_text") or _group_source_english(group))
    semantic_guard_details = _temporal_mask_semantic_guard_details(source_for_guard, text)
    semantic_guard_missing = [
        f"{item['kind']}:{item['literal']} x{int(item['missing_count'])}"
        for item in semantic_guard_details
        if item.get("blocking", True)
    ]
    semantic_guard_pass = not semantic_guard_missing
    direction = _timing_recast_direction(
        measured_ms,
        int(geometry["source_window_ms"]),
        max_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        mouth_close_early_tolerance_ms=int(geometry["mouth_close_early_tolerance_ms"]),
        mouth_close_late_tolerance_ms=int(geometry["mouth_close_late_tolerance_ms"]),
    )
    rank = _timing_candidate_rank(
        measured_ms,
        int(geometry["source_window_ms"]),
        int(geometry["preferred_raw_ms"]),
        max_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        mouth_close_early_tolerance_ms=int(geometry["mouth_close_early_tolerance_ms"]),
        mouth_close_late_tolerance_ms=int(geometry["mouth_close_late_tolerance_ms"]),
    )
    # Required-rush is measured against the sentence's own end PLUS whatever safe gap
    # already exists before the next sentence starts (mouth_close_late_tolerance_ms is
    # the same protected-gap-overflow allowance _timing_recast_direction already uses
    # internally to decide whether compression is even needed -- this just lets the
    # REPORTED/RANKED number agree with that, instead of pretending no gap exists).
    # "gap_aware" when a real usable gap exists, "sentence_end" when there is none
    # (mode 1 is simply what happens automatically when there is no gap to use).
    late_tolerance_ms = int(geometry["mouth_close_late_tolerance_ms"])
    speed_fit_target_ms = int(geometry["source_window_ms"]) + late_tolerance_ms
    return {
        "candidate_id": candidate_id,
        "spoken_text": text,
        "path": str(output_path),
        "measured_ms": measured_ms,
        "required_speed_percent": round(_temporal_required_rush_percent(measured_ms, speed_fit_target_ms), 3),
        "speed_fit_mode": _speed_fit_mode_label(late_tolerance_ms),
        "direction": direction,
        "fit": bool(direction is None),
        "semantic_guard_pass": bool(semantic_guard_pass),
        "semantic_guard_missing": list(semantic_guard_missing),
        "semantic_guard_requirements": list(semantic_guard_details),
        "rank": [int(value) for value in rank],
    }


def _measure_temporal_batch(
    *,
    tts: AzureTTSBackend,
    groups: Sequence[Mapping[str, Any]],
    candidates_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    geometries: Mapping[str, Mapping[str, Any]],
    voice_assignments: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    round_number: int,
    force: bool,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    tasks = []
    for group in groups:
        group_id = str(group["group_id"])
        voice_info = voice_assignments[str(group["speaker_id"])]
        for candidate in candidates_by_group[group_id]:
            tasks.append((group, voice_info, candidate, geometries[group_id]))
    measured: dict[str, list[dict[str, Any]]] = {str(group["group_id"]): [] for group in groups}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(tasks)))) as pool:
        futures = {
            pool.submit(
                _measure_temporal_candidate,
                tts=tts,
                group=group,
                voice_info=voice_info,
                candidate=candidate,
                geometry=geometry,
                output_root=output_root,
                round_number=round_number,
                force=force,
            ): str(group["group_id"])
            for group, voice_info, candidate, geometry in tasks
        }
        for future in as_completed(futures):
            group_id = futures[future]
            measured[group_id].append(future.result())
    # Restore model candidate order after concurrent synthesis.
    order = {
        group_id: {str(item["candidate_id"]): index for index, item in enumerate(candidates)}
        for group_id, candidates in candidates_by_group.items()
    }
    for group_id in measured:
        measured[group_id].sort(key=lambda item: order[group_id].get(str(item["candidate_id"]), 999))
    return measured


def _temporal_residual_candidate_wire(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact linguistic feedback; do not send Azure milliseconds to Grok."""
    direction = item.get("direction")
    semantic_pass = bool(item.get("semantic_guard_pass", True))
    missing = list(item.get("semantic_guard_missing") or [])
    if not semantic_pass:
        rejection = "semantic_guard"
    elif direction == "expand":
        rejection = "undersized"
    else:
        # Overlong is deliberately not a rejection; Mathula will speed it up.
        rejection = "none"
    return {
        "candidate_id": item["candidate_id"],
        "spoken_text": item["spoken_text"],
        "estimated_syllables": int(item.get("estimated_syllables") or 0),
        "undersized": bool(direction == "expand"),
        "overlong_allowed": bool(direction == "compress"),
        "semantic_guard_pass": semantic_pass,
        "semantic_guard_missing": missing,
        "semantic_guard_requirements": list(item.get("semantic_guard_requirements") or []),
        "rejection_reason": rejection,
    }


def _temporal_residual_best_rank(item: Mapping[str, Any]) -> tuple[int, ...]:
    """Prefer semantic completeness before acoustic closeness during repair search."""
    requirements = list(item.get("semantic_guard_requirements") or [])
    missing_count = sum(int(value.get("missing_count") or 0) for value in requirements)
    if not requirements and not bool(item.get("semantic_guard_pass", True)):
        # Backward-compatible fallback for older/in-memory candidate records.
        missing_count = max(1, len(list(item.get("semantic_guard_missing") or [])))
    return (missing_count, *(int(x) for x in item.get("rank") or (999999,)))


def _choose_temporal_candidate(measured: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not measured:
        return None

    # Hard semantic fidelity first. Timing is asymmetric: undersized speech is not
    # committable because Mathula never slows speech; overlong speech IS committable
    # at any required acceleration. +12% is only the quality-warning threshold.
    semantic = [dict(item) for item in measured if item.get("semantic_guard_pass", True)]
    if not semantic:
        return None

    first = semantic[0] if semantic and str(semantic[0].get("candidate_id")) == str(measured[0].get("candidate_id")) else None
    if first is not None and first.get("direction") is None:
        return first

    normal_fit = [item for item in semantic if item.get("direction") is None]
    if normal_fit:
        return min(normal_fit, key=lambda item: tuple(int(x) for x in item["rank"]))

    # A faithful overlong candidate is preferable to semantic damage or pipeline
    # termination. Pick the least-rushed candidate. Never accept a materially early
    # candidate here; that still goes through expansion repair.
    rush = [item for item in semantic if item.get("direction") == "compress"]
    if rush:
        return min(rush, key=lambda item: int(item.get("measured_ms") or 0))
    return None


def _temporal_candidate_pace_name(candidate_id: str) -> str:
    """Friendly label for a candidate letter at any ladder depth.

    "a"/"b"/"c" keep their established names; a ladder that goes deeper (a group whose
    content needed more than two rounds of compaction) gets "concise+N" for however many
    extra rounds beyond "c" it took, rather than inventing more fixed names.
    """
    named = {"a": "natural", "b": "compact", "c": "concise"}
    key = str(candidate_id).strip().lower()
    if key in named:
        return named[key]
    depth_beyond_c = ord(key) - ord("c") if len(key) == 1 and key.isalpha() else 0
    return f"concise+{max(1, depth_beyond_c)}"


def _choose_sentence_candidate(
    measured: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Choose the best candidate for ONE sentence, independent of its window siblings.

    Every candidate letter for a sentence is a fully faithful translation of the same
    content -- "a" is Phase A's natural translation, "b"/"b2"/... are Phase B's successive
    targeted-compaction attempts (see TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT) -- they differ
    only in how aggressively they compact wording, never in what they say.
    That is a real, deliberate difference from the old multi-sentence-block design this
    superseded, where "b" for one block was drafted AND selected jointly with "b" for
    its neighbors, and picking different letters across blocks risked real incoherence
    (blocks could share content across their shared boundary). A sentence's own
    candidates never disagree on content with a sibling sentence's choice, so there is
    no coherence reason left to force every sentence in a window to share one letter.

    Confirmed real production bug this fixes: the old joint-column selector scored a
    whole window by its single WORST sentence. Compacting a sentence that is already
    undersized makes it MORE undersized, not better (shortening an already-short
    rendering only widens the gap from its target) -- so any window containing even
    one naturally undersized sentence (measured ~27% of real sentences) always looked
    worse at "b"/"c" than at "a" for the WHOLE window, vetoing compaction for every
    genuinely oversized sibling sentence in that same window (measured ~59% of real
    sentences -- the actual majority needing a shorter option).

    The only hard disqualifier is the deterministic semantic guard, exactly as before.
    Among the candidates that pass it, pick the one with the best (lowest) rank tuple
    -- see _timing_candidate_rank, whose (category, magnitude, target_distance) tuple
    already prefers a normal fit over a rushed candidate over an undersized one,
    without ever treating "undersized" as unusable.
    """
    valid = [dict(item) for item in measured if bool(item.get("semantic_guard_pass", True))]
    if not valid:
        return None
    return min(valid, key=lambda item: tuple(int(value) for value in (item.get("rank") or (0, 0, 0))))

def _temporal_mask_groups_for_render(
    pass2: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    artifact_groups = list(pass2.get("temporal_mask_groups") or [])
    if not artifact_groups:
        raise ValueError("Temporal-mask Pass 2 has no temporal_mask_groups")
    segment_by_id = {str(item["segment_id"]): item for item in segments}
    expected = [str(item["segment_id"]) for item in segments]
    seen: list[str] = []
    phrase_groups: list[dict[str, Any]] = []
    translations: dict[str, str] = {segment_id: "" for segment_id in expected}
    for raw in artifact_groups:
        ids = [str(value) for value in raw.get("segment_ids") or []]
        if not ids or any(segment_id not in segment_by_id for segment_id in ids):
            raise ValueError(f"Temporal-mask group {raw.get('group_id')} has invalid segment IDs")
        seen.extend(ids)
        source_items = [segment_by_id[segment_id] for segment_id in ids]
        text = str(raw.get("spoken_text") or "").strip()
        if not text:
            raise ValueError(f"Temporal-mask group {raw.get('group_id')} has empty spoken_text")
        # Pass 2 produces one combined Zulu rendering per group, not a
        # per-original-segment breakdown, so there is no real "each segment's own
        # portion" to assign for a merged multi-segment group. Give every member
        # segment the same full group text rather than leaving segment_ids[1:]
        # empty: an empty translation silently excludes a real segment from any
        # downstream consumer that treats "no text" as "not a real block" --
        # confirmed in production, where a merged group's second segment (a real,
        # contentful segment) disappeared from TikTok hook evidence candidates
        # entirely, and the AI then cited it anyway from elsewhere in its context,
        # failing evidence-block validation against a pool that had silently
        # dropped it.
        for segment_id in ids:
            translations[segment_id] = text
        phrase_groups.append({
            "group_id": str(raw["group_id"]),
            "speaker_id": str(raw["speaker_id"]),
            "segment_ids": ids,
            "start_ms": int(raw["start_ms"]),
            "source_end_ms": int(raw["source_end_ms"]),
            "source_span_ms": int(raw["source_span_ms"]),
            "source_text": " ".join(str(segment_by_id[segment_id]["restored_text"]) for segment_id in ids),
            "discourse_unit_kind": raw.get("discourse_unit_kind"),
            "temporal_mask_group": True,
            "source_segments": [
                {"segment_id": segment_id, "source_text": str(segment_by_id[segment_id]["restored_text"])}
                for segment_id in ids
            ],
            "parts": [{"type": "text", "segment_id": ids[0], "text": text}],
        })
    if seen != expected:
        raise ValueError("Temporal-mask groups no longer reconstruct the restored source segment order")
    return phrase_groups, translations


def _run_native_pass2_temporal_mask(
    *,
    job: JobManifest,
    job_root: Path,
    provider: FoundryGrokProvider,
    tts: AzureTTSBackend,
    force: bool,
    progress: Callable[[str], None] | None,
    batch_size: int,
    candidate_count: int,
    residual_rounds: int,
    tts_workers: int,
    preferred_raw_speed_percent: int,
) -> dict[str, Any]:
    """Two-phase per-sentence temporal-mask translation: translate once, compact only what needs it.

    Phase A translates every sentence exactly once (TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT, no
    ladder, no numeric length target) in forward chronological windows, then Azure TTS measures
    every sentence for real. Sentences that fit (or come out undersized) are committed immediately.
    Sentences that measure oversized are queued for Phase B (TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT),
    which is called ONLY for that subset, batched, handing Grok each sentence's real measured
    required_speed_percent as a concrete target instead of a blind linguistic guess. Phase B runs
    for up to candidate_count rounds (its ceiling repurposed as a round cap): each round only
    carries forward whatever is still oversized and has not declined further compaction, and every
    round's new attempt is compared back against the ORIGINAL Phase-A "a" via
    _choose_sentence_candidate, never chained against the previous round's attempt, so a bad round
    can never be worse than staying at "a". This replaces the old single-call natural-to-concise
    ladder: cheaper (Phase B never runs for sentences that already fit) and higher-quality (Phase B
    compacts against a real number, not a guess).
    """
    del residual_rounds
    paths = native_dub_paths(job_root)
    # Pass 2 does not yet know per-speaker voice assignments (those are resolved later
    # in native-dub) -- measurement only needs a REAL, consistent isiZulu voice to rank
    # candidates relative to each other, not the exact final persona, since native-dub's
    # own Phase A re-synthesizes the chosen wording with the correct voice regardless.
    measurement_voice_info = {"selected_voice": str(tts.default_voice), "base_prosody": {}}
    measurement_audio_root = paths.root / "pass2_candidate_measurements"
    if not paths.pass1.is_file():
        raise ValueError("Run native-translate pass 1 first; pass1_restored_transcript.json is missing")
    pass1 = read_json(paths.pass1)
    segments = list(pass1.get("segments") or [])
    if not segments:
        raise ValueError("Pass-1 artifact has no restored segments")
    context_ledger, cached_context_usage = _load_existing_context_ledger(
        job_root=job_root, pass1=pass1, segments=segments, progress=progress,
    )
    if context_ledger is None:
        context_ledger, context_usage = _ensure_context_ledger(
            job_root=job_root, provider=provider, pass1=pass1, segments=segments, progress=progress,
        )
    else:
        context_usage = dict(cached_context_usage)

    sentence_records = _build_pass1_sentence_records(segments, job_root=job_root)
    atomic_write_json(
        paths.pass1_sentences,
        {
            "schema_version": "mathula-native-pass1-sentences-v1",
            "source_pass1_sha256": checksum(paths.pass1) if paths.pass1.is_file() else None,
            "sentence_count": len(sentence_records),
            "records": sentence_records,
            "completed_at": utcnow(),
        },
    )
    full_source_transcript = " ".join(
        str(item.get("restored_text") or item.get("source_text") or "").strip()
        for item in segments
        if str(item.get("restored_text") or item.get("source_text") or "").strip()
    )

    groups = _build_temporal_mask_source_groups(sentence_records)
    if not groups:
        raise ValueError("No temporal-mask source groups were created")
    if not (1 <= int(candidate_count) <= 8):
        raise ValueError(
            "Temporal-mask mode's candidate cap must be between 1 and 8 -- it is a ceiling on how "
            "many Phase-B compaction rounds a still-oversized sentence may go through, not an "
            "exact count Grok must fill"
        )
    source_pace_ratios = _english_source_pace_ratios(groups)

    input_hash = _hash_payload({
        "pass1_sha256": checksum(paths.pass1),
        "context_ledger_sha256": checksum(paths.context_ledger) if paths.context_ledger.is_file() else None,
        "full_source_transcript_sha256": hashlib.sha256(full_source_transcript.encode("utf-8")).hexdigest(),
        "prompt_version": TEMPORAL_MASK_PROMPT_VERSION,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "window_budget_fraction": TEMPORAL_MASK_WINDOW_BUDGET_FRACTION,
        "window_output_calibration": _TEMPORAL_MASK_OUTPUT_CALIBRATION,
        "max_window_items": TEMPORAL_MASK_MAX_WINDOW_ITEMS,
        "max_compaction_rounds": int(candidate_count),
        "provider_self_review_cycles": 4,
        "azure_measurement_in_pass2": True,
        "python_temporal_residual_rounds": 0,
        "rush_warning_threshold_percent": DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "two_phase_translate_then_compact": True,
    })
    if paths.pass2.is_file() and not force:
        existing = read_json(paths.pass2)
        if existing.get("input_sha256") == input_hash and existing.get("temporal_mask_mode"):
            _emit_progress(progress, "[native pass 2] Reusing matching two-phase temporal-mask checkpoint")
            return existing

    geometries: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(groups):
        next_start = int(groups[index + 1]["start_ms"]) if index + 1 < len(groups) else int(group["source_end_ms"])
        geometries[str(group["group_id"])] = _temporal_mask_geometry(
            group,
            next_source_start_ms=next_start,
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
            max_natural_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
            mouth_tolerance_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
            max_protected_gap_overflow_ms=DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS,
            min_protected_pause_ms=DEFAULT_MIN_PROTECTED_PAUSE_MS,
        )

    usage_input = int(context_usage.get("input_tokens") or 0)
    usage_output = int(context_usage.get("output_tokens") or 0)
    usage_attempts = int(context_usage.get("attempts") or 0)
    usage_cached = 0
    model = provider.config.deployment
    output_budget = max(2048, int(getattr(provider.config, "max_output_tokens", 8192)))
    dynamic_output_budget = int(output_budget * TEMPORAL_MASK_WINDOW_BUDGET_FRACTION)
    provider_calls = 0
    phase_a_calls = 0
    phase_b_calls = 0

    resolved: dict[str, dict[str, Any]] = {}
    self_check_by_group: dict[str, dict[str, Any]] = {}
    revision_cycles_by_group: dict[str, Any] = {}
    window_contract_by_group: dict[str, dict[str, Any]] = {}

    _emit_progress(
        progress,
        f"[native pass 2] Phase A: translating {len(groups)} sentences; windows sized dynamically "
        f"from real estimated output tokens (up to {dynamic_output_budget:,} tokens/call, "
        f"{int(TEMPORAL_MASK_WINDOW_BUDGET_FRACTION * 100)}% of the {output_budget:,}-token ceiling, "
        f"capped at {TEMPORAL_MASK_MAX_WINDOW_ITEMS} sentences/call regardless of budget headroom) "
        "instead of a fixed sentence count; Grok self-QA up to 4 internal revision cycles; "
        "one faithful natural rendering per sentence, no speculative compaction ladder",
    )

    # ---- Phase A: translate every sentence once, natural only ----
    accepted_for_context: list[dict[str, Any]] = []

    def _send_phase_a_indices(indices: Sequence[int]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """Translate exactly groups[indices[0]:indices[-1]+1] in one provider call.

        Depends only on ``indices`` (a contiguous range), never on outer-loop state that
        assumes the original, un-split window size -- _retry_batch_call calls this
        recursively on progressively smaller contiguous sub-ranges of the same original
        window when the full window's response truncates.
        """
        batch_slice = [groups[i] for i in indices]
        group_ids = [str(group["group_id"]) for group in batch_slice]
        masks = [
            _temporal_mask_wire(
                groups,
                i,
                accepted_previous=accepted_for_context[-2:],
                geometry=geometries[str(groups[i]["group_id"])],
                candidate_count=1,
                source_pace_ratio=source_pace_ratios.get(str(groups[i]["group_id"]), 1.0),
            )
            for i in indices
        ]
        window_contract = _temporal_sentence_semantic_window(
            groups, mutable_start=indices[0], mutable_count=len(indices), accepted=accepted_for_context
        )
        response = provider.complete_json(
            operation="native_temporal_mask_translate_window",
            system_prompt=TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT,
            # Field order matters for real money: Azure Foundry's prompt cache is a
            # strict shared-PREFIX match (confirmed empirically against this exact
            # endpoint -- a byte-identical leading block gets served from cache even
            # when a later, different tail follows it, but any early divergence kills
            # caching for everything textually after that point). context_ledger and
            # full_source_transcript together are the bulk of this payload's tokens
            # and are IDENTICAL on every window call in this run; confirmed_translation_pitfalls
            # and window_contract are likewise constant per run. Putting all four
            # first -- before the genuinely per-window fields (continuity_context,
            # sentence_semantic_window, masks) -- lets that whole static block get
            # cached after the first call instead of re-billed in full on every one.
            payload={
                "context_ledger": context_ledger,
                "full_source_transcript": full_source_transcript,
                "confirmed_translation_pitfalls": list(CONFIRMED_TRANSLATION_PITFALLS),
                "window_contract": {
                    "current_plus_following_window": True,
                    # Window size varies call-to-call based on real estimated content length
                    # (see sentence_semantic_window.actual_block_count/blocks for THIS call's
                    # real count) -- there is no fixed sentence ceiling to state here, and this
                    # field must stay a constant boolean (not a per-call number) so it stays in
                    # the byte-identical, cacheable static prefix of this payload.
                    "window_size_is_dynamic_per_call": True,
                    "tail_windows_allowed": True,
                    "single_natural_translation_per_sentence": True,
                    "internal_self_review_cycles": 4,
                    "back_translation_qa_required": True,
                    "azure_measurement_deferred": False,
                    "rush_warning_threshold_percent": DEFAULT_MAX_NATURAL_SPEED_PERCENT,
                    "rush_threshold_is_non_fatal": True,
                    "further_targeted_compaction_may_follow_in_a_separate_pass": True,
                },
                "continuity_context": [
                    {
                        "group_id": item["group_id"],
                        "speaker_id": item["speaker_id"],
                        "english": item["source_text"],
                        "zulu": item["spoken_text"],
                    }
                    for item in accepted_for_context[-2:]
                ],
                "sentence_semantic_window": window_contract,
                "masks": masks,
            },
            schema=_temporal_mask_schema(1, expected_group_ids=group_ids, require_self_check=False),
            max_output_tokens=min(
                output_budget, max(1800, int(_estimate_temporal_translate_window_tokens(batch_slice) * 1.4))
            ),
        )
        candidates_by_group = _normalize_temporal_mask_response(response.data, batch_slice, candidate_count=1)
        internal_revision_cycles = (
            response.data.get("internal_revision_cycles") if isinstance(response.data, Mapping) else None
        )
        per_group = {
            group_id: {
                "candidate": candidates_by_group[group_id][0],
                "internal_revision_cycles": internal_revision_cycles,
                "window_contract": window_contract,
                "model": response.model,
            }
            for group_id in group_ids
        }
        usage = {
            "input_tokens": int(response.input_tokens),
            "output_tokens": int(response.output_tokens),
            "attempts": int(response.attempts),
            "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
        }
        return per_group, usage

    index = 0
    while index < len(groups):
        # Estimate-driven, not a fixed sentence count -- but bounded by
        # TEMPORAL_MASK_MAX_WINDOW_ITEMS regardless of how much budget headroom the
        # estimate thinks is available (see its definition: a real run once trusted the
        # estimate alone into a 154-of-155-sentence single call). _shrink_batch_to_budget
        # then grows the actual window off the calibrated real estimated token cost until
        # dynamic_output_budget is hit, the ceiling is hit, or the transcript runs out.
        batch = _shrink_batch_to_budget(
            groups[index:index + TEMPORAL_MASK_MAX_WINDOW_ITEMS],
            budget=dynamic_output_budget,
            base_overhead=80,
            estimate_fn=lambda group: int(
                _estimate_text_output_tokens(str(group.get("source_text") or "")) * _TEMPORAL_MASK_OUTPUT_CALIBRATION
            ),
        )
        indices = list(range(index, index + len(batch)))
        group_ids = [str(group["group_id"]) for group in batch]
        provider_calls += 1
        phase_a_calls += 1
        _emit_progress(
            progress,
            f"[native pass 2] Phase A window {index + 1}-{index + len(batch)}/{len(groups)}: "
            f"{batch[0]['group_id']}..{batch[-1]['group_id']} — one Grok call; internal back-translation QA",
        )
        per_group_results, usage = _retry_batch_call(
            _send_phase_a_indices, indices, operation="native_temporal_mask_translate_window", progress=progress,
        )
        usage_input += int(usage.get("input_tokens", 0))
        usage_output += int(usage.get("output_tokens", 0))
        usage_attempts += int(usage.get("attempts", 0))
        usage_cached += int(usage.get("cached_tokens", 0))
        candidates_by_group = {
            group_id: [per_group_results[group_id]["candidate"]] for group_id in group_ids
        }

        for group in batch:
            group_id = str(group["group_id"])
            result = per_group_results[group_id]
            model = result["model"]
            candidate = result["candidate"]
            text = str(candidate["spoken_text"]).strip()
            # Phase A no longer reports a self_check (see TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
            # and _temporal_mask_schema's require_self_check=False) -- the in-call self-report was
            # confirmed unreliable in production and the independent, Python-orchestrated
            # run_native_qa_back_translation audit is the real backstop. _normalize_temporal_mask_
            # response still fills a passing placeholder here so downstream record shape is unchanged.
            self_check_by_group[group_id] = candidate["self_check"]
            revision_cycles_by_group[group_id] = result["internal_revision_cycles"]
            window_contract_by_group[group_id] = result["window_contract"]
            acronym_warnings = [
                f"{item['kind']}:{item['literal']} x{int(item['missing_count'])}"
                for item in _temporal_mask_semantic_guard_details(str(group.get("source_text") or ""), text)
                if not item.get("blocking", True)
            ]
            if acronym_warnings:
                _emit_progress(
                    progress,
                    f"\x1b[31m[native pass 2] WARNING {group_id} Phase-A acronym surface not preserved "
                    f"exactly: {acronym_warnings}; continuing with natural Zulu wording.\x1b[0m",
                )

        voice_assignments = {str(group["speaker_id"]): measurement_voice_info for group in batch}
        measured_by_group = _measure_temporal_batch(
            tts=tts,
            groups=batch,
            candidates_by_group=candidates_by_group,
            geometries=geometries,
            voice_assignments=voice_assignments,
            output_root=measurement_audio_root,
            round_number=0,
            force=force,
            workers=tts_workers,
        )
        failed_groups: dict[str, list[str]] = {}
        for group in batch:
            group_id = str(group["group_id"])
            measured = measured_by_group.get(group_id, [])
            chosen = _choose_sentence_candidate(measured)
            if chosen is None:
                failed_groups[group_id] = [
                    f"{item['candidate_id']}:missing={item.get('semantic_guard_missing')}"
                    for item in measured
                    if not item.get("semantic_guard_pass", True)
                ]
                continue
            resolved[group_id] = chosen
            accepted_for_context.append({
                "group_id": group_id,
                "speaker_id": str(group["speaker_id"]),
                "source_text": str(group.get("source_text") or ""),
                "spoken_text": str(chosen["spoken_text"]),
            })
        if failed_groups:
            raise RuntimeError(
                f"Sentence temporal-mask window {batch[0]['group_id']}..{batch[-1]['group_id']}: "
                f"Phase-A translation failed the semantic guard for {len(failed_groups)} sentence(s); "
                f"failures={failed_groups}. Python will not make a residual repair call."
            )
        index += len(batch)

    oversized_group_ids = [
        str(group["group_id"]) for group in groups
        if resolved[str(group["group_id"])].get("direction") == "compress"
    ]
    _emit_progress(
        progress,
        f"[native pass 2] Phase A complete: {len(groups)} sentence(s) translated, "
        f"{len(oversized_group_ids)} measured oversized and queued for Phase B compaction",
    )

    # ---- Phase B: compact only the oversized subset, using the real measured gap ----
    groups_by_id = {str(group["group_id"]): group for group in groups}
    pending = list(oversized_group_ids)
    round_number = 1

    def _send_phase_b_chunk(chunk_group_ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """Compact exactly these group_ids in one provider call (round_number read at call
        time from the enclosing round loop, so a retry-halved sub-chunk still reports the
        correct round)."""
        chunk = list(chunk_group_ids)
        original_a = {group_id: resolved[group_id] for group_id in chunk}
        repairs = [
            _temporal_compact_wire(groups_by_id[group_id], original_a[group_id], geometry=geometries[group_id])
            for group_id in chunk
        ]
        response = provider.complete_json(
            operation="native_temporal_mask_compact_batch",
            system_prompt=TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT,
            payload={
                "context_ledger": context_ledger,
                "full_source_transcript": full_source_transcript,
                "confirmed_translation_pitfalls": list(CONFIRMED_TRANSLATION_PITFALLS),
                "compaction_contract": {
                    "round_number": int(round_number),
                    "internal_self_review_cycles": 4,
                    "back_translation_qa_required": True,
                    "may_decline_further_compaction": True,
                },
                "repairs": repairs,
            },
            schema=_temporal_compact_schema(chunk),
            max_output_tokens=min(
                output_budget, max(1200, int(_estimate_temporal_compact_chunk_tokens(repairs) * 1.4))
            ),
        )
        results_by_group = _normalize_temporal_compact_response(response.data, chunk)
        per_group = {
            group_id: {"result": results_by_group[group_id], "model": response.model} for group_id in chunk
        }
        usage = {
            "input_tokens": int(response.input_tokens),
            "output_tokens": int(response.output_tokens),
            "attempts": int(response.attempts),
            "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
        }
        return per_group, usage

    while pending and round_number <= int(candidate_count):
        _emit_progress(
            progress,
            f"[native pass 2] Phase B round {round_number}/{int(candidate_count)}: "
            f"compacting {len(pending)} still-oversized sentence(s)",
        )
        still_pending: list[str] = []
        chunk_start = 0
        while chunk_start < len(pending):
            # Every repair item is already independent, so bounding by
            # TEMPORAL_MASK_MAX_WINDOW_ITEMS here is a pure safety ceiling, not a
            # discourse-coherence concern -- same estimate-driven-but-bounded shape as
            # Phase A's window above.
            chunk = _shrink_batch_to_budget(
                pending[chunk_start:chunk_start + TEMPORAL_MASK_MAX_WINDOW_ITEMS],
                budget=dynamic_output_budget,
                base_overhead=60,
                estimate_fn=lambda group_id: int(_estimate_text_output_tokens(
                    str(resolved[group_id].get("spoken_text") or "")
                ) * _TEMPORAL_MASK_OUTPUT_CALIBRATION
                ),
            )
            original_a = {group_id: resolved[group_id] for group_id in chunk}
            provider_calls += 1
            phase_b_calls += 1
            per_group_results, usage = _retry_batch_call(
                _send_phase_b_chunk, chunk, operation="native_temporal_mask_compact_batch", progress=progress,
            )
            usage_input += int(usage.get("input_tokens", 0))
            usage_output += int(usage.get("output_tokens", 0))
            usage_attempts += int(usage.get("attempts", 0))
            usage_cached += int(usage.get("cached_tokens", 0))

            candidates_by_group = {}
            declined: dict[str, bool] = {}
            for group_id in chunk:
                result = per_group_results[group_id]["result"]
                model = per_group_results[group_id]["model"]
                declined[group_id] = bool(result.get("declined_further_compaction"))
                candidate_id = "b" if round_number == 1 else f"b{round_number}"
                candidates_by_group[group_id] = [{
                    "candidate_id": candidate_id,
                    "spoken_text": str(result["spoken_text"]).strip(),
                    "self_check": result["self_check"],
                }]

            voice_assignments = {
                str(groups_by_id[group_id]["speaker_id"]): measurement_voice_info for group_id in chunk
            }
            measured_by_group = _measure_temporal_batch(
                tts=tts,
                groups=[groups_by_id[group_id] for group_id in chunk],
                candidates_by_group=candidates_by_group,
                geometries=geometries,
                voice_assignments=voice_assignments,
                output_root=measurement_audio_root,
                round_number=round_number,
                force=force,
                workers=tts_workers,
            )
            for group_id in chunk:
                candidate_self_check = candidates_by_group[group_id][0]["self_check"]
                if not candidate_self_check["matches_source"]:
                    _emit_red_warning(
                        progress,
                        f"[native pass 2] WARNING {group_id} Phase-B round {round_number} compaction "
                        "failed its own back-translation self-check: "
                        f"{candidate_self_check['discrepancy_summary'] or 'no explanation given'}; "
                        "still measuring it for selection -- flag for translation review if chosen.",
                    )
                candidate_measured = measured_by_group.get(group_id, [])
                best = _choose_sentence_candidate([original_a[group_id], *candidate_measured])
                if best is None:
                    # Neither the original "a" nor this round's compaction passes the semantic
                    # guard -- "a" itself already passed the guard in Phase A, so this can only
                    # happen if the measured-candidate list is malformed; fall back to "a" rather
                    # than lose the sentence.
                    best = original_a[group_id]
                if str(best.get("candidate_id")) == candidates_by_group[group_id][0]["candidate_id"]:
                    self_check_by_group[group_id] = candidate_self_check
                resolved[group_id] = best
                still_oversized = best.get("direction") == "compress"
                if still_oversized and not declined[group_id] and round_number < int(candidate_count):
                    still_pending.append(group_id)
                _emit_progress(
                    progress,
                    f"[native pass 2] Phase B round {round_number} COMMIT {group_id}: "
                    f"candidate={best.get('candidate_id')}, measured={best.get('measured_ms')}ms, "
                    f"window={geometries[group_id]['source_window_ms']}ms; "
                    f"still_oversized={bool(still_oversized)}, declined={declined[group_id]}",
                )
            chunk_start += len(chunk)
        pending = still_pending
        round_number += 1

    if pending:
        _emit_progress(
            progress,
            f"[native pass 2] Phase B round cap reached with {len(pending)} sentence(s) still "
            "oversized; committing each at its best-measured candidate and letting native-dub rush the audio",
        )

    # ---- Assemble final accepted records in chronological order ----
    accepted: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["group_id"])
        geometry = geometries[group_id]
        chosen = resolved[group_id]
        chosen_candidate_id = str(chosen["candidate_id"])
        pace_name = _temporal_candidate_pace_name(chosen_candidate_id)
        self_check = self_check_by_group[group_id]
        semantic_guard_warnings = [
            f"{item['kind']}:{item['literal']} x{int(item['missing_count'])}"
            for item in chosen.get("semantic_guard_requirements") or []
            if not item.get("blocking", True)
        ]
        required_speed_percent = float(chosen.get("required_speed_percent") or 0.0)
        speed_fit_mode = chosen.get("speed_fit_mode") or "sentence_end"
        rush_warning = None
        if required_speed_percent > DEFAULT_MAX_NATURAL_SPEED_PERCENT:
            rush_warning = {
                "type": "measured_rush_warning",
                "measured_ms": int(chosen.get("measured_ms") or 0),
                "source_window_ms": int(geometry["source_window_ms"]),
                "estimated_required_rush_percent": required_speed_percent,
                "speed_fit_mode": speed_fit_mode,
                "fatal": False,
                "tts_policy": "rushed_speed",
            }
            _emit_red_warning(
                progress,
                f"[native pass 2] WARNING {group_id} candidate '{chosen_candidate_id}' ({pace_name}) "
                f"measured {chosen.get('measured_ms')}ms against a {geometry['source_window_ms']}ms window "
                f"(+{required_speed_percent:.1f}% rush, {speed_fit_mode}); keeping wording and continuing -- "
                "native-dub will rush the audio to fit.",
            )
        record = {
            "group_id": group_id,
            "speaker_id": str(group["speaker_id"]),
            "segment_ids": [str(value) for value in group["segment_ids"]],
            "start_ms": int(group["start_ms"]),
            "source_end_ms": int(group["source_end_ms"]),
            "source_span_ms": int(group["source_span_ms"]),
            "source_text": str(group.get("source_text") or ""),
            "spoken_text": str(chosen["spoken_text"]),
            "undersized": bool(chosen.get("direction") == "expand"),
            "semantic_guard_warnings": semantic_guard_warnings,
            "selected_candidate_id": chosen_candidate_id,
            "selected_candidate_pace": pace_name,
            "selected_candidate_path": chosen.get("path"),
            "speed_fit_mode": speed_fit_mode,
            "measured_ms": int(chosen.get("measured_ms") or 0),
            "temporal_mask": dict(geometry),
            "sentence_semantic_window": window_contract_by_group.get(group_id),
            "internal_revision_cycles": revision_cycles_by_group.get(group_id),
            "self_check": self_check,
            "azure_timing_deferred": False,
            "rush_warning": rush_warning,
        }
        accepted.append(record)
        _emit_progress(
            progress,
            f"[native pass 2] COMMIT {group_id}: candidate={chosen_candidate_id} ({pace_name}), "
            f"measured={record['measured_ms']}ms, window={geometry['source_window_ms']}ms; "
            f"rush_warning={bool(record.get('rush_warning'))}",
        )

    translations = []
    for group in accepted:
        for position, segment_id in enumerate(group["segment_ids"]):
            translations.append({
                "segment_id": str(segment_id),
                "spoken_text": str(group["spoken_text"]) if position == 0 else "",
                "temporal_mask_group_id": str(group["group_id"]),
            })

    artifact = {
        "schema_version": TEMPORAL_MASK_TRANSLATION_SCHEMA_VERSION,
        "prompt_version": TEMPORAL_MASK_PROMPT_VERSION,
        "provider": provider.provider,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "model": model,
        "input_sha256": input_hash,
        "source_pass1_path": str(paths.pass1),
        "source_pass1_sha256": checksum(paths.pass1),
        "context_ledger_path": str(paths.context_ledger),
        "temporal_mask_mode": True,
        "rolling_syllable_mode": False,
        "timing_optimized": False,
        "timing_optimization_mode": "two_phase_translate_then_targeted_compact_downstream_azure_fit",
        "candidate_count": int(candidate_count),
        "max_compaction_rounds": int(candidate_count),
        "window_size_is_dynamic_per_call": True,
        "window_budget_fraction": TEMPORAL_MASK_WINDOW_BUDGET_FRACTION,
        "requested_batch_size_ignored_for_window_contract": int(batch_size),
        "python_temporal_residual_rounds": 0,
        "provider_internal_self_review_cycles": 4,
        "tts_rate_percent": 0,
        "max_natural_speed_percent": DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "mouth_close_tolerance_ms": MOUTH_CLOSE_TARGET_TOLERANCE_MS,
        "context_contract": {
            "semantic_unit": "one_sentence_per_group_translated_once_then_targeted_compaction",
            "english_zulu_equivalence": "combined meaning and comprehension for every sentence",
            "speaker_attribution_preserved_per_block": True,
            "locked_zulu_immutable": True,
            "phase_b_compares_every_attempt_against_original_phase_a": True,
            "commit_requires_azure_measurement": False,
            "azure_measurement_deferred_to_native_dub": True,
            "overlong_speed_threshold_is_warning_only": True,
            "undersized_syllables_are_warning_only": True,
            "undersized_syllable_tts_policy": "normal_speed",
            "python_temporal_residual_calls": 0,
        },
        "token_strategy": {
            "english_blocks_sent_once_per_window": True,
            "previous_two_committed_zulu_sent_once_per_window": True,
            "per_mask_context_text_repetition": False,
            "phase_b_only_calls_for_sentences_measured_oversized": True,
            "provider_internal_backtranslation_qa": True,
            "provider_internal_self_review_cycles": 4,
            "python_residual_loop": False,
            "azure_milliseconds_sent_to_grok_in_phase_b_only": True,
        },
        "temporal_mask_groups": accepted,
        "translations": translations,
        "summary": {
            "mask_count": len(accepted),
            "source_segment_count": len(segments),
            "sentence_window_calls": int(provider_calls),
            "phase_a_calls": int(phase_a_calls),
            "phase_b_calls": int(phase_b_calls),
            "phase_b_sentence_count": len(oversized_group_ids),
            "python_temporal_residual_calls": 0,
            "rollback_mask_count": 0,
            "undersized_syllable_count": sum(1 for group in accepted if group.get("undersized")),
            "oversized_syllable_count": sum(1 for group in accepted if group.get("rush_warning")),
        },
        "usage": {
            "input_tokens": int(usage_input),
            "output_tokens": int(usage_output),
            "cached_input_tokens": int(usage_cached),
            "attempts": int(usage_attempts),
            "context_ledger_usage": context_usage,
        },
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.pass2, artifact)
    _emit_progress(
        progress,
        f"[native pass 2] Saved two-phase temporal-mask translation: {len(accepted)} committed sentence(s), "
        f"{phase_a_calls} Phase-A call(s), {phase_b_calls} Phase-B call(s) — {paths.pass2}",
    )
    return artifact


def _apply_referent_audit(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    pass2: dict[str, Any],
    force: bool,
    progress: Callable[[str], None] | None,
    batch_size: int,
    max_repair_rounds: int,
    max_workers: int = DEFAULT_REFERENT_AUDIT_WORKERS,
) -> dict[str, Any]:
    """Run the independent referent-fidelity audit/repair pass and persist it.

    The canonical Pass-2 translation is only mutated here, as the final step of
    Pass 2 itself and before it is ever consumed by timing repair or TTS -- not
    afterward. Once this returns, the persisted translation is what timing
    repair treats as immutable (see the Pass-2 contract at the top of this
    module); this function is what makes that canonical text correct in the
    first place.
    """

    paths = native_dub_paths(job_root)
    existing_summary = pass2.get("referent_audit_summary")
    if (
        not force
        and isinstance(existing_summary, Mapping)
        and existing_summary.get("input_sha256") == pass2.get("input_sha256")
        and existing_summary.get("schema_version") == REFERENT_AUDIT_SCHEMA_VERSION
    ):
        _emit_progress(progress, "[referent audit] Reusing matching completed checkpoint")
        return pass2
    if isinstance(existing_summary, Mapping) and existing_summary.get("schema_version") not in (
        None,
        REFERENT_AUDIT_SCHEMA_VERSION,
    ):
        _emit_progress(
            progress,
            "[referent audit] Existing checkpoint is from an older audit schema "
            f"({existing_summary.get('schema_version')!r}); re-auditing with "
            f"{REFERENT_AUDIT_SCHEMA_VERSION!r}",
        )

    registry: EntityRegistry | None
    try:
        registry = EntityRegistry()
    except (FileNotFoundError, ValueError) as exc:
        _emit_progress(
            progress,
            f"[referent audit] WARNING: entity registry unavailable ({exc}); "
            "continuing with the independent audit call only",
        )
        registry = None

    result = audit_and_repair_pass2(
        provider=provider,
        pass2=pass2,
        registry=registry,
        batch_size=batch_size,
        max_repair_rounds=max_repair_rounds,
        max_workers=max_workers,
        progress=progress,
    )
    pass2["referent_audit_summary"] = {
        "schema_version": result["schema_version"],
        "input_sha256": pass2.get("input_sha256"),
        "groups_checked": result["groups_checked"],
        "groups_flagged": len(result["flagged_group_ids"]),
        "repairs_applied": len(result["repairs"]),
        "unresolved_group_ids": result["unresolved_group_ids"],
        "usage": result["usage"],
    }
    atomic_write_json(
        paths.referent_audit,
        {
            "schema_version": result["schema_version"],
            "pass2_input_sha256": pass2.get("input_sha256"),
            "completed_at": utcnow(),
            **result,
        },
    )
    atomic_write_json(paths.pass2, pass2)
    return pass2


def run_native_pass2(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    audit_referents: bool = False,
    referent_audit_batch_size: int = DEFAULT_REFERENT_AUDIT_BATCH_SIZE,
    max_referent_repair_rounds: int = DEFAULT_MAX_REFERENT_REPAIR_ROUNDS,
    referent_audit_workers: int = DEFAULT_REFERENT_AUDIT_WORKERS,
    rolling_syllable: bool = False,
    rolling_syllable_batch_size: int = DEFAULT_ROLLING_SYLLABLE_BATCH_SIZE,
    temporal_mask: bool = False,
    temporal_mask_batch_size: int = DEFAULT_TEMPORAL_MASK_BATCH_SIZE,
    temporal_mask_candidates: int = DEFAULT_TEMPORAL_MASK_CANDIDATES,
    temporal_mask_residual_rounds: int = DEFAULT_TEMPORAL_MASK_RESIDUAL_ROUNDS,
    temporal_mask_tts_workers: int = DEFAULT_TEMPORAL_MASK_TTS_WORKERS,
    job: JobManifest | None = None,
    tts: AzureTTSBackend | None = None,
    preferred_raw_speed_percent: int = DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
    content_syllables_per_second: float = DEFAULT_ZULU_CONTENT_SYLLABLES_PER_SECOND,
    azure_utterance_overhead_ms: int = DEFAULT_AZURE_UTTERANCE_OVERHEAD_MS,
    syllable_tolerance_percent: int = DEFAULT_SYLLABLE_TOLERANCE_PERCENT,
) -> dict[str, Any]:
    if rolling_syllable and temporal_mask:
        raise ValueError("Choose either rolling-syllable or temporal-mask Pass 2, not both")
    if temporal_mask:
        if job is None or tts is None:
            raise ValueError("Temporal-mask Pass 2 requires the job manifest and Azure TTS backend")
        pass2 = _run_native_pass2_temporal_mask(
            job=job,
            job_root=job_root,
            provider=provider,
            tts=tts,
            force=force,
            progress=progress,
            batch_size=int(temporal_mask_batch_size),
            candidate_count=int(temporal_mask_candidates),
            residual_rounds=int(temporal_mask_residual_rounds),
            tts_workers=int(temporal_mask_tts_workers),
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
        )
    elif rolling_syllable:
        pass2 = _run_native_pass2_rolling_syllable(
            job_root=job_root,
            provider=provider,
            force=force,
            progress=progress,
            rolling_batch_size=rolling_syllable_batch_size,
            preferred_raw_speed_percent=preferred_raw_speed_percent,
            content_syllables_per_second=content_syllables_per_second,
            azure_utterance_overhead_ms=azure_utterance_overhead_ms,
            syllable_tolerance_percent=syllable_tolerance_percent,
        )
    else:
        pass2 = _run_native_pass2_standard(
            job_root=job_root, provider=provider, force=force, progress=progress
        )
    if audit_referents:
        pass2 = _apply_referent_audit(
            job_root=job_root,
            provider=provider,
            pass2=pass2,
            force=force,
            progress=progress,
            batch_size=int(referent_audit_batch_size),
            max_repair_rounds=int(max_referent_repair_rounds),
            max_workers=int(referent_audit_workers),
        )
    return pass2


def run_native_translation(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    audit_referents: bool = False,
    referent_audit_batch_size: int = DEFAULT_REFERENT_AUDIT_BATCH_SIZE,
    max_referent_repair_rounds: int = DEFAULT_MAX_REFERENT_REPAIR_ROUNDS,
    referent_audit_workers: int = DEFAULT_REFERENT_AUDIT_WORKERS,
    rolling_syllable: bool = False,
    rolling_syllable_batch_size: int = DEFAULT_ROLLING_SYLLABLE_BATCH_SIZE,
    temporal_mask: bool = False,
    temporal_mask_batch_size: int = DEFAULT_TEMPORAL_MASK_BATCH_SIZE,
    temporal_mask_candidates: int = DEFAULT_TEMPORAL_MASK_CANDIDATES,
    temporal_mask_residual_rounds: int = DEFAULT_TEMPORAL_MASK_RESIDUAL_ROUNDS,
    temporal_mask_tts_workers: int = DEFAULT_TEMPORAL_MASK_TTS_WORKERS,
    job: JobManifest | None = None,
    tts: AzureTTSBackend | None = None,
    content_syllables_per_second: float = DEFAULT_ZULU_CONTENT_SYLLABLES_PER_SECOND,
    azure_utterance_overhead_ms: int = DEFAULT_AZURE_UTTERANCE_OVERHEAD_MS,
    syllable_tolerance_percent: int = DEFAULT_SYLLABLE_TOLERANCE_PERCENT,
) -> dict[str, Any]:
    pass1 = run_native_pass1(job_root=job_root, provider=provider, force=force, progress=progress)
    pass2 = run_native_pass2(
        job_root=job_root,
        provider=provider,
        force=force,
        progress=progress,
        audit_referents=audit_referents,
        referent_audit_batch_size=referent_audit_batch_size,
        max_referent_repair_rounds=max_referent_repair_rounds,
        referent_audit_workers=referent_audit_workers,
        rolling_syllable=rolling_syllable,
        rolling_syllable_batch_size=rolling_syllable_batch_size,
        temporal_mask=temporal_mask,
        temporal_mask_batch_size=temporal_mask_batch_size,
        temporal_mask_candidates=temporal_mask_candidates,
        temporal_mask_residual_rounds=temporal_mask_residual_rounds,
        temporal_mask_tts_workers=temporal_mask_tts_workers,
        job=job,
        tts=tts,
        content_syllables_per_second=content_syllables_per_second,
        azure_utterance_overhead_ms=azure_utterance_overhead_ms,
        syllable_tolerance_percent=syllable_tolerance_percent,
    )
    return {
        "schema_version": NATIVE_DUB_SCHEMA_VERSION,
        "pass1_path": str(native_dub_paths(job_root).pass1),
        "pass2_path": str(native_dub_paths(job_root).pass2),
        "pass1_corrections": len(pass1["corrections"]),
        "pass2_segments": len(pass2["translations"]),
        "provider": provider.provider,
        "model": pass2["model"],
        "automatic_web_search": False,
        "rolling_syllable_mode": bool(pass2.get("rolling_syllable_mode")),
        "temporal_mask_mode": bool(pass2.get("temporal_mask_mode")),
        "syllable_summary": pass2.get("syllable_summary"),
        "temporal_mask_summary": pass2.get("summary"),
        "referent_audit_summary": pass2.get("referent_audit_summary"),
    }


def _timing_quality_flags(
    measured: Sequence[Mapping[str, Any]],
    *,
    min_overrun_ms: int,
    min_rush_percent: float,
) -> list[dict[str, Any]]:
    """Flag real synthesis units whose natural (pre-speed-fit) overrun is severe in
    BOTH absolute and relative terms -- see REVIEW_REQUIRED_MIN_OVERRUN_MS's comment
    for why neither signal alone is reliable. Non-fatal: this only ever adds review
    metadata, never blocks rendering.
    """
    flags: list[dict[str, Any]] = []
    for item in measured:
        overrun_ms = int(item.get("natural_overrun_ms") or 0)
        rush_percent = float(item.get("raw_speed_percent") or 0.0)
        if overrun_ms > int(min_overrun_ms) and rush_percent > float(min_rush_percent):
            flags.append({
                "group_id": str(item["group_id"]),
                "natural_overrun_ms": overrun_ms,
                "raw_speed_percent": round(rush_percent, 1),
            })
    return flags


QA_BACK_TRANSLATION_SYSTEM_PROMPT = r"""You are Mathula TV's independent QA back-translation reviewer.

You will be given pairs of (original English, committed isiZulu translation) for individual sentences
OR, when several original sentences were merged into one economical translated unit, for the WHOLE
merged unit's combined English and combined isiZulu -- both cases have ALREADY been translated and
approved for a dubbed broadcast. Your job is to catch translation drift that a human reviewer would
otherwise have to find by hand -- you are a SEPARATE verification pass, not the translator; do not try
to improve or re-translate anything. A merged unit's combined isiZulu is allowed to restate or reorder
content relative to the original sentence order -- judge only whether every fact/name/number/date/
negation/proposition present anywhere in the combined English still appears somewhere in the combined
isiZulu, not whether sentence order was preserved.

For EACH pair:
1. Translate the isiZulu back into English as literally and faithfully as you can. Do not smooth over
   or "fix" anything -- a literal back-translation is more useful for catching drift than a polished one.
2. Compare your back-translation against the ORIGINAL English sentence (not the isiZulu -- you are
   checking whether the ORIGINAL meaning survived, not whether the isiZulu reads naturally on its own).
3. Report matches_original=true only if the core meaning is intact: the same question is being asked
   (or the same statement is being made), the same person/agent is doing the action, negation is
   preserved (present vs. absent), facts/names/numbers/dates are unchanged, no proposition was added
   or dropped, and every legally/evidentially significant ACT keeps its EXACT precise implication (see
   the term-precision rule below). Minor rewording that preserves all of this is fine and should be
   matches_original=true.

   TERM-PRECISION RULE, confirmed by a real test failure: the translation step now has real license to
   retell naturally rather than transfer clause-by-clause, which means a fluent, natural-sounding
   back-translation can still be WRONG on the specific act being described. "Intercepted" (implies
   something was tampered with or blocked while in transit) is NOT the same claim as "seized" or
   "confiscated" (implies custody was taken) -- a real natural retelling made exactly this substitution.
   When the original describes a specific legal, evidentiary, or procedural act (intercepted/seized/
   arrested/detained/searched/subpoenaed/charged/convicted/etc.), check that your back-translation
   names the SAME specific act, not merely a topically-similar one -- a "close enough gist" is not
   sufficient here even though it would be for ordinary paraphrase elsewhere in this sentence.
4. If ANYTHING above changed, report matches_original=false and a discrepancy_summary that NAMES THE
   EXACT WORDS on each side that differ (for example: "original asks when the listener spoke;
   back-translation asks when the listener SAID they would speak -- a different question"). A vague
   or generic summary is not acceptable -- if you cannot point to a specific word or clause that
   changed the meaning, the meaning did not change: report matches_original=true instead of guessing.
5. Apply the drift-judgment criteria below -- they cover internal-repetition detection, the central
   test for deciding whether a vanished phrase is decoration or a proposition, and named categories
   that are NOT drift on their own (pro-drop grammar, discourse filler, greetings, lexical variance).
   A verdict driven by those criteria that comes out matches_original=false still needs the exact-words
   discrepancy_summary required by rule 4.

6. For every verdict, also classify it with a category and a severity (this applies even when
   matches_original=true -- use category="none" and severity="none" for a true match):
   - category: "accuracy" (a fact, name, number, or date changed, INCLUDING a legally/evidentially
     significant act swapped for a different specific act per the term-precision rule above -- e.g.
     "intercepted" rendered back as "seized" is category="accuracy", not "fluency" or "other", even
     though the sentence reads naturally), "negation" (polarity flipped --
     treat this as its own category, never folded into "accuracy", because a missed negation flip
     is the single most dangerous class of drift this audit exists to catch), "omission" (a
     proposition present in the English is entirely absent from the isiZulu), "addition" (the
     isiZulu asserts something the English never said), "attribution" (a different person/agent is
     doing or saying the thing), "structure" (internal repetition or another structural defect per
     the repetition rule above, independent of meaning), "fluency" (the isiZulu itself is broken or
     unnatural, not a fidelity issue), "other" (a real drift that doesn't fit the above).
   - severity: "critical" (a fact, name, number, negation, or agent is simply WRONG -- a listener
     would be told something false), "major" (a real, confirmed drift per the rules above that
     changes what's being claimed or asked, but isn't an outright factual error -- e.g. a materially
     weakened or strengthened claim), "minor" (a genuine but narrow drift a careful human reviewer
     would likely let stand without a re-take -- a small nuance shift that doesn't change the core
     claim, question, agent, or any fact). Reserve "minor" for cases you would not bother a human
     about; when in doubt between "minor" and "major", choose "major" -- a missed real issue is worse
     than an over-cautious escalation.

""" + QA_DRIFT_JUDGMENT_CRITERIA + r"""

Watch specifically for changed speaker attribution WITHIN a sentence, not just between sentences:
an English clause like "X, who [someone] told me was Y" nests a reporting verb ("told", "said",
"indicated") inside a relative clause, and isiZulu's relative concord can attach to either the
person doing the telling or the person being described. Confirmed real defect: "General
Lincoln...Mr. Brown Mogotsi, who HE (Lincoln) told me was close to the minister" was committed as
isiZulu whose relative clause back-translates to "Mogotsi, who TOLD ME he was close to the
minister" -- the teller and the person described swapped. When back-translating this pattern,
explicitly identify who the grammatical subject of the embedded reporting verb is in your
back-translation and confirm it is the same person as in the original English; a swap here is
category="attribution".

confirmed_translation_pitfalls lists other hand-confirmed real mistranslations from this same
production pipeline, each with the specific wording that caused it -- actively check whether the
committed isiZulu exhibits the pattern each entry describes, do not treat the list as background
trivia to skim past.

context_ledger (summary, entities, speaker_roles) is whole-programme background for judging
apparent attribution/reference changes correctly, not just spotting them. English speech
routinely uses a generic or colloquial "you" (e.g. "and then you email the minister") to mean the
SAME already-established person the sentence has been describing in third person, not a genuine
second-person address to someone else -- confirmed real case: a back-translation that rendered
such a "you" as "he" was flagged as a critical attribution change, when context_ledger showed only
ONE person (a single witness) plausibly performing that whole chain of actions, making the
"you"/"he" difference a register/colloquialism shift, not a changed agent. Before flagging
category="attribution", check whether context_ledger's entities/speaker_roles support treating the
two references as the SAME person; only flag attribution when the evidence actually points to two
DIFFERENT people, not merely because the literal words differ.

preceding_english, when present on a pair, is the immediately preceding sentence's own original
English -- real discourse context for a short or elliptical sentence whose full meaning depends on
it. English routinely drops a predicate that was just stated and expects the listener to recover it
(e.g. "I will concede that I could have worded it better, Commissioner." followed immediately by "I
could have.", which means "I could have [worded it better]", not a bare, contentless claim).
Confirmed real defect: a back-translation that surfaced the isiZulu's necessarily-explicit recovered
predicate (isiZulu cannot leave a "could have" bare the way English can -- the counterfactual
construction requires an actual verb) was flagged as category="addition", treating a correct,
context-grounded completion as fabricated content. Before flagging "addition" on a short sentence,
check whether preceding_english already states the content the isiZulu makes explicit; only flag
addition when the extra content is NOT recoverable from preceding_english (i.e. it is genuinely
invented, not merely spelled out).

Return exactly one verdict per requested ref, in any order. Never omit, duplicate, or invent a ref.
Every verdict, including matches_original=true, needs a discrepancy_summary: for a true match, a
short phrase is enough (for example "faithful, no meaningful difference"); for a false match, it
must name the specific change per rule 4 above. Never leave it blank. Every verdict also needs a
category and a severity per rule 6 above.
"""


def _qa_back_translation_schema(refs: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": len(refs),
                "maxItems": len(refs),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "ref", "back_translation", "matches_original", "discrepancy_summary",
                        "category", "severity",
                    ],
                    "properties": {
                        "ref": {"type": "string", "enum": list(refs)},
                        "back_translation": {"type": "string", "minLength": 1},
                        "matches_original": {"type": "boolean"},
                        "discrepancy_summary": {"type": "string", "minLength": 1},
                        "category": {"type": "string", "enum": list(_QA_CATEGORY_ENUM)},
                        "severity": {"type": "string", "enum": list(_QA_SEVERITY_ENUM)},
                    },
                },
            },
        },
    }


def _qa_sentence_is_negation_sensitive(english_text: str) -> bool:
    """True when the English source carries a negation cue worth extra scrutiny.

    isiZulu marks negation through verb-prefix infixes shared with potential/conditional
    mood -- a genuine source of ambiguity that produced a real, only partially-resolved
    audit flag in production (see _QA_NEGATION_CUE_PATTERN). Used to floor a flagged
    sentence's severity so a negation-bearing drift can never settle at "minor".
    """
    return bool(_QA_NEGATION_CUE_PATTERN.search(english_text or ""))


def _qa_severity_at_least(current: str, floor: str) -> str:
    if _QA_SEVERITY_RANK.get(current, 0) >= _QA_SEVERITY_RANK.get(floor, 0):
        return current
    return floor


def _qa_severity_max(a: str, b: str) -> str:
    return a if _QA_SEVERITY_RANK.get(a, -1) >= _QA_SEVERITY_RANK.get(b, -1) else b


def _qa_sentence_pairs_for_group(group: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return (ref, original_english, committed_isizulu) for one committed group.

    One entry per sentence when the English/Zulu sentence counts agree (the common
    case), so QA feedback points at exactly which sentence drifted, not just which
    group. Falls back to one whole-group entry when counts disagree -- mirrors
    build_island_phrase_groups's own fallback, and is deliberately conservative:
    QA reviews whatever text actually exists rather than guessing a pairing.
    """
    group_id = str(group.get("group_id") or "")
    source_text = str(group.get("source_text") or "")
    spoken_text = str(group.get("spoken_text") or "")
    english_sentences = split_into_sentences(source_text, ENGLISH_TITLE_ABBREVIATIONS)
    zulu_sentences = split_into_sentences(spoken_text, ZULU_TITLE_ABBREVIATIONS)
    if len(english_sentences) > 1 and len(english_sentences) == len(zulu_sentences):
        return [
            (f"{group_id}__s{index:02d}", en, zu)
            for index, (en, zu) in enumerate(zip(english_sentences, zulu_sentences))
        ]
    return [(group_id, source_text, spoken_text)]


def _request_qa_back_translation_batch(
    *,
    provider: FoundryGrokProvider,
    items: Sequence[tuple[str, str, str]],
    context_ledger: Mapping[str, Any] | None = None,
    preceding_english_by_group_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    refs = [ref for ref, _, _ in items]
    preceding_by_ref = preceding_english_by_group_id or {}
    response = provider.complete_json(
        operation="native_qa_back_translation_batch",
        system_prompt=QA_BACK_TRANSLATION_SYSTEM_PROMPT,
        # confirmed_translation_pitfalls first: a module-level constant, byte-identical
        # across every batch/job/run, so ordering it before the per-batch pairs keeps
        # the longest possible cacheable prefix (same reasoning as the temporal-mask
        # window payload and the candidate-translate payload). context_ledger is
        # per-job constant (not per-batch), so it goes next, still before pairs.
        payload={
            "zulu_grammar_foundation": ZULU_GRAMMAR_FOUNDATION,
            "confirmed_translation_pitfalls": list(CONFIRMED_TRANSLATION_PITFALLS),
            "context_ledger": dict(context_ledger or {}),
            "pairs": [
                {
                    "ref": ref, "original_english": en, "committed_isizulu": zu,
                    **(
                        {"preceding_english": preceding_by_ref[_qa_ref_base_group_id(ref)]}
                        if _qa_ref_base_group_id(ref) in preceding_by_ref
                        else {}
                    ),
                }
                for ref, en, zu in items
            ],
        },
        schema=_qa_back_translation_schema(refs),
        # 260/item was tuned against grok-4.3's typical verbosity; confirmed
        # real 2026-08-30 that it truncates a dense 15-item batch's response
        # under gpt-5.6-sol-1 (finish_reason "length" at exactly 260*15=3900,
        # regardless of the provider's own configured max_output_tokens ceiling
        # -- this per-item estimate, not that ceiling, was the real limiter).
        max_output_tokens=min(
            max(1200, 450 * len(items)),
            int(getattr(provider.config, "max_output_tokens", 8192)),
        ),
    )
    returned = list(response.data.get("verdicts") or [])
    by_ref: dict[str, Mapping[str, Any]] = {}
    for item in returned:
        ref = str(item.get("ref") or "")
        if ref in refs and ref not in by_ref:
            by_ref[ref] = item
    # A missing/duplicate/unknown ref here is the same class of problem this
    # session's own batch-repair fix addressed: a provider response that doesn't
    # exactly match what was requested must never crash the whole QA run. A
    # sentence that didn't get a verdict is flagged for review instead of silently
    # dropped -- "we couldn't verify this one" is itself useful QA signal.
    result: dict[str, dict[str, Any]] = {}
    for ref in refs:
        item = by_ref.get(ref)
        if item is None:
            result[ref] = {
                "back_translation": "",
                "matches_original": False,
                "discrepancy_summary": "No verdict returned for this sentence in its batch.",
                # An unverified sentence must never be silently treated as low-stakes --
                # "we couldn't check this one" is escalated, not downgraded.
                "category": "other",
                "severity": "critical",
                "verdict_missing": True,
            }
        else:
            matches = bool(item.get("matches_original"))
            category = str(item.get("category") or "").strip().lower()
            if category not in _QA_CATEGORY_ENUM:
                category = "none" if matches else "other"
            severity = str(item.get("severity") or "").strip().lower()
            if severity not in _QA_SEVERITY_ENUM:
                # A provider response that omits severity (or an older/simulated caller
                # that doesn't know about it yet) defaults a real flag to "major" rather
                # than "minor" -- an unclassified issue is never auto-accepted as low-stakes.
                severity = "none" if matches else "major"
            result[ref] = {
                "back_translation": str(item.get("back_translation") or ""),
                "matches_original": matches,
                "discrepancy_summary": str(item.get("discrepancy_summary") or ""),
                "category": category,
                "severity": severity,
                "verdict_missing": False,
            }
    return result, {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }


QA_DIRECT_ADEQUACY_SYSTEM_PROMPT = r"""You are Mathula TV's independent QA direct-adequacy reviewer.

You will be given pairs of (original English, committed isiZulu translation) for individual sentences
OR, when several original sentences were merged into one economical translated unit, for the WHOLE
merged unit's combined English and combined isiZulu -- both cases have ALREADY been translated and
approved for a dubbed broadcast. A merged unit's combined isiZulu is allowed to restate or reorder
content relative to the original sentence order -- judge only whether every fact/name/number/date/
negation/proposition present anywhere in the combined English still appears somewhere in the combined
isiZulu, not whether sentence order was preserved. This is a SEPARATE verification method from
back-translation QA -- do NOT translate the isiZulu back into English and diff it. Instead,
read the isiZulu directly (as a fluent isiZulu/English bilingual would) and judge, in one step, whether
it conveys the same core meaning as the English: the same question or claim, the same agent, the same
negation, the same facts/names/numbers/dates, no proposition added or dropped.

This exists because back-translation is itself a translation step and can introduce or mask exactly the
kind of drift it's meant to catch -- a systematic blind spot in how a model reads a specific isiZulu
construction can produce a self-consistent round-trip that looks clean even when it isn't. Judging the
isiZulu directly against the English, without an intermediate re-translation, is a genuinely different
lens on the same question, not a repeat of the same check.

For EACH pair:
1. Read the isiZulu directly against the English source. Do not produce or rely on any back-translation.
2. Report matches_original=true only if the core meaning is intact per the criteria above, INCLUDING
   term precision: when the English names a specific legally/evidentially significant act (intercepted/
   seized/arrested/detained/searched/subpoenaed/charged/convicted/etc.), the isiZulu must convey that
   SAME specific act, not a topically-similar but different one -- confirmed real drift: a natural
   retelling of "intercepted" as isiZulu meaning "seized/held" reads fluently but asserts a different
   act. Minor rewording, register shifts, or the isiZulu's own natural phrasing that preserves all of
   this is fine.
3. If ANYTHING changed, report matches_original=false and an issue_summary that NAMES THE EXACT WORDS
   or clause on each side that differ. A vague or generic summary is not acceptable -- if you cannot
   point to a specific word or clause that changed the meaning, report matches_original=true instead.
4. Apply the same drift-judgment criteria used elsewhere in this pipeline's QA -- they cover internal-
   repetition detection, the central test for deciding whether a vanished phrase is decoration or a
   proposition, and named categories that are NOT drift on their own (pro-drop grammar, discourse
   filler, greetings, lexical variance).
5. Classify every verdict with a category and a severity, same definitions as this pipeline's
   back-translation QA: category is one of "accuracy", "negation", "omission", "addition",
   "attribution", "structure", "fluency", "other", or "none" for a true match; severity is "critical"
   (a fact/name/number/negation/agent is simply wrong), "major" (a real drift changing what's claimed
   or asked), "minor" (a narrow drift a careful human reviewer would likely let stand), or "none" for a
   true match. When in doubt between "minor" and "major", choose "major".

""" + QA_DRIFT_JUDGMENT_CRITERIA + r"""

Return exactly one verdict per requested ref, in any order. Never omit, duplicate, or invent a ref.
Every verdict needs an issue_summary: for a true match, a short phrase is enough (for example
"faithful, no meaningful difference"); for a false match, it must name the specific change. Never
leave it blank. Every verdict also needs a category and a severity.
"""


def _qa_direct_adequacy_schema(refs: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": len(refs),
                "maxItems": len(refs),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ref", "matches_original", "issue_summary", "category", "severity"],
                    "properties": {
                        "ref": {"type": "string", "enum": list(refs)},
                        "matches_original": {"type": "boolean"},
                        "issue_summary": {"type": "string", "minLength": 1},
                        "category": {"type": "string", "enum": list(_QA_CATEGORY_ENUM)},
                        "severity": {"type": "string", "enum": list(_QA_SEVERITY_ENUM)},
                    },
                },
            },
        },
    }


def _request_qa_direct_adequacy_batch(
    *, provider: FoundryGrokProvider, items: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Independent second opinion on the same pairs -- source vs. target directly, no
    back-translation step. See QA_DIRECT_ADEQUACY_SYSTEM_PROMPT for why this is a
    genuinely different check rather than a repeat of _request_qa_back_translation_batch.
    """
    refs = [ref for ref, _, _ in items]
    response = provider.complete_json(
        operation="native_qa_direct_adequacy_batch",
        system_prompt=QA_DIRECT_ADEQUACY_SYSTEM_PROMPT,
        payload={
            "pairs": [
                {"ref": ref, "original_english": en, "committed_isizulu": zu}
                for ref, en, zu in items
            ],
        },
        schema=_qa_direct_adequacy_schema(refs),
        max_output_tokens=min(
            max(1000, 200 * len(items)),
            int(getattr(provider.config, "max_output_tokens", 8192)),
        ),
    )
    returned = list(response.data.get("verdicts") or [])
    by_ref: dict[str, Mapping[str, Any]] = {}
    for item in returned:
        ref = str(item.get("ref") or "")
        if ref in refs and ref not in by_ref:
            by_ref[ref] = item
    result: dict[str, dict[str, Any]] = {}
    for ref in refs:
        item = by_ref.get(ref)
        if item is None:
            result[ref] = {
                "matches_original": False,
                "issue_summary": "No verdict returned for this sentence in its batch.",
                "category": "other",
                "severity": "critical",
                "verdict_missing": True,
            }
        else:
            matches = bool(item.get("matches_original"))
            category = str(item.get("category") or "").strip().lower()
            if category not in _QA_CATEGORY_ENUM:
                category = "none" if matches else "other"
            severity = str(item.get("severity") or "").strip().lower()
            if severity not in _QA_SEVERITY_ENUM:
                severity = "none" if matches else "major"
            result[ref] = {
                "matches_original": matches,
                "issue_summary": str(item.get("issue_summary") or ""),
                "category": category,
                "severity": severity,
                "verdict_missing": False,
            }
    return result, {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }


def _chunk_sequence(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[index:index + size]) for index in range(0, len(values), max(1, size))]


QA_BACK_TRANSLATION_REPAIR_SYSTEM_PROMPT = r"""You are repairing a specific, confirmed isiZulu back-translation drift.

For each group you receive the FULL English source sentence, the FULL current (flawed) isiZulu \
translation, and one or more specific issues: an independent back-translation of the current isiZulu \
that did not match the English, together with a discrepancy_summary naming exactly what changed \
(a name, number, negation, attribution, question/answer intent, or other proposition). Return a \
corrected isiZulu translation for the WHOLE group that fixes every named issue while preserving \
everything else -- register, phrasing, length -- as close to the current isiZulu as possible. Do \
not introduce new errors, do not omit any fact present in the English, and do not simply restate the \
English source as though translating from scratch -- fix only what the named issues actually flag."""


def _qa_repair_schema(expected_group_ids: Sequence[str]) -> dict[str, Any]:
    requested_ids = [str(value) for value in expected_group_ids]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("QA repair schema received duplicate expected group IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["repairs"],
        "properties": {
            "repairs": {
                "type": "array",
                "minItems": len(requested_ids),
                "maxItems": len(requested_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["group_id", "corrected_zulu"],
                    "properties": {
                        "group_id": {"type": "string", "enum": requested_ids},
                        "corrected_zulu": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def _request_qa_repair_batch(
    *, provider: FoundryGrokProvider, items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, int]]:
    """Send a batch of confirmed drift issues for repair; raise loudly on any identity mismatch.

    Unlike ``_request_qa_back_translation_batch`` (which fills a missing verdict with a
    placeholder rather than crashing an audit that already found real issues elsewhere),
    a repair response that doesn't exactly match what was requested indicates the provider
    ignored the structured-output contract -- the same class of problem this session's own
    referent-audit repair batch (``referent_audit.py:_request_repair_batch``) treats as fatal
    rather than silently dropping a correction.
    """
    expected_ids = [str(item["group_id"]) for item in items]
    response = provider.complete_json(
        operation="native_qa_repair_batch",
        system_prompt=QA_BACK_TRANSLATION_REPAIR_SYSTEM_PROMPT,
        payload={"repairs": list(items)},
        schema=_qa_repair_schema(expected_ids),
        # Same per-item headroom fix as the QA back-translation call above.
        max_output_tokens=min(
            max(1200, 450 * len(items)),
            int(getattr(provider.config, "max_output_tokens", 8192)),
        ),
    )
    returned = list(response.data.get("repairs") or [])
    expected_set = set(expected_ids)
    by_id: dict[str, str] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        group_id = str(item.get("group_id") or "")
        corrected = str(item.get("corrected_zulu") or "").strip()
        if group_id in by_id:
            duplicate_ids.append(group_id)
            continue
        if group_id not in expected_set:
            unknown_ids.append(group_id)
            continue
        if not corrected:
            continue
        by_id[group_id] = corrected

    missing_ids = [group_id for group_id in expected_ids if group_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Native QA repair batch identity mismatch: " + "; ".join(details))

    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return by_id, usage


def _apply_qa_group_correction(pass2: dict[str, Any], group: dict[str, Any], corrected_zulu: str) -> None:
    """Mutate pass2's canonical translation in place for one repaired group.

    ``group`` must be the SAME dict object stored in ``pass2["temporal_mask_groups"]`` (callers
    get this for free by reading groups directly off ``pass2`` rather than deep-copying them), so
    setting ``group["spoken_text"]`` here already updates the canonical structure -- mirrors
    ``referent_audit.py:_apply_group_correction``'s carrier/blank segment convention for keeping
    ``pass2["translations"]`` in sync with a multi-segment group's single spoken_text.
    """
    segment_ids = [str(value) for value in group.get("segment_ids") or []]
    translations_by_id = {str(item["segment_id"]): item for item in pass2.get("translations") or []}
    if segment_ids:
        carrier_id, *blank_ids = segment_ids
        if carrier_id in translations_by_id:
            translations_by_id[carrier_id]["spoken_text"] = corrected_zulu
        for blank_id in blank_ids:
            if blank_id in translations_by_id:
                translations_by_id[blank_id]["spoken_text"] = ""
    group["spoken_text"] = corrected_zulu


def _qa_ref_base_group_id(ref: str) -> str:
    return ref.split("__s", 1)[0]


def _run_qa_batch_verdicts(
    *,
    provider: FoundryGrokProvider,
    batch: Sequence[tuple[str, str, str]],
    direct_adequacy: bool,
    context_ledger: Mapping[str, Any] | None = None,
    preceding_english_by_group_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """One batch's back-translation (and, if enabled, direct-adequacy) verdicts.

    Called from worker threads by _run_qa_verdicts -- returns plain values only,
    no shared-state mutation, matching referent_audit.py's own convention that
    worker threads only make provider calls and return results.
    """
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    bt_verdicts, bt_usage = _request_qa_back_translation_batch(
        provider=provider, items=batch, context_ledger=context_ledger,
        preceding_english_by_group_id=preceding_english_by_group_id,
    )
    for key in usage:
        usage[key] += bt_usage.get(key, 0)
    da_verdicts: dict[str, dict[str, Any]] = {}
    if direct_adequacy:
        da_verdicts, da_usage = _request_qa_direct_adequacy_batch(provider=provider, items=batch)
        for key in usage:
            usage[key] += da_usage.get(key, 0)
    return bt_verdicts, da_verdicts, usage


def _run_qa_verdicts(
    *,
    provider: FoundryGrokProvider,
    target_groups: Sequence[Mapping[str, Any]],
    batch_size: int,
    direct_adequacy: bool,
    workers: int = DEFAULT_QA_AUDIT_WORKERS,
    progress: Callable[[str], None] | None = None,
    context_ledger: Mapping[str, Any] | None = None,
    preceding_english_by_group_id: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Back-translate (and optionally direct-adequacy-check) every sentence in
    `target_groups`, batched and run IN PARALLEL across batches.

    Extracted from what was previously run_native_qa_back_translation's internal
    _audit() closure -- a plain sequential for-loop, the last unparallelized
    fan-out in this file. QA batches are fully independent of each other (each
    is its own provider call over a distinct slice of sentences), so this uses
    the exact ThreadPoolExecutor pattern already proven in referent_audit.py's
    audit_and_repair_pass2: workers only make provider calls and return values;
    all merging into bt_verdicts/da_verdicts/usage_totals happens on the main
    thread inside the as_completed loop, so no locking is needed.

    ``preceding_english_by_group_id``: real preceding-sentence grounding,
    matching what natural-translation generation already receives
    (_translate_natural_via_grok) -- confirmed real need 2026-08-30: a short
    elliptical sentence's correctly-recovered content (e.g. "I could have."
    completing the immediately preceding sentence's own explicit "...worded it
    better") was flagged as fabricated "added content" by this QA check, which
    -- unlike generation -- had never seen the preceding sentence at all. The
    caller passes this keyed by the AUTHORITATIVE, full-job group order (not
    inferred from `target_groups`'s own order, which may be a subset such as a
    repair recheck) so a subset's lookups still resolve to the real neighbor.

    Returns (flagged_results, usage_totals); a pair with no drift is simply
    absent from the flagged list -- matching the original closure's behavior.
    """
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    pairs: list[tuple[str, str, str]] = []
    for group in target_groups:
        pairs.extend(_qa_sentence_pairs_for_group(group))
    if not pairs:
        return [], usage_totals

    pairs_by_ref = {ref: (en, zu) for ref, en, zu in pairs}
    batches = _chunk_sequence(pairs, batch_size)
    bt_verdicts: dict[str, dict[str, Any]] = {}
    da_verdicts: dict[str, dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(batches)))) as pool:
        futures = {
            pool.submit(
                _run_qa_batch_verdicts, provider=provider, batch=batch,
                direct_adequacy=direct_adequacy, context_ledger=context_ledger,
                preceding_english_by_group_id=preceding_english_by_group_id,
            ): len(batch)
            for batch in batches
        }
        for future in as_completed(futures):
            batch_bt, batch_da, usage = future.result()
            bt_verdicts.update(batch_bt)
            da_verdicts.update(batch_da)
            for key in usage_totals:
                usage_totals[key] += usage.get(key, 0)
            completed += 1
            _emit_progress(
                progress,
                f"[native qa] Batch {completed}/{len(batches)} complete ({futures[future]} sentence(s))",
            )

    results: list[dict[str, Any]] = []
    for ref, (en, zu) in pairs_by_ref.items():
        bt = bt_verdicts[ref]
        da = da_verdicts.get(ref)
        if bt["matches_original"] and (da is None or da["matches_original"]):
            continue
        checks_flagged: list[str] = []
        severity = "minor"
        category = "other"
        discrepancy_parts: list[str] = []
        if not bt["matches_original"]:
            checks_flagged.append("back_translation")
            severity = _qa_severity_max(severity, bt["severity"])
            if bt["category"] != "none":
                category = bt["category"]
            if bt["discrepancy_summary"]:
                discrepancy_parts.append(bt["discrepancy_summary"])
        direct_adequacy_issue = ""
        if da is not None and not da["matches_original"]:
            checks_flagged.append("direct_adequacy")
            severity = _qa_severity_max(severity, da["severity"])
            if category == "other" and da["category"] != "none":
                category = da["category"]
            direct_adequacy_issue = da["issue_summary"]
            if direct_adequacy_issue:
                discrepancy_parts.append(f"direct-adequacy: {direct_adequacy_issue}")
        negation_sensitive = _qa_sentence_is_negation_sensitive(en)
        if negation_sensitive:
            severity = _qa_severity_at_least(severity, "major")
        results.append({
            "ref": ref,
            "original_english": en,
            "committed_isizulu": zu,
            "matches_original": False,
            "back_translation": bt["back_translation"],
            "direct_adequacy_issue": direct_adequacy_issue,
            "discrepancy_summary": " | ".join(discrepancy_parts) or "no verdict returned",
            "category": category,
            "severity": severity,
            "negation_sensitive": negation_sensitive,
            "checks_flagged": checks_flagged,
            "verdict_missing": bool(bt.get("verdict_missing")) or bool(da and da.get("verdict_missing")),
        })
    return results, usage_totals


def run_native_qa_back_translation(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    force: bool = False,
    batch_size: int = DEFAULT_QA_BACK_TRANSLATION_BATCH_SIZE,
    repair: bool = True,
    max_repair_rounds: int = DEFAULT_QA_MAX_REPAIR_ROUNDS,
    direct_adequacy: bool = False,
    qa_audit_workers: int = DEFAULT_QA_AUDIT_WORKERS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Independently back-translate every committed sentence, flag drift, and repair it.

    This is a genuine, separate, Python-orchestrated verification pass -- not the
    same thing as the translation prompt's own internal self-QA loop (see
    TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT's internal_revision_cycles), which cannot be
    confirmed to have actually run or actually caught anything -- and, as of this
    version, is no longer even self-reported by Phase A (see
    _temporal_mask_schema's require_self_check). Confirmed gap in production: a
    translation whose internal revision cycles supposedly ran still shipped a
    back-translation that materially changed the question or reversed a negation.
    Runs on the committed Pass 2 text alone (no TTS, no audio) so it can catch
    semantic drift before any Azure synthesis cost is spent on it.

    Every flagged sentence carries an MQM-lite ``category``/``severity`` pair instead of
    a bare pass/fail (see QA_REPAIR_SEVERITY_FLOOR) -- only ``major``/``critical`` findings
    are auto-repaired; a ``minor`` finding is reported in ``accepted_minor`` rather than
    forcing a repair round, on the theory that a careful human reviewer would let it stand.
    A negation-bearing English source is floored at ``major`` regardless of what a check
    itself assigns, since isiZulu's negation/mood infixes make a missed negation flip the
    single most dangerous class of drift here (see _qa_sentence_is_negation_sensitive).

    When ``direct_adequacy`` is enabled, every pair is ALSO judged by a second, genuinely
    independent method: reading the isiZulu directly against the English with no
    back-translation step (see QA_DIRECT_ADEQUACY_SYSTEM_PROMPT). Back-translation is
    itself a translation step and can share the same model's blind spots in both
    directions, self-consistently hiding a real drift; a sentence is flagged if EITHER
    method finds a problem. This doubles the Grok calls this function makes, so it
    defaults to off at this layer -- the native-qa CLI command turns it on by default for
    real runs; existing callers that don't pass it keep the original single-method cost.

    When ``repair`` is enabled (the default), flagged sentences at or above the severity
    floor are batched back for a targeted correction -- mirroring referent_audit.py's
    audit_and_repair_pass2 shape -- and RE-VERIFIED with the same audit before being
    accepted, for up to ``max_repair_rounds`` rounds. Each round only carries forward
    groups still flagged after the previous round's correction and re-check;
    ``pass2.json`` is mutated in place (same convention as _apply_referent_audit) so a
    corrected translation becomes the new canonical Pass-2 text before any downstream
    TTS/render consumes it. Whatever is still flagged after the round cap is reported in
    ``flagged``/``unresolved_group_ids``, not silently dropped.

    The actual audit/direct-adequacy fan-out is ``_run_qa_verdicts`` (module level, not
    a closure here) -- its QA batches are fully independent of each other and run in
    parallel across ``qa_audit_workers``, the same ThreadPoolExecutor pattern already
    proven in referent_audit.py.
    """
    paths = native_dub_paths(job_root)
    if not paths.pass2.is_file():
        raise ValueError("Run native-translate before native-qa")
    pass2 = read_json(paths.pass2)
    if not pass2.get("temporal_mask_mode"):
        raise ValueError("native-qa currently supports temporal-mask translations only")
    groups = list(pass2.get("temporal_mask_groups") or [])
    if not groups:
        raise ValueError("Pass 2 has no temporal_mask_groups to review")
    groups_by_id = {str(group.get("group_id")): group for group in groups}

    input_hash = _hash_payload({
        "pass2_input_sha256": pass2.get("input_sha256"),
        "schema_version": QA_BACK_TRANSLATION_SCHEMA_VERSION,
        "repair_enabled": bool(repair),
        "max_repair_rounds": int(max_repair_rounds) if repair else 0,
        "direct_adequacy_enabled": bool(direct_adequacy),
    })
    if paths.qa_back_translation.is_file() and not force:
        existing = read_json(paths.qa_back_translation)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native qa] Reusing matching back-translation QA checkpoint")
            return existing

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    ordered_group_ids = [str(group.get("group_id")) for group in groups]
    preceding_english_by_group_id = {
        ordered_group_ids[i]: str(groups_by_id[ordered_group_ids[i - 1]].get("source_text") or "")
        for i in range(1, len(ordered_group_ids))
    }

    def _audit(target_groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Thin wrapper: run the parallel QA engine and fold its usage into
        this call's running total. The engine itself is module-level
        (_run_qa_verdicts), not a closure, so it can be reused/tested/called
        from elsewhere without going through this function."""
        results, usage = _run_qa_verdicts(
            provider=provider,
            target_groups=target_groups,
            batch_size=batch_size,
            direct_adequacy=direct_adequacy,
            workers=qa_audit_workers,
            progress=progress,
            preceding_english_by_group_id=preceding_english_by_group_id,
        )
        for key in usage_totals:
            usage_totals[key] += usage.get(key, 0)
        return results

    def _partition_by_repair_floor(
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split flagged sentences by group, not by sentence.

        Repair always rewrites a WHOLE group's Zulu text at once, so a group containing
        even one major/critical sentence must be repaired in full -- accepting only its
        minor sibling sentences as-is would leave them describing text a repair round is
        about to overwrite. A group is accepted without repair only when every one of
        its flagged sentences is "minor".
        """
        by_group: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_group.setdefault(_qa_ref_base_group_id(item["ref"]), []).append(dict(item))
        repair_eligible: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        for group_items in by_group.values():
            target = repair_eligible if any(
                item["severity"] in QA_REPAIR_SEVERITY_FLOOR for item in group_items
            ) else accepted
            target.extend(group_items)
        return repair_eligible, accepted

    total_sentence_count = sum(len(_qa_sentence_pairs_for_group(group)) for group in groups)
    _emit_progress(
        progress,
        f"[native qa] Back-translating {total_sentence_count} committed sentence(s) across {len(groups)} group(s)"
        + (" (with independent direct-adequacy cross-check)" if direct_adequacy else ""),
    )
    initial_flagged = _audit(groups)
    flagged, accepted_minor = _partition_by_repair_floor(initial_flagged)
    _emit_progress(
        progress,
        f"[native qa] Initial audit: {len(initial_flagged)}/{total_sentence_count} sentence(s) flagged"
        f" ({len(flagged)} major/critical, {len(accepted_minor)} minor accepted without repair)"
        if initial_flagged
        else f"[native qa] Initial audit: all {total_sentence_count} sentence(s) passed",
    )
    for item in accepted_minor:
        _emit_progress(
            progress,
            f"[native qa] {item['ref']}: minor drift accepted without repair -- {item['discrepancy_summary']}",
        )

    repairs_log: list[dict[str, Any]] = []
    if repair and flagged:
        unresolved_group_ids = sorted({_qa_ref_base_group_id(item["ref"]) for item in flagged})
        for repair_round in range(1, max(0, int(max_repair_rounds)) + 1):
            if not unresolved_group_ids:
                break
            _emit_progress(
                progress,
                f"[native qa] Repair round {repair_round}/{max_repair_rounds}: "
                f"{len(unresolved_group_ids)} group(s) still flagged",
            )
            flagged_by_group: dict[str, list[dict[str, Any]]] = {}
            for item in flagged:
                flagged_by_group.setdefault(_qa_ref_base_group_id(item["ref"]), []).append(item)

            repair_items = [
                {
                    "group_id": group_id,
                    "english": str(groups_by_id[group_id].get("source_text") or ""),
                    "current_zulu": str(groups_by_id[group_id].get("spoken_text") or ""),
                    "issues": [
                        {
                            "sentence_ref": item["ref"],
                            "original_english": item["original_english"],
                            "back_translation": item["back_translation"],
                            "discrepancy_summary": item["discrepancy_summary"],
                        }
                        for item in flagged_by_group[group_id]
                    ],
                }
                for group_id in unresolved_group_ids
            ]
            for batch in _chunk_sequence(repair_items, batch_size):
                corrected_by_id, usage = _request_qa_repair_batch(provider=provider, items=batch)
                for key in usage_totals:
                    usage_totals[key] += usage.get(key, 0)
                for item in batch:
                    group_id = item["group_id"]
                    group = groups_by_id[group_id]
                    previous_zulu = str(group.get("spoken_text") or "")
                    corrected = corrected_by_id[group_id]
                    _apply_qa_group_correction(pass2, group, corrected)
                    repairs_log.append({
                        "group_id": group_id,
                        "repair_round": repair_round,
                        "english": item["english"],
                        "previous_zulu": previous_zulu,
                        "corrected_zulu": corrected,
                    })

            recheck_flagged = _audit([groups_by_id[group_id] for group_id in unresolved_group_ids])
            flagged, newly_minor = _partition_by_repair_floor(recheck_flagged)
            accepted_minor.extend(newly_minor)
            for item in newly_minor:
                _emit_progress(
                    progress,
                    f"[native qa] {item['ref']}: minor drift accepted without repair -- "
                    f"{item['discrepancy_summary']}",
                )
            newly_resolved = len(unresolved_group_ids) - len({
                _qa_ref_base_group_id(item["ref"]) for item in flagged
            })
            _emit_progress(
                progress,
                f"[native qa] Repair round {repair_round} re-check: {newly_resolved} group(s) "
                f"resolved, {len({_qa_ref_base_group_id(item['ref']) for item in flagged})} still flagged",
            )
            unresolved_group_ids = sorted({_qa_ref_base_group_id(item["ref"]) for item in flagged})

        if unresolved_group_ids:
            _emit_progress(
                progress,
                f"[native qa] WARNING: {len(unresolved_group_ids)} group(s) could not be repaired "
                f"within {max_repair_rounds} round(s): {', '.join(unresolved_group_ids)}",
            )

    for item in flagged:
        checks = "+".join(item.get("checks_flagged") or ["back_translation"])
        _emit_red_warning(
            progress,
            f"[native qa] WARNING {item['ref']} [{item['severity']}/{item['category']}, via {checks}]: "
            f"{item.get('discrepancy_summary') or 'no verdict returned'}. "
            f"EN: {item['original_english']!r} ZU: {item['committed_isizulu']!r} "
            f"back-translation: {item.get('back_translation')!r}",
        )

    if repairs_log:
        atomic_write_json(paths.pass2, pass2)

    result = {
        "schema_version": QA_BACK_TRANSLATION_SCHEMA_VERSION,
        "input_sha256": input_hash,
        "pass2_input_sha256": pass2.get("input_sha256"),
        "model": provider.config.deployment,
        "provider": provider.provider,
        "direct_adequacy_enabled": bool(direct_adequacy),
        "sentence_count": total_sentence_count,
        "flagged_count": len(flagged),
        "requires_review": bool(flagged),
        "flagged": flagged,
        "accepted_minor_count": len(accepted_minor),
        "accepted_minor": accepted_minor,
        "repairs": repairs_log,
        "unresolved_group_ids": sorted({_qa_ref_base_group_id(item["ref"]) for item in flagged}),
        "usage": usage_totals,
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.qa_back_translation, result)
    _emit_progress(
        progress,
        f"[native qa] COMPLETE: {len(repairs_log)} repair(s) applied across "
        f"{len({item['group_id'] for item in repairs_log})} group(s); "
        f"{len(flagged)}/{total_sentence_count} sentence(s) still flagged for review"
        if repairs_log or flagged
        else f"[native qa] COMPLETE: all {total_sentence_count} sentence(s) passed back-translation QA",
    )
    return result


def render_native_dub(
    *,
    job: JobManifest,
    job_root: Path,
    tts: AzureTTSBackend,
    timing_repair_provider: FoundryGrokProvider | None = None,
    editorial_provider: FoundryGrokProvider | None = None,
    force: bool = False,
    apply_opening_cut: bool = True,
    render_hook_overlay: bool = True,
    voice_map_path: Path | None = None,
    max_join_gap_ms: int = 900,
    max_phrase_source_ms: int = 45_000,
    repair_threshold_ms: int = 0,
    min_same_speaker_rest_ms: int = 0,
    preferred_raw_speed_percent: int = DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
    max_natural_speed_percent: int = DEFAULT_MAX_NATURAL_SPEED_PERCENT,
    mouth_close_lag_tolerance_ms: int = MOUTH_CLOSE_TARGET_TOLERANCE_MS,
    protected_gap_overflow: bool = True,
    max_protected_gap_overflow_ms: int = DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS,
    min_protected_pause_ms: int = DEFAULT_MIN_PROTECTED_PAUSE_MS,
    enable_island_timing: bool = True,
    enable_sdk_group_synthesis: bool = False,
    enable_turn_group_synthesis: bool = False,
    sdk_boundary: AzureSpeechSDKBoundary | None = None,
    lite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Render an untouched-video dub from the one natural translation.

    ``lite=True`` caps ffmpeg's own encode threads at half the machine's
    logical CPU count and lowers the ffmpeg subprocess's OS scheduling
    priority, for running a render in the background without pegging the
    user's CPU (see the module-level comment above _lite_mode_cpu_threads
    for why this is an approximation, not a literal enforced ceiling).

    ``enable_turn_group_synthesis=True`` (Phase 14) synthesizes and speed-fits
    each multi-member speaker turn as ONE unit instead of fitting every
    sentence independently -- fixes a confirmed real defect where a turn's
    last sentence drifts audibly past the true mouth-close (see
    `_synthesize_speaker_turns_with_bookmarks`). Independent of
    `enable_sdk_group_synthesis` (a different, still-unvalidated mechanism for
    sub-sentence islands); requires `sdk_boundary`, same as that flag does.
    """
    paths = native_dub_paths(job_root)
    if not paths.pass2.is_file():
        raise ValueError("Run native-translate before native-dub")
    # Phase 17: cheap, disk-only, provider-free -- render_native_dub may run as
    # a separate CLI invocation/process from build_candidate_pool, so any
    # in-memory job-scoped pronunciation dictionary from that earlier process
    # is gone; reload it from the artifact build_candidate_pool already wrote.
    # Degrades silently (default job-agnostic dictionary) when the artifact
    # doesn't exist -- a job predating Phase 17, or research was skipped.
    if paths.pronunciation_research.is_file():
        _apply_native_pronunciation_research_to_job_cache(job_root, read_json(paths.pronunciation_research))
    pass1 = read_json(paths.pass1)
    pass2 = read_json(paths.pass2)
    if isinstance(pass1.get("context_ledger"), Mapping):
        context_ledger = dict(pass1["context_ledger"])
    elif paths.context_ledger.is_file():
        cached_context = read_json(paths.context_ledger)
        context_ledger = dict(cached_context.get("context_ledger") or {})
    else:
        context_ledger = _deterministic_context_ledger(list(pass1.get("segments") or []))
    segments = list(pass1["segments"])
    if pass2.get("temporal_mask_mode"):
        # Temporal-mask groups reference sentence-level segment_ids (e.g.
        # "seg-00001__s00"), not the original diarization segment_ids in
        # pass1["segments"] -- reload the sentence records Pass 2 derived and
        # persisted (see _build_pass1_sentence_records) so segment_by_id lookups
        # actually match what the committed groups point at.
        if not paths.pass1_sentences.is_file():
            raise ValueError(
                "Temporal-mask Pass 2 is missing pass1_sentences.json; re-run native-translate "
                "--temporal-mask --pass 2 to regenerate it"
            )
        segments = list(read_json(paths.pass1_sentences).get("records") or [])
        if not segments:
            raise ValueError("pass1_sentences.json has no sentence records")
        phrase_groups, translations = _temporal_mask_groups_for_render(pass2, segments)
        _emit_progress(
            progress,
            f"[native dub] Reusing {len(phrase_groups)} committed temporal-mask phrase group(s); "
            "renderer will not regroup their wording",
        )
    else:
        translations = {item["segment_id"]: item["spoken_text"] for item in pass2["translations"]}
        if {item["segment_id"] for item in segments} != set(translations):
            raise ValueError("Native translation no longer matches the restored transcript segment set")
        phrase_groups = None

    source_video = resolve_job_source_video(job, job_root)
    required_speakers = {item["speaker_id"] for item in segments}
    resolved_voice_path: Path | None = voice_map_path
    if voice_map_path is None:
        resolved_voice_path = ensure_native_voice_assignments(
            job=job,
            job_root=job_root,
            segments=segments,
            context_ledger=context_ledger,
            provider=(editorial_provider or timing_repair_provider),
            progress=progress,
        )
    voice_assignments = load_native_voice_assignments(
        job_root,
        required_speakers=required_speakers,
        explicit_path=resolved_voice_path,
    )
    if phrase_groups is None:
        phrase_groups = build_phrase_groups(
            segments,
            translations,
            max_join_gap_ms=max_join_gap_ms,
            max_phrase_source_ms=max_phrase_source_ms,
        )
    source_duration_ms = int(round(float(job.source_duration) * 1000))
    if source_duration_ms <= 0:
        source_duration_ms = max(item["end_ms"] for item in segments)

    if not 0 <= int(preferred_raw_speed_percent) <= 12:
        raise ValueError("Native preferred raw speed target must be between 0% and 12%")
    if not 0 <= int(max_natural_speed_percent) <= 15:
        raise ValueError("Native hard-sync speed bound must be between 0% and 15%")
    if int(preferred_raw_speed_percent) > int(max_natural_speed_percent):
        raise ValueError("Preferred raw speed target cannot exceed maximum natural speed")
    if not 0 <= int(mouth_close_lag_tolerance_ms) <= MOUTH_CLOSE_HARD_TOLERANCE_MS:
        raise ValueError(
            f"Native mouth-close lag tolerance must be between 0 and {MOUTH_CLOSE_HARD_TOLERANCE_MS} ms"
        )
    if not 0 <= int(max_protected_gap_overflow_ms) <= PROTECTED_GAP_OVERFLOW_HARD_MAX_MS:
        raise ValueError(
            f"Native protected gap overflow must be between 0 and {PROTECTED_GAP_OVERFLOW_HARD_MAX_MS} ms"
        )
    if not 0 <= int(min_protected_pause_ms) <= 2000:
        raise ValueError("Native protected pause must be between 0 and 2000 ms")
    if enable_sdk_group_synthesis and sdk_boundary is None:
        raise ValueError("enable_sdk_group_synthesis requires an sdk_boundary")
    if enable_turn_group_synthesis and sdk_boundary is None:
        raise ValueError("enable_turn_group_synthesis requires an sdk_boundary")

    dub_started = time.monotonic()
    phrase_count = len(phrase_groups)
    _emit_progress(
        progress,
        f"[native dub] Prepared {phrase_count} hard-sync phrase group(s); "
        f"raw TTS target +{int(preferred_raw_speed_percent)}%, "
        f"rush warning threshold +{int(max_natural_speed_percent)}% (non-fatal), "
        f"mouth-close lag tolerance {int(mouth_close_lag_tolerance_ms)} ms; "
        + (
            f"protected gap overflow <= {int(max_protected_gap_overflow_ms)} ms "
            f"with >= {int(min_protected_pause_ms)} ms pause"
            if protected_gap_overflow
            else "protected gap overflow disabled"
        ),
    )
    paths.audio_dir.mkdir(parents=True, exist_ok=True)

    turn_finalized: dict[str, dict[str, Any]] = {}
    if enable_turn_group_synthesis:
        turn_finalized = _synthesize_speaker_turns_with_bookmarks(
            phrase_groups=phrase_groups,
            source_duration_ms=source_duration_ms,
            voice_assignments=voice_assignments,
            tts=tts,
            sdk_boundary=sdk_boundary,
            audio_dir=paths.audio_dir,
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
            max_natural_speed_percent=int(max_natural_speed_percent),
            mouth_close_lag_tolerance_ms=int(mouth_close_lag_tolerance_ms),
            protected_gap_overflow=bool(protected_gap_overflow),
            max_protected_gap_overflow_ms=int(max_protected_gap_overflow_ms),
            min_protected_pause_ms=int(min_protected_pause_ms),
            force=force,
            progress=progress,
            job_root=job_root,
        )

    island_counts: dict[str, int] = {str(group["group_id"]): 1 for group in phrase_groups}
    timing_input_groups = phrase_groups
    presynthesized: dict[str, AzureTTSResult] = {}
    if enable_island_timing and pass2.get("temporal_mask_mode"):
        word_index = _load_raw_word_index_for_job(job_root)
        # Turn-finalized whole-groups must never be split into sub-sentence
        # islands -- their bookmark boundary is already decided per-sentence at
        # the turn level, keyed by the whole-sentence group_id.
        groups_for_island_split = [
            group for group in phrase_groups if str(group["group_id"]) not in turn_finalized
        ]
        split_groups, island_counts = _split_phrase_groups_with_islands(
            groups_for_island_split, word_index, progress=progress,
        )
        turn_finalized_whole_groups = [
            group for group in phrase_groups if str(group["group_id"]) in turn_finalized
        ]
        for group in turn_finalized_whole_groups:
            island_counts[str(group["group_id"])] = 1
        # Re-sort by start_ms is load-bearing: Phase A's next_source_start_ms
        # reads phrase_groups[index + 1]["start_ms"], so this list must stay in
        # true chronological order or every neighboring group's late-tolerance
        # computation silently corrupts.
        timing_input_groups = sorted(
            split_groups + turn_finalized_whole_groups, key=lambda g: int(g["start_ms"]),
        )
        if enable_sdk_group_synthesis:
            presynthesized = _presynthesize_islands_with_bookmarks(
                timing_input_groups=timing_input_groups,
                island_counts=island_counts,
                voice_assignments=voice_assignments,
                tts=tts,
                sdk_boundary=sdk_boundary,
                audio_dir=paths.audio_dir,
                force=force,
                progress=progress,
                job_root=job_root,
            )

    measured, repair_records = _run_batch_adaptive_timing_controller(
        phrase_groups=timing_input_groups,
        source_duration_ms=source_duration_ms,
        voice_assignments=voice_assignments,
        tts=tts,
        context_ledger=context_ledger,
        audio_dir=paths.audio_dir,
        preferred_raw_speed_percent=int(preferred_raw_speed_percent),
        max_natural_speed_percent=int(max_natural_speed_percent),
        mouth_close_lag_tolerance_ms=int(mouth_close_lag_tolerance_ms),
        protected_gap_overflow=bool(protected_gap_overflow),
        max_protected_gap_overflow_ms=int(max_protected_gap_overflow_ms),
        min_protected_pause_ms=int(min_protected_pause_ms),
        force=force,
        progress=progress,
        presynthesized=presynthesized,
        turn_finalized=turn_finalized,
        job_root=job_root,
    )

    # Computed on the pre-restitch, per-real-synthesis-unit `measured` list (one entry
    # per island where a group was split, one entry per whole group otherwise) so a
    # severe overrun buried inside an otherwise-fine multi-island group is never
    # diluted or hidden once islands are stitched back into one group-level record.
    timing_quality_flags = _timing_quality_flags(
        measured,
        min_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS,
        min_rush_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT,
    )
    if timing_quality_flags:
        worst = max(timing_quality_flags, key=lambda item: item["natural_overrun_ms"])
        _emit_red_warning(
            progress,
            f"[native dub] WARNING: {len(timing_quality_flags)} unit(s) exceeded the timing-quality review "
            f"threshold ({REVIEW_REQUIRED_MIN_OVERRUN_MS} ms and {REVIEW_REQUIRED_MIN_RUSH_PERCENT:.0f}% rush); "
            f"worst is {worst['group_id']} at {worst['natural_overrun_ms']} ms / {worst['raw_speed_percent']:.1f}%. "
            "This job will be marked needs_timing_review.",
        )

    if any(count > 1 for count in island_counts.values()):
        measured, island_records = _restitch_island_measurements(
            measured, island_counts,
            audio_dir=paths.audio_dir,
            max_natural_speed_percent=int(max_natural_speed_percent),
            progress=progress,
        )
        atomic_write_json(paths.speech_islands, {
            "schema_version": SPEECH_ISLANDS_SCHEMA_VERSION,
            "pass2_input_sha256": pass2.get("input_sha256"),
            "completed_at": utcnow(),
            "groups": island_records,
        })

    scheduled = _schedule_source_timeline(
        measured,
        min_same_speaker_rest_ms=min_same_speaker_rest_ms,
    )
    maximum_scheduled_end_ms = max(
        (int(item["scheduled_end_ms"]) for item in scheduled),
        default=0,
    )
    if maximum_scheduled_end_ms > source_duration_ms:
        _emit_red_warning(
            progress,
            "[native dub] WARNING: hard-synced speech extends beyond the untouched source video by "
            f"{maximum_scheduled_end_ms - source_duration_ms} ms after speed fitting; timing is non-fatal. "
            "The untouched video duration remains authoritative.",
        )

    clips = [
        {
            "unit_id": item["group_id"],
            "speaker_id": item["speaker_id"],
            "path": item["path"],
            "start_ms": item["scheduled_start_ms"],
            "duration_ms": item["tts_duration_ms"],
        }
        for item in scheduled
    ]
    render_duration_ms = source_duration_ms
    _emit_progress(progress, "[native dub] Mixing natural dialogue on the untouched source timeline...")
    dialogue, mix_wall_seconds = _run_with_progress_heartbeat(
        progress=progress,
        label="[native dub] dialogue mix",
        operation=lambda: render_dialogue_tracks(
            clips,
            paths.root / "audio" / "speaker_tracks",
            paths.dialogue_bus,
            total_duration_ms=render_duration_ms,
            sample_rate=tts.sample_rate,
        ),
    )
    _emit_progress(progress, f"[native dub] Dialogue mix complete in {mix_wall_seconds:.1f}s")
    _emit_progress(progress, "[native dub] Rendering dubbed master without changing video timing...")
    master_started = time.monotonic()
    if progress is None:
        render_video(source_video, paths.dialogue_bus, paths.video)
    else:
        _render_native_master_with_ffmpeg_progress(
            source_video=source_video,
            dialogue_bus=paths.dialogue_bus,
            output_video=paths.video,
            progress=progress,
            lite=lite,
        )
    master_wall_seconds = time.monotonic() - master_started
    _emit_progress(progress, f"[native dub] Dubbed master render complete in {master_wall_seconds:.1f}s")

    publication: dict[str, Any] | None = None
    final_video = paths.video
    if render_hook_overlay:
        if editorial_provider is None:
            raise ValueError(
                "Native dub hook overlay requires a Foundry Grok editorial provider"
            )
        _emit_progress(progress, "[native dub] Rendering existing production hook/footer + SEO package...")
        publication = render_native_hook_publication(
            job=job,
            job_root=job_root,
            master_video=paths.video,
            segments=segments,
            translations=translations,
            provider=editorial_provider,
            force=force,
            apply_opening_cut=apply_opening_cut,
            progress=progress,
            lite=lite,
        )
        final_video = Path(str(publication["output"]["path"]))

    timing_plan = {
        "schema_version": TIMING_PLAN_SCHEMA_VERSION,
        "source_video": str(source_video),
        "source_video_sha256": checksum(source_video),
        "source_video_untouched": True,
        "video_retimed": False,
        "video_speed_changed": False,
        "tts_rate_percent": 0,
        "timing_policy": "target_overflow_then_adaptive_speed_with_protected_source_gap",
        "source_gap_borrowing_allowed": bool(protected_gap_overflow),
        "source_gap_borrowing_policy": "bounded_existing_gap_preserve_pause_no_shift" if protected_gap_overflow else "disabled",
        "max_protected_gap_overflow_ms": int(max_protected_gap_overflow_ms),
        "min_protected_pause_ms": int(min_protected_pause_ms),
        "later_block_shift_allowed": False,
        "slowing_allowed": False,
        "pass2_soft_duration_target_enabled": True,
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "post_tts_speed_method": "ffmpeg_atempo_pitch_preserving",
        "max_natural_speed_percent": int(max_natural_speed_percent),
        "mouth_close_target_tolerance_ms": int(mouth_close_lag_tolerance_ms),
        "mouth_close_tolerance_policy": "configured_early_plus_bounded_protected_gap_late",
        "mouth_close_hard_tolerance_ms": MOUTH_CLOSE_HARD_TOLERANCE_MS,
        "canonical_translation_sha256": checksum(paths.pass2),
        "phrase_group_count": len(scheduled),
        "groups": scheduled,
        "timing_repairs": repair_records,
        "timing_repair_strategy": "none_undersized_speech_accepted_at_natural_speed",
        "timing_quality_review_required": bool(timing_quality_flags),
        "timing_quality_flags": timing_quality_flags,
        "dialogue": dialogue,
        "native_master_video": str(paths.video),
        "native_master_video_sha256": checksum(paths.video),
        "hook_overlay_enabled": bool(render_hook_overlay),
        "hook_edit_enabled": bool(render_hook_overlay and apply_opening_cut),
        "hook_publication": publication,
        "output_video": str(final_video),
        "output_video_sha256": checksum(final_video),
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.timing_repairs, {
        "schema_version": TIMING_REPAIR_SCHEMA_VERSION,
        "canonical_translation_sha256": checksum(paths.pass2),
        "repairs": repair_records,
        "completed_at": utcnow(),
    })
    atomic_write_json(paths.timing_plan, timing_plan)
    _emit_progress(
        progress,
        f"[native dub] COMPLETE in {time.monotonic() - dub_started:.1f}s — output: {final_video}",
    )
    return timing_plan


def render_native_hook_publication(
    *,
    job: JobManifest,
    job_root: Path,
    master_video: Path,
    segments: Sequence[Mapping[str, Any]],
    translations: Mapping[str, str],
    provider: FoundryGrokProvider,
    force: bool,
    apply_opening_cut: bool,
    progress: Callable[[str], None] | None = None,
    lite: bool = False,
) -> dict[str, Any]:
    """Render the existing production footer hook over the native dub master.

    This reuses Mathula's production TikTok hook selector and title-panel
    renderer.  It never retimes the picture. ``apply_opening_cut`` has the
    same semantics as production ``dub-azure --hook-edit``: disabling it keeps
    the full video but DOES NOT disable the persistent footer hook card.
    """
    from .tiktok_editor import (
        caption_without_hashtags,
        load_publication_title_authority,
        render_tiktok_hook_edit,
        select_tiktok_hook,
    )

    paths = native_dub_paths(job_root)
    blocks: list[dict[str, Any]] = []
    english_segments: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        translated = str(translations[segment_id]).strip()
        restored = str(
            segment.get("restored_text")
            or segment.get("source_text")
            or ""
        ).strip()
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        blocks.append(
            {
                "block_id": segment_id,
                "speaker_id": str(segment.get("speaker_id") or ""),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "source_text": restored,
                "translated_text": translated,
                "tts_text": translated,
            }
        )
        english_segments.append(
            {
                "segment_id": segment_id,
                "speaker_id": str(segment.get("speaker_id") or ""),
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "source_text": restored,
            }
        )

    analysis_root = Path(job_root) / "analysis"
    classification_path = analysis_root / "domain_classification.json"
    context_path = analysis_root / "context.json"
    classification = read_json(classification_path) if classification_path.is_file() else {}
    context = read_json(context_path) if context_path.is_file() else {}
    # Reuse the cached whole-program discourse ledger so Grok editorial calls do
    # not need duplicate full English + isiZulu transcript payloads.
    if paths.context_ledger.is_file():
        ledger_artifact = read_json(paths.context_ledger)
        ledger_value = ledger_artifact.get("context_ledger")
        if isinstance(ledger_value, Mapping):
            context = dict(context)
            context["native_context_ledger"] = dict(ledger_value)
    elif paths.pass1.is_file():
        pass1_value = read_json(paths.pass1)
        ledger_value = pass1_value.get("context_ledger")
        if isinstance(ledger_value, Mapping):
            context = dict(context)
            context["native_context_ledger"] = dict(ledger_value)
    english_transcript = {
        "schema_version": "mathula-native-publication-source-v1",
        "language": "en-ZA",
        "segments": english_segments,
    }
    source_duration_seconds = float(job.source_duration or 0.0)
    if source_duration_seconds <= 0:
        source_duration_seconds = max(
            (int(item["end_ms"]) for item in segments), default=0
        ) / 1000.0

    checkpoint_reuse_expected = (not force and paths.editorial_response_checkpoint.is_file())
    _emit_progress(
        progress,
        "[native dub] Editorial package: reusing saved Grok checkpoint..."
        if checkpoint_reuse_expected
        else "[native dub] Editorial package: requesting compact Grok selection/SEO...",
    )
    selection = select_tiktok_hook(
        provider=provider,
        job_id=job.job_id,
        target_language=job.target_language,
        blocks=blocks,
        seo={},
        source_duration_seconds=source_duration_seconds,
        english_transcript=english_transcript,
        classification=classification,
        context=context,
        publication_title_authority=load_publication_title_authority(Path(job_root)),
        response_checkpoint_path=(
            None if force else paths.editorial_response_checkpoint
        ),
    )
    _emit_progress(progress, "[native dub] Editorial package ready; writing publication metadata...")
    atomic_write_json(
        paths.publication_selection,
        {
            "schema_version": "mathula-native-hook-selection-v1",
            "provider": provider.provider,
            "model": provider.config.deployment,
            "automatic_web_search": False,
            "token_strategy": {
                "compact_grok_editorial_prompt": True,
                "deduplicated_transcript_payload": True,
                "cached_context_ledger": True,
                "full_json_schema_sent_to_model": False,
            },
            "selection": selection,
            "completed_at": utcnow(),
        },
    )

    # select_tiktok_hook is the production single editorial call: it generates
    # the footer title, opening anchor AND the searchable TikTok SEO package.
    # Persist the SEO separately inside native_dub so this experimental path
    # never overwrites translation/tiktok_*.json from the legacy production
    # workflow. Web research remains manual-only; the editorial provider has
    # no server tools enabled.
    editorial_seo = selection.get("editorial_seo")
    if not isinstance(editorial_seo, Mapping):
        raise ValueError("Native editorial selection returned no SEO package")
    seo_package = dict(editorial_seo)
    seo_package["native_dub"] = {
        "schema_version": "mathula-native-seo-link-v1",
        "automatic_web_search": False,
        "approved_translation_immutable": True,
        "source_selection": str(paths.publication_selection),
    }
    atomic_write_json(paths.seo_package, seo_package)

    caption = str(
        seo_package.get("caption") or seo_package.get("tiktok_caption") or ""
    ).strip()
    hashtags = [
        str(value).strip()
        for value in (
            seo_package.get("topic_hashtags")
            or seo_package.get("tiktok_hashtags")
            or seo_package.get("hashtags")
            or []
        )
        if str(value).strip()
    ]
    search_keywords = [
        str(value).strip()
        for value in (seo_package.get("search_keywords") or [])
        if str(value).strip()
    ]
    paths.seo_caption.write_text(caption + ("\n" if caption else ""), encoding="utf-8")
    paths.seo_hashtags.write_text(" ".join(hashtags) + ("\n" if hashtags else ""), encoding="utf-8")
    paths.seo_search_keywords.write_text("\n".join(search_keywords) + ("\n" if search_keywords else ""), encoding="utf-8")

    title_text = caption_without_hashtags(str(selection.get("hook_text") or ""))
    if not title_text:
        raise ValueError("Native hook selection returned an empty footer hook")
    _emit_progress(
        progress,
        "[native dub] Rendering production footer/hook overlay"
        + (" with opening cut..." if apply_opening_cut else " on the full video..."),
    )
    publication_started = time.monotonic()
    result = render_tiktok_hook_edit(
        master_video=Path(master_video),
        output_path=paths.publication_video,
        manifest_path=paths.publication_manifest,
        hook_text_path=paths.hook_text,
        selection=selection,
        translation_path=paths.pass2,
        title_text=title_text,
        apply_opening_cut=apply_opening_cut,
        publication_audio=None,
        progress_callback=(
            (lambda event: _emit_native_ffmpeg_progress(
                progress, label="publication encode", event=event
            ))
            if progress is not None
            else None
        ),
        popen_factory=(_lite_mode_popen_factory() if lite else subprocess.Popen),
        runner=(
            functools.partial(subprocess.run, **_lite_mode_subprocess_kwargs())
            if lite else subprocess.run
        ),
        cpu_threads=(_lite_mode_cpu_threads() if lite else None),
    )
    publication_wall_seconds = time.monotonic() - publication_started
    _emit_progress(progress, f"[native dub] Hook/footer render complete in {publication_wall_seconds:.1f}s")
    result["native_publication"] = {
        "schema_version": "mathula-native-hook-publication-v1",
        "automatic_web_search": False,
        "video_retimed": False,
        "video_speed_changed": False,
        "tts_rate_percent": 0,
        "hook_overlay_preserved_from_production_renderer": True,
        "seo_preserved_from_production_editorial_call": True,
        "seo_package_path": str(paths.seo_package),
        "seo_caption_path": str(paths.seo_caption),
        "seo_hashtags_path": str(paths.seo_hashtags),
        "seo_search_keywords_path": str(paths.seo_search_keywords),
    }
    return result


def resolve_job_source_video(job: JobManifest, job_root: Path) -> Path:
    candidates: list[Path] = []
    if job.local_source_path:
        candidates.append(Path(job.local_source_path))
    input_dir = Path(job_root) / "input"
    if input_dir.is_dir():
        if job.source_filename:
            candidates.append(input_dir / Path(job.source_filename).name)
        candidates.extend(sorted(input_dir.glob("*.mp4")))
        candidates.extend(sorted(input_dir.glob("*.webm")))
        candidates.extend(sorted(input_dir.glob("*.mkv")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise FileNotFoundError(
        f"Could not resolve source video for job {job.job_id}; checked manifest local_source_path and {input_dir}"
    )


def _native_voice_samples(
    segments: Sequence[Mapping[str, Any]],
    *,
    only_speakers: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[str]] = {}
    for item in segments:
        speaker = str(item.get("speaker_id") or "").strip()
        if only_speakers is not None and speaker not in only_speakers:
            continue
        text = str(item.get("restored_text") or item.get("source_text") or "").strip()
        if speaker and text:
            by_speaker.setdefault(speaker, []).append(re.sub(r"\s+", " ", text))
    result: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        values = by_speaker[speaker]
        # Token-saving but representative: longest utterance plus first/last when distinct.
        indexes = [0, len(values) - 1, max(range(len(values)), key=lambda i: len(values[i]))]
        selected: list[str] = []
        for index in indexes:
            value = values[index][:420]
            if value and value not in selected:
                selected.append(value)
            if len(selected) >= 3:
                break
        result.append({"speaker_id": speaker, "samples": selected})
    return result


def _stable_pitch_personas(families: Mapping[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family in ("masculine", "feminine"):
        speakers = sorted(s for s, value in families.items() if value == family)
        count = len(speakers)
        if not count:
            continue
        if count == 1:
            offsets = [0]
        else:
            offsets = [round(-6 + (12 * i / (count - 1))) for i in range(count)]
        for speaker, offset in zip(speakers, offsets):
            result[speaker] = int(max(-6, min(6, offset)))
    return result


def _native_work_dir(job_root: Path) -> Path:
    root = Path(job_root)
    if root.parent.name.casefold() == "jobs":
        return root.parent.parent
    # Test/custom layouts may pass the job directory directly.
    return root.parent


def _native_cache_work_dirs(job_root: Path) -> list[Path]:
    """Return cache roots in priority order.

    The first entry is the active job's work dir.  The project-local ``working``
    directory is also searched so a Windows migration with a stale Linux
    MATHULA_TV_WORK_DIR can still reuse the caches copied into C:\\mathula-tv\\working.
    """
    candidates = [_native_work_dir(job_root), Path(__file__).resolve().parents[2] / "working"]
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).casefold()
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _normalise_identity_name(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())
    return " ".join(text.split())


def _iter_resolution_assignments(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    raw_assignments = value.get("assignments")
    if isinstance(raw_assignments, Mapping):
        for speaker, item in raw_assignments.items():
            if isinstance(item, Mapping) and item.get("selected_voice"):
                assignments[str(speaker)] = dict(item)
    speakers = value.get("speakers")
    if isinstance(speakers, list):
        for item in speakers:
            if not isinstance(item, Mapping):
                continue
            speaker = str(item.get("speaker_id") or "").strip()
            voice = str(item.get("selected_voice") or "").strip()
            status = str(item.get("status") or item.get("resolution_status") or "resolved")
            if speaker and voice and status not in {"unresolved", "failed"}:
                assignments[speaker] = dict(item)
    return assignments


def _voice_family_from_assignment(item: Mapping[str, Any]) -> str:
    family = str(item.get("voice_family") or "").strip().lower()
    if family in {"masculine", "feminine"}:
        return family
    voice = str(item.get("selected_voice") or "").casefold()
    if "thando" in voice:
        return "feminine"
    if "themba" in voice:
        return "masculine"
    return "unknown"


def _trusted_job_voice_cache(
    path: Path,
    *,
    required: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, Mapping):
        return {}
    schema = str(value.get("schema_version") or "")
    # Refresh the earlier native Grok-only cache once so the legacy classifier
    # and cross-job caches are actually consulted. Legacy direct-dub caches are
    # already evidence-rich and remain authoritative.
    if schema == "mathula-native-voice-resolution-v1":
        return {}
    assignments = _iter_resolution_assignments(value)
    selected: dict[str, dict[str, Any]] = {}
    for speaker in required:
        if speaker not in assignments:
            continue
        item = dict(assignments[speaker])
        if not str(item.get("selected_voice") or "").strip():
            continue
        base = dict(item.get("base_prosody") or {})
        base["legacy_cached_rate_percent_ignored"] = int(base.get("rate_percent") or 0)
        base["rate_percent"] = 0
        base["native_rate_policy"] = "hard_locked_zero_percent"
        item["base_prosody"] = base
        selected[speaker] = item
    return selected

def _historical_identity_voice_cache(work_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Index safe named mappings from earlier jobs across local cache roots.

    Diarization IDs are intentionally never reused across jobs. Only records
    carrying a stable identified name/person identity participate.
    """
    result: dict[str, dict[str, Any]] = {}
    rank_by_name: dict[str, int] = {}
    source_rank = {
        "known_speaker_signature": 50,
        "explicit_voice_map": 45,
        "contextual_speaker_identity": 40,
        "historical_identified_voice_cache": 35,
        "mapped_voice_family": 20,
        "native_context_voice_family": 10,
    }
    seen_paths: set[str] = set()
    for work_dir in work_dirs:
        jobs_root = Path(work_dir) / "jobs"
        if not jobs_root.is_dir():
            continue
        for path in jobs_root.glob("*/direct_dub/voice_resolution.json"):
            path_key = str(path.resolve(strict=False)).casefold()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            try:
                value = read_json(path)
            except Exception:
                continue
            if not isinstance(value, Mapping):
                continue
            for item in _iter_resolution_assignments(value).values():
                name = _normalise_identity_name(item.get("identified_name"))
                known = item.get("known_speaker")
                if not name and isinstance(known, Mapping):
                    name = _normalise_identity_name(known.get("display_name"))
                if not name:
                    continue
                family = _voice_family_from_assignment(item)
                voice = str(item.get("selected_voice") or "").strip()
                if family not in {"masculine", "feminine"} or not voice:
                    continue
                status = str(item.get("resolution_status") or item.get("status") or "")
                rank = source_rank.get(status, 5)
                if rank < rank_by_name.get(name, -1):
                    continue
                rank_by_name[name] = rank
                result[name] = {
                    "selected_voice": voice,
                    "voice_family": family,
                    "voice_family_confidence": float(item.get("voice_family_confidence") or 1.0),
                    "voice_family_source": "historical_identified_voice_cache",
                    "historical_resolution_status": status,
                    "historical_source_path": str(path),
                }
    return result

def _load_profiles_by_id(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in value.get("speakers", []) if isinstance(value, Mapping) else []:
        if isinstance(item, Mapping) and item.get("speaker_id"):
            result[str(item["speaker_id"])] = item
    return result


def _canonical_transcript_for_voice_resolution(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, Mapping):
            continue
        value = dict(item)
        if "source_text" not in value:
            value["source_text"] = str(value.get("restored_text") or "")
        values.append(value)
    return {"segments": values}


def _preferred_cache_work_dir(work_dirs: Sequence[Path], relative: str) -> Path:
    for root in work_dirs:
        if (Path(root) / relative).exists():
            return Path(root)
    return Path(work_dirs[0])


def _native_contextual_identities(
    *,
    job: JobManifest,
    job_root: Path,
    transcript: Mapping[str, Any],
    required: Sequence[str],
    work_dir: Path,
    progress: Callable[[str], None] | None,
) -> dict[str, Mapping[str, Any]]:
    output = Path(job_root) / "analysis" / "contextual_speaker_identities.json"
    try:
        result = resolve_contextual_speaker_identities(
            job_id=job.job_id,
            transcript=transcript,
            name_research={},
            required_speaker_ids=required,
            output_path=output,
            registry_path=Path(work_dir) / "contextual_speaker_identities" / "registry.json",
            voice_inference_provider_factory=None,
            voice_inference_speaker_ids=[],
        )
        return {str(k): dict(v) for k, v in result.items() if isinstance(v, Mapping)}
    except Exception as exc:
        _emit_progress(progress, f"[native dub] Legacy contextual voice classification unavailable: {exc}")
        return {}


def _native_legacy_acoustic_evidence(
    *,
    job: JobManifest,
    job_root: Path,
    transcript: Mapping[str, Any],
    required: Sequence[str],
    progress: Callable[[str], None] | None,
    min_confidence: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str | None]:
    analysis_root = Path(job_root) / "analysis"
    diarization_path = analysis_root / "azure_diarization.json"
    audio_path = Path(job_root) / "audio" / "analysis_mono.wav"
    acoustic_path = analysis_root / "acoustic_analysis.json"
    if not diarization_path.is_file() or not audio_path.is_file():
        return {}, {}, "analysis audio or Azure diarization is unavailable"
    diarization = read_json(diarization_path)
    # Native voice analysis treats canonical Pass-1 segment timings as the
    # authoritative reel source.  Azure diarization remains useful for raw-ID
    # mapping, but classifier extraction must not depend on a provider-specific
    # ``turns`` layout once canonical segments already exist.
    canonical_turns: dict[str, list[Mapping[str, Any]]] = {}
    for item in transcript.get("segments", []):
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        if speaker:
            canonical_turns.setdefault(speaker, []).append(item)
    azure_turn_count = len([
        item for item in diarization.get("turns", []) if isinstance(item, Mapping)
    ])
    canonical_turn_count = sum(len(items) for items in canonical_turns.values())
    if canonical_turn_count:
        _emit_progress(
            progress,
            f"[native dub] Voice reels: using {canonical_turn_count} canonical Pass-1 timing segment(s) "
            f"across {len(canonical_turns)} speaker(s); Azure legacy turns={azure_turn_count}",
        )
    mapping = build_speaker_id_mapping(transcript, diarization)
    existing_raw = read_json(acoustic_path) if acoustic_path.is_file() else {}
    existing = canonicalize_voice_family_evidence(existing_raw, mapping)
    unresolved = []
    for speaker in required:
        item = existing.get(speaker, {})
        family = str(item.get("voice_family") or "unknown").lower()
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if family not in {"masculine", "feminine"} or confidence < min_confidence:
            unresolved.append(speaker)
    if not unresolved:
        return existing, mapping, None

    try:
        _emit_progress(
            progress,
            f"[native dub] Running legacy ECAPA voice-family classifier for {len(unresolved)} unresolved speaker(s)...",
        )
        raw = ensure_canonical_voice_family_analysis(
            job_id=job.job_id,
            analysis_audio_path=audio_path,
            diarization=diarization,
            speaker_mapping=mapping,
            required_speaker_ids=unresolved,
            output_path=acoustic_path,
            speakers_root=Path(job_root) / "audio" / "speakers",
            canonical_speaker_turns=canonical_turns,
            confidence_threshold=min_confidence,
            force=False,
        )
        canonical = canonicalize_voice_family_evidence(raw, mapping)
        # direct_voice_analysis intentionally isolates failures per speaker so one
        # bad/short reel does not kill a legacy dub.  Native resolution must still
        # surface those failures here; otherwise an unavailable local classifier
        # looks like six ordinary low-confidence speakers and Mathula wastes a
        # Grok call.
        failures_value = raw.get("failures") if isinstance(raw, Mapping) else None
        failure_messages: list[str] = []
        if isinstance(failures_value, Mapping):
            for speaker in unresolved:
                record = failures_value.get(speaker)
                if not isinstance(record, Mapping):
                    continue
                message = str(record.get("error") or "voice-family classifier failed").strip()
                failure_messages.append(f"{speaker}: {message}")
                _emit_progress(
                    progress,
                    f"[native dub] Legacy ECAPA {speaker}: classifier failure: {message}",
                )
        if failure_messages:
            return canonical, mapping, "; ".join(failure_messages)
        return canonical, mapping, None
    except Exception as exc:
        return existing, mapping, str(exc)


def _native_known_speaker_matches(
    *,
    job_root: Path,
    transcript: Mapping[str, Any],
    required: Sequence[str],
    work_dir: Path,
    mapping: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    registry = Path(work_dir) / "known_speakers" / "registry.json"
    audio_path = Path(job_root) / "audio" / "analysis_mono.wav"
    diarization_path = Path(job_root) / "analysis" / "azure_diarization.json"
    if not registry.is_file() or not audio_path.is_file() or not diarization_path.is_file():
        return {}, {}
    diarization = read_json(diarization_path)
    canonical_turns: dict[str, list[Mapping[str, Any]]] = {}
    for item in transcript.get("segments", []):
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        if speaker:
            canonical_turns.setdefault(speaker, []).append(item)
    matches: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        reels = ensure_canonical_speaker_reels(
            analysis_audio_path=audio_path,
            diarization=diarization,
            speaker_mapping=mapping,
            required_speaker_ids=required,
            speakers_root=Path(job_root) / "audio" / "speakers",
            canonical_speaker_turns=canonical_turns,
            force=False,
        )
    except Exception as exc:
        return {}, {speaker: str(exc) for speaker in required}
    for speaker, reel in reels.items():
        if str(reel.get("status") or "ready") != "ready" or not reel.get("speaker_audio_path"):
            errors[speaker] = str(reel.get("error") or "speaker reel unavailable")
            continue
        try:
            match = identify_known_speaker(Path(str(reel["speaker_audio_path"])), work_dir=work_dir)
            if match is not None:
                matches[speaker] = match
                _emit_progress(
                    progress,
                    f"[native dub] Reused known-speaker cache: {speaker} -> {match.display_name} ({match.azure_voice})",
                )
        except Exception as exc:
            errors[speaker] = str(exc)
    return matches, errors


def ensure_native_voice_assignments(
    *,
    job: JobManifest,
    job_root: Path,
    segments: Sequence[Mapping[str, Any]],
    context_ledger: Mapping[str, Any],
    provider: FoundryGrokProvider | None,
    progress: Callable[[str], None] | None = None,
    min_confidence: float = 0.70,
) -> Path:
    """Resolve native voices through the legacy evidence hierarchy first.

    Precedence: current trusted job cache -> verified known-speaker signature ->
    historical named voice cache -> authoritative/contextual transcript identity ->
    legacy ECAPA acoustic classifier -> existing speaker profile -> Grok for only
    the still-unresolved speakers. Unknown never defaults to Themba.
    """
    job_root = Path(job_root)
    output_path = job_root / "direct_dub" / "voice_resolution.json"
    required = sorted({str(item.get("speaker_id") or "").strip() for item in segments if item.get("speaker_id")})
    if not required:
        raise ValueError("Native dub has no speakers to resolve")

    current_cached = _trusted_job_voice_cache(output_path, required=required)
    if set(required) <= set(current_cached):
        _emit_progress(progress, f"[native dub] Reusing cached voice resolution for all {len(required)} speaker(s)")
        # Persist the native hard-zero rate policy even when the source cache was
        # produced by the older direct renderer.
        cached_value = read_json(output_path)
        if isinstance(cached_value, Mapping):
            cached_value = dict(cached_value)
            cached_value["native_tts_rate_percent"] = 0
            cached_value["assignments"] = current_cached
            if isinstance(cached_value.get("speakers"), list):
                cached_value["speakers"] = [
                    ({**dict(item), **current_cached[str(item.get("speaker_id"))]}
                     if isinstance(item, Mapping) and str(item.get("speaker_id") or "") in current_cached
                     else item)
                    for item in cached_value["speakers"]
                ]
            atomic_write_json(output_path, cached_value)
        return output_path

    to_resolve = [speaker for speaker in required if speaker not in current_cached]
    if current_cached:
        _emit_progress(
            progress,
            f"[native dub] Reusing {len(current_cached)} current-job cached voice mapping(s); "
            f"resolving {len(to_resolve)} remaining speaker(s)...",
        )

    work_dirs = _native_cache_work_dirs(job_root)
    work_dir = work_dirs[0]
    contextual_cache_work_dir = _preferred_cache_work_dir(
        work_dirs, "contextual_speaker_identities/registry.json"
    )
    known_cache_work_dir = _preferred_cache_work_dir(work_dirs, "known_speakers/registry.json")
    transcript = _canonical_transcript_for_voice_resolution(segments)
    contextual = _native_contextual_identities(
        job=job,
        job_root=job_root,
        transcript=transcript,
        required=to_resolve,
        work_dir=contextual_cache_work_dir,
        progress=progress,
    )
    historical_by_name = _historical_identity_voice_cache(work_dirs)

    # Build the raw/canonical mapping even when the classifier is unavailable;
    # known-speaker recognition and cached acoustic evidence use the same mapping.
    diarization_path = job_root / "analysis" / "azure_diarization.json"
    if diarization_path.is_file():
        mapping = build_speaker_id_mapping(transcript, read_json(diarization_path))
    else:
        mapping = {"raw_to_canonical": {}, "canonical_to_raw": {}, "mappings": []}

    known_matches, known_errors = _native_known_speaker_matches(
        job_root=job_root,
        transcript=transcript,
        required=to_resolve,
        work_dir=known_cache_work_dir,
        mapping=mapping,
        progress=progress,
    )

    profiles = _load_profiles_by_id(job_root / "analysis" / "speaker_profiles.json")
    # Do not run the acoustic model for speakers already resolved by a stronger
    # identity/context cache. This keeps the old classifier precise and cheap.
    acoustic_required: list[str] = []
    for speaker in to_resolve:
        if speaker in known_matches:
            continue
        context = contextual.get(speaker)
        profile = profiles.get(speaker, {})
        identified_name = (
            str(context.get("identified_name") or "").strip()
            if isinstance(context, Mapping)
            else ""
        )
        if identified_name and _normalise_identity_name(identified_name) in historical_by_name:
            continue
        identity_gender = str(profile.get("identity_gender") or "").lower()
        if identity_gender in {"male", "female"}:
            continue
        if isinstance(context, Mapping):
            contextual_family = str(context.get("voice_family") or "unknown").lower()
            try:
                contextual_confidence = float(context.get("voice_family_confidence") or 0.99)
            except (TypeError, ValueError):
                contextual_confidence = 0.0
            if contextual_family in {"masculine", "feminine"} and contextual_confidence >= min_confidence:
                continue
        acoustic_required.append(speaker)

    acoustic, acoustic_mapping, acoustic_error = _native_legacy_acoustic_evidence(
        job=job,
        job_root=job_root,
        transcript=transcript,
        required=acoustic_required,
        progress=progress,
        min_confidence=min_confidence,
    )
    if acoustic_mapping:
        mapping = acoustic_mapping
    if acoustic_error and acoustic_required:
        unavailable_markers = (
            "not installed locally",
            "optional PyTorch classifier dependencies",
            "No module named",
            "Neither torchaudio nor librosa available",
        )
        if any(marker in acoustic_error for marker in unavailable_markers):
            raise ValueError(
                "Legacy acoustic voice-family classification is required before Grok fallback, "
                "but the classifier is unavailable. Install the optional voice dependencies with "
                '`python -m pip install -e ".[speaker-recognition]" --no-build-isolation`, then run '
                "`python scripts/install_voice_gender_classifier.py` once. Original classifier error: "
                + acoustic_error
            )
        _emit_progress(
            progress,
            "[native dub] Legacy ECAPA classifier could not resolve every speaker; "
            f"remaining speakers may use transcript context/Grok: {acoustic_error}",
        )

    if acoustic_required:
        for speaker in acoustic_required:
            item = acoustic.get(speaker, {}) if isinstance(acoustic, Mapping) else {}
            family = str(item.get("voice_family") or "unknown").lower()
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            evidence = item.get("evidence") if isinstance(item.get("evidence"), Mapping) else {}
            probabilities = (
                evidence.get("ecapa_probabilities")
                if isinstance(evidence.get("ecapa_probabilities"), Mapping)
                else item.get("probabilities")
            )
            if family in {"masculine", "feminine"}:
                _emit_progress(
                    progress,
                    f"[native dub] Legacy ECAPA {speaker}: {family} ({confidence:.2f})",
                )
            elif isinstance(probabilities, Mapping):
                try:
                    pm = float(probabilities.get("masculine") or 0.0)
                    pf = float(probabilities.get("feminine") or 0.0)
                    _emit_progress(
                        progress,
                        f"[native dub] Legacy ECAPA {speaker}: below {min_confidence:.2f} confidence "
                        f"(masculine={pm:.2f}, feminine={pf:.2f}); leaving unresolved for fallback",
                    )
                except (TypeError, ValueError):
                    _emit_progress(progress, f"[native dub] Legacy ECAPA {speaker}: unresolved")
            else:
                _emit_progress(progress, f"[native dub] Legacy ECAPA {speaker}: unresolved")
    assignments: dict[str, dict[str, Any]] = {speaker: dict(item) for speaker, item in current_cached.items()}
    resolution_records: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for speaker in to_resolve:
        family = "unknown"
        confidence = 0.0
        source = "unresolved"
        selected_voice: str | None = None
        identified_name: str | None = None
        resolution_status = "unresolved"
        known = known_matches.get(speaker)
        context = contextual.get(speaker)
        acoustic_item = acoustic.get(speaker, {})
        profile = profiles.get(speaker, {})
        cache_record: Mapping[str, Any] | None = None

        if known is not None:
            identified_name = str(known.display_name)
            family = "masculine" if known.gender == "male" else "feminine"
            confidence = float(known.score)
            source = "speechbrain_ecapa_known_speaker"
            selected_voice = str(known.azure_voice)
            resolution_status = "known_speaker_signature"
        else:
            if isinstance(context, Mapping):
                identified_name = str(context.get("identified_name") or "").strip() or None
            if identified_name:
                cache_record = historical_by_name.get(_normalise_identity_name(identified_name))
            if cache_record is not None:
                family = str(cache_record.get("voice_family") or "unknown")
                confidence = float(cache_record.get("voice_family_confidence") or 1.0)
                source = "historical_identified_voice_cache"
                selected_voice = str(cache_record.get("selected_voice") or "").strip() or None
                resolution_status = "historical_identified_voice_cache"
            else:
                identity_gender = str(profile.get("identity_gender") or "").lower()
                if identity_gender in {"male", "female"}:
                    family = "masculine" if identity_gender == "male" else "feminine"
                    confidence = float(profile.get("identity_confidence") or 1.0)
                    source = str(profile.get("identity_gender_source") or "authoritative_identity_gender")
                    resolution_status = "authoritative_speaker_profile"
                elif isinstance(context, Mapping):
                    contextual_family = str(context.get("voice_family") or "unknown").lower()
                    contextual_confidence = float(context.get("voice_family_confidence") or 0.99)
                    if contextual_family in {"masculine", "feminine"} and contextual_confidence >= min_confidence:
                        family = contextual_family
                        confidence = contextual_confidence
                        source = str(context.get("voice_family_source") or "contextual_transcript_identity_gender")
                        resolution_status = (
                            "contextual_speaker_identity" if identified_name else "contextual_transcript_voice_family"
                        )

        if family == "unknown":
            acoustic_family = str(acoustic_item.get("voice_family") or "unknown").lower()
            try:
                acoustic_confidence = float(acoustic_item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                acoustic_confidence = 0.0
            if acoustic_family in {"masculine", "feminine"} and acoustic_confidence >= min_confidence:
                family = acoustic_family
                confidence = acoustic_confidence
                evidence = acoustic_item.get("evidence") if isinstance(acoustic_item.get("evidence"), Mapping) else {}
                source = str(evidence.get("method") or acoustic_item.get("method") or "ecapa_voice_family_classifier")
                resolution_status = "legacy_ecapa_voice_family_classifier"

        if family == "unknown":
            profile_family = str(profile.get("tts_voice_family") or "unknown").lower()
            profile_source = str(profile.get("tts_voice_family_source") or "")
            try:
                profile_confidence = float(profile.get("acoustic_confidence") or 0.0)
            except (TypeError, ValueError):
                profile_confidence = 0.0
            authoritative = profile_source in {"authoritative_identity_gender", "authoritative_speaker_registry"}
            if profile_family in {"masculine", "feminine"} and (authoritative or profile_confidence >= min_confidence):
                family = profile_family
                confidence = 1.0 if authoritative else profile_confidence
                source = profile_source or "speaker_profile"
                resolution_status = "speaker_profile"

        if family in {"masculine", "feminine"}:
            if not selected_voice:
                selected_voice = "zu-ZA-ThandoNeural" if family == "feminine" else "zu-ZA-ThembaNeural"
            assignment = {
                "speaker_id": speaker,
                "identified_name": identified_name,
                "selected_voice": selected_voice,
                "voice_family": family,
                "voice_family_confidence": round(confidence, 6),
                "voice_family_source": source,
                "base_prosody": {"rate_percent": 0, "pitch_percent": 0, "volume_percent": 0},
                "source_features": collect_source_features(profile, acoustic_item),
                "resolution_status": resolution_status,
            }
            if known is not None:
                assignment["known_speaker"] = known.to_dict()
            if isinstance(context, Mapping):
                assignment["contextual_identity"] = dict(context)
            if cache_record is not None:
                assignment["historical_voice_cache"] = dict(cache_record)
            assignments[speaker] = assignment
        else:
            unresolved.append(speaker)

    grok_usage = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    if unresolved:
        if provider is None:
            raise ValueError(
                "Native voice resolution remains unresolved after legacy cache/classification for: "
                + ", ".join(unresolved)
                + ". Provide --voice-map; Mathula will not default unknown voices."
            )
        _emit_progress(
            progress,
            f"[native dub] Legacy cache/classifier resolved {len(required) - len(unresolved)}/{len(required)} speaker(s); "
            f"asking Grok only for {len(unresolved)} unresolved speaker(s)...",
        )
        unresolved_set = set(unresolved)
        response = provider.complete_json(
            operation="native_voice_resolution",
            system_prompt=VOICE_RESOLUTION_SYSTEM_PROMPT,
            payload={
                "context_ledger": {
                    "speaker_roles": [
                        item for item in list(context_ledger.get("speaker_roles") or [])[:12]
                        if str(item.get("speaker_id") or "") in unresolved_set
                    ],
                    "entities": list(context_ledger.get("entities") or [])[:16],
                },
                "speakers": _native_voice_samples(segments, only_speakers=unresolved_set),
            },
            schema=_VOICE_RESOLUTION_SCHEMA,
            max_output_tokens=min(700, int(getattr(provider.config, "max_output_tokens", 8192))),
        )
        rows = response.data.get("voices")
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping) and row.get("speaker_id"):
                speaker = str(row["speaker_id"]).strip()
                if speaker in by_id:
                    raise ValueError(f"Native voice resolution duplicated {speaker}")
                by_id[speaker] = row
        missing = sorted(unresolved_set - set(by_id))
        extra = sorted(set(by_id) - unresolved_set)
        if missing or extra:
            raise ValueError(
                "Native Grok voice-resolution speaker set mismatch; "
                + (f"missing: {', '.join(missing)}. " if missing else "")
                + (f"unexpected: {', '.join(extra)}." if extra else "")
            )
        still_unresolved: list[str] = []
        for speaker in unresolved:
            row = by_id[speaker]
            family = str(row.get("voice_family") or "unknown")
            confidence = float(row.get("confidence") or 0.0)
            if family not in {"masculine", "feminine"} or confidence < min_confidence:
                still_unresolved.append(speaker)
                continue
            assignments[speaker] = {
                "speaker_id": speaker,
                "identified_name": (
                    str(contextual.get(speaker, {}).get("identified_name") or "").strip() or None
                    if isinstance(contextual.get(speaker), Mapping) else None
                ),
                "selected_voice": "zu-ZA-ThandoNeural" if family == "feminine" else "zu-ZA-ThembaNeural",
                "voice_family": family,
                "voice_family_confidence": round(confidence, 6),
                "voice_family_source": "grok_transcript_context_fallback",
                "base_prosody": {"rate_percent": 0, "pitch_percent": 0, "volume_percent": 0},
                "source_features": collect_source_features(profiles.get(speaker, {}), acoustic.get(speaker, {})),
                "resolution_status": "native_context_voice_family_fallback",
            }
        if still_unresolved:
            raise ValueError(
                "Native voice resolution could not confidently map: "
                + ", ".join(still_unresolved)
                + ". Provide --voice-map for these speakers; Mathula will not default unknown voices."
            )
        grok_usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "attempts": response.attempts,
        }

    # Reuse the old source-relative persona separation, but hard-lock native TTS
    # rate back to 0% so voice identity never changes cadence in this pipeline.
    assignments = apply_direct_azure_personas(assignments)
    for speaker, assignment in assignments.items():
        base = dict(assignment.get("base_prosody") or {})
        old_rate = int(base.get("rate_percent") or 0)
        base["legacy_persona_rate_percent_ignored"] = old_rate
        base["rate_percent"] = 0
        base["native_rate_policy"] = "hard_locked_zero_percent"
        assignment["base_prosody"] = base
        assignments[speaker] = assignment

    for speaker in required:
        item = assignments[speaker]
        resolution_records.append({"status": "resolved", **item})
        _emit_progress(
            progress,
            f"[native dub] Voice {speaker}: {item['selected_voice']} "
            f"({item.get('voice_family')}, {item.get('voice_family_source')}, "
            f"confidence {float(item.get('voice_family_confidence') or 0.0):.2f}, "
            f"pitch {int((item.get('base_prosody') or {}).get('pitch_percent') or 0):+d}%, rate +0%)",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, {
        "schema_version": NATIVE_VOICE_RESOLUTION_SCHEMA_VERSION,
        "prompt_version": VOICE_RESOLUTION_PROMPT_VERSION,
        "persona_version": DIRECT_AZURE_PERSONA_VERSION,
        "job_id": job.job_id,
        "provider": provider.provider if provider is not None and grok_usage["attempts"] else None,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION if grok_usage["attempts"] else None,
        "model": provider.config.deployment if provider is not None and grok_usage["attempts"] else None,
        "automatic_web_search": False,
        "resolution_precedence": [
            "trusted_current_job_cache",
            "speechbrain_known_speaker_signature",
            "historical_identified_voice_cache",
            "authoritative_or_contextual_identity",
            "legacy_ecapa_voice_family_classifier",
            "speaker_profile",
            "grok_unresolved_only",
        ],
        "token_strategy": "no_grok_when_legacy_evidence_resolves;unresolved_speakers_only",
        "tts_rate_percent": 0,
        "speaker_id_mapping": mapping,
        "sources": {
            "known_speaker_registry": str(Path(known_cache_work_dir) / "known_speakers" / "registry.json"),
            "contextual_identity_registry": str(Path(contextual_cache_work_dir) / "contextual_speaker_identities" / "registry.json"),
            "acoustic_analysis": str(job_root / "analysis" / "acoustic_analysis.json"),
            "speaker_profiles": str(job_root / "analysis" / "speaker_profiles.json"),
            "historical_jobs_roots": [str(Path(root) / "jobs") for root in work_dirs],
            "legacy_ecapa_error": acoustic_error,
            "known_speaker_errors": known_errors,
        },
        "assignments": assignments,
        "speakers": resolution_records,
        "unresolved_speakers": [],
        "unknown_defaults_to_masculine": False,
        "usage": grok_usage,
        "completed_at": utcnow(),
    })
    _emit_progress(progress, f"[native dub] Saved voice resolution: {output_path}")
    return output_path

def load_native_voice_assignments(
    job_root: Path,
    *,
    required_speakers: set[str],
    explicit_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = Path(explicit_path) if explicit_path else Path(job_root) / "direct_dub" / "voice_resolution.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Voice mapping is required for native dub: {path}. "
            "Run voice resolution first or pass --voice-map."
        )
    value = read_json(path)
    assignments: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping) and isinstance(value.get("speakers"), list):
        for item in value["speakers"]:
            if not isinstance(item, Mapping):
                continue
            speaker = str(item.get("speaker_id") or "").strip()
            voice = str(item.get("selected_voice") or "").strip()
            status = str(item.get("status") or item.get("resolution_status") or "resolved")
            if speaker and voice and status not in {"unresolved", "failed"}:
                assignments[speaker] = dict(item)
    elif isinstance(value, Mapping):
        # Explicit operator convenience map: {"SPEAKER_00": "zu-ZA-..."}
        for speaker, voice_value in value.items():
            if isinstance(voice_value, str) and voice_value.strip():
                assignments[str(speaker)] = {
                    "speaker_id": str(speaker),
                    "selected_voice": voice_value.strip(),
                    "base_prosody": {},
                    "status": "explicit_override",
                }
            elif isinstance(voice_value, Mapping) and voice_value.get("selected_voice"):
                assignments[str(speaker)] = dict(voice_value)
    missing = sorted(required_speakers - set(assignments))
    if missing:
        raise ValueError(
            "Native dub refuses to collapse unmapped speakers onto one voice; missing mappings for: "
            + ", ".join(missing)
        )
    return assignments


def build_phrase_groups(
    segments: Sequence[Mapping[str, Any]],
    translations: Mapping[str, str],
    *,
    max_join_gap_ms: int,
    max_phrase_source_ms: int,
) -> list[dict[str, Any]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for segment in segments:
        if not current:
            current = [segment]
            continue
        previous = current[-1]
        same_speaker = segment["speaker_id"] == previous["speaker_id"]
        gap_ms = int(segment["start_ms"]) - int(previous["end_ms"])
        span_ms = int(segment["end_ms"]) - int(current[0]["start_ms"])
        # Phrase grouping remains articulation-based. Protected gap overflow, when
        # enabled later, never changes grouping or a later phrase's source onset.
        if (
            same_speaker
            and -200 <= gap_ms <= min(0, max_join_gap_ms)
            and span_ms <= max_phrase_source_ms
        ):
            current.append(segment)
        else:
            groups.append(current)
            current = [segment]
    if current:
        groups.append(current)

    result: list[dict[str, Any]] = []
    for index, items in enumerate(groups, 1):
        parts: list[dict[str, Any]] = []
        for pos, item in enumerate(items):
            if pos:
                gap = max(0, int(item["start_ms"]) - int(items[pos - 1]["end_ms"]))
                if gap >= 80:
                    parts.append({"type": "break", "duration_ms": min(gap, 2000)})
            parts.append({
                "type": "text",
                "segment_id": item["segment_id"],
                "text": translations[item["segment_id"]],
            })
        result.append({
            "group_id": f"native_phrase_{index:04d}",
            "speaker_id": items[0]["speaker_id"],
            "segment_ids": [item["segment_id"] for item in items],
            "start_ms": int(items[0]["start_ms"]),
            "source_end_ms": int(items[-1]["end_ms"]),
            "source_span_ms": int(items[-1]["end_ms"]) - int(items[0]["start_ms"]),
            "source_segments": [
                {"segment_id": item["segment_id"], "source_text": item["restored_text"]}
                for item in items
            ],
            "parts": parts,
        })
    return result


def _hard_sync_speed_factor(measured_duration_ms: int, source_window_ms: int) -> float:
    """Pitch-preserving speed required to make one overflowing phrase fit its own source window."""
    measured = int(measured_duration_ms)
    window = int(source_window_ms)
    if measured <= 0 or window <= 0:
        raise ValueError("Hard-sync durations must be positive")
    return max(1.0, measured / window)


def _temporal_required_rush_percent(measured_duration_ms: int, source_window_ms: int) -> float:
    """Percent pitch-preserving speed-up required to land measured audio on its own source window.

    Expressed as the same ratio as ``_hard_sync_speed_factor``, just as a percentage above
    100% rather than a raw multiplier (0.0 when the audio already fits; never negative,
    since speech is never slowed).
    """
    return (_hard_sync_speed_factor(measured_duration_ms, source_window_ms) - 1.0) * 100.0


def _speed_fit_mode_label(late_tolerance_ms: int) -> str:
    """Name which of the two speed-fit modes applies to a sentence.

    "gap_aware": a real, already-safe gap exists before the next sentence starts
    (mouth_close_late_tolerance_ms / _effective_protected_late_allowance), so the
    speed-fit target extends into it -- part of any overflow is absorbed by existing
    silence instead of pure pitch-preserving acceleration. "sentence_end": there is no
    usable gap, so the target is just the sentence's own end (unchanged from before this
    was added). Selection is automatic per sentence, never a global flag.
    """
    return "gap_aware" if int(late_tolerance_ms) > 0 else "sentence_end"


def _effective_mouth_close_tolerance(
    configured_tolerance_ms: int,
    source_end_ms: int,
    next_source_start_ms: int,
) -> int:
    """Never let sync tolerance create a new overlap with the next source block."""
    configured = max(0, int(configured_tolerance_ms))
    free_room = max(0, int(next_source_start_ms) - int(source_end_ms))
    return min(configured, free_room)


def _effective_protected_late_allowance(
    configured_tolerance_ms: int,
    source_end_ms: int,
    next_source_start_ms: int,
    *,
    protected_gap_overflow: bool,
    max_protected_gap_overflow_ms: int,
    min_protected_pause_ms: int,
) -> int:
    """Return the maximum safe late finish without moving the next source onset.

    The configured mouth-close tolerance remains the baseline.  When protected gap
    overflow is enabled, Mathula may additionally occupy existing post-articulation
    silence, capped by ``max_protected_gap_overflow_ms`` and only after reserving
    ``min_protected_pause_ms`` before the next source onset.
    """
    free_gap_ms = max(0, int(next_source_start_ms) - int(source_end_ms))
    baseline_ms = min(max(0, int(configured_tolerance_ms)), free_gap_ms)
    if not protected_gap_overflow:
        return baseline_ms
    protected_room_ms = max(0, free_gap_ms - max(0, int(min_protected_pause_ms)))
    overflow_ms = min(max(0, int(max_protected_gap_overflow_ms)), protected_room_ms)
    return min(free_gap_ms, max(baseline_ms, overflow_ms))


def _mouth_close_sync_ok(
    end_error_ms: int,
    *,
    early_tolerance_ms: int,
    late_tolerance_ms: int,
) -> bool:
    """Return True when a mouth-close error is inside asymmetric safe bounds."""
    error = int(end_error_ms)
    early = max(0, int(early_tolerance_ms))
    late = max(0, int(late_tolerance_ms))
    return -early <= error <= late


def _timing_recast_direction(
    measured_duration_ms: int,
    source_window_ms: int,
    *,
    max_speed_percent: int,
    mouth_close_early_tolerance_ms: int,
    mouth_close_late_tolerance_ms: int,
) -> str | None:
    """Return expand/compress only when speed fitting cannot satisfy mouth-close sync."""
    measured = int(measured_duration_ms)
    window = int(source_window_ms)
    early_tolerance = int(mouth_close_early_tolerance_ms)
    late_tolerance = int(mouth_close_late_tolerance_ms)
    if measured < window - early_tolerance:
        return "expand"
    maximum = 1.0 + int(max_speed_percent) / 100.0
    latest_acceptable_end = window + late_tolerance
    minimum_speed = max(1.0, measured / max(1, latest_acceptable_end))
    if minimum_speed > maximum + 1e-9:
        return "compress"
    return None


def _timing_target_score(
    measured_duration_ms: int,
    source_window_ms: int,
    target_raw_duration_ms: int,
    *,
    max_speed_percent: int,
    mouth_close_early_tolerance_ms: int,
    mouth_close_late_tolerance_ms: int,
) -> int:
    """Lower is better; infeasible early/late speech is heavily penalised."""
    measured = int(measured_duration_ms)
    window = int(source_window_ms)
    target = int(target_raw_duration_ms)
    early_tolerance = int(mouth_close_early_tolerance_ms)
    late_tolerance = int(mouth_close_late_tolerance_ms)
    maximum = 1.0 + int(max_speed_percent) / 100.0
    latest_raw_that_max_speed_can_rescue = int(round((window + late_tolerance) * maximum))
    penalty = 0
    if measured < window - early_tolerance:
        penalty += (window - early_tolerance - measured) * 20
    elif measured > latest_raw_that_max_speed_can_rescue:
        penalty += (measured - latest_raw_that_max_speed_can_rescue) * 20
    return penalty + abs(measured - target)


def _timing_candidate_rank(
    measured_duration_ms: int,
    source_window_ms: int,
    target_raw_duration_ms: int,
    *,
    max_speed_percent: int,
    mouth_close_early_tolerance_ms: int,
    mouth_close_late_tolerance_ms: int,
) -> tuple[int, int, int]:
    """Asymmetric ordering with unlimited-rush timing policy.

    Lower tuples are better:
      0. fits at or below the +max_speed quality-warning threshold;
      1. overlong, but mechanically rescuable by stronger pitch-preserving acceleration;
      2. materially early, which would require forbidden slowing/dead-air acceptance.

    ``max_speed_percent`` is therefore used to classify quality, not safety. A rushed
    candidate is preferred to an undersized candidate because the former can still be
    landed exactly on the source mouth-close by atempo.
    """
    measured = int(measured_duration_ms)
    window = int(source_window_ms)
    target = int(target_raw_duration_ms)
    early_tolerance = int(mouth_close_early_tolerance_ms)
    late_tolerance = int(mouth_close_late_tolerance_ms)
    direction = _timing_recast_direction(
        measured,
        window,
        max_speed_percent=int(max_speed_percent),
        mouth_close_early_tolerance_ms=early_tolerance,
        mouth_close_late_tolerance_ms=late_tolerance,
    )
    if direction is None:
        return (0, abs(measured - target), abs(measured - window))
    if direction == "compress":
        # Rank the required rush against the SAME extended boundary (window + late
        # tolerance) that _timing_recast_direction already used internally to decide
        # compression was even needed -- keeps classification and ranking consistent.
        required_rush_milli = int(
            round(_temporal_required_rush_percent(measured, window + late_tolerance) * 1000.0)
        )
        return (1, required_rush_milli, abs(measured - target))

    early_miss = max(0, window - early_tolerance - measured)
    return (2, early_miss, abs(measured - target))


_PAUSE_AWARE_MIN_GAP_DETECT_MS = 50
_PAUSE_AWARE_GAP_FLOOR_MS = 30


def _detect_internal_silence_runs(
    frames: bytes,
    *,
    sample_width: int,
    channels: int,
    framerate: int,
    threshold: int,
    min_run_ms: int,
) -> list[tuple[int, int]]:
    """Frame-index (start, end) ranges of near-silent runs strictly INSIDE the
    buffer -- a run touching the very first or last frame is edge silence
    (already handled by canonicalize_pcm_wav's trimming elsewhere), not a
    real inter-clause/inter-sentence breathing pause. Uses the SAME amplitude
    definition as azure_tts._frame_is_near_silent/SILENCE_AMPLITUDE_THRESHOLD
    so "near-silent" means one consistent thing across this codebase.

    Only 16-bit PCM is supported (the only sample width this pipeline's TTS
    output ever uses -- see OUTPUT_FORMATS); any other width degrades to "no
    gaps detected" rather than guessing at an unfamiliar layout.
    """
    frame_size = channels * sample_width
    total_frames = len(frames) // frame_size
    if total_frames == 0 or sample_width != 2:
        return []
    samples = array.array("h")
    samples.frombytes(frames[: total_frames * frame_size])
    if sys.byteorder == "big":
        samples.byteswap()
    is_silent = [
        all(-threshold <= samples[i * channels + c] <= threshold for c in range(channels))
        for i in range(total_frames)
    ]
    min_run_frames = int(round(min_run_ms * framerate / 1000.0))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < total_frames:
        if is_silent[i]:
            j = i
            while j < total_frames and is_silent[j]:
                j += 1
            if i > 0 and j < total_frames and (j - i) >= min_run_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _remap_ms_through_segments(old_ms: float, segments: Sequence[Mapping[str, Any]]) -> int:
    """Map a timestamp on the ORIGINAL timeline to its position on a
    piecewise-linearly-retimed timeline, given the {old_start_ms, old_end_ms,
    new_start_ms, new_end_ms} segment list _pause_aware_speed_fit_wav
    returns. A value at or past the last segment's end (e.g. a bookmark
    exactly at the source's own trailing edge) clamps to that segment's own
    new end rather than extrapolating.
    """
    if not segments:
        return int(round(old_ms))
    for segment in segments:
        start, end = float(segment["old_start_ms"]), float(segment["old_end_ms"])
        if start <= old_ms <= end:
            span = end - start
            if span <= 0:
                return int(round(segment["new_start_ms"]))
            fraction = (old_ms - start) / span
            return int(round(
                float(segment["new_start_ms"])
                + fraction * (float(segment["new_end_ms"]) - float(segment["new_start_ms"]))
            ))
    last = segments[-1]
    return int(round(float(last["new_end_ms"])))


def _pause_aware_speed_fit_wav(
    *,
    source_path: Path,
    output_path: Path,
    target_duration_ms: int,
    max_speed_percent: int,
    allowed_late_ms: int = MOUTH_CLOSE_TARGET_TOLERANCE_MS,
    allowed_early_ms: int = MOUTH_CLOSE_TARGET_TOLERANCE_MS,
    gap_floor_ms: int = _PAUSE_AWARE_GAP_FLOOR_MS,
    min_gap_detect_ms: int = _PAUSE_AWARE_MIN_GAP_DETECT_MS,
    silence_amplitude_threshold: int = SILENCE_AMPLITUDE_THRESHOLD,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Pitch-preserving speed fit that protects real inter-clause pauses,
    instead of _speed_fit_overflow_wav's single uniform atempo factor across
    the whole clip.

    Real production defect (job 33cd7b46..., confirmed via real Azure STT
    word timestamps): a genuine ~80ms natural pause at a comma boundary
    survived at 0ms after _speed_fit_overflow_wav's uniform compression --
    the words remained intelligible, but the lost breathing room between two
    back-to-back rhetorical clauses read as an audibly rushed run-on. This
    matches published dubbing research (Oktem, Farrus & Bonafonte 2019,
    "Prosodic Phrase Alignment for Machine Dubbing": real professional dubs
    preserve pauses at high rates -- a pause >=400ms survives into the dub
    91% of the time -- by fitting a duration "bending ratio" to the SPEECH
    within each voiced segment while handling pause duration separately,
    never scaling silence and speech by the same factor. The same segment-
    differentiated principle underlies commercial "smart speed" audio
    playback (detect silence, treat it differently from speech).

    Detects real internal near-silent runs (_detect_internal_silence_runs,
    the same amplitude definition already used for edge-trimming elsewhere
    in this file), splits the clip into alternating speech/gap segments, and
    puts the compression burden on speech: each gap shrinks toward the flat
    uniform ratio but never below ``gap_floor_ms``, and whatever time gaps
    can't give up gets made up by compressing speech somewhat harder than a
    flat uniform factor would have required -- mirrors the cited paper's own
    per-segment "bending ratio" applied only to voiced content.

    No internal gaps detected, or gaps alone would already meet/exceed the
    target (a pathological case never seen in real-job testing): degrades to
    exactly _speed_fit_overflow_wav's plain uniform behavior -- never worse
    than the mechanism it replaces.

    Returns the same shape as _speed_fit_overflow_wav (``speed_factor``/
    ``speed_percent`` describe the SPEECH-only factor actually applied) PLUS
    a ``segments`` key for the caller to remap bookmark offsets through
    (see _remap_ms_through_segments) instead of a flat ratio.
    """
    target = int(target_duration_ms)
    late_tolerance = int(allowed_late_ms)
    early_tolerance = int(allowed_early_ms)
    if target <= 0:
        raise ValueError("Hard-sync target duration must be positive")

    before = wav_duration_ms(Path(source_path))
    if before <= target:
        return {
            "schema_version": "mathula-native-pause-aware-speed-fit-v1",
            "method": "none",
            "source_path": str(source_path),
            "output_path": str(source_path),
            "before_duration_ms": before,
            "target_duration_ms": target,
            "final_duration_ms": before,
            "speed_factor": 1.0,
            "speed_percent": 0.0,
            "end_error_ms": before - target,
            "mouth_close_sync_ok": _mouth_close_sync_ok(
                before - target, early_tolerance_ms=early_tolerance, late_tolerance_ms=late_tolerance,
            ),
            "segments": [
                {"old_start_ms": 0, "old_end_ms": before, "new_start_ms": 0, "new_end_ms": before}
            ],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _fallback_uniform() -> dict[str, Any]:
        result = _speed_fit_overflow_wav(
            source_path=source_path, output_path=output_path, target_duration_ms=target,
            max_speed_percent=max_speed_percent, allowed_late_ms=late_tolerance,
            allowed_early_ms=early_tolerance, runner=runner,
        )
        final_ms = int(result["final_duration_ms"])
        result["schema_version"] = "mathula-native-pause-aware-speed-fit-v1"
        result["segments"] = [
            {"old_start_ms": 0, "old_end_ms": before, "new_start_ms": 0, "new_end_ms": final_ms}
        ]
        return result

    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(params.nframes)
    framerate, channels, sample_width = params.framerate, params.nchannels, params.sampwidth
    frame_size = channels * sample_width
    total_frames = len(frames) // frame_size

    gap_runs = _detect_internal_silence_runs(
        frames, sample_width=sample_width, channels=channels, framerate=framerate,
        threshold=silence_amplitude_threshold, min_run_ms=min_gap_detect_ms,
    )
    if not gap_runs:
        return _fallback_uniform()

    uniform_factor = target / before  # duration ratio (target/natural), < 1

    segments: list[dict[str, Any]] = []
    cursor = 0
    for gap_start, gap_end in gap_runs:
        if gap_start > cursor:
            segments.append({"kind": "speech", "start_frame": cursor, "end_frame": gap_start})
        segments.append({"kind": "gap", "start_frame": gap_start, "end_frame": gap_end})
        cursor = gap_end
    if cursor < total_frames:
        segments.append({"kind": "speech", "start_frame": cursor, "end_frame": total_frames})

    def _frame_to_ms(frame: int) -> float:
        return frame * 1000.0 / framerate

    gap_new_ms: dict[int, float] = {}
    gap_new_ms_total = 0.0
    gap_natural_ms_total = 0.0
    for index, segment in enumerate(segments):
        if segment["kind"] != "gap":
            continue
        natural_ms = _frame_to_ms(segment["end_frame"]) - _frame_to_ms(segment["start_frame"])
        gap_natural_ms_total += natural_ms
        new_ms = min(natural_ms, max(float(gap_floor_ms), natural_ms * uniform_factor))
        gap_new_ms[index] = new_ms
        gap_new_ms_total += new_ms

    speech_natural_ms_total = before - gap_natural_ms_total
    speech_target_ms_total = target - gap_new_ms_total
    if speech_target_ms_total <= 0 or speech_natural_ms_total <= 0:
        return _fallback_uniform()

    speech_factor = speech_natural_ms_total / speech_target_ms_total  # > 1 = speed up

    rendered_frame_buffers: list[bytes] = []
    remap_segments: list[dict[str, Any]] = []
    new_cursor_ms = 0.0
    with tempfile.TemporaryDirectory(prefix="pause_aware_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for position, segment in enumerate(segments):
            old_start_ms = _frame_to_ms(segment["start_frame"])
            old_end_ms = _frame_to_ms(segment["end_frame"])
            raw_bytes = frames[segment["start_frame"] * frame_size: segment["end_frame"] * frame_size]
            if segment["kind"] == "gap":
                new_ms = gap_new_ms[position]
                keep_frames = int(round(new_ms * framerate / 1000.0))
                segment_bytes = raw_bytes[: keep_frames * frame_size]
                new_duration_ms = keep_frames * 1000.0 / framerate
            else:
                seg_input_path = temp_dir / f"seg_{position:04d}_in.wav"
                seg_output_path = temp_dir / f"seg_{position:04d}_out.wav"
                with wave.open(str(seg_input_path), "wb") as handle:
                    handle.setparams((channels, sample_width, framerate, 0, "NONE", "not compressed"))
                    handle.writeframes(raw_bytes)
                runner(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(seg_input_path), "-filter:a", atempo_chain(speech_factor),
                        "-c:a", "pcm_s16le", str(seg_output_path),
                    ],
                    check=True,
                )
                with wave.open(str(seg_output_path), "rb") as handle:
                    seg_params = handle.getparams()
                    segment_bytes = handle.readframes(seg_params.nframes)
                new_duration_ms = len(segment_bytes) / frame_size * 1000.0 / framerate
            rendered_frame_buffers.append(segment_bytes)
            remap_segments.append({
                "old_start_ms": old_start_ms, "old_end_ms": old_end_ms,
                "new_start_ms": new_cursor_ms, "new_end_ms": new_cursor_ms + new_duration_ms,
            })
            new_cursor_ms += new_duration_ms

    with wave.open(str(output_path), "wb") as out:
        out.setparams((channels, sample_width, framerate, 0, "NONE", "not compressed"))
        for buffer in rendered_frame_buffers:
            out.writeframes(buffer)

    final_ms = wav_duration_ms(output_path)
    end_error_ms = final_ms - target
    warning_threshold = 1.0 + int(max_speed_percent) / 100.0
    return {
        "schema_version": "mathula-native-pause-aware-speed-fit-v1",
        "method": "pause_aware_atempo",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "before_duration_ms": before,
        "target_duration_ms": target,
        "allowed_early_ms": early_tolerance,
        "allowed_late_ms": late_tolerance,
        "final_duration_ms": final_ms,
        "speed_factor": round(float(speech_factor), 8),
        "speed_percent": round((float(speech_factor) - 1.0) * 100.0, 3),
        "gap_count": len(gap_runs),
        "gap_natural_ms_total": round(gap_natural_ms_total, 1),
        "gap_protected_ms_total": round(gap_new_ms_total, 1),
        "speed_warning_threshold_percent": int(max_speed_percent),
        "speed_warning_exceeded": bool(float(speech_factor) > warning_threshold + 1e-9),
        "end_error_ms": end_error_ms,
        "mouth_close_sync_ok": _mouth_close_sync_ok(
            end_error_ms, early_tolerance_ms=early_tolerance, late_tolerance_ms=late_tolerance,
        ),
        "segments": remap_segments,
    }


def _speed_fit_overflow_wav(
    *,
    source_path: Path,
    output_path: Path,
    target_duration_ms: int,
    max_speed_percent: int,
    allowed_late_ms: int = MOUTH_CLOSE_TARGET_TOLERANCE_MS,
    allowed_early_ms: int = MOUTH_CLOSE_TARGET_TOLERANCE_MS,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Pitch-preserving speed fit to the source mouth-close.

    The exact source end is always preferred. ``max_speed_percent`` is a QUALITY
    WARNING threshold, not a hard ceiling: faithful overlong speech may be accelerated
    beyond it and must never terminate the pipeline merely for requiring a rush. Small
    encoder/filter rounding is refined toward the mouth-close. Speech is never slowed.
    """
    target = int(target_duration_ms)
    late_tolerance = int(allowed_late_ms)
    early_tolerance = int(allowed_early_ms)
    if target <= 0:
        raise ValueError("Hard-sync target duration must be positive")
    if not 0 <= int(max_speed_percent) <= 30:
        raise ValueError("Hard-sync maximum speed percent must be between 0 and 30")
    if not 0 <= late_tolerance <= PROTECTED_GAP_OVERFLOW_HARD_MAX_MS:
        raise ValueError(
            "Hard-sync/protected-gap late allowance must be between 0 and "
            f"{PROTECTED_GAP_OVERFLOW_HARD_MAX_MS} ms"
        )
    if not 0 <= early_tolerance <= MOUTH_CLOSE_HARD_TOLERANCE_MS:
        raise ValueError(
            f"Hard-sync early tolerance must be between 0 and {MOUTH_CLOSE_HARD_TOLERANCE_MS} ms"
        )

    before = wav_duration_ms(Path(source_path))
    exact_required = _hard_sync_speed_factor(before, target)
    warning_threshold = 1.0 + int(max_speed_percent) / 100.0
    if before <= target:
        return {
            "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
            "method": "none",
            "source_path": str(source_path),
            "output_path": str(source_path),
            "before_duration_ms": before,
            "target_duration_ms": target,
            "allowed_early_ms": early_tolerance,
            "allowed_late_ms": late_tolerance,
            "final_duration_ms": before,
            "speed_factor": 1.0,
            "speed_percent": 0.0,
            "end_error_ms": before - target,
            "mouth_close_sync_ok": _mouth_close_sync_ok(
                before - target,
                early_tolerance_ms=early_tolerance,
                late_tolerance_ms=late_tolerance,
            ),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer exact landing. FFmpeg atempo can quantize a few milliseconds to either
    # side of the target, so refine in both directions. Late refinement speeds up;
    # early refinement eases the speed down. Never slow below the original rate.
    factor = max(1.0, exact_required * 1.0002)
    for attempt in range(5):
        runner(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-filter:a",
                atempo_chain(factor),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
        )
        final_ms = wav_duration_ms(output_path)
        end_error_ms = final_ms - target
        if _mouth_close_sync_ok(
            end_error_ms,
            early_tolerance_ms=early_tolerance,
            late_tolerance_ms=late_tolerance,
        ):
            break
        if attempt < 4:
            refined = factor * final_ms / max(1, target)
            factor = max(1.0, refined)
    final_ms = wav_duration_ms(output_path)
    end_error_ms = final_ms - target
    # Never terminate merely because the required acceleration is high. If FFmpeg
    # quantization still leaves a tiny late error after refinement, report it to the
    # caller; the controller emits a warning rather than converting timing into a fatal
    # semantic rewrite loop.
    return {
        "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
        "method": "ffmpeg_atempo",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "before_duration_ms": before,
        "target_duration_ms": target,
        "allowed_early_ms": early_tolerance,
        "allowed_late_ms": late_tolerance,
        "final_duration_ms": final_ms,
        "speed_factor": round(float(factor), 8),
        "speed_percent": round((float(factor) - 1.0) * 100.0, 3),
        "speed_warning_threshold_percent": int(max_speed_percent),
        "speed_warning_exceeded": bool(float(factor) > warning_threshold + 1e-9),
        "end_error_ms": end_error_ms,
        "mouth_close_sync_ok": _mouth_close_sync_ok(
            end_error_ms,
            early_tolerance_ms=early_tolerance,
            late_tolerance_ms=late_tolerance,
        ),
    }


def _pad_wav_with_trailing_silence(
    *,
    source_path: Path,
    output_path: Path,
    target_duration_ms: int,
) -> dict[str, Any]:
    """Append trailing silence to close a small residual shortfall.

    Closing the LAST fraction of a syllable's worth of timing through wording is not
    achievable -- any real edit adds or removes at least a full syllable, overshooting
    a gap this small in either direction. This is the "expand" side's equivalent of
    ``_speed_fit_overflow_wav``: a real audio-level fix instead of asking the model
    for a precision natural language editing cannot deliver. Only ever called for a
    gap already confirmed small and confirmed to fit inside existing free room before
    the next block (see the caller's ``MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS`` bound) --
    this is not a substitute for real repair, only a last-mile rounding fix.
    """
    target = int(target_duration_ms)
    before = wav_duration_ms(Path(source_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    padding_ms = max(0, target - before)
    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(params.nframes)
    silence_frames = int(round(padding_ms * params.framerate / 1000.0))
    with wave.open(str(output_path), "wb") as dst:
        dst.setparams((params.nchannels, params.sampwidth, params.framerate, 0, "NONE", "not compressed"))
        dst.writeframes(frames)
        dst.writeframes(b"\0" * silence_frames * params.nchannels * params.sampwidth)
    final_ms = wav_duration_ms(output_path)
    return {
        "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
        "method": "trailing_silence_pad",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "before_duration_ms": before,
        "target_duration_ms": target,
        "padding_ms": padding_ms,
        "final_duration_ms": final_ms,
        "speed_factor": 1.0,
        "speed_percent": 0.0,
        "end_error_ms": final_ms - target,
        "mouth_close_sync_ok": True,
    }


_NATIVE_DUB_PRONUNCIATION_DICTIONARY: PronunciationDictionary | None = None
# Phase 17: a job-scoped dictionary augmented with web-researched pronunciation
# entries (see _ensure_native_pronunciation_research/_apply_native_pronunciation_
# research_to_job_cache), keyed by str(job_root). Populated once per job -- by
# build_candidate_pool directly in-process, or by render_native_dub reading the
# cached pronunciation_research.json artifact back off disk when it runs as a
# separate process/invocation. A job_root with no entry here (research never
# ran, or this is a job predating Phase 17) safely falls back to the plain
# job-agnostic dictionary below -- never a crash, never a behavior change for
# an unregistered job.
_NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB: dict[str, PronunciationDictionary] = {}


def _native_dub_pronunciation_dictionary() -> PronunciationDictionary:
    global _NATIVE_DUB_PRONUNCIATION_DICTIONARY
    if _NATIVE_DUB_PRONUNCIATION_DICTIONARY is None:
        _NATIVE_DUB_PRONUNCIATION_DICTIONARY = with_default_organisation_initialisms(
            PronunciationDictionary(dictionary_version="v1", language="zu-ZA")
        )
    return _NATIVE_DUB_PRONUNCIATION_DICTIONARY


def _apply_native_pronunciation_research_to_job_cache(
    job_root: Path, research: Mapping[str, Any] | None,
) -> PronunciationDictionary:
    """Build and cache this job's pronunciation dictionary, augmented with any
    accepted web-researched entries (Phase 17). Reuses pronunciation.py's own
    already-existing job-scoping machinery (with_web_researched_organisation_
    pronunciations/with_job_overrides) -- the exact same mechanism the legacy
    pipeline already uses for this. A real job_id (not the default None) is
    given up front so with_job_overrides never has to synthesize one.
    """
    base = with_default_organisation_initialisms(
        PronunciationDictionary(dictionary_version="v1", language="zu-ZA", job_id=str(job_root))
    )
    # Phase 24: self_supervised_corrections is shaped identically to
    # accepted_corrections (same fields with_web_researched_organisation_
    # pronunciations reads) -- merged into one list here so that function
    # needs no knowledge of which source a record came from, only whether it
    # carries real web evidence or a confirmed round-trip pass (see its own
    # relaxed evidence gate).
    merged_research: Mapping[str, Any] | None = research
    if isinstance(research, Mapping):
        self_supervised = research.get("self_supervised_corrections") or ()
        if self_supervised:
            merged_research = {
                **research,
                "accepted_corrections": [
                    *(research.get("accepted_corrections") or ()), *self_supervised,
                ],
            }
    dictionary = with_web_researched_organisation_pronunciations(base, merged_research)
    _NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB[str(job_root)] = dictionary
    return dictionary


def _native_dub_tts_ready_text(text: str, *, job_root: Path | str | None = None) -> str:
    """Apply the reviewed pronunciation dictionary's TTS-only substitutions.

    Never touches the committed/display text (captions, QA back-translation,
    the numeric-literal guard, referent audit all continue to see the exact
    text committed by Pass 2) -- only the string actually handed to Azure for
    synthesis is affected. See pronunciation.py's PronunciationDictionary.apply.

    ``job_root``, when given AND already registered in
    ``_NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB`` (Phase 17), selects that
    job's own web-research-augmented dictionary instead of the plain
    job-agnostic default -- omitting it, or passing an unregistered job_root,
    preserves today's exact behavior byte-for-byte.
    """
    dictionary = _native_dub_pronunciation_dictionary()
    if job_root is not None:
        dictionary = _NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB.get(str(job_root), dictionary)
    return dictionary.apply(text).tts_text


def _synthesize_group(
    *,
    tts: AzureTTSBackend,
    group: Mapping[str, Any],
    voice_info: Mapping[str, Any],
    output_path: Path,
    preferred_ms: int,
    maximum_ms: int,
    force: bool,
    job_root: Path | None = None,
) -> AzureTTSResult:
    typed_parts = []
    plain_text: list[str] = []
    for part in group["parts"]:
        if part["type"] == "break":
            typed_parts.append(BreakPart(int(part["duration_ms"])))
        else:
            text = str(part["text"])
            typed_parts.append(TextPart(_native_dub_tts_ready_text(text, job_root=job_root)))
            plain_text.append(text)
    prosody = voice_info.get("base_prosody") if isinstance(voice_info.get("base_prosody"), Mapping) else {}
    # Rate is always exactly zero for this experimental architecture. Stable
    # pitch/volume persona is retained because it does not time-compress speech.
    pitch = bounded_prosody_int(prosody.get("pitch_percent"), default=0, lower=-10, upper=10)
    volume = bounded_prosody_int(prosody.get("volume_percent"), default=0, lower=-3, upper=3)
    request = AzureTTSRequest(
        turn_id=str(group["group_id"]),
        speaker_id=str(group["speaker_id"]),
        text=" ".join(plain_text),
        preferred_duration_ms=max(1, int(preferred_ms)),
        maximum_duration_ms=max(1, int(maximum_ms)),
        voice=str(voice_info["selected_voice"]),
        language="zu-ZA",
        rate_percent=0,
        pitch_percent=pitch,
        volume_percent=volume,
        parts=tuple(typed_parts),
    )
    return tts.synthesize(request, output_path, force=force)


def _synthesize_group_recovering_stale_cache(
    *,
    tts: AzureTTSBackend,
    group: Mapping[str, Any],
    voice_info: Mapping[str, Any],
    output_path: Path,
    preferred_ms: int,
    maximum_ms: int,
    force: bool,
    progress: Callable[[str], None] | None,
    job_root: Path | None = None,
) -> AzureTTSResult:
    """Synthesize a group's initial raw TTS, auto-recovering from a stale-cache
    idempotency conflict at this group's fixed output path.

    Phase A always writes to a FIXED path (``{group_id}.wav``), unlike the repair
    path's content-addressed naming (``{group_id}.repairN.{digest}.wav``) -- so
    when a group's committed text changes between runs (a translation fix, or --
    confirmed in production -- an island-derivation fix changing how a group's
    islands are cut), the cached artifact at that fixed path no longer matches the
    current request and ``AzureTTSBackend`` correctly refuses to silently
    overwrite it. Previously this propagated straight up and killed the entire
    render (confirmed twice in production this session, including immediately
    after landing the abbreviation-aware sentence-splitter fix). A
    ``request_mismatch`` conflict specifically means the STORED content is for
    genuinely different input, not that anything is corrupted -- regenerating is
    the correct, safe response, so this retries once with ``force=True`` instead
    of crashing. An ``artifact_integrity`` conflict (a rarer, different failure
    mode -- an incomplete or tampered write) is deliberately NOT auto-recovered:
    that signals a real problem worth surfacing, not silently papering over.
    """
    try:
        return _synthesize_group(
            tts=tts, group=group, voice_info=voice_info, output_path=output_path,
            preferred_ms=preferred_ms, maximum_ms=maximum_ms, force=force, job_root=job_root,
        )
    except AzureTTSIdempotencyError as exc:
        if force or exc.conflict_kind != "request_mismatch":
            raise
        _emit_red_warning(
            progress,
            f"[native dub] {group['group_id']}: cached synthesis at {output_path.name} is for "
            f"different content than the current committed text ({exc}); regenerating this one "
            "group instead of failing the whole render.",
        )
        return _synthesize_group(
            tts=tts, group=group, voice_info=voice_info, output_path=output_path,
            preferred_ms=preferred_ms, maximum_ms=maximum_ms, force=True, job_root=job_root,
        )


def _group_source_english(group: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get("source_text") or "").strip()
        for item in group.get("source_segments", [])
        if str(item.get("source_text") or "").strip()
    )


def _group_spoken_zulu(group: Mapping[str, Any]) -> str:
    return " ".join(
        str(part.get("text") or "").strip()
        for part in group.get("parts", [])
        if part.get("type") == "text" and str(part.get("text") or "").strip()
    )


def _protected_timing_literals(group: Mapping[str, Any]) -> list[str]:
    """Cheap deterministic guardrail: expose high-risk source literals to the recaster."""
    source = _group_source_english(group)
    if not source:
        return []
    found: list[str] = []
    # Numbers/dates/money/percentages, acronyms, month names, and multi-token
    # title-cased entity surfaces. Avoid protecting every sentence-initial word.
    protected_pattern = (
        r"(?<!\w)(?:R|\$)?\d[\d,.]*(?:%|st|nd|rd|th)?(?!\w)"
        r"|\b[A-Z]{2,}(?:-[A-Z]{2,})*\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
        r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    )
    for match in re.finditer(protected_pattern, source):
        value = match.group(0).strip().rstrip(",.")
        if value and value not in found:
            found.append(value)
    return found[:24]


def _source_supports_aposiopesis(group: Mapping[str, Any]) -> bool:
    source = _group_source_english(group).strip()
    if not source:
        return False
    if source.endswith(("...", "…", "--", "—", "-")):
        return True
    lowered = source.lower()
    return any(token in lowered for token in ("[interrupted]", "[trails off]", "[inaudible]"))


def _source_supports_auxesis(group: Mapping[str, Any]) -> bool:
    source = _group_source_english(group).strip()
    if not source:
        return False
    if "!" in source:
        return True
    lowered = source.lower()
    markers = (
        "very ", "really ", "absolutely", "extremely", "so urgent", "so much",
        "more and more", "worse", "best", "worst", "total catastrophe",
    )
    return any(marker in lowered for marker in markers)


def _load_raw_word_index_for_job(
    job_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]] | None:
    """Load word/segment lookups for speech-island derivation, or None if unavailable.

    Older jobs (or ones reconstructed from an already-authoritative transcript with
    no raw Azure diarization behind it) may not have this file. Island-level timing
    simply does not apply there -- every group falls back to whole-group handling,
    exactly as it worked before islands existed.
    """
    raw_path = Path(job_root) / "analysis" / "transcript_en_raw.json"
    if not raw_path.is_file():
        return None
    return load_raw_word_index(read_json(raw_path))


def _split_phrase_groups_with_islands(
    phrase_groups: Sequence[Mapping[str, Any]],
    word_index: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
    *,
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Expand groups into sentence-level island pseudo-groups where safely possible.

    Returns the flattened, chronologically-ordered list to hand directly to the
    existing measure-and-repair controller unmodified, plus a per-original-group
    island count (1 means the group was not split and is handled as a single unit,
    exactly as before).
    """
    flattened: list[dict[str, Any]] = []
    island_counts: dict[str, int] = {}
    if word_index is None:
        for group in phrase_groups:
            flattened.append(dict(group))
            island_counts[str(group["group_id"])] = 1
        return flattened, island_counts

    words_by_id, segments_by_id = word_index
    split_group_count = 0
    total_islands = 0
    for group in phrase_groups:
        group_id = str(group["group_id"])
        pseudo_groups = build_island_phrase_groups(group, words_by_id, segments_by_id)
        if pseudo_groups is None:
            flattened.append(dict(group))
            island_counts[group_id] = 1
            continue
        flattened.extend(pseudo_groups)
        island_counts[group_id] = len(pseudo_groups)
        split_group_count += 1
        total_islands += len(pseudo_groups)

    if split_group_count:
        _emit_progress(
            progress,
            f"[native dub] Speech islands: split {split_group_count}/{len(phrase_groups)} phrase group(s) "
            f"into {total_islands} sentence-level island(s) for per-sentence timing measurement and repair.",
        )
    return flattened, island_counts


def _island_comparison_record(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    """One comparable row: this unit's English source next to its final Zulu text and timing.

    Used for both real islands and whole (unsplit) groups, so a job's
    ``speech_islands.json`` gives a complete island-for-island (or, where a
    group wasn't split, group-for-group) comparison table across the entire
    job -- not just the subset that happened to split -- for reviewing
    translation quality/timing without re-deriving it from Pass 1/2 by hand.
    """
    source_window_ms = int(item.get("source_window_ms") or 1)
    natural_ms = int(item.get("natural_tts_duration_ms") or item["tts_duration_ms"])
    return {
        "island_id": item["group_id"],
        "index": index,
        "start_ms": int(item["start_ms"]),
        "source_span_ms": source_window_ms,
        "source_text": str(item.get("source_text") or ""),
        "spoken_text": _group_spoken_zulu(item),
        "alignment_method": item.get("alignment_method"),
        "natural_tts_duration_ms": natural_ms,
        "tts_duration_ms": int(item["tts_duration_ms"]),
        "required_rush_percent": round(max(0.0, (natural_ms / max(1, source_window_ms) - 1.0) * 100.0), 1),
        "direction": (
            "compress" if natural_ms > source_window_ms
            else "expand" if natural_ms < source_window_ms
            else None
        ),
        "timing_repair_used": bool(item.get("timing_repair_used")),
        "synthesis_method": item.get("synthesis_method", "rest_per_island"),
        "path": str(item["path"]),
    }


def _restitch_island_measurements(
    measured: Sequence[Mapping[str, Any]],
    island_counts: Mapping[str, int],
    *,
    audio_dir: Path,
    max_natural_speed_percent: int,
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge per-island measured entries back into one entry per original group.

    ``measured`` is whatever the existing measure-and-repair controller produced
    when run over the flattened island/whole-group list from
    ``_split_phrase_groups_with_islands``. Island entries carry ``parent_group_id``/
    ``island_index`` (see ``build_island_phrase_groups``); whole-group entries
    (islands weren't safely derivable for that group) carry neither and pass
    through unchanged, so every downstream consumer keeps seeing exactly one
    measured entry per real group_id, identical in shape to today.
    """
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for entry in measured:
        parent_id = str(entry.get("parent_group_id") or entry["group_id"])
        by_parent.setdefault(parent_id, []).append(entry)

    restitched: list[dict[str, Any]] = []
    island_records: list[dict[str, Any]] = []
    for group_id, entries in by_parent.items():
        if island_counts.get(group_id, 1) <= 1 or len(entries) <= 1:
            entry = entries[0]
            restitched.append(dict(entry))
            island_records.append({
                "group_id": group_id,
                "island_count": 1,
                "islands": [_island_comparison_record(entry, index=0)],
                "stitched_path": str(entry["path"]),
                "stitched_duration_ms": int(entry["tts_duration_ms"]),
            })
            continue

        ordered = sorted(entries, key=lambda item: int(item.get("island_index") or 0))
        island_paths = [Path(str(item["path"])) for item in ordered]
        stitched_output = audio_dir / f"{group_id}.islands-stitched.wav"
        stitch_result = stitch_islands_to_group_wav(island_paths, stitched_output)
        stitched_duration_ms = int(stitch_result["final_duration_ms"])

        first, last = ordered[0], ordered[-1]
        parent_start_ms = int(first.get("parent_start_ms") or first["start_ms"])
        parent_source_end_ms = int(last.get("parent_source_end_ms") or last["source_end_ms"])
        parent_source_span_ms = int(last.get("parent_source_span_ms") or (parent_source_end_ms - parent_start_ms))
        parent_speaker_id = str(first.get("parent_speaker_id") or first["speaker_id"])
        parent_segment_ids = list(first.get("parent_segment_ids") or [item["group_id"] for item in ordered])

        selected_path = stitched_output
        final_duration_ms = stitched_duration_ms
        speed_fit = last.get("speed_fit")
        late_tolerance_ms = int(last.get("mouth_close_late_tolerance_ms") or 0)
        early_tolerance_ms = int(first.get("mouth_close_early_tolerance_ms") or 0)
        speed_fit_mode = _speed_fit_mode_label(late_tolerance_ms)
        if stitched_duration_ms > parent_source_span_ms + late_tolerance_ms:
            # Same gap_aware/sentence_end target-extension as the main per-group render
            # path -- target the sentence's own end plus whatever safe gap already
            # exists before the next sentence, and shrink allowed_late_ms to a small
            # residual since the gap is now baked into the target itself.
            speed_fit = _speed_fit_overflow_wav(
                source_path=stitched_output,
                output_path=audio_dir / f"{group_id}.islands-stitched.hard-sync.wav",
                target_duration_ms=parent_source_span_ms + late_tolerance_ms,
                max_speed_percent=int(max_natural_speed_percent),
                allowed_late_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
                allowed_early_ms=early_tolerance_ms,
            )
            speed_fit["speed_fit_mode"] = speed_fit_mode
            selected_path = Path(str(speed_fit["output_path"]))
            final_duration_ms = int(speed_fit["final_duration_ms"])
            _emit_progress(
                progress,
                f"[native dub] {group_id} stitched islands +{float(speed_fit['speed_percent']):.1f}% "
                f"({speed_fit_mode}) safety speed fit -> {final_duration_ms} ms (window {parent_source_span_ms} ms)",
            )

        mouth_close_error_ms = final_duration_ms - parent_source_span_ms
        restitched.append({
            **last,
            "group_id": group_id,
            "speaker_id": parent_speaker_id,
            "segment_ids": parent_segment_ids,
            "start_ms": parent_start_ms,
            "source_end_ms": parent_source_end_ms,
            "source_window_ms": parent_source_span_ms,
            "source_span_ms": parent_source_span_ms,
            "path": str(selected_path),
            "tts_duration_ms": final_duration_ms,
            "speed_fit": speed_fit,
            "mouth_close_end_error_ms": mouth_close_error_ms,
            "island_split_used": True,
            "island_count": len(ordered),
        })
        island_records.append({
            "group_id": group_id,
            "island_count": len(ordered),
            "islands": [
                _island_comparison_record(item, index=int(item.get("island_index") or position))
                for position, item in enumerate(ordered)
            ],
            "stitched_path": str(selected_path),
            "stitched_duration_ms": final_duration_ms,
        })

    restitched.sort(key=lambda item: int(item["start_ms"]))
    return restitched, island_records


def _synthesize_speaker_turns_with_bookmarks(
    *,
    phrase_groups: Sequence[Mapping[str, Any]],
    source_duration_ms: int,
    voice_assignments: Mapping[str, Mapping[str, Any]],
    tts: AzureTTSBackend,
    sdk_boundary: AzureSpeechSDKBoundary,
    audio_dir: Path,
    preferred_raw_speed_percent: int,
    max_natural_speed_percent: int,
    mouth_close_lag_tolerance_ms: int,
    protected_gap_overflow: bool,
    max_protected_gap_overflow_ms: int,
    min_protected_pause_ms: int,
    force: bool,
    progress: Callable[[str], None] | None,
    job_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Phase 14: for every multi-member speaker turn, synthesize the WHOLE
    turn's already-final Zulu text in ONE bookmarked Speech SDK call (one
    voice throughout -- _build_speaker_turns only ever merges same-speaker
    sentences), make ONE speed-fit decision against the turn's own aggregate
    window, and slice the result back into real per-sentence WAVs.

    This exists because independent per-sentence fitting has a confirmed real
    defect: a turn's LAST member is judged against whatever comes right after
    it (often seconds away), lands in "gap_aware" mode, and drifts audibly
    past the true mouth-close while its siblings are compressed harder.
    Fitting the whole turn as one unit removes the per-member divergence at
    its source -- there is no longer a per-member fit decision to diverge.

    A bridge member (a short interjection like "All right.") is treated
    exactly like any other member here -- it does NOT disqualify its turn.
    Bridge classification only ever mattered for a sentence judged against
    its OWN tiny native window in isolation (a "collision-only" timing model
    was built specifically to spare it from that); once timing is decided at
    the aggregate-turn level instead, a bridge's own window is never
    independently judged at all, so the reason it needed different treatment
    disappears. That collision-only model stays real and necessary only for a
    bridge that IS an entire turn on its own (no same-speaker neighbor to
    merge with -- nothing to aggregate it into), which the `len(member_ids) >
    1` eligibility check below still correctly leaves on the independent
    path. An earlier version of this function excluded ANY turn containing a
    bridge member anywhere from one-shot synthesis entirely -- confirmed on a
    real job to silently revert 85+ otherwise-ordinary seconds of a turn back
    to the older per-sentence path over a single 3-word opener at its start.

    Returns {group_id: {"path", "voice", "natural_duration_ms",
    "final_duration_ms", "turn_speed_fit", "turn_mouth_close_sync_ok"}} for
    every member of every turn that fully finalized. A turn either fully
    finalizes or fully falls back -- never half-and-half, so no member is ever
    left with a stale internal boundary from a partially-completed turn. Any
    failure is scoped to the ONE turn that failed (caught, logged, skipped);
    those members simply flow through the existing independent per-sentence
    path in `_run_batch_adaptive_timing_controller`, exactly like
    `_presynthesize_islands_with_bookmarks`'s own per-group degrade pattern.
    """
    groups_by_id = {str(g["group_id"]): g for g in phrase_groups}
    turns = _build_speaker_turns(phrase_groups)
    finalized: dict[str, dict[str, Any]] = {}
    for turn_index, turn in enumerate(turns):
        all_member_ids = [str(gid) for gid in turn["member_group_ids"]]
        next_turn_start_ms = (
            int(turns[turn_index + 1]["start_ms"]) if turn_index + 1 < len(turns) else int(source_duration_ms)
        )
        # Every member of a turn is treated uniformly here, bridge or not: once
        # timing is decided at the aggregate-turn level rather than per
        # sentence, a bridge's own tiny native window never gets independently
        # judged at all, which is the entire reason it needed a different
        # ("collision-only") timing model in the first place -- that model
        # stays real and necessary only for a bridge that IS its own whole
        # turn (no same-speaker neighbor to merge with, so nothing to
        # aggregate it into), which `len(member_ids) > 1` below still
        # correctly leaves on the independent path. An earlier version of
        # this function excluded any turn containing a bridge ANYWHERE from
        # one-shot synthesis entirely, which silently reverted the other 85+
        # seconds of an otherwise-ordinary real turn to the older per-sentence
        # path over one 3-word opener -- confirmed on a real job.
        for run_index, member_ids in enumerate([all_member_ids]):
            if len(member_ids) <= 1:
                continue
            run_tag = f"{turn_index:04d}"
            next_index_in_turn = all_member_ids.index(member_ids[-1]) + 1
            next_source_start_ms = (
                int(groups_by_id[all_member_ids[next_index_in_turn]]["start_ms"])
                if next_index_in_turn < len(all_member_ids) else next_turn_start_ms
            )
            try:
                # Computed directly via the same primitive Phase A uses per-sentence
                # (_effective_protected_late_allowance), NOT the decision-time
                # _speaker_turn_geometry helper -- that helper hardcodes module-level
                # default tolerance constants, silently ignoring whatever this render
                # was actually configured with (mouth_close_lag_tolerance_ms,
                # max_protected_gap_overflow_ms, min_protected_pause_ms are all real,
                # render-time CLI-configurable values).
                turn_start_ms = int(groups_by_id[member_ids[0]]["start_ms"])
                turn_source_end_ms = int(groups_by_id[member_ids[-1]]["source_end_ms"])
                source_window_ms = max(1, turn_source_end_ms - turn_start_ms)
                early_tolerance_ms = int(mouth_close_lag_tolerance_ms)
                late_tolerance_ms = _effective_protected_late_allowance(
                    int(mouth_close_lag_tolerance_ms), turn_source_end_ms, next_source_start_ms,
                    protected_gap_overflow=bool(protected_gap_overflow),
                    max_protected_gap_overflow_ms=int(max_protected_gap_overflow_ms),
                    min_protected_pause_ms=int(min_protected_pause_ms),
                )
                free_gap_ms = max(0, next_source_start_ms - turn_source_end_ms)
                speaker_id = str(groups_by_id[member_ids[0]]["speaker_id"])
                voice_info = voice_assignments[speaker_id]
                prosody = voice_info.get("base_prosody") if isinstance(voice_info.get("base_prosody"), Mapping) else {}
                pitch = bounded_prosody_int(prosody.get("pitch_percent"), default=0, lower=-10, upper=10)
                volume = bounded_prosody_int(prosody.get("volume_percent"), default=0, lower=-3, upper=3)
                voice = str(voice_info["selected_voice"])
                # A per-clause variable-rate SSML mechanism (multiple sibling
                # <prosody> elements, one per punctuation-bounded clause) was
                # tried here and reverted (2026-09-03): a real A/B listening
                # test found it sounded audibly worse (subtle but confirmed
                # "random sluggishness and rushing") than a single flat
                # <prosody> wrapper around the whole turn -- and a follow-up
                # test found the single-wrapper version sounds IDENTICAL to no
                # <prosody> element at all. Splitting one continuous utterance
                # into multiple prosody-scoped SSML elements disrupts Azure's
                # own natural inter-clause prosodic continuity independent of
                # what rate values are chosen, so plain per-member text with
                # one flat pitch/volume for the whole turn is both simpler and
                # measurably more natural.
                islands = [
                    (gid, _native_dub_tts_ready_text(_group_spoken_zulu(groups_by_id[gid]), job_root=job_root))
                    for gid in member_ids
                ]
                # Bounded regardless of turn size -- see _measure_speaker_turn_total's
                # own comment: joining every member id overflows Windows' 255-char
                # filename limit once a real, uncapped turn grows past a dozen or so
                # sentences (confirmed by a real job run in Phase 13).
                output_path = audio_dir / f"turn_{member_ids[0]}_to_{member_ids[-1]}_n{len(member_ids)}{GROUP_WAV_SUFFIX}"
                group_result = synthesize_group_with_bookmarks(
                    sdk_boundary, islands, parent_group_id=f"turn{run_tag}",
                    voice=voice, allowed_voices=tts.configured_voices, output_path=output_path,
                    force=force, pitch_percent=pitch, volume_percent=volume,
                    sample_rate=tts.sample_rate, channels=tts.channels, sample_width=tts.sample_width,
                    ssml_bounds=tts.ssml_bounds,
                )
                # NOT write_group_slices (which hardcodes the default trailing-
                # silence trim, correct for the DIFFERENT island-presynthesis use
                # case but wrong here): confirmed real bug on job 33cd7b46...
                # trimming each interior member's own natural post-sentence pause
                # (Azure's real, deliberate silence between sentences in one
                # continuous take) silently deleted ~3.2s of real pacing across a
                # 6-member turn, so the final audio ran ~3s short of its own real
                # fitted target -- the turn's own real natural pauses are exactly
                # what should be PRESERVED here, not stripped, since the whole
                # point of one-shot synthesis is to keep them.
                natural_slices = slice_wav_by_bookmarks(
                    Path(group_result.group_wav_path), member_ids, group_result.bookmark_offsets_ms, audio_dir,
                    output_suffix=".turn-sync-slice.wav", trim_digital_silence_ms=0,
                )
                natural_duration_ms = int(group_result.group_duration_ms)

                direction = _timing_recast_direction(
                    natural_duration_ms, source_window_ms,
                    max_speed_percent=int(max_natural_speed_percent),
                    mouth_close_early_tolerance_ms=early_tolerance_ms,
                    mouth_close_late_tolerance_ms=late_tolerance_ms,
                )

                final_slices: Mapping[str, Mapping[str, Any]] = natural_slices
                final_duration_ms = natural_duration_ms
                if direction == "expand":
                    shortfall_ms = max(0, source_window_ms - natural_duration_ms)
                    if shortfall_ms <= MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS and shortfall_ms <= free_gap_ms:
                        padded_path = audio_dir / f"turn{run_tag}.silence-pad.wav"
                        pad_result = _pad_wav_with_trailing_silence(
                            source_path=Path(group_result.group_wav_path),
                            output_path=padded_path, target_duration_ms=source_window_ms,
                        )
                        final_duration_ms = int(pad_result["final_duration_ms"])
                        # Padding only ever extends the very tail, after the LAST
                        # member's real content -- every earlier member's boundary
                        # and audio is completely unaffected, so keep their already-
                        # correctly-trimmed natural slices unchanged. Only the last
                        # member is re-sliced, from the padded audio, WITHOUT
                        # trailing-silence trimming (trim_digital_silence_ms=0) --
                        # the padding just added IS real, deliberate trailing
                        # silence, and the default trim (meant to strip Azure's own
                        # unwanted inter-sentence pauses) would otherwise silently
                        # strip it right back off.
                        last_gid = member_ids[-1]
                        padded_slices = slice_wav_by_bookmarks(
                            padded_path, member_ids, group_result.bookmark_offsets_ms, audio_dir,
                            output_suffix=".turn-sync-slice.wav", trim_digital_silence_ms=0,
                        )
                        final_slices = dict(natural_slices)
                        final_slices[last_gid] = padded_slices[last_gid]
                        speed_fit = pad_result
                    else:
                        speed_fit = {
                            "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
                            "method": "none", "source_path": group_result.group_wav_path,
                            "output_path": group_result.group_wav_path, "before_duration_ms": natural_duration_ms,
                            "target_duration_ms": source_window_ms, "final_duration_ms": natural_duration_ms,
                            "speed_factor": 1.0, "speed_percent": 0.0,
                            "end_error_ms": natural_duration_ms - source_window_ms, "mouth_close_sync_ok": True,
                        }
                elif natural_duration_ms > source_window_ms:
                    # Target the TRUE window, not source_window_ms + late_tolerance_ms.
                    # Confirmed real user insight: gap-overflow tolerance and
                    # compression are not independent budgets to spend in parallel --
                    # once a turn has genuine content to compress (this branch only
                    # fires when it does), it has the headroom to close the WHOLE
                    # gap via compression, so it should demand tight mouth-close
                    # sync rather than settling for the looser, gap-borrowed target
                    # and drifting the remainder into trailing silence. Gap-overflow
                    # stays fully available in the OTHER branches (expand/fits) for
                    # short, bridge-like content that has no compression headroom to
                    # spend in the first place -- that's the real distinction, not a
                    # blanket "always prefer silence over compression" rule.
                    #
                    # A TTS-rate-first design (request the speed-up from Azure's own
                    # SSML rate before falling back to atempo) was tried and reverted
                    # (2026-09-03): a real A/B render showed no perceptible difference
                    # over plain atempo, and TTS-rate costs a genuine second Azure
                    # Speech SDK call per compressed turn versus atempo's free,
                    # already-synthesized-audio post-processing -- not worth the extra
                    # cost for an inaudible difference.
                    speed_fit = _pause_aware_speed_fit_wav(
                        source_path=Path(group_result.group_wav_path),
                        output_path=audio_dir / f"turn{run_tag}.hard-sync.wav",
                        target_duration_ms=source_window_ms,
                        max_speed_percent=int(max_natural_speed_percent),
                        allowed_late_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS, allowed_early_ms=early_tolerance_ms,
                    )
                    final_duration_ms = int(speed_fit["final_duration_ms"])
                    # Piecewise remap through the pause-aware segment map, NOT a flat
                    # ratio: gaps and speech were compressed by DIFFERENT factors (see
                    # _pause_aware_speed_fit_wav), so a single before/after ratio would
                    # misplace every bookmark after the first protected gap.
                    rescaled_offsets = {
                        gid: _remap_ms_through_segments(offset_ms, speed_fit["segments"])
                        for gid, offset_ms in group_result.bookmark_offsets_ms.items()
                    }
                    # trim_digital_silence_ms=0 -- same reasoning as the natural
                    # slices above: the real inter-sentence pauses (now uniformly
                    # compressed along with everything else) must survive into the
                    # final slices, or the concatenated total falls short of the
                    # real fitted target this atempo pass just computed.
                    final_slices = slice_wav_by_bookmarks(
                        Path(speed_fit["output_path"]), member_ids, rescaled_offsets, audio_dir,
                        output_suffix=".turn-sync-slice.wav", trim_digital_silence_ms=0,
                    )
                else:
                    speed_fit = {
                        "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
                        "method": "none", "source_path": group_result.group_wav_path,
                        "output_path": group_result.group_wav_path, "before_duration_ms": natural_duration_ms,
                        "target_duration_ms": source_window_ms, "final_duration_ms": natural_duration_ms,
                        "speed_factor": 1.0, "speed_percent": 0.0,
                        "end_error_ms": natural_duration_ms - source_window_ms,
                        "mouth_close_sync_ok": _mouth_close_sync_ok(
                            natural_duration_ms - source_window_ms,
                            early_tolerance_ms=early_tolerance_ms, late_tolerance_ms=late_tolerance_ms,
                        ),
                    }

                turn_mouth_close_sync_ok = bool(speed_fit.get("mouth_close_sync_ok", True))
                # Sync the turn's OUTER boundaries to the real lip movement (the first
                # member's true source start, the last member's real audio end) --
                # interior boundaries between members are NOT real visual events (a
                # mouth doesn't stop and restart between sentences in one continuous
                # utterance), so they're free to reflow to match the real, natural
                # internal pacing of the one true synthesis, rather than being pinned
                # to Phase 13's independent-REST-based per-sentence ESTIMATE. Placing
                # each slice at that stale estimate instead of its own real position is
                # exactly what produced a confirmed real defect on the first attempt at
                # this mechanism: audible silence gaps appearing INSIDE a turn where
                # the audio used to run continuously (up to 1015ms on job 33cd7b46...).
                cursor_ms = turn_start_ms
                for gid in member_ids:
                    slice_info = final_slices[gid]
                    slice_path = Path(slice_info["path"] if isinstance(slice_info, Mapping) else slice_info.path)
                    natural_info = natural_slices[gid]
                    natural_ms = int(
                        natural_info["duration_ms"] if isinstance(natural_info, Mapping) else natural_info.duration_ms
                    )
                    slice_ms = int(
                        slice_info["duration_ms"] if isinstance(slice_info, Mapping) else slice_info.duration_ms
                    )
                    member_start_ms = cursor_ms
                    cursor_ms += slice_ms
                    finalized[gid] = {
                        "path": str(slice_path),
                        "voice": voice,
                        "natural_duration_ms": natural_ms,
                        "final_duration_ms": slice_ms,
                        "start_ms": member_start_ms,
                        "source_end_ms": cursor_ms,
                        "turn_speed_fit": dict(speed_fit),
                        "turn_mouth_close_sync_ok": turn_mouth_close_sync_ok,
                        "request_hash": group_result.request_hash,
                    }
                _emit_progress(
                    progress,
                    f"[native dub] Turn synthesis: {len(member_ids)} sentence(s) "
                    f"({member_ids[0]}..{member_ids[-1]}) synthesized in one call, "
                    f"direction={direction or 'fit'}, speed={float(speed_fit.get('speed_percent') or 0.0):+.1f}%, "
                    f"saving {max(0, len(member_ids) - 1)} redundant Azure call(s).",
                )
            except Exception as exc:  # noqa: BLE001 - a turn-level failure must degrade, never crash the job
                _emit_red_warning(
                    progress,
                    f"[native dub] Turn synthesis FAILED for {member_ids[0]}..{member_ids[-1]} ({exc}); "
                    "falling back to independent per-sentence synthesis for this turn.",
                )
                for gid in member_ids:
                    finalized.pop(gid, None)
                continue
    return finalized


def _presynthesize_islands_with_bookmarks(
    *,
    timing_input_groups: Sequence[Mapping[str, Any]],
    island_counts: Mapping[str, int],
    voice_assignments: Mapping[str, Mapping[str, Any]],
    tts: AzureTTSBackend,
    sdk_boundary: AzureSpeechSDKBoundary,
    audio_dir: Path,
    force: bool,
    progress: Callable[[str], None] | None,
    job_root: Path | None = None,
) -> dict[str, AzureTTSResult]:
    """Pre-synthesize every multi-island group in one bookmarked Speech SDK call each.

    Runs once, before Phase A of ``_run_batch_adaptive_timing_controller``, and pays
    Azure's fixed per-utterance overhead once per group instead of once per island
    (see azure_tts_sdk.py's module docstring). Any failure is scoped to the ONE group
    that failed: that group's islands simply aren't in the returned map, so they fall
    through to today's unmodified per-island REST path in Phase A. This pre-pass must
    never hard-fail the job -- it is a pure optimization over an already-correct path.
    """
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for island in timing_input_groups:
        parent_id = str(island.get("parent_group_id") or "")
        if not parent_id or int(island_counts.get(parent_id, 1)) <= 1:
            continue
        by_parent.setdefault(parent_id, []).append(island)

    presynthesized: dict[str, AzureTTSResult] = {}
    for parent_id, islands in by_parent.items():
        ordered = sorted(islands, key=lambda item: int(item.get("island_index", 0)))
        speaker_id = str(ordered[0]["speaker_id"])
        voice_info = voice_assignments[speaker_id]
        prosody = voice_info.get("base_prosody") if isinstance(voice_info.get("base_prosody"), Mapping) else {}
        pitch = bounded_prosody_int(prosody.get("pitch_percent"), default=0, lower=-10, upper=10)
        volume = bounded_prosody_int(prosody.get("volume_percent"), default=0, lower=-3, upper=3)
        voice = str(voice_info["selected_voice"])
        island_texts = [
            (str(item["group_id"]), _native_dub_tts_ready_text(_group_spoken_zulu(item), job_root=job_root))
            for item in ordered
        ]

        try:
            group_result = synthesize_group_with_bookmarks(
                sdk_boundary,
                island_texts,
                parent_group_id=parent_id,
                voice=voice,
                allowed_voices=tts.configured_voices,
                output_path=audio_dir / f"{parent_id}{GROUP_WAV_SUFFIX}",
                force=force,
                pitch_percent=pitch,
                volume_percent=volume,
                sample_rate=tts.sample_rate,
                channels=tts.channels,
                sample_width=tts.sample_width,
                ssml_bounds=tts.ssml_bounds,
            )
            slices = write_group_slices(group_result, audio_dir, force=force)
        except Exception as exc:  # noqa: BLE001 - any failure here must degrade, never crash the job
            _emit_red_warning(
                progress,
                f"[native dub] SDK bookmark pre-pass FAILED for {parent_id} ({exc}); "
                "falling back to per-island REST synthesis for this group.",
            )
            continue

        _emit_progress(
            progress,
            f"[native dub] SDK bookmark pre-pass: {parent_id} ({len(ordered)} islands) synthesized in one call, "
            f"saving {max(0, len(ordered) - 1)} redundant Azure call(s).",
        )
        for item in ordered:
            island_id = str(item["group_id"])
            slice_result = slices[island_id]
            zulu_text = _group_spoken_zulu(item)
            duration_ms = max(1, int(slice_result.duration_ms))
            # preferred/maximum_duration_ms and warnings/ssml_* fields below are never
            # read by native_dub.py's own control flow (only duration_ms, output_path,
            # and voice are) -- they exist to satisfy AzureTTSResult's dataclass
            # contract with honest values, not to drive downstream repair logic.
            presynthesized[island_id] = AzureTTSResult(
                backend="azure_tts_sdk_bookmark_group",
                turn_id=island_id,
                speaker_id=speaker_id,
                voice=voice,
                language="zu-ZA",
                input_text_hash=hashlib.sha256(zulu_text.encode("utf-8")).hexdigest(),
                ssml_hash=group_result.request_hash,
                output_path=slice_result.path,
                sample_rate=tts.sample_rate,
                channels=tts.channels,
                sample_width=tts.sample_width,
                duration_ms=duration_ms,
                preferred_duration_ms=duration_ms,
                maximum_duration_ms=duration_ms,
                rate_percent=0,
                attempt=group_result.attempt,
                sha256=slice_result.sha256,
                warnings=(),
                request_hash=group_result.request_hash,
                ssml_path=group_result.ssml_path,
                manifest_path=slice_result.manifest_path,
                ssml_summary=(
                    f"sdk_bookmark_group parent={parent_id} voice={voice}; "
                    f"islands={len(ordered)}; pitch={pitch:+d}%; volume={volume:+d}%"
                ),
                idempotent_reuse=bool(group_result.idempotent_reuse or slice_result.idempotent_reuse),
            )
    return presynthesized


def _run_batch_adaptive_timing_controller(
    *,
    phrase_groups: Sequence[Mapping[str, Any]],
    source_duration_ms: int,
    voice_assignments: Mapping[str, Mapping[str, Any]],
    tts: AzureTTSBackend,
    context_ledger: Mapping[str, Any],
    audio_dir: Path,
    preferred_raw_speed_percent: int,
    max_natural_speed_percent: int,
    mouth_close_lag_tolerance_ms: int,
    protected_gap_overflow: bool,
    max_protected_gap_overflow_ms: int,
    min_protected_pause_ms: int,
    force: bool,
    progress: Callable[[str], None] | None,
    presynthesized: Mapping[str, AzureTTSResult] | None = None,
    turn_finalized: Mapping[str, Mapping[str, Any]] | None = None,
    job_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure all phrases, accepting both directions non-fatally: overlong speech is
    pitch-preserving speed-fit (rushed) and undersized speech plays at its own
    natural pace (no wording repair or expansion is attempted for either direction).

    ``turn_finalized`` (Phase 14) carries per-group records already fully
    synthesized AND speed-fit as part of a whole speaker turn (see
    `_synthesize_speaker_turns_with_bookmarks`) -- a group_id present here is
    NEVER also in `presynthesized` (turn-finalized supersedes it). Phase A
    skips synthesis entirely for these groups; Phase C skips its own
    direction/speed-fit decision entirely for them too, since re-fitting an
    already turn-fitted slice against its own narrow window would silently
    reproduce the exact "two independently-safe mechanisms compounding" defect
    Phase 13.1 was reverted for -- see that section's own docstring.
    """
    states: list[dict[str, Any]] = []
    phrase_count = len(phrase_groups)
    presynthesized = presynthesized or {}
    turn_finalized = turn_finalized or {}
    max_speed_factor = 1.0 + int(max_natural_speed_percent) / 100.0

    # Phase A: canonical raw TTS for the entire dub. No AI call is made here.
    for index, group in enumerate(phrase_groups):
        phrase_number = index + 1
        source_window_ms = int(group["source_span_ms"])
        if source_window_ms <= 0:
            raise RuntimeError(f"{group['group_id']} has an invalid source timing window")
        next_source_start_ms = (
            int(phrase_groups[index + 1]["start_ms"])
            if index + 1 < phrase_count else int(source_duration_ms)
        )
        early_tolerance_ms = int(mouth_close_lag_tolerance_ms)
        baseline_late_tolerance_ms = _effective_mouth_close_tolerance(
            int(mouth_close_lag_tolerance_ms), int(group["source_end_ms"]), next_source_start_ms
        )
        # A "bridge" sentence (a short interjection like "All right.") is judged only
        # on whether it collides with its neighbors, never on fitting its own tiny
        # native window -- see Phase 9. Handing _effective_protected_late_allowance a
        # much larger overflow budget achieves exactly that, safely: its result is
        # always clamped to the real free gap, so this can never overshoot into a
        # neighbor's own speech regardless of the override's size.
        effective_max_protected_gap_overflow_ms = (
            BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS
            if str(group.get("discourse_unit_kind") or "") == "bridge"
            else int(max_protected_gap_overflow_ms)
        )
        late_tolerance_ms = _effective_protected_late_allowance(
            int(mouth_close_lag_tolerance_ms),
            int(group["source_end_ms"]),
            next_source_start_ms,
            protected_gap_overflow=bool(protected_gap_overflow),
            max_protected_gap_overflow_ms=effective_max_protected_gap_overflow_ms,
            min_protected_pause_ms=int(min_protected_pause_ms),
        )
        target_raw_ms = int(round(source_window_ms * (1.0 + int(preferred_raw_speed_percent) / 100.0)))
        maximum_raw_rescuable_ms = int(round((source_window_ms + late_tolerance_ms) * max_speed_factor))
        _emit_progress(
            progress,
            f"[native dub] TTS {phrase_number}/{phrase_count} "
            f"({int(round((phrase_number - 1) * 100 / max(1, phrase_count)))}%): "
            f"{group['group_id']} ({group['speaker_id']}) — mouth window "
            f"{source_window_ms / 1000.0:.2f}s; raw target {target_raw_ms / 1000.0:.2f}s "
            f"(+{int(preferred_raw_speed_percent)}%)",
        )
        voice_info = voice_assignments[group["speaker_id"]]
        turn_entry = turn_finalized.get(str(group["group_id"]))
        if turn_entry is not None:
            result = AzureTTSResult(
                backend="azure_tts_sdk_bookmark_group", turn_id=str(group["group_id"]),
                speaker_id=str(group["speaker_id"]), voice=str(turn_entry["voice"]), language="zu-ZA",
                input_text_hash="", ssml_hash=str(turn_entry.get("request_hash") or ""),
                output_path=str(turn_entry["path"]), sample_rate=tts.sample_rate, channels=tts.channels,
                sample_width=tts.sample_width, duration_ms=int(turn_entry["natural_duration_ms"]),
                preferred_duration_ms=int(turn_entry["natural_duration_ms"]),
                maximum_duration_ms=int(turn_entry["natural_duration_ms"]), rate_percent=0, attempt=1,
                sha256="", warnings=(), request_hash=str(turn_entry.get("request_hash") or ""),
                ssml_path="", manifest_path="", ssml_summary="sdk_turn_group",
            )
            synth_wall_seconds = 0.0
        elif group["group_id"] in presynthesized:
            result = presynthesized[group["group_id"]]
            synth_wall_seconds = 0.0
        else:
            result, synth_wall_seconds = _run_with_progress_heartbeat(
                progress=progress,
                label=f"[native dub] TTS {phrase_number}/{phrase_count} {group['group_id']}",
                operation=lambda group=group, voice_info=voice_info, target_raw_ms=target_raw_ms, maximum_raw_rescuable_ms=maximum_raw_rescuable_ms: _synthesize_group_recovering_stale_cache(
                    tts=tts,
                    group=group,
                    voice_info=voice_info,
                    output_path=audio_dir / f"{group['group_id']}.wav",
                    preferred_ms=target_raw_ms,
                    maximum_ms=maximum_raw_rescuable_ms,
                    force=force,
                    progress=progress,
                    job_root=job_root,
                ),
            )
        _emit_progress(
            progress,
            f"[native dub] TTS {phrase_number}/{phrase_count} "
            f"({int(round(phrase_number * 100 / max(1, phrase_count)))}%) complete in "
            f"{synth_wall_seconds:.1f}s; raw={result.duration_ms / 1000.0:.2f}s "
            f"({(int(result.duration_ms) / source_window_ms - 1.0) * 100:+.1f}% vs mouth window)",
        )
        state = {
            "index": index,
            "canonical_group": group,
            "chosen_group": group,
            "canonical_result": result,
            "chosen_result": result,
            "voice_info": voice_info,
            "source_window_ms": source_window_ms,
            "next_source_start_ms": next_source_start_ms,
            "early_tolerance_ms": early_tolerance_ms,
            "baseline_late_tolerance_ms": baseline_late_tolerance_ms,
            "late_tolerance_ms": late_tolerance_ms,
            "free_gap_ms": max(0, next_source_start_ms - int(group["source_end_ms"])),
            "protected_gap_overflow_budget_ms": max(0, late_tolerance_ms - baseline_late_tolerance_ms),
            "target_raw_ms": target_raw_ms,
            "maximum_raw_rescuable_ms": maximum_raw_rescuable_ms,
            "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
            "max_natural_speed_percent": int(max_natural_speed_percent),
            "repair_rounds": [],
            "unresolved_early_mouth_close_ms": 0,
            "is_turn_finalized": turn_entry is not None,
            "turn_entry": turn_entry,
        }
        states.append(state)
        if turn_entry is not None:
            # Already fitted as part of a whole-turn decision (see
            # _synthesize_speaker_turns_with_bookmarks) -- judging this member
            # against its own narrow window here would be exactly the framing
            # that mechanism exists to discard, and Phase C skips it entirely
            # for this group too (see below).
            continue
        direction = _timing_recast_direction(
            int(result.duration_ms), source_window_ms,
            max_speed_percent=int(max_natural_speed_percent),
            mouth_close_early_tolerance_ms=early_tolerance_ms,
            mouth_close_late_tolerance_ms=late_tolerance_ms,
        )
        if direction == "compress":
            required_speed = _temporal_required_rush_percent(
                int(result.duration_ms), source_window_ms + late_tolerance_ms
            )
            _emit_red_warning(
                progress,
                f"[native dub] WARNING {group['group_id']}: natural speech would require +{required_speed:.1f}% "
                f"speed (> +{int(max_natural_speed_percent)}% quality threshold); no compression repair is required.",
            )
        elif direction is not None:
            _emit_native_timing_failure_diagnostic(
                progress=progress, stage="initial measured failure", group=group,
                measured_duration_ms=int(result.duration_ms), source_window_ms=source_window_ms,
                target_raw_ms=target_raw_ms, early_tolerance_ms=early_tolerance_ms,
                late_tolerance_ms=late_tolerance_ms, max_speed_percent=int(max_natural_speed_percent),
                timing_direction=direction, next_source_start_ms=next_source_start_ms,
            )

    # Phase C: resolve safe early misses, reject unsafe late misses, then speed-fit.
    measured: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for state in states:
        group = state["canonical_group"]
        chosen_group = state["chosen_group"]
        chosen_result = state["chosen_result"]
        source_window_ms = int(state["source_window_ms"])
        if state.get("is_turn_finalized"):
            # Phase 14: this group's fit decision was already made once, for the
            # WHOLE turn (see _synthesize_speaker_turns_with_bookmarks). Never
            # re-decide or re-process its audio here -- only build the same
            # output-dict shape every other group produces, from the already-
            # final numbers, so downstream consumers can't tell the difference.
            entry = state["turn_entry"]
            final_duration_ms = int(entry["final_duration_ms"])
            natural_duration_ms = int(entry["natural_duration_ms"])
            # Real cumulative placement within the turn's own real, natural
            # synthesis (see _synthesize_speaker_turns_with_bookmarks) -- NOT the
            # stale independent-REST-based per-sentence estimate `group` still
            # carries. Only the turn's OUTER boundaries (first member's real
            # start, last member's real end) are synced to the true lip
            # movement; interior boundaries reflow to the real audio, which is
            # what keeps a turn playing as one continuous, gap-free utterance.
            member_start_ms = int(entry["start_ms"])
            member_source_end_ms = int(entry["source_end_ms"])
            member_source_window_ms = max(1, member_source_end_ms - member_start_ms)
            mouth_close_error_ms = final_duration_ms - member_source_window_ms
            speed_fit = dict(entry["turn_speed_fit"])
            speed_fit["speed_fit_mode"] = _speed_fit_mode_label(int(state["late_tolerance_ms"]))
            measured.append({
                **chosen_group, "path": str(entry["path"]), "voice": str(entry["voice"]),
                "start_ms": member_start_ms,
                "canonical_raw_tts_duration_ms": natural_duration_ms,
                "natural_tts_duration_ms": natural_duration_ms, "tts_duration_ms": final_duration_ms,
                "source_window_ms": member_source_window_ms, "source_end_ms": member_source_end_ms,
                "source_span_ms": member_source_window_ms,
                "raw_target_duration_ms": int(state["target_raw_ms"]),
                "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
                "raw_speed_percent": round((natural_duration_ms / member_source_window_ms - 1.0) * 100.0, 3),
                "natural_overrun_ms": max(0, natural_duration_ms - member_source_window_ms),
                "source_gap_borrowed_ms": max(0, mouth_close_error_ms),
                "protected_gap_overflow_ms": max(0, mouth_close_error_ms),
                "protected_gap_overflow_budget_ms": int(state.get("protected_gap_overflow_budget_ms") or 0),
                "protected_pause_remaining_ms": max(0, int(state.get("free_gap_ms") or 0) - max(0, mouth_close_error_ms)),
                "next_block_shift_ms": 0, "tts_rate_percent": 0,
                "speed_fit": speed_fit, "mouth_close_end_error_ms": mouth_close_error_ms,
                "mouth_close_tolerance_ms": int(mouth_close_lag_tolerance_ms),
                "mouth_close_early_tolerance_ms": int(state["early_tolerance_ms"]),
                "mouth_close_late_tolerance_ms": int(state["late_tolerance_ms"]),
                # Reflects the TURN's own aggregate fit result, not this member's
                # individually-recomputed number -- an interior member can look
                # "out of sync" against its own narrow window even when the
                # turn's shared speed factor is correctly optimized for the
                # whole turn, and that must not spuriously trigger human review.
                "mouth_close_sync_ok": bool(entry.get("turn_mouth_close_sync_ok", True)),
                "unresolved_early_mouth_close_ms": 0,
                "timing_repair_used": False,
                "hard_window_fit": mouth_close_error_ms <= int(state["late_tolerance_ms"]),
                "synthesis_method": "sdk_turn_group",
            })
            continue
        direction = _timing_recast_direction(
            int(chosen_result.duration_ms), source_window_ms,
            max_speed_percent=int(max_natural_speed_percent),
            mouth_close_early_tolerance_ms=int(state["early_tolerance_ms"]),
            mouth_close_late_tolerance_ms=int(state["late_tolerance_ms"]),
        )
        silence_pad_applied: dict[str, Any] | None = None
        if direction == "expand":
            shortfall_ms = max(0, source_window_ms - int(chosen_result.duration_ms))
            state["unresolved_early_mouth_close_ms"] = shortfall_ms
            free_gap_ms = int(state.get("free_gap_ms") or 0)
            if shortfall_ms <= MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS and shortfall_ms <= free_gap_ms:
                # Closing this last fraction of a syllable's worth of timing through
                # wording is not achievable -- any real edit adds or removes at least a
                # full syllable, overshooting a gap this small either way. Close it with
                # real audio instead, the same way overlong speech is closed with a real
                # speed fit rather than more wording.
                pad_result = _pad_wav_with_trailing_silence(
                    source_path=Path(str(chosen_result.output_path)),
                    output_path=audio_dir / f"{chosen_group['group_id']}.silence-pad.wav",
                    target_duration_ms=source_window_ms,
                )
                chosen_result = replace(
                    chosen_result,
                    output_path=str(pad_result["output_path"]),
                    duration_ms=int(pad_result["final_duration_ms"]),
                )
                state["chosen_result"] = chosen_result
                state["unresolved_early_mouth_close_ms"] = 0
                silence_pad_applied = pad_result
                _emit_progress(
                    progress,
                    f"[native dub] {chosen_group['group_id']} padded +{shortfall_ms} ms trailing silence "
                    f"to close a residual shortfall too small for wording repair to hit precisely "
                    f"(window {source_window_ms} ms, free gap {free_gap_ms} ms available)",
                )
            else:
                # Natural speech is genuinely shorter than its window. Mathula no
                # longer forces wording repair or expansion to close this gap -- a
                # shorter, faithful, natural-paced rendering is preferred over
                # padding wording just to fill time (the rhetorical-expansion
                # mechanism this used to trigger was the exact root cause of a real
                # production bug where an island's expansion repair restated most
                # of a sibling island's content to hit a syllable floor). Play at
                # natural speed and accept the early mouth-close; timing in this
                # direction is now non-fatal, symmetric with "compress".
                _emit_red_warning(
                    progress,
                    f"[native dub] {chosen_group['group_id']}: natural speech is {shortfall_ms} ms shorter than "
                    "its window; playing at natural speed instead of stretching or padding wording. Timing "
                    "remains non-fatal and the pipeline continues.",
                )
        elif direction == "compress":
            required_speed = _temporal_required_rush_percent(
                int(chosen_result.duration_ms), source_window_ms + int(state["late_tolerance_ms"])
            )
            _emit_red_warning(
                progress,
                f"[native dub] WARNING {group['group_id']}: requires +{required_speed:.1f}% speed "
                f"(> +{int(max_natural_speed_percent)}% quality threshold); rendering rushed and continuing.",
            )
            # Deliberately fall through to unbounded pitch-preserving speed fit.

        selected_path = Path(str(chosen_result.output_path))
        final_duration_ms = int(chosen_result.duration_ms)
        speed_fit = {
            "schema_version": "mathula-native-hard-sync-speed-v2-mouth-close-tolerance",
            "method": "none", "source_path": str(selected_path), "output_path": str(selected_path),
            "before_duration_ms": final_duration_ms, "target_duration_ms": source_window_ms,
            "allowed_early_ms": int(state["early_tolerance_ms"]), "allowed_late_ms": int(state["late_tolerance_ms"]),
            "final_duration_ms": final_duration_ms, "speed_factor": 1.0, "speed_percent": 0.0,
            "end_error_ms": final_duration_ms - source_window_ms,
            "mouth_close_sync_ok": _mouth_close_sync_ok(final_duration_ms - source_window_ms, early_tolerance_ms=int(state["early_tolerance_ms"]), late_tolerance_ms=int(state["late_tolerance_ms"])),
            # No speed change was needed at all here -- either it genuinely fit the bare
            # sentence end (sentence_end) or it only fit once the safe gap before the next
            # sentence is counted (gap_aware), same mode-labeling rule as the actual
            # speed-fit branch below.
            "speed_fit_mode": _speed_fit_mode_label(state["late_tolerance_ms"]),
        }
        if silence_pad_applied is not None:
            speed_fit = silence_pad_applied
        elif final_duration_ms > source_window_ms + int(state["late_tolerance_ms"]):
            # Speed-fit target is the sentence's own end PLUS the already-tracked safe
            # gap before the next sentence (mode 2, "gap_aware") -- if a gap exists, some
            # of the overflow is absorbed by existing silence instead of pure
            # acceleration; with no gap, this is identical to the old strict target
            # (mode 1, "sentence_end"). allowed_late_ms shrinks to a small residual
            # quantization tolerance here since the gap is now baked into the target
            # itself -- passing the full gap again would double-count it and risk
            # landing into the next sentence's actual speech.
            speed_fit = _speed_fit_overflow_wav(
                source_path=selected_path, output_path=audio_dir / f"{group['group_id']}.hard-sync.wav",
                target_duration_ms=source_window_ms + int(state["late_tolerance_ms"]),
                max_speed_percent=int(max_natural_speed_percent),
                allowed_late_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS, allowed_early_ms=int(state["early_tolerance_ms"]),
            )
            speed_fit["speed_fit_mode"] = _speed_fit_mode_label(state["late_tolerance_ms"])
            selected_path = Path(str(speed_fit["output_path"]))
            final_duration_ms = int(speed_fit["final_duration_ms"])
            _emit_progress(progress, f"[native dub] {group['group_id']} adaptive speed +{float(speed_fit['speed_percent']):.1f}% ({speed_fit['speed_fit_mode']}) -> end error {int(speed_fit['end_error_ms']):+d} ms (mouth tolerance -{int(state['early_tolerance_ms'])}/+{int(state['late_tolerance_ms'])} ms)")
            if bool(speed_fit.get("speed_warning_exceeded")):
                _emit_red_warning(
                    progress,
                    f"[native dub] WARNING {group['group_id']}: rendered at +{float(speed_fit['speed_percent']):.1f}% "
                    f"speed, above +{int(max_natural_speed_percent)}% quality threshold; pipeline continues.",
                )
        mouth_close_error_ms = final_duration_ms - source_window_ms
        if mouth_close_error_ms > int(state["late_tolerance_ms"]):
            _emit_red_warning(
                progress,
                f"[native dub] WARNING {group['group_id']}: mouth-close remains {mouth_close_error_ms} ms late after "
                "speed fit; timing remains non-fatal and the pipeline continues.",
            )
        measured.append({
            **chosen_group, "path": str(selected_path), "voice": chosen_result.voice,
            "canonical_raw_tts_duration_ms": int(state["canonical_result"].duration_ms),
            "natural_tts_duration_ms": int(chosen_result.duration_ms), "tts_duration_ms": final_duration_ms,
            "source_window_ms": source_window_ms, "source_end_ms": int(group["source_end_ms"]),
            "raw_target_duration_ms": int(state["target_raw_ms"]), "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
            "raw_speed_percent": round((int(chosen_result.duration_ms) / source_window_ms - 1.0) * 100.0, 3),
            "natural_overrun_ms": max(0, int(chosen_result.duration_ms) - source_window_ms),
            "source_gap_borrowed_ms": max(0, mouth_close_error_ms),
            "protected_gap_overflow_ms": max(0, mouth_close_error_ms),
            "protected_gap_overflow_budget_ms": int(state.get("protected_gap_overflow_budget_ms") or 0),
            "protected_pause_remaining_ms": max(0, int(state.get("free_gap_ms") or 0) - max(0, mouth_close_error_ms)),
            "next_block_shift_ms": 0, "tts_rate_percent": 0,
            "speed_fit": speed_fit, "mouth_close_end_error_ms": mouth_close_error_ms,
            "mouth_close_tolerance_ms": int(mouth_close_lag_tolerance_ms),
            "mouth_close_early_tolerance_ms": int(state["early_tolerance_ms"]),
            "mouth_close_late_tolerance_ms": int(state["late_tolerance_ms"]),
            "mouth_close_sync_ok": _mouth_close_sync_ok(mouth_close_error_ms, early_tolerance_ms=int(state["early_tolerance_ms"]), late_tolerance_ms=int(state["late_tolerance_ms"])),
            "unresolved_early_mouth_close_ms": int(state["unresolved_early_mouth_close_ms"]),
            "timing_repair_used": chosen_group is not group,
            "hard_window_fit": mouth_close_error_ms <= int(state["late_tolerance_ms"]),
            "synthesis_method": (
                "sdk_bookmark_group" if chosen_result.backend == "azure_tts_sdk_bookmark_group" else "rest_per_island"
            ),
        })
        if state["repair_rounds"]:
            repair_records.append({
                "group_id": group["group_id"], "source_window_ms": source_window_ms,
                "target_raw_duration_ms": int(state["target_raw_ms"]),
                "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
                "canonical_duration_ms": int(state["canonical_result"].duration_ms),
                "final_natural_duration_ms": int(chosen_result.duration_ms),
                "max_natural_speed_percent": int(max_natural_speed_percent),
                "mouth_close_early_tolerance_ms": int(state["early_tolerance_ms"]),
                "mouth_close_late_tolerance_ms": int(state["late_tolerance_ms"]),
                "rounds": state["repair_rounds"], "accepted": chosen_group is not group,
                "token_strategy": "mixed_failure_batch_plus_cached_context_ledger",
            })

    measured.sort(key=lambda item: int(item["start_ms"]))
    return measured, repair_records


def _emit_native_timing_failure_diagnostic(
    *,
    progress: Callable[[str], None] | None,
    stage: str,
    group: Mapping[str, Any],
    measured_duration_ms: int,
    source_window_ms: int,
    target_raw_ms: int,
    early_tolerance_ms: int,
    late_tolerance_ms: int,
    max_speed_percent: int,
    timing_direction: str,
    next_source_start_ms: int,
) -> None:
    measured = int(measured_duration_ms)
    window = max(1, int(source_window_ms))
    error_ms = measured - window
    required_speed_percent = max(0.0, (measured / window - 1.0) * 100.0)
    source_end_ms = int(group.get("source_end_ms") or (int(group.get("start_ms") or 0) + window))
    free_gap_ms = max(0, int(next_source_start_ms) - source_end_ms)
    english = _group_source_english(group) or "<missing source text>"
    zulu = _group_spoken_zulu(group) or "<missing spoken text>"
    _emit_progress(
        progress,
        "\x1b[31m"
        f"[native dub] TIMING FAILURE {group.get('group_id')} — {stage}\n"
        f"  speaker: {group.get('speaker_id')} | segments: {', '.join(map(str, group.get('segment_ids', [])))}\n"
        f"  English: {english}\n"
        f"  isiZulu: {zulu}\n"
        f"  timing: source={window} ms ({window / 1000.0:.3f}s), "
        f"measured={measured} ms ({measured / 1000.0:.3f}s), "
        f"end_error={error_ms:+d} ms, raw_target={int(target_raw_ms)} ms, "
        f"tolerance=-{int(early_tolerance_ms)}/+{int(late_tolerance_ms)} ms\n"
        f"  boundary: source_end={source_end_ms} ms, next_onset={int(next_source_start_ms)} ms, "
        f"free_gap={free_gap_ms} ms | direction={timing_direction}, "
        f"required_speed={required_speed_percent:+.2f}%, max_speed=+{int(max_speed_percent)}%"
        "\x1b[0m",
    )


def _schedule_source_timeline(
    measured: Sequence[Mapping[str, Any]],
    *,
    min_same_speaker_rest_ms: int = 0,
) -> list[dict[str, Any]]:
    """Schedule every phrase at its immutable source start; never accumulate lag.

    ``min_same_speaker_rest_ms`` is retained only for call compatibility. A phrase
    may already contain a bounded protected-gap overflow approved by the timing
    controller, but the next phrase always keeps its immutable source onset.
    """
    scheduled: list[dict[str, Any]] = []
    for item in measured:
        source_start = int(item["start_ms"])
        source_end = int(item.get("source_end_ms") or (source_start + int(item["source_span_ms"])))
        duration_ms = int(item["tts_duration_ms"])
        end = source_start + duration_ms
        late_allowance_ms = int(
            item.get("mouth_close_late_tolerance_ms")
            if item.get("mouth_close_late_tolerance_ms") is not None
            else (item.get("mouth_close_tolerance_ms") or 0)
        )
        schedule_late_warning_ms = max(0, end - (source_end + late_allowance_ms))
        value = dict(item)
        value.update(
            schedule_late_warning_ms=schedule_late_warning_ms,
            source_start_ms=source_start,
            source_end_ms=source_end,
            scheduled_start_ms=source_start,
            scheduled_end_ms=end,
            same_speaker_lag_ms=0,
            source_gap_borrowed_ms=int(item.get("source_gap_borrowed_ms") or 0),
            protected_gap_overflow_ms=int(item.get("protected_gap_overflow_ms") or 0),
            next_block_shift_ms=0,
            end_alignment_error_ms=end - source_end,
        )
        scheduled.append(value)
    return scheduled


def _load_manual_overrides(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": MANUAL_WEB_OVERRIDE_SCHEMA_VERSION, "entries": []}
    value = read_json(path)
    # Installed artifact contains provenance in addition to the contract.
    core = {
        "schema_version": value.get("schema_version"),
        "entries": value.get("entries"),
    }
    jsonschema.validate(core, _WEB_OVERRIDE_SCHEMA)
    return core


def _normalized_segments(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = transcript.get("segments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("English transcript has no segments")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_start = -1
    for index, item in enumerate(raw, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Transcript segment {index} is not an object")
        segment_id = str(item.get("segment_id") or f"seg-{index:05d}")
        if segment_id in seen:
            raise ValueError(f"Duplicate transcript segment_id: {segment_id}")
        seen.add(segment_id)
        speaker = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        text = str(item.get("source_text") or item.get("text") or "").strip()
        start_ms = _time_ms(item, "start", "start_ms")
        end_ms = _time_ms(item, "end", "end_ms")
        if not speaker or not text or end_ms <= start_ms:
            raise ValueError(f"Transcript segment {segment_id} is missing speaker/text/timing")
        if start_ms < previous_start:
            raise ValueError("Transcript segments are not in chronological order")
        previous_start = start_ms
        result.append({
            "segment_id": segment_id,
            "speaker_id": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "source_text": text,
        })
    return result


def _time_ms(item: Mapping[str, Any], seconds_key: str, ms_key: str) -> int:
    if ms_key in item and item.get(ms_key) is not None:
        return int(round(float(item[ms_key])))
    return int(round(float(item.get(seconds_key) or 0.0) * 1000))


def _compact_wire_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    text_key: str,
    soft_timing_speed_percent: int | None = None,
) -> list[dict[str, Any]]:
    """Compact provider wire with optional soft raw-TTS duration intent.

    ``src_ms`` is the source articulation/block window. ``target_raw_ms`` is
    deliberately a little longer so final sync can be solved by pitch-preserving
    acceleration instead of slowing speech. It is only a language-planning hint.
    """
    result: list[dict[str, Any]] = []
    for item in segments:
        text = str(
            item.get(text_key)
            or item.get("source_text")
            or item.get("restored_text")
            or ""
        ).strip()
        row: dict[str, Any] = {
            "id": str(item.get("segment_id") or ""),
            "spk": str(item.get("speaker_id") or ""),
            "text": text,
        }
        if soft_timing_speed_percent is not None:
            start_ms = int(item.get("start_ms") or 0)
            end_ms = int(item.get("end_ms") or 0)
            source_ms = max(1, end_ms - start_ms)
            row["src_ms"] = source_ms
            row["target_raw_ms"] = int(
                round(source_ms * (1.0 + int(soft_timing_speed_percent) / 100.0))
            )
        result.append(row)
    return result


def _compact_manual_overrides(overrides: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    entries = overrides.get("entries")
    if not isinstance(entries, list):
        return result
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        heard = str(item.get("heard_as") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if heard and canonical:
            result.append({"heard": heard, "canonical": canonical})
    return result


def _deterministic_context_ledger(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Small fallback ledger for old checkpoints/test doubles without another AI call."""
    speaker_ids: list[str] = []
    for item in segments:
        speaker = str(item.get("speaker_id") or "").strip()
        if speaker and speaker not in speaker_ids:
            speaker_ids.append(speaker)
    # Extract repeated/canonical-looking names and acronyms without interpreting them.
    counts: dict[str, int] = {}
    entity_pattern = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){1,3})\b")
    for item in segments:
        text = str(item.get("restored_text") or item.get("source_text") or "")
        for match in entity_pattern.findall(text):
            value = " ".join(match.split()).strip()
            if len(value) >= 2:
                counts[value] = counts.get(value, 0) + 1
    entities = [
        {"name": name, "role": "mentioned in source transcript"}
        for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
    ]
    # Evenly sampled source anchors are only continuity pointers, not AI summaries.
    beats: list[dict[str, Any]] = []
    if segments:
        slots = min(8, len(segments))
        positions = sorted({round(i * (len(segments) - 1) / max(1, slots - 1)) for i in range(slots)})
        for pos in positions:
            item = segments[pos]
            text = str(item.get("restored_text") or item.get("source_text") or "").strip()
            note = re.sub(r"\s+", " ", text)[:220]
            if note:
                beats.append({"segment_ids": [str(item["segment_id"])], "note": note})
    return {
        "summary": "",
        "entities": entities,
        "speaker_roles": [{"speaker_id": speaker, "role": ""} for speaker in speaker_ids[:12]],
        "story_beats": beats,
    }


def _normalize_context_ledger(
    value: Any,
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _deterministic_context_ledger(segments)
    ledger = {
        "summary": str(value.get("summary") or "").strip()[:1200],
        "entities": list(value.get("entities") or [])[:24],
        "speaker_roles": list(value.get("speaker_roles") or [])[:12],
        "story_beats": list(value.get("story_beats") or [])[:8],
    }
    jsonschema.validate(ledger, _CONTEXT_LEDGER_CONTRACT)

    # Story beats are discourse spans, not fixed-size windows.  Grok may
    # legitimately associate one beat with more than four source segments.
    # Canonicalize the references deterministically after schema validation:
    # every id must exist, duplicates are removed, and source order wins over
    # provider ordering.  This keeps the cached ledger stable without imposing
    # an arbitrary semantic span ceiling on the model.
    source_order = {str(item["segment_id"]): idx for idx, item in enumerate(segments)}
    valid_ids = set(source_order)
    canonical_beats: list[dict[str, Any]] = []
    for beat in ledger["story_beats"]:
        raw_ids = [str(x) for x in beat["segment_ids"]]
        unknown = [segment_id for segment_id in raw_ids if segment_id not in valid_ids]
        if unknown:
            raise ValueError(
                "Context ledger returned unknown story-beat segment IDs: "
                + ", ".join(dict.fromkeys(unknown))
            )
        unique_ids = sorted(set(raw_ids), key=source_order.__getitem__)
        canonical_beats.append(
            {
                "segment_ids": unique_ids,
                "note": str(beat["note"]).strip(),
            }
        )
    ledger["story_beats"] = canonical_beats
    jsonschema.validate(ledger, _CONTEXT_LEDGER_CONTRACT)
    return ledger


def _write_context_ledger(
    *,
    paths: NativeDubPaths,
    pass1: Mapping[str, Any],
    ledger: Mapping[str, Any],
    usage: Mapping[str, Any],
    generated_with_pass1: bool,
    generated_with_pass2: bool = False,
) -> None:
    atomic_write_json(paths.context_ledger, {
        "schema_version": CONTEXT_LEDGER_SCHEMA_VERSION,
        "source_pass1_sha256": checksum(paths.pass1),
        "context_ledger": dict(ledger),
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "attempts": int(usage.get("attempts") or 0),
        },
        "generated_with_pass1": bool(generated_with_pass1),
        "generated_with_pass2": bool(generated_with_pass2),
        "completed_at": pass1.get("completed_at") or utcnow(),
    })


_CASE_STATE_CONTEXT_MAX_CHARS = 12000


def _load_case_state_context(
    *, job_root: Path, progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Load and compact working/case_state/master_case_state.json (see
    master_case_context.py) -- a project-level "Story Memory" built by
    scripts/sync_madlanga_case.py, SEPARATE from this job's own per-job
    context_ledger/glossary. Returns None when the file doesn't exist (this
    must never be a hard requirement -- most jobs won't have one).

    `job_root` is a job directory (working/jobs/<job_id>); the case-state
    directory sits two levels up, alongside "jobs" itself.
    """
    working_root = job_root.parent.parent
    case_state = master_case_context.load_master_case_state(working_root)
    if not case_state:
        return None
    compact = master_case_context.compact_case_context(case_state, max_chars=_CASE_STATE_CONTEXT_MAX_CHARS)
    _emit_progress(
        progress,
        f"[native candidate pool] Loaded case-state context ({len(compact.get('standing_threads') or [])} "
        "standing thread(s)) for translation/QA grounding.",
    )
    return compact


def _load_existing_context_ledger(
    *,
    job_root: Path,
    pass1: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    paths = native_dub_paths(job_root)
    embedded = pass1.get("context_ledger")
    if isinstance(embedded, Mapping):
        ledger = _normalize_context_ledger(embedded, segments)
        if not paths.context_ledger.is_file():
            _write_context_ledger(
                paths=paths,
                pass1=pass1,
                ledger=ledger,
                usage={"input_tokens": 0, "output_tokens": 0, "attempts": 0},
                generated_with_pass1=True,
            )
        return ledger, {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    if paths.context_ledger.is_file():
        cached = read_json(paths.context_ledger)
        if cached.get("source_pass1_sha256") == checksum(paths.pass1):
            ledger = _normalize_context_ledger(cached.get("context_ledger"), segments)
            usage = cached.get("usage") if isinstance(cached.get("usage"), Mapping) else {}
            _emit_progress(progress, "[native context] Reusing cached compact discourse ledger")
            return ledger, {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "attempts": int(usage.get("attempts") or 0),
            }
    return None, {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def _ensure_context_ledger(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    pass1: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    paths = native_dub_paths(job_root)
    existing, usage = _load_existing_context_ledger(
        job_root=job_root,
        pass1=pass1,
        segments=segments,
        progress=progress,
    )
    if existing is not None:
        return existing, usage

    # Old Pass-1 checkpoints predate the ledger. Production Grok pays one full-
    # transcript input cost once, caches a small context artifact, and all later
    # bounded calls reuse it instead of repeating the programme transcript.
    if isinstance(provider, FoundryGrokProvider):
        _emit_progress(
            progress,
            "[native context] Old Pass-1 checkpoint has no context ledger; generating one compact cached ledger...",
        )
        restored = [
            {
                "segment_id": item["segment_id"],
                "speaker_id": item["speaker_id"],
                "source_text": item.get("restored_text") or item.get("source_text") or "",
            }
            for item in segments
        ]
        overrides = _load_manual_overrides(paths.web_overrides)
        response = provider.complete_json(
            operation="native_context_ledger",
            system_prompt=CONTEXT_LEDGER_SYSTEM_PROMPT,
            payload={
                "segments": _compact_wire_segments(restored, text_key="source_text"),
                "manual_overrides": _compact_manual_overrides(overrides),
            },
            schema=_CONTEXT_ONLY_SCHEMA,
            max_output_tokens=min(1600, int(getattr(provider.config, "max_output_tokens", 8192))),
        )
        ledger = _normalize_context_ledger(response.data.get("context_ledger"), segments)
        usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "attempts": response.attempts,
        }
    else:
        ledger = _deterministic_context_ledger(segments)
        usage = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}

    _write_context_ledger(
        paths=paths,
        pass1=pass1,
        ledger=ledger,
        usage=usage,
        generated_with_pass1=False,
    )
    return ledger, usage


def _normalize_zulu_glossary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"terms": [], "do_not_translate": [], "register": {}}
    glossary = {
        "terms": list(value.get("terms") or [])[:40],
        "do_not_translate": [str(x) for x in list(value.get("do_not_translate") or [])[:30]],
        "register": dict(value.get("register") or {}),
    }
    jsonschema.validate(glossary, _ZULU_GLOSSARY_CONTRACT)
    return glossary


def _write_zulu_glossary(
    *, paths: NativeDubPaths, pass1_sha256: str, glossary: Mapping[str, Any], usage: Mapping[str, Any],
) -> None:
    atomic_write_json(paths.zulu_glossary, {
        "schema_version": ZULU_GLOSSARY_SCHEMA_VERSION,
        "prompt_version": ZULU_GLOSSARY_PROMPT_VERSION,
        "source_pass1_sha256": pass1_sha256,
        "zulu_terminology_glossary": dict(glossary),
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "attempts": int(usage.get("attempts") or 0),
        },
        "completed_at": utcnow(),
    })


def _ensure_zulu_glossary(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    pass1: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    context_ledger: Mapping[str, Any],
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Decide the whole-programme Zulu terminology glossary once, from the full
    transcript -- see ZULU_GLOSSARY_SYSTEM_PROMPT for why this replaces rolling
    continuity_context in the candidate-pool architecture. Cached the same way
    context_ledger is: keyed off the Pass-1 checksum, so re-running native-translate
    on an unchanged transcript never re-spends this call.
    """
    paths = native_dub_paths(job_root)
    pass1_sha256 = checksum(paths.pass1)
    if paths.zulu_glossary.is_file() and not force:
        cached = read_json(paths.zulu_glossary)
        # Real gap found 2026-09-08: this only ever compared source_pass1_sha256,
        # so a PROMPT change (e.g. fixing a bad "register" description that then
        # gets treated as BINDING by every later translation call) silently never
        # took effect on an unchanged transcript without also passing --force.
        if (
            cached.get("source_pass1_sha256") == pass1_sha256
            and cached.get("prompt_version") == ZULU_GLOSSARY_PROMPT_VERSION
        ):
            glossary = _normalize_zulu_glossary(cached.get("zulu_terminology_glossary"))
            usage = cached.get("usage") if isinstance(cached.get("usage"), Mapping) else {}
            _emit_progress(progress, "[native glossary] Reusing cached Zulu terminology glossary")
            return glossary, {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "attempts": int(usage.get("attempts") or 0),
            }

    _emit_progress(progress, "[native glossary] Deciding whole-programme Zulu terminology and register...")
    restored = [
        {
            "segment_id": item["segment_id"],
            "speaker_id": item["speaker_id"],
            "source_text": item.get("restored_text") or item.get("source_text") or "",
        }
        for item in segments
    ]
    response = provider.complete_json(
        operation="native_zulu_glossary",
        system_prompt=ZULU_GLOSSARY_SYSTEM_PROMPT,
        payload={
            "context_ledger": dict(context_ledger),
            "segments": _compact_wire_segments(restored, text_key="source_text"),
        },
        schema=_ZULU_GLOSSARY_SCHEMA,
        max_output_tokens=min(3600, int(getattr(provider.config, "max_output_tokens", 8192))),
    )
    glossary = _normalize_zulu_glossary(response.data.get("zulu_terminology_glossary"))
    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    _write_zulu_glossary(paths=paths, pass1_sha256=pass1_sha256, glossary=glossary, usage=usage)
    _emit_progress(
        progress,
        f"[native glossary] Decided {len(glossary['terms'])} term(s), "
        f"{len(glossary['do_not_translate'])} do-not-translate item(s)",
    )
    return glossary, usage


# --- Phase 17: web-researched pronunciation ----------------------------------------
# Reuses the EXISTING AzureResponsesWebResearchProvider (azure_web_research.py, real
# Azure OpenAI Responses API web_search tool, already used in production by the
# legacy autocorrect_research.py pipeline) as a narrow, additive side-channel for
# entity/pronunciation research ONLY -- never touches translation, never touches
# FoundryGrokProvider (native_dub.py's own tool-free Chat-Completions transport).
#
# The output shape is deliberately reused verbatim from autocorrect_research.py's
# own already-validated schema/validator (_NAME_RESEARCH_OUTPUT_SCHEMA/
# validate_researched_corrections) rather than a hand-duplicated one, since
# pronunciation.py's with_web_researched_organisation_pronunciations was built
# entirely around that shape's field names (grounded_pronunciation_evidence_urls,
# pronunciation_confidence, pronunciation_mode, etc.) -- reusing it guarantees
# compatibility by construction instead of risking field-name drift.
#
# Load-bearing, confirmed by direct read of validate_researched_corrections
# (autocorrect_research.py): a candidate with any needs_identity_research value
# OTHER than exactly False is routed into the "full identity-correction" branch,
# which rejects outright whenever canonical_text comes back unchanged from the
# candidate's own name -- BEFORE it ever looks at a pronunciation field. Every
# candidate here therefore carries needs_identity_research=False and ONLY
# target_locale_pronunciation as a research goal: this phase is pronunciation-only,
# full stop, enforced by the reused validator itself. Spelling/identity correction
# stays owned by Pass-1 autocorrection and the existing manual_web_overrides.json
# channel (install_manual_web_overrides, explicitly "Never performs research").
NATIVE_PRONUNCIATION_RESEARCH_PROMPT_VERSION = "native-pronunciation-research-v1-tts-only"
# Sending all (up to 40) candidates in one request was real, observed to
# repeatedly hit Azure's per-minute token quota (HTTP 429) on a session with
# already-heavy GPT usage from translation/QA -- a request's size relative to
# the CURRENT remaining quota matters, not just its absolute size. Splitting
# into smaller batches (each with its own retry/backoff already built into
# AzureResponsesWebResearchProvider) reduces each individual request's token
# footprint, making it more likely to fit even when the quota is tight, at the
# cost of a few more round trips. One batch's failure never discards another
# batch's already-accepted corrections (see the per-batch try/except below).
_PRONUNCIATION_RESEARCH_BATCH_SIZE = 5
NATIVE_PRONUNCIATION_RESEARCH_SYSTEM_PROMPT = r"""You are Mathula TV's native-pipeline entity
pronunciation research agent for isiZulu (zu-ZA) narration. The input is a bounded list of
candidate_entities: names and organisation/institution names that appear, UNCHANGED, inside an
isiZulu broadcast translation -- they are code-switched English-origin terms, never translated.
Their canonical spelling is ALREADY FIXED and must never be changed -- you are researching ONLY
how a zu-ZA neural text-to-speech voice should be made to pronounce each one correctly, never
correcting or re-spelling anything.

For each candidate_entities[i], use web search to find direct pronunciation evidence: the named
person's own speech, official hearings/broadcasts, or native/source-language coverage where the
entity is spoken aloud. Determine pronunciation_language from evidence rather than guessing from
the name's appearance. Preserve the entity's attested source-language pronunciation; never
Anglicise, isiZulu-ise, or substitute a more familiar name.

Return pronunciation_mode=initialism only when letters are spoken separately; acronym when spoken
as one word; word_name for an ordinary name; otherwise unknown. pronunciation_tts_text is HIDDEN
application text fed only to the zu-ZA voice -- a short, plain-text phonetic respelling using the
smallest conservative adjustment needed for that voice to say the name correctly (for example
"Jastis Koleji" for "Justice College"). Never change canonical_text -- return it identical to the
supplied name, exactly. Return pronunciation_confidence and pronunciation_reason.
pronunciation_evidence_urls must directly support HOW the entity is said; official spelling alone
is not pronunciation evidence. If direct pronunciation evidence is unavailable, omit
pronunciation_tts_text, use mode unknown, and list the entity_id in unresolved_candidates rather
than inventing a phonetic spelling.

Return one strict JSON object: schema_version, corrections, unresolved_candidates. For each
correction return entity_id, canonical_text (unchanged), entity_type, pronunciation_mode,
pronunciation_evidence_urls, pronunciation_tts_text, pronunciation_confidence, pronunciation_reason,
pronunciation_language, and pronunciation_ipa when grounded. Never propose an identity or spelling
change. Never research, add, or merge an entity not present in candidate_entities."""


# --- Phase 24: self-supervised pronunciation fallback for entities with no
# web evidence ----------------------------------------------------------------
# Real user-reported production defect (job fb3d08b63fed4d90922b08f7e325b906,
# 2026-09-07): "Godfrey Gidi" (a local EFF mayoral candidate) was correctly
# sent to the web-research pipeline above (confirmed by direct inspection of
# context_ledger.json -- entity #2 of 7, no candidate-selection bug) but came
# back with no accepted correction, because a local municipal candidate has no
# indexed web audio of himself speaking his own name -- a fundamental limit of
# web search, not a defect any code fix to that mechanism can close. Per the
# user's explicit instruction ("not patching, make the system flexible enough
# to handle such issues automatically"), this closes the gap generally: for
# ANY entity that ends up with no accepted correction, generate a couple of
# phonetic respelling candidates from the model's own linguistic knowledge
# (no web search tool -- this is a pronunciation guess, not an identity
# lookup, so it's cheap and shares none of the web-search rate-limit risk)
# and empirically verify each one with the EXACT SAME real TTS+STT round-trip
# judge Phase 18 already built (_measure_pronunciation_round_trip) -- adopting
# a candidate only if it measurably beats the raw spelling. No new judge is
# invented, only a new candidate source feeding the one already proven.
NATIVE_PRONUNCIATION_SELF_SUPERVISED_PROMPT_VERSION = "native-pronunciation-self-supervised-v1"
NATIVE_PRONUNCIATION_SELF_SUPERVISED_SYSTEM_PROMPT = r"""You are Mathula TV's pronunciation-
respelling assistant for isiZulu (zu-ZA) narration. The input is a bounded list of entities: names
or organisation/institution names that appear, UNCHANGED, inside an isiZulu broadcast translation
-- code-switched English-origin (or other non-Zulu) terms, never translated. Web search already
tried and failed to find direct pronunciation evidence for each one (no indexed audio of the
person/entity actually speaking, common for local or lesser-known names).

When an entity carries a raw_asr_hint, treat it as the single strongest evidence available: it is
the ORIGINAL, unedited speech-to-text transcription of this exact same moment in this job's own
source audio, BEFORE spelling correction -- literally what the real automatic transcription engine
heard when it listened to the actual person say this name, before a later spelling-correction step
normalised it to the tidier canonical_text. It is not a guess and not a web source; it is a direct,
job-specific phonetic clue to how the name actually sounds -- e.g. a canonical_text of "Godfrey
Gidi" with a raw_asr_hint mentioning "Godrej Gade" is telling you the real audio sounds much closer
to "Godrej Gade" than to a standard English reading of "Godfrey Gidi". Build your respelling to
match what raw_asr_hint suggests the name actually sounds like, not the possibly-more-conventional
canonical spelling. When no raw_asr_hint is given, fall back to ordinary linguistic knowledge of how
the name is most likely pronounced by its own language/culture of origin -- NOT web search.

For each entity, propose 0, 1, or 2 DISTINCT short, plain-text phonetic respellings -- hidden text
fed only to a zu-ZA neural voice, never shown to a viewer and never changing the visible/canonical
spelling. Each respelling should nudge the voice toward a natural pronunciation of the name using
Zulu-friendly spelling conventions (for example doubling a vowel for a full, clearly-articulated
syllable, or spelling a name the way an English speaker's name-sound would be written out
phonetically) -- never a wildly different-sounding invention. Returning zero respellings ("no
confident guess") is a fully valid, encouraged answer when you are not confident -- a missed
respelling costs nothing (the entity simply keeps its plain spelling), while a bad invented one
could make things worse.

Return one strict JSON object: schema_version, candidates. Each candidates[] item: entity_id (copy
exactly from the input), respellings (0-2 short plain-text strings, most-likely-correct first).
Never invent, merge, or research an entity not present in the input. Never propose a change to the
visible spelling itself."""

_SELF_SUPERVISED_PRONUNCIATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "candidates"],
    "properties": {
        "schema_version": {"const": "mathula-pronunciation-self-supervised-v1"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity_id", "respellings"],
                "properties": {
                    "entity_id": {"type": "string", "minLength": 1},
                    "respellings": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}


def _find_raw_asr_hint_for_entity(
    entity_name: str, pass1_corrections: Sequence[Mapping[str, Any]],
) -> str | None:
    """Real, already-computed, free evidence for how a name actually sounds:
    Pass 1's own whole-transcript restoration (run_native_pass1) records
    every name/entity spelling correction it made, keeping BOTH the original
    (raw ASR, i.e. what real speech-to-text heard directly from the actual
    audio) and corrected (canonical) text side by side. Real confirmed case,
    job fb3d08b63fed4d90922b08f7e325b906: the canonical "Godfrey Gidi" was
    originally transcribed as "Godrej Gade" -- much closer to the name's real
    sound than the tidied-up canonical spelling, per direct user feedback
    ("the English audio has correct pronunciation of 'Gardee', can't learn
    phonetics from [spelling] there"). Returns the whole original_text (not
    an isolated span -- the model is given both full sentences and can find
    the relevant part itself, avoiding a fragile word-diff alignment here).
    """
    name = entity_name.casefold().strip()
    if not name:
        return None
    for correction in pass1_corrections:
        if not isinstance(correction, Mapping):
            continue
        if str(correction.get("reason_code") or "") != "name_or_entity":
            continue
        corrected = str(correction.get("corrected_text") or "")
        if name in corrected.casefold():
            original = str(correction.get("original_text") or "").strip()
            if original:
                return original
    return None


def _request_self_supervised_pronunciation_candidates(
    *, provider: FoundryGrokProvider, entities: Sequence[Mapping[str, Any]],
    pass1_corrections: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """One batched, tool-free (no web search) request for phonetic respelling
    guesses on entities the web-research pass could not resolve. Identity
    mismatch (an entity_id in the response not present in the request) is
    silently dropped rather than fatal -- this is an advisory candidate
    source; the real safety gate is the round-trip verification that follows,
    not this request's own fidelity.
    """
    if not entities:
        return {}, {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    requested_ids = {str(item.get("entity_id") or "") for item in entities}

    def _payload_item(item: Mapping[str, Any]) -> dict[str, Any]:
        name = str(item.get("representative_text") or item.get("canonical_name_hint") or "")
        built = {
            "entity_id": str(item.get("entity_id") or ""),
            "name": name,
            "entity_type": str(item.get("entity_type") or "other_name"),
        }
        hint = _find_raw_asr_hint_for_entity(name, pass1_corrections)
        if hint:
            built["raw_asr_hint"] = hint
        return built

    response = provider.complete_json(
        operation="native_pronunciation_self_supervised",
        system_prompt=NATIVE_PRONUNCIATION_SELF_SUPERVISED_SYSTEM_PROMPT,
        payload={"candidates": [_payload_item(item) for item in entities]},
        schema=_SELF_SUPERVISED_PRONUNCIATION_SCHEMA,
    )
    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    by_entity: dict[str, list[str]] = {}
    for item in response.data.get("candidates") or ():
        if not isinstance(item, Mapping):
            continue
        entity_id = str(item.get("entity_id") or "")
        if entity_id not in requested_ids:
            continue
        respellings = [
            str(value).strip() for value in item.get("respellings") or () if str(value).strip()
        ]
        if respellings:
            by_entity[entity_id] = respellings[:2]
    return by_entity, usage


def _native_pronunciation_candidate_id(name: str) -> str:
    """Stable, run-independent match key -- confirmed via direct read of
    autocorrect_research.py's _candidate_key/_resolve_candidate to be the exact
    field (entity_id) those functions use to reconcile a response item back to
    the candidate that requested it.
    """
    return hashlib.sha256(name.casefold().strip().encode("utf-8")).hexdigest()[:16]


def _native_pronunciation_research_candidates(
    *, context_ledger: Mapping[str, Any], glossary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build this job's pronunciation-research candidate list from two
    already-computed, job-scoped sources -- no new detection step needed:

    - context_ledger["entities"][].name (person/witness names).
    - glossary["do_not_translate"][] and glossary["terms"][] where
      kind in {"name","organisation"} AND english == zulu (classified but left
      untranslated) -- both are the "survives verbatim into committed Zulu"
      category. A terms[] entry whose zulu genuinely differs from its english
      is excluded: PronunciationDictionary.apply can only ever substitute a
      string that actually appears in the text handed to TTS, so researching a
      translated form's pronunciation has no possible application point.

    Deduped by casefold().strip(), capped at 40 (same order of magnitude as the
    existing glossary call).
    """
    names: list[tuple[str, str]] = []  # (name, entity_type)
    for entity in context_ledger.get("entities") or ():
        if not isinstance(entity, Mapping):
            continue
        name = str(entity.get("name") or "").strip()
        if name:
            names.append((name, "person"))
    for value in glossary.get("do_not_translate") or ():
        name = str(value or "").strip()
        if name:
            names.append((name, "organisation"))
    for term in glossary.get("terms") or ():
        if not isinstance(term, Mapping):
            continue
        kind = str(term.get("kind") or "").casefold()
        english = str(term.get("english") or "").strip()
        zulu = str(term.get("zulu") or "").strip()
        if kind in {"name", "organisation"} and english and english.casefold() == zulu.casefold():
            names.append((english, "person" if kind == "name" else "organisation"))

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for name, entity_type in names:
        key = name.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append({
            "entity_id": _native_pronunciation_candidate_id(name),
            "representative_text": name,
            "canonical_name_hint": name,
            "aliases": [name],
            "needs_identity_research": False,
            "needs_pronunciation_research": True,
            "research_goals": ["target_locale_pronunciation"],
            "entity_type": entity_type,
        })
        if len(candidates) >= 40:
            break
    return candidates


# Phase 18: automated pronunciation round-trip verification. Web research
# reasons from text/web sources only -- it never listens to what the actual
# zu-ZA Azure voice produces. Confirmed by a real failure this session: the
# researcher explicitly decided a real Setswana-origin surname "retains its
# local pronunciation" (no respelling needed), and that textual judgment call
# was acoustically wrong for both configured voices, only caught because a
# human listened to the render. This closes that gap deterministically: a
# real TTS+STT round trip on the SAME short carrier phrase, once with the raw
# canonical spelling and once with the researched respelling, scored against
# the canonical name's own text -- promoted to a hidden TTS alias only if the
# round trip actually confirms the respelling helps. A known, honestly-stated
# limitation (not solved by this mechanism): STT itself normalizes/simplifies
# fine articulatory distinctions, so a subtle correction (this session's own
# "Mogotsi" vs "Mokgotsi" case) can round-trip as similarly-scored either way
# -- this catches gross mispronunciation reliably, not every nuance a human
# ear would catch.
_PRONUNCIATION_ROUND_TRIP_MIN_SCORE = 0.7
_PRONUNCIATION_ROUND_TRIP_CARRIER_TEMPLATE = "Kukhulunywa ngo{name} kulesi sigaba."


def _pronunciation_round_trip_score(recovered_text: str, canonical_text: str) -> float:
    """Best-substring similarity in [0, 1] between what STT actually heard and
    the real name.

    A whole-string ratio against the full recovered carrier sentence was
    tried first and rejected: the fixed carrier words ("kukhulunywa ngo...
    kulesi sigaba") dominate the comparison and dilute the actual name signal
    -- a real test against this session's own Justice College data scored a
    clearly-garbled recovery ("chastique koleke") at 0.36 and a clearly-clean
    one ("justice college") at only 0.34, statistically useless. Instead,
    score every window of the recovered text sized like the canonical name
    (anchored at each of difflib's own matching-block offsets, the standard
    "partial ratio" technique) and take the best one -- confirmed on the same
    data to separate the clean case (1.0) from the garbled one (0.6).
    """
    haystack = re.sub(r"[^\w\s]", "", recovered_text.casefold()).strip()
    needle = re.sub(r"[^\w\s]", "", canonical_text.casefold()).strip()
    if not needle or not haystack:
        return 0.0
    best = difflib.SequenceMatcher(None, haystack, needle).ratio()
    matcher = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        start = max(0, block.a - block.b)
        end = min(len(haystack), start + len(needle))
        start = max(0, end - len(needle))
        window = haystack[start:end]
        best = max(best, difflib.SequenceMatcher(None, window, needle).ratio())
    return best


def _synthesize_and_score_round_trip(
    *,
    tts: AzureTTSBackend,
    stt_backend: AzureFastTranscriptionBackend,
    output_path: Path,
    turn_id: str,
    voice: str,
    text: str,
    canonical_text: str,
) -> tuple[float, str]:
    request = AzureTTSRequest(
        turn_id=turn_id, speaker_id="pronunciation_verify", text=text,
        preferred_duration_ms=6000, maximum_duration_ms=15000, voice=voice,
    )
    tts.synthesize(request, output_path, force=True)
    raw = stt_backend.transcribe(output_path, locale="zu-ZA")
    data = normalize_stt_response(raw)
    recovered = " ".join(str(item.get("text") or "") for item in data.get("words") or ())
    return _pronunciation_round_trip_score(recovered, canonical_text), recovered


def _measure_pronunciation_round_trip(
    *,
    tts: AzureTTSBackend,
    stt_backend: AzureFastTranscriptionBackend,
    job_root: Path,
    entity_id: str,
    canonical_text: str,
    pronunciation_tts_text: str,
    voices: Sequence[str],
    progress: Callable[[str], None] | None = None,
    repeats_per_voice: int = 2,
) -> dict[str, Any]:
    """Real TTS+STT round trip for one accepted correction, across every
    configured voice -- which specific voice this entity's speaker will get
    isn't known yet at this stage of the pipeline, so every configured voice
    is tested, matching this session's own manual calibration practice for
    Justice College/Lee Segeels Ncube/Brown Mogotsi. A synthesis/STT failure
    for one voice is skipped, not fatal -- the verdict uses whichever voices
    actually produced a result.

    Real, confirmed finding, job fb3d08b63fed4d90922b08f7e325b906: Azure
    neural TTS is not deterministic even for identical input text -- the
    SAME raw spelling ("Godfrey Gidi") round-tripped a clean 1.0 on one real
    call and badly mangled on others, purely from synthesis variance, not
    from anything about the text itself. A single sample per voice can
    therefore give a misleadingly confident (or misleadingly bad) verdict.
    repeats_per_voice (default 2, so 4 real Azure calls per voice: 2 raw + 2
    candidate) takes the MEAN score across repeats for each voice first --
    a robust per-voice estimate that averages out this real synthesis
    noise -- THEN the max across voices, preserving the original "does at
    least one configured voice do well" semantics for genuine per-voice
    differences (which are real, not noise, and should not be averaged
    away). raw_best_score/candidate_best_score keep their established field
    names for API compatibility; their value is now this more robust
    mean-then-max estimate rather than a single raw sample.
    """
    audio_dir = native_dub_paths(job_root).root / "pronunciation_round_trip"
    audio_dir.mkdir(parents=True, exist_ok=True)
    per_voice: list[dict[str, Any]] = []
    for voice in voices:
        raw_text = _PRONUNCIATION_ROUND_TRIP_CARRIER_TEMPLATE.format(name=canonical_text)
        candidate_text = _PRONUNCIATION_ROUND_TRIP_CARRIER_TEMPLATE.format(name=pronunciation_tts_text)
        raw_samples: list[tuple[float, str]] = []
        candidate_samples: list[tuple[float, str]] = []
        try:
            for repeat in range(max(1, repeats_per_voice)):
                raw_samples.append(_synthesize_and_score_round_trip(
                    tts=tts, stt_backend=stt_backend,
                    output_path=audio_dir / f"{entity_id}_{voice}_raw_{repeat}.wav",
                    turn_id=f"pron_verify_{entity_id}_{voice}_raw_{repeat}", voice=voice, text=raw_text,
                    canonical_text=canonical_text,
                ))
                candidate_samples.append(_synthesize_and_score_round_trip(
                    tts=tts, stt_backend=stt_backend,
                    output_path=audio_dir / f"{entity_id}_{voice}_candidate_{repeat}.wav",
                    turn_id=f"pron_verify_{entity_id}_{voice}_candidate_{repeat}", voice=voice, text=candidate_text,
                    canonical_text=canonical_text,
                ))
        except Exception as exc:  # noqa: BLE001 - one voice's failure must never block the others
            _emit_progress(
                progress,
                f"[native pronunciation] Round-trip check for '{canonical_text}' failed on {voice} "
                f"({exc}); skipping that voice.",
            )
            continue
        raw_mean = sum(score for score, _ in raw_samples) / len(raw_samples)
        candidate_mean = sum(score for score, _ in candidate_samples) / len(candidate_samples)
        per_voice.append({
            "voice": voice, "raw_score": raw_mean, "raw_transcript": raw_samples[-1][1],
            "candidate_score": candidate_mean, "candidate_transcript": candidate_samples[-1][1],
            "raw_sample_scores": [score for score, _ in raw_samples],
            "candidate_sample_scores": [score for score, _ in candidate_samples],
        })
    if not per_voice:
        return {"verified": None, "raw_best_score": None, "candidate_best_score": None, "per_voice": []}
    raw_best = max(item["raw_score"] for item in per_voice)
    candidate_best = max(item["candidate_score"] for item in per_voice)
    # Never let the respelling regress vs. the raw spelling, and require either
    # a genuine improvement or an already-solid absolute score -- a flat "must
    # clear this fixed floor" rule (tried first) wrongly rejected a real,
    # meaningful improvement that started too low to reach the floor even
    # after doubling (KZN: 0.333 -> 0.667), while a flat "+0.1 over raw" rule
    # (tried second) would have wrongly rejected several real ties at an
    # already-good score that can never show a margin once raw sits near the
    # ceiling (e.g. "modus operandi" tying raw and candidate at 0.929). This
    # rule reproduces every other real verdict from this session's own first
    # live run unchanged and fixes only the KZN case.
    verified = candidate_best >= raw_best and (
        candidate_best > raw_best or candidate_best >= _PRONUNCIATION_ROUND_TRIP_MIN_SCORE
    )
    return {
        "verified": verified, "raw_best_score": raw_best, "candidate_best_score": candidate_best,
        "per_voice": per_voice,
    }


def _verify_pronunciation_corrections_round_trip(
    *,
    accepted_corrections: Sequence[Mapping[str, Any]],
    tts: AzureTTSBackend,
    stt_backend: AzureFastTranscriptionBackend,
    voices: Sequence[str],
    job_root: Path,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Annotate every accepted correction with a real round-trip verdict.

    Never drops a correction outright -- an unverified one is kept in the
    artifact (audit trail: a human can see what was tried and why it didn't
    confirm) with round_trip_verified=False, so
    with_web_researched_organisation_pronunciations's promotion gate can
    refuse to build a hidden alias from it without losing the record.
    """
    verified_list: list[dict[str, Any]] = []
    for raw_item in accepted_corrections:
        item = dict(raw_item)
        pronunciation_tts_text = str(item.get("pronunciation_tts_text") or "").strip()
        canonical_text = str(item.get("canonical_text") or "").strip()
        if not pronunciation_tts_text or not canonical_text:
            verified_list.append(item)
            continue
        result = _measure_pronunciation_round_trip(
            tts=tts, stt_backend=stt_backend, job_root=job_root,
            entity_id=str(item.get("entity_id") or _native_pronunciation_candidate_id(canonical_text)),
            canonical_text=canonical_text, pronunciation_tts_text=pronunciation_tts_text,
            voices=voices, progress=progress,
        )
        if result["verified"] is not None:
            item["round_trip_verified"] = bool(result["verified"])
        item["round_trip_raw_score"] = result["raw_best_score"]
        item["round_trip_candidate_score"] = result["candidate_best_score"]
        item["round_trip_detail"] = result["per_voice"]
        _emit_progress(
            progress,
            f"[native pronunciation] Round-trip '{canonical_text}' -> '{pronunciation_tts_text}': "
            f"raw={_round_or_none(result['raw_best_score'])}, "
            f"candidate={_round_or_none(result['candidate_best_score'])}, verified={result.get('verified')}",
        )
        verified_list.append(item)
    return verified_list


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


_NATIVE_PRONUNCIATION_CACHE_FILENAME = "native_pronunciation_cache.json"


def _native_pronunciation_cache_path(job_root: Path) -> Path:
    """Project-level (NOT per-job) cache of web-researched pronunciation
    corrections, at working/context/native_pronunciation_cache.json --
    sibling to the existing politics_context.json/commission_master_case_
    state.json project-level context files (see config.py). Deliberately
    separate from that commission "Story Memory" file: this one is fully
    owned and written by this pipeline (the case-state file is a one-way
    external sync from scripts/sync_madlanga_case.py and must never be
    written to from here).
    """
    return job_root.parent.parent / "context" / _NATIVE_PRONUNCIATION_CACHE_FILENAME


def _load_native_pronunciation_cache(job_root: Path) -> dict[str, dict[str, Any]]:
    path = _native_pronunciation_cache_path(job_root)
    try:
        data = read_json(path)
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _save_native_pronunciation_cache(job_root: Path, cache: Mapping[str, Mapping[str, Any]]) -> None:
    path = _native_pronunciation_cache_path(job_root)
    try:
        atomic_write_json(path, dict(cache))
    except Exception:
        pass  # never a hard failure -- losing this cache only costs a future re-research, not correctness


def _ensure_native_pronunciation_research(
    *,
    job_root: Path,
    context_ledger: Mapping[str, Any],
    glossary: Mapping[str, Any],
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    tts: AzureTTSBackend | None = None,
    stt_backend: AzureFastTranscriptionBackend | None = None,
    skip_round_trip_verification: bool = False,
    grok_provider: FoundryGrokProvider | None = None,
    pass1_corrections: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], dict[str, int]]:
    """Research pronunciation for this job's code-switched proper nouns via a
    real web-search call, cached the same way _ensure_zulu_glossary is
    (checksum-keyed, force bypass). Never a hard failure -- any provider
    error (missing config, network, quota) degrades to an empty, non-fatal
    artifact, exactly like NLLB's own degraded:bool handling and Phase 16's
    per-window fallback. A job with no candidates never spends on a call.

    Phase 18: when both ``tts`` and ``stt_backend`` are supplied (and
    ``skip_round_trip_verification`` is False), every accepted correction with
    a ``pronunciation_tts_text`` is additionally verified with a real TTS+STT
    round trip (see _verify_pronunciation_corrections_round_trip) before being
    written -- web research reasons from text/web sources only and never
    listens to what the actual Azure voice produces, a real gap confirmed this
    session (research explicitly decided a real Setswana-origin surname
    "retains its local pronunciation" with no respelling needed, which was
    acoustically wrong). Omitting ``tts``/``stt_backend`` (e.g. no
    AZURE_SPEECH_KEY configured) preserves the exact pre-Phase-18 behavior --
    corrections are written unverified, and pronunciation.py's promotion gate
    treats an absent round_trip_verified field as verified, matching every
    cached artifact from before this feature existed.
    """
    paths = native_dub_paths(job_root)
    source_hash = _hash_payload({
        "context_ledger": context_ledger, "glossary": glossary,
        "skip_round_trip_verification": bool(skip_round_trip_verification),
    })
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    if paths.pronunciation_research.is_file() and not force:
        cached = read_json(paths.pronunciation_research)
        if cached.get("source_hash") == source_hash:
            _emit_progress(progress, "[native pronunciation] Reusing cached pronunciation research")
            _apply_native_pronunciation_research_to_job_cache(job_root, cached)
            usage = cached.get("usage") if isinstance(cached.get("usage"), Mapping) else {}
            for key in usage_totals:
                usage_totals[key] = int(usage.get(key) or 0)
            return cached, usage_totals

    all_candidates = _native_pronunciation_research_candidates(context_ledger=context_ledger, glossary=glossary)

    # Two real, confirmed skips, checked in order of authority. (1) A name
    # already in pronunciation.py's own permanent, human-reviewed dictionary
    # needs no research at all -- that entry already wins over anything a
    # fresh web search could propose, so researching it can only waste money
    # (confirmed real case, job 8372120960474ef6b1d75af7de51a605: fresh
    # research on "Lincoln"/"General Lincoln" -- already hand-fixed to
    # "Linken" earlier the same day via real STT round-trip testing --
    # produced WORSE candidates that round-trip verification correctly
    # rejected anyway, after paying for them). (2) A name already resolved
    # by a PRIOR job's real web search (this project-level cache) needs no
    # research either -- confirmed real case: a smaller re-cut of an
    # already-processed video re-researched the exact same 13 entities from
    # scratch, none of them new.
    hardcoded_keys = {str(name).casefold().strip() for name in _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS}
    project_cache = _load_native_pronunciation_cache(job_root)
    candidates: list[dict[str, Any]] = []
    cache_hit_accepted: list[dict[str, Any]] = []
    skipped_hardcoded = 0
    for candidate in all_candidates:
        key = str(candidate.get("representative_text") or "").casefold().strip()
        if key in hardcoded_keys:
            skipped_hardcoded += 1
            continue
        # force=True bypasses the cross-job cache too (same meaning as every
        # other --force in this pipeline: ignore any checkpoint and redo the
        # real work), NOT just the per-job artifact check above -- a user
        # explicitly forcing a re-research (e.g. suspecting a cached entry
        # is stale) must actually get fresh research, not a silent cache hit.
        cached_entry = None if force else project_cache.get(key)
        if isinstance(cached_entry, Mapping):
            cache_hit_accepted.append(dict(cached_entry))
            continue
        candidates.append(candidate)
    if skipped_hardcoded or cache_hit_accepted:
        _emit_progress(
            progress,
            f"[native pronunciation] Skipping {skipped_hardcoded} already-hardcoded and "
            f"{len(cache_hit_accepted)} already-cached entity/entities -- only {len(candidates)} "
            "genuinely new candidate(s) need real research.",
        )

    if not candidates:
        artifact = {
            "schema_version": NATIVE_PRONUNCIATION_RESEARCH_PROMPT_VERSION,
            "source_hash": source_hash, "status": "empty" if not cache_hit_accepted else "completed",
            "candidate_count": len(all_candidates),
            "accepted_corrections": cache_hit_accepted, "rejected_corrections": [],
            "self_supervised_corrections": [],
            "usage": usage_totals, "completed_at": utcnow(),
        }
        atomic_write_json(paths.pronunciation_research, artifact)
        _apply_native_pronunciation_research_to_job_cache(job_root, artifact)
        return artifact, usage_totals

    _emit_progress(
        progress,
        f"[native pronunciation] Researching pronunciation for {len(candidates)} entity/entities...",
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    self_supervised: list[dict[str, Any]] = []
    status = "completed"
    provider = None
    try:
        provider = AzureResponsesWebResearchProvider.from_environment()
    except Exception as exc:  # noqa: BLE001 - research must degrade, never fail the job
        status = "degraded"
        _emit_red_warning(
            progress,
            f"[native pronunciation] Pronunciation research failed ({exc}); continuing without it "
            "-- entities keep their default/plain pronunciation.",
        )

    if provider is not None:
        batches = _chunk_sequence(candidates, _PRONUNCIATION_RESEARCH_BATCH_SIZE)
        for batch_index, batch in enumerate(batches, start=1):
            try:
                response = provider.research_json(
                    system_prompt=NATIVE_PRONUNCIATION_RESEARCH_SYSTEM_PROMPT,
                    payload={"candidate_entities": batch},
                    output_schema=_NAME_RESEARCH_OUTPUT_SCHEMA,
                )
                actual_evidence_urls = [
                    str(item.get("url") or "").strip()
                    for item in (response.metadata.server_tool_evidence or ())
                    if isinstance(item, Mapping) and str(item.get("url") or "").strip()
                ]
                batch_accepted, batch_rejected = validate_researched_corrections(
                    response.data, candidates=batch, actual_evidence_urls=actual_evidence_urls,
                )
                accepted.extend(batch_accepted)
                rejected.extend(batch_rejected)
                usage_totals["input_tokens"] += int(response.metadata.usage.get("input_tokens") or 0)
                usage_totals["output_tokens"] += int(response.metadata.usage.get("output_tokens") or 0)
                usage_totals["attempts"] += int(response.metadata.request_attempts)
            except Exception as exc:  # noqa: BLE001 - one batch's failure must never lose the others
                status = "degraded"
                _emit_red_warning(
                    progress,
                    f"[native pronunciation] Batch {batch_index}/{len(batches)} "
                    f"({len(batch)} entity/entities) failed ({exc}); continuing with the rest -- "
                    "these entities keep their default/plain pronunciation.",
                )

        if accepted and not skip_round_trip_verification and tts is not None and stt_backend is not None:
            voices = tuple(dict.fromkeys(str(v).strip() for v in (tts.configured_voices or ()) if str(v).strip()))
            if voices:
                try:
                    accepted = _verify_pronunciation_corrections_round_trip(
                        accepted_corrections=accepted, tts=tts, stt_backend=stt_backend,
                        voices=voices, job_root=job_root, progress=progress,
                    )
                except Exception as exc:  # noqa: BLE001 - verification must degrade, never fail research
                    _emit_red_warning(
                        progress,
                        f"[native pronunciation] Round-trip verification failed ({exc}); "
                        "accepted corrections kept unverified.",
                    )

        # Phase 24: self-supervised fallback for any candidate that STILL has
        # no correction after real web research -- covers every drop reason
        # uniformly (no web evidence found, validation rejection, or a batch
        # that failed outright), since in every one of those cases the entity
        # currently gets zero pronunciation help. Only runs when round-trip
        # verification is actually possible (tts/stt_backend configured) --
        # an unverified guess is exactly the "invented phonetic spelling"
        # risk the web-research prompt itself is told never to take, so this
        # mechanism must not either.
        accepted_ids = {str(item.get("entity_id") or "") for item in accepted}
        unresolved = [c for c in candidates if str(c.get("entity_id") or "") not in accepted_ids]
        if (
            unresolved and grok_provider is not None and not skip_round_trip_verification
            and tts is not None and stt_backend is not None
        ):
            voices = tuple(dict.fromkeys(str(v).strip() for v in (tts.configured_voices or ()) if str(v).strip()))
            if voices:
                try:
                    guesses, guess_usage = _request_self_supervised_pronunciation_candidates(
                        provider=grok_provider, entities=unresolved, pass1_corrections=pass1_corrections,
                    )
                    usage_totals["input_tokens"] += guess_usage["input_tokens"]
                    usage_totals["output_tokens"] += guess_usage["output_tokens"]
                    usage_totals["attempts"] += guess_usage["attempts"]
                except Exception as exc:  # noqa: BLE001 - a guess failure must never fail research
                    guesses = {}
                    _emit_red_warning(
                        progress,
                        f"[native pronunciation] Self-supervised respelling request failed ({exc}); "
                        f"{len(unresolved)} entity/entities keep their default/plain pronunciation.",
                    )
                for entity in unresolved:
                    entity_id = str(entity.get("entity_id") or "")
                    canonical_text = str(entity.get("representative_text") or "").strip()
                    respellings = guesses.get(entity_id) or []
                    if not canonical_text or not respellings:
                        continue
                    best_candidate: str | None = None
                    best_result: dict[str, Any] | None = None
                    for respelling in respellings:
                        result = _measure_pronunciation_round_trip(
                            tts=tts, stt_backend=stt_backend, job_root=job_root,
                            entity_id=f"{entity_id}_guess", canonical_text=canonical_text,
                            pronunciation_tts_text=respelling, voices=voices, progress=progress,
                        )
                        if result["verified"] and (
                            best_result is None
                            or (result["candidate_best_score"] or 0) > (best_result["candidate_best_score"] or 0)
                        ):
                            best_candidate, best_result = respelling, result
                    _emit_progress(
                        progress,
                        f"[native pronunciation] Self-supervised guess for '{canonical_text}': "
                        f"{len(respellings)} candidate(s) tried, "
                        f"{'adopted ' + repr(best_candidate) if best_candidate else 'none verified'}.",
                    )
                    if best_candidate and best_result:
                        self_supervised.append({
                            "entity_id": entity_id,
                            "canonical_text": canonical_text,
                            "entity_type": str(entity.get("entity_type") or "other_name"),
                            "pronunciation_mode": "word_name",
                            "pronunciation_tts_text": best_candidate,
                            "pronunciation_confidence": 0.9,
                            "correction_mode": "self_supervised_no_evidence",
                            "round_trip_verified": True,
                            "round_trip_raw_score": best_result["raw_best_score"],
                            "round_trip_candidate_score": best_result["candidate_best_score"],
                        })

        # Write-through: persist every NEWLY researched correction into the
        # project-level cache (keyed by its own representative_text, the
        # same key used to check the cache above) so the next job -- e.g. a
        # different cut of the same source video -- never pays to
        # re-discover it. Cached regardless of round_trip_verified: a
        # correctly-rejected suggestion is just as worth remembering as an
        # accepted one, since re-researching it again would only rediscover
        # the same rejection at the same real cost. Self-supervised guesses
        # are cached the same way -- a future job's cache hit lands in
        # cache_hit_accepted regardless of original provenance, which is
        # fine: the relaxed round-trip-evidence gate in
        # with_web_researched_organisation_pronunciations checks the
        # confirmed round-trip fields, not which list a record arrived
        # through.
        if accepted or self_supervised:
            for record in (*accepted, *self_supervised):
                key = str(record.get("representative_text") or record.get("canonical_text") or "").casefold().strip()
                if key:
                    project_cache[key] = dict(record)
            _save_native_pronunciation_cache(job_root, project_cache)

    accepted = cache_hit_accepted + accepted
    artifact = {
        "schema_version": NATIVE_PRONUNCIATION_RESEARCH_PROMPT_VERSION,
        "source_hash": source_hash, "status": status, "candidate_count": len(all_candidates),
        "accepted_corrections": accepted, "rejected_corrections": rejected,
        "self_supervised_corrections": self_supervised,
        "usage": usage_totals, "completed_at": utcnow(),
    }
    atomic_write_json(paths.pronunciation_research, artifact)
    _apply_native_pronunciation_research_to_job_cache(job_root, artifact)
    _emit_progress(
        progress,
        f"[native pronunciation] {len(accepted)}/{len(all_candidates)} entity/entities got a "
        "web-grounded pronunciation.",
    )
    return artifact, usage_totals


def _glossary_conformance(glossary: Mapping[str, Any], english_source: str, zulu_candidate: str) -> int:
    """Count how many glossary terms whose ENGLISH form appears in this sentence's
    source also have their canonical ZULU form present in a candidate. Used only as
    a low-priority selection tie-break/reported metric -- deterministic string
    substitution on agglutinative Zulu output would risk breaking noun-class
    concord, so this is observational, never corrective.
    """
    source = str(english_source or "")
    candidate = str(zulu_candidate or "")
    conforming = 0
    for term in glossary.get("terms", []):
        english = str(term.get("english") or "")
        zulu = str(term.get("zulu") or "")
        if not english or not zulu:
            continue
        if english.casefold() in source.casefold() and zulu.casefold() in candidate.casefold():
            conforming += 1
    return conforming


# --- Candidate translation, QA, TTS measurement, and speaker-turn rebalancing --
# Phase 9 of the --candidate-pool architecture: translate each sentence's ONE natural
# English text through Grok, QA-check/repair it, measure survivors for real, then
# group sentences into real speaker turns and rebalance (Phase 13) for timing fit
# and pacing evenness. Writes pass2_candidate_pool.json ONLY -- see the approved
# plan for why this never touches pass2_natural_translation.json directly.


def _candidate_translate_schema(expected_unit_ids: Sequence[str]) -> dict[str, Any]:
    requested_ids = [str(value) for value in expected_unit_ids]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Candidate translate schema received duplicate expected unit IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["translations"],
        "properties": {
            "translations": {
                "type": "array",
                "minItems": len(requested_ids),
                "maxItems": len(requested_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["unit_id", "zulu_text", "register_notes", "clauses", "shortened_candidates"],
                    "properties": {
                        "unit_id": {"type": "string", "enum": requested_ids},
                        "zulu_text": {"type": "string", "minLength": 1},
                        "register_notes": {"type": "string"},
                        "clauses": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["clause_id", "english_text", "rank"],
                                "properties": {
                                    "clause_id": {"type": "string", "minLength": 1, "maxLength": 8},
                                    "english_text": {"type": "string", "minLength": 1},
                                    "rank": {"type": "integer", "minimum": 1},
                                },
                            },
                        },
                        "shortened_candidates": {
                            "type": "array",
                            "maxItems": _MAX_SHORTENED_CANDIDATES_PER_SEGMENT,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["dropped_clause_ids", "isizulu_text"],
                                "properties": {
                                    "dropped_clause_ids": {
                                        "type": "array",
                                        "minItems": 0,
                                        "items": {"type": "string"},
                                    },
                                    "isizulu_text": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _request_candidate_translation_batch(
    *,
    provider: FoundryGrokProvider,
    units: Sequence[Mapping[str, Any]],
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """units: [{"unit_id", "english_text", "preceding_english"?, "reference_vocabulary"?}, ...].
    "preceding_english" is optional per-unit, real grounding (not a described
    pattern) for a sentence whose correct translation depends on the
    immediately preceding sentence's content (elliptical denials, anaphora).
    "reference_vocabulary" is an optional {english_word: [isiZulu options]}
    hint from a general-purpose bilingual lexicon (see zulu_lexicon.py) --
    supplementary, never authoritative; see CANDIDATE_TRANSLATE_SYSTEM_PROMPT's
    own rules for how both may be used. Identity mismatch is fatal: this feeds
    a deterministic downstream pipeline, not a best-effort audit.

    Returns {unit_id: {"zulu_text": str, "shortened_candidates": [...]}}.
    Real gap found on job fb3d08b63fed4d90922b08f7e325b906, 2026-09-07: this
    was the ONLY translation path with no droppable-content mechanism at
    all (unlike TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT's clauses/
    shortened_candidates) -- so any sentence that fell back here (a failed
    turn-block merge, or --skip-turn-block-translation) permanently lost
    its chance at _trim_by_fact_priority, regardless of how obviously
    droppable its content was (e.g. "Despite the cold and wet weather," a
    pure scene-setting clause). Now mirrors that same mechanism per-unit,
    canonicalized via the same _canonicalize_segment_clauses/
    _canonicalize_shortened_candidates already proven for it -- a
    malformed/missing clauses shape degrades this ONE unit's
    shortened_candidates to empty (never fatal to the whole batch), since a
    unit's own droppable-content offer is exactly as optional here as it
    already is for a turn-block segment.
    """
    expected_ids = [str(item["unit_id"]) for item in units]
    response = provider.complete_json(
        operation="native_candidate_translate_batch",
        system_prompt=CANDIDATE_TRANSLATE_SYSTEM_PROMPT,
        # Field order matters for real money: Azure Foundry's prompt cache is a
        # strict shared-PREFIX match (see the identical reasoning at the
        # temporal-mask window payload above). zulu_grammar_foundation and
        # confirmed_translation_pitfalls are both module-level constants,
        # identical across every job and every call ever made;
        # zulu_terminology_glossary/context_ledger are constant across every
        # batch WITHIN one job. Ordering from most- to least-stable keeps the
        # longest possible byte-identical prefix cacheable across batches.
        payload={
            "zulu_grammar_foundation": ZULU_GRAMMAR_FOUNDATION,
            "confirmed_translation_pitfalls": list(CONFIRMED_TRANSLATION_PITFALLS),
            "zulu_terminology_glossary": dict(glossary),
            "context_ledger": dict(context_ledger or {}),
            "units": [
                {
                    "unit_id": str(item["unit_id"]), "english_text": str(item["english_text"]),
                    **(
                        {"preceding_english": str(item["preceding_english"])}
                        if item.get("preceding_english")
                        else {}
                    ),
                    **(
                        {"reference_vocabulary": dict(item["reference_vocabulary"])}
                        if item.get("reference_vocabulary")
                        else {}
                    ),
                }
                for item in units
            ],
        },
        schema=_candidate_translate_schema(expected_ids),
        # 450/unit was calibrated when each unit's response was just a bare
        # zulu_text; confirmed real 2026-09-08 that it truncates even a
        # 2-unit batch (finish_reason "length" at exactly the 1200 floor)
        # now that every unit also carries a full clauses/shortened_candidates
        # ladder (added 2026-09-07, see this function's own docstring) --
        # match the sibling turn-block-translate call's per-item budget
        # (700/unit), which already handles the identical response shape.
        max_output_tokens=min(
            max(1600, 700 * len(units)),
            int(getattr(provider.config, "max_output_tokens", 8192)),
        ),
    )
    returned = list(response.data.get("translations") or [])
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        unit_id = str(item.get("unit_id") or "")
        if unit_id in by_id:
            duplicate_ids.append(unit_id)
            continue
        if unit_id not in expected_set:
            unknown_ids.append(unit_id)
            continue
        zulu_text = str(item.get("zulu_text") or "").strip()
        clauses = _canonicalize_segment_clauses(item.get("clauses"))
        shortened_candidates = (
            _canonicalize_shortened_candidates(
                item.get("shortened_candidates"), isizulu_text=zulu_text,
                clause_ids=[c["clause_id"] for c in clauses],
            )
            if clauses is not None and zulu_text
            else []
        )
        by_id[unit_id] = {
            "zulu_text": zulu_text, "clauses": clauses, "shortened_candidates": shortened_candidates,
            "register_notes": str(item.get("register_notes") or "").strip(),
        }

    missing_ids = [unit_id for unit_id in expected_ids if unit_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Candidate translate batch identity mismatch: " + "; ".join(details))

    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return by_id, usage


def _turn_block_translate_schema(expected_window_ids: Sequence[str]) -> dict[str, Any]:
    requested_ids = [str(value) for value in expected_window_ids]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Turn block translate schema received duplicate expected window IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["windows"],
        "properties": {
            "windows": {
                "type": "array",
                "minItems": len(requested_ids),
                "maxItems": len(requested_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["window_id", "segments"],
                    "properties": {
                        "window_id": {"type": "string", "enum": requested_ids},
                        "segments": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "start_index", "end_index", "communicative_goal", "register_notes",
                                    "clauses", "isizulu_text", "shortened_candidates",
                                ],
                                "properties": {
                                    "start_index": {"type": "integer", "minimum": 1},
                                    "end_index": {"type": "integer", "minimum": 1},
                                    "communicative_goal": {"type": "string", "minLength": 1},
                                    "register_notes": {"type": "string"},
                                    "clauses": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 16,
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["clause_id", "english_text", "rank"],
                                            "properties": {
                                                "clause_id": {"type": "string", "minLength": 1, "maxLength": 8},
                                                "english_text": {"type": "string", "minLength": 1},
                                                "rank": {"type": "integer", "minimum": 1},
                                            },
                                        },
                                    },
                                    "isizulu_text": {"type": "string", "minLength": 1},
                                    "shortened_candidates": {
                                        "type": "array",
                                        "maxItems": _MAX_SHORTENED_CANDIDATES_PER_SEGMENT,
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["dropped_clause_ids", "isizulu_text"],
                                            "properties": {
                                                "dropped_clause_ids": {
                                                    "type": "array",
                                                    "minItems": 0,
                                                    "items": {"type": "string"},
                                                },
                                                "isizulu_text": {"type": "string", "minLength": 1},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _request_turn_block_translation_batch(
    *,
    provider: FoundryGrokProvider,
    windows: Sequence[Mapping[str, Any]],
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """windows: [{"window_id", "members": [{"index", "english_text"}, ...],
    "preceding_english"?, "recent_turns_digest"?, "speaker_role"?,
    "reference_vocabulary"?, "target_syllables"?, "min_syllables"?,
    "max_syllables"?}, ...]. Returns {window_id: raw segments as returned --
    NOT yet checked for exact contiguous tiling, see
    _validate_turn_block_segments}. Identity mismatch on window_id is fatal
    (same convention as every other candidate-translate-family function);
    per-window segment SHAPE problems are the caller's job to detect and
    degrade, not this function's.
    """
    expected_ids = [str(w["window_id"]) for w in windows]
    response = provider.complete_json(
        operation="native_turn_block_translate_batch",
        system_prompt=TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT,
        payload={
            "confirmed_translation_pitfalls": list(CONFIRMED_TRANSLATION_PITFALLS),
            "zulu_terminology_glossary": dict(glossary),
            "context_ledger": dict(context_ledger or {}),
            "windows": [
                {
                    "window_id": str(w["window_id"]),
                    "members": [
                        {"index": int(m["index"]), "english_text": str(m["english_text"])}
                        for m in w["members"]
                    ],
                    **(
                        {"preceding_english": str(w["preceding_english"])}
                        if w.get("preceding_english") else {}
                    ),
                    **(
                        {"recent_turns_digest": str(w["recent_turns_digest"])}
                        if w.get("recent_turns_digest") else {}
                    ),
                    **(
                        {"speaker_role": str(w["speaker_role"])}
                        if w.get("speaker_role") else {}
                    ),
                    **(
                        {"reference_vocabulary": dict(w["reference_vocabulary"])}
                        if w.get("reference_vocabulary") else {}
                    ),
                    **(
                        {"target_syllables": int(w["target_syllables"])}
                        if w.get("target_syllables") is not None else {}
                    ),
                    **(
                        {"min_syllables": int(w["min_syllables"])}
                        if w.get("min_syllables") is not None else {}
                    ),
                    **(
                        {"max_syllables": int(w["max_syllables"])}
                        if w.get("max_syllables") is not None else {}
                    ),
                }
                for w in windows
            ],
        },
        schema=_turn_block_translate_schema(expected_ids),
        max_output_tokens=min(
            max(1500, 700 * sum(len(w["members"]) for w in windows)),
            int(getattr(provider.config, "max_output_tokens", 8192)),
        ),
    )
    returned = list(response.data.get("windows") or [])
    expected_set = set(expected_ids)
    by_id: dict[str, list[dict[str, Any]]] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        window_id = str(item.get("window_id") or "")
        if window_id in by_id:
            duplicate_ids.append(window_id)
            continue
        if window_id not in expected_set:
            unknown_ids.append(window_id)
            continue
        by_id[window_id] = list(item.get("segments") or [])

    missing_ids = [window_id for window_id in expected_ids if window_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Turn block translate batch identity mismatch: " + "; ".join(details))

    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return by_id, usage


def _canonicalize_segment_clauses(raw_clauses: Any) -> list[dict[str, Any]] | None:
    """A segment's clauses must be a non-empty list of {clause_id, english_text,
    rank} with unique, non-empty clause_ids -- the structural requirement that
    makes clause ranking (and therefore the reduction ladder) mandatory rather
    than optional. Returns None on any malformed shape (missing field, wrong
    type, duplicate id) so the caller can treat this as the same kind of
    recoverable per-window failure as a bad start_index/end_index.
    """
    clauses: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in list(raw_clauses or []):
        try:
            clause_id = str(raw["clause_id"]).strip()
            english_text = str(raw["english_text"]).strip()
            rank = int(raw["rank"])
        except (KeyError, TypeError, ValueError):
            return None
        if not clause_id or not english_text or clause_id in seen_ids:
            return None
        seen_ids.add(clause_id)
        clauses.append({"clause_id": clause_id, "english_text": english_text, "rank": rank})
    if not clauses:
        return None
    return clauses


def _canonicalize_shortened_candidates(
    raw_candidates: Any, *, isizulu_text: str, clause_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Keep only candidates that are actually usable: {dropped_clause_ids,
    isizulu_text}, every dropped_clause_id referencing a real clause on this
    segment (a candidate naming an unknown id is silently dropped, not fatal
    to the segment), non-empty text after stripping, deduplicated by text,
    capped at _MAX_SHORTENED_CANDIDATES_PER_SEGMENT, and genuinely different
    from isizulu_text itself (a "shortened" candidate identical to the
    original offers nothing). Unlike a span, a candidate's isizulu_text is a
    COMPLETE alternative -- no substring-of-the-original check is needed or
    wanted, since the whole point is that the model may have re-smoothed
    wording at the cut seam. Order is preserved -- it encodes priority
    (lightest cut first), and dropped_clause_ids is kept for the audit trail
    a promoted trim always carries.

    ``dropped_clause_ids`` may be EMPTY: a restructure-only candidate (every
    clause kept, just reworded into a tighter, less convoluted construction
    -- see TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT's own restructuring-tier
    guidance) is a legitimate, distinct kind of candidate from a drop, and is
    still rejected here if its text doesn't actually differ from the
    original (a "restructured" candidate identical to isizulu_text offers
    nothing, same as an empty-drop candidate would).
    """
    original = isizulu_text.strip()
    known_ids = {str(value) for value in clause_ids}
    candidates: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for raw in list(raw_candidates or [])[: _MAX_SHORTENED_CANDIDATES_PER_SEGMENT * 2]:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("isizulu_text") or "").strip()
        dropped = [str(value).strip() for value in (raw.get("dropped_clause_ids") or [])]
        dropped = [value for value in dropped if value]
        if not text or text in seen_text or text == original:
            continue
        if known_ids and not set(dropped).issubset(known_ids):
            continue
        seen_text.add(text)
        candidates.append({"dropped_clause_ids": dropped, "isizulu_text": text})
        if len(candidates) >= _MAX_SHORTENED_CANDIDATES_PER_SEGMENT:
            break
    return candidates


def _validate_turn_block_segments(
    *, member_count: int, segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """A window's raw response is valid ONLY if it contains EXACTLY ONE segment
    covering the whole window (start_index=1, end_index=member_count) -- Phase
    23 removed the earlier multi-segment tiling contract entirely (see the
    deleted _turn_block_segment_shows_real_consolidation): a window is always
    retold and ranked as one whole, so there is no longer a "did this partial
    merge earn its keep" question for multiple segments to answer. The segment
    must also carry a real, non-empty clause breakdown (see
    _canonicalize_segment_clauses -- this is what makes "nothing droppable
    here" a conclusion reachable only after ranking, never a way to skip it).
    Any violation (more than one segment, wrong range, empty text,
    malformed/missing clauses) is a recoverable, per-window failure -- returns
    None so the caller degrades the whole window to the per-sentence fallback
    path, never a partial promotion.

    The surviving segment also carries canonicalized ``shortened_candidates``
    (see _canonicalize_shortened_candidates) -- a candidate-level problem
    never fails the segment, since an empty list is always a safe, valid
    answer (it just means every clause here ranked among the most essential
    -- see the prompt's own rule for when that is actually true).
    """
    if len(segments) != 1:
        return None
    segment = segments[0]
    try:
        clauses = _canonicalize_segment_clauses(segment.get("clauses"))
        if clauses is None:
            return None
        isizulu_text = str(segment["isizulu_text"]).strip()
        start_index = int(segment["start_index"])
        end_index = int(segment["end_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if start_index != 1 or end_index != member_count or not isizulu_text:
        return None
    return [{
        "start_index": start_index,
        "end_index": end_index,
        "communicative_goal": str(segment.get("communicative_goal") or "").strip(),
        "register_notes": str(segment.get("register_notes") or "").strip(),
        "clauses": clauses,
        "isizulu_text": isizulu_text,
        "shortened_candidates": _canonicalize_shortened_candidates(
            segment.get("shortened_candidates"),
            isizulu_text=isizulu_text,
            clause_ids=[c["clause_id"] for c in clauses],
        ),
    }]


def _build_turn_block_group(
    *,
    group_id: str,
    members: Sequence[Mapping[str, Any]],
    shortened_candidates: Sequence[Mapping[str, Any]] = (),
    clauses: Sequence[Mapping[str, Any]] | None = None,
    communicative_goal: str | None = None,
    register_notes: str | None = None,
) -> dict[str, Any]:
    """Build one committed block-group dict from a contiguous run of raw,
    per-sentence groups (`members`, in order). Mirrors _speaker_turn_geometry's
    own "first member's start, last member's end" pattern, already confirmed
    reusable for an arbitrary member-id list, not turn-specific. A size-1
    `members` list (nothing merged) produces a group byte-shape-identical to
    _build_temporal_mask_source_groups's own output, plus member_group_ids.

    ``shortened_candidates`` (empty by default -- the per-sentence fallback
    path has no candidates of its own to offer) is generated alongside this
    block's own translation and carried on the block itself so
    _trim_by_fact_priority can find it later without any separate lookup.
    Each entry is ``{"dropped_clause_ids": [...], "isizulu_text": "..."}``
    (see _canonicalize_shortened_candidates).

    ``clauses`` (the same ranked breakdown that produced shortened_candidates,
    see _canonicalize_segment_clauses) is carried alongside for the same
    reason. Real confirmed bug (job fb3d08b63fed4d90922b08f7e325b906,
    2026-09-09): this parameter did not exist until _filter_safe_shortened_
    candidates grew a clauses-aware mode to stop rejecting a candidate for
    dropping a literal that lives ONLY inside a clause it deliberately
    dropped -- without this field actually reaching the group dict
    _trim_by_fact_priority reads, that fix's own precondition
    (``clauses is not None``) was never true, so it silently never took
    effect. Every call site must now pass the same clauses the shortened_
    candidates it's also passing came from.

    ``communicative_goal`` and ``register_notes`` are the model's own,
    already-required-by-schema explanation of its translation/ranking
    choices (see CANDIDATE_TRANSLATE_SYSTEM_PROMPT/TURN_BLOCK_TRANSLATE_
    SYSTEM_PROMPT's own reasoning-trace instructions) -- carried through for
    the same "log the reason behind a decision" purpose as the fallback-
    reason logging in _translate_turn_blocks_via_grok, per real user
    direction, 2026-09-09: "isn't there a way to also log what the model is
    thinking." ``communicative_goal`` only exists in the turn-block schema
    (a per-sentence fallback unit has no separate goal-identification step);
    ``register_notes`` exists in both.
    """
    member_ids = [str(member["group_id"]) for member in members]
    segment_ids: list[str] = []
    source_segments: list[dict[str, Any]] = []
    for member in members:
        segment_ids.extend(str(value) for value in member["segment_ids"])
        source_segments.extend(dict(segment) for segment in member["source_segments"])
    return {
        "group_id": group_id,
        "speaker_id": str(members[0]["speaker_id"]),
        "segment_ids": segment_ids,
        "member_group_ids": member_ids,
        "start_ms": int(members[0]["start_ms"]),
        "source_end_ms": int(members[-1]["source_end_ms"]),
        "source_span_ms": int(members[-1]["source_end_ms"]) - int(members[0]["start_ms"]),
        "source_segments": source_segments,
        "source_text": " ".join(str(member["source_text"]) for member in members),
        "shortened_candidates": list(shortened_candidates),
        "clauses": list(clauses) if clauses else None,
        "communicative_goal": communicative_goal or None,
        "register_notes": register_notes or None,
    }


def _translate_turn_blocks_via_grok(
    *,
    provider: FoundryGrokProvider,
    groups: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    natural_english_by_group: Mapping[str, str],
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
    preferred_raw_speed_percent: int = DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
    max_members_per_window: int = _TURN_BLOCK_TRANSLATE_MAX_MEMBERS,
    batch_size: int = DEFAULT_TURN_BLOCK_TRANSLATE_BATCH_SIZE,
    workers: int = DEFAULT_CANDIDATE_POOL_WORKERS,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """Phase 16 (Phase 23 revision): translate each speaker turn as one or more
    WINDOWS (bounded, consecutive chunks of up to max_members_per_window
    original sentences -- a call-size limit only, never a merge-eligibility
    decision). Each window is ALWAYS translated and clause-ranked as ONE whole
    retelling unit in a single Grok call (see TURN_BLOCK_TRANSLATE_SYSTEM_
    PROMPT's STRUCTURAL RULE and _validate_turn_block_segments, which now
    requires exactly one segment covering the whole window) -- there is no
    longer a separate "did this merge earn its keep" question for any guard to
    answer; the model's own clause ranking (with its hard rule protecting
    names/numbers/dates/claims) is what prevents information loss, not a
    post-hoc consolidation check. Replaces _translate_natural_via_grok as
    build_candidate_pool's primary translation step; that function is kept,
    unchanged, as the per-sentence fallback for any window that fails
    structural or literal-preservation validation -- a rejected window
    degrades to exactly today's proven mechanism for exactly its own members,
    never a partial or silently-broken result.

    Returns (block_groups, candidate_by_block_group_id, usage_totals).
    block_groups is in true chronological order BY CONSTRUCTION -- the final
    assembly walks `turns`/members in their own already-chronological order,
    so no sort step is ever needed. A single-member turn, or a window that
    fails validation, degenerates to exactly today's one-block-per-sentence
    shape (plus member_group_ids=[gid]) -- there is no separate code path for
    "nothing to merge"; it is the size-1 case of the same mechanism.
    """
    groups_by_id = {str(g["group_id"]): g for g in groups}
    ordered_group_ids = [str(g["group_id"]) for g in groups]
    preceding_english_by_group_id = {
        ordered_group_ids[i]: str(groups_by_id[ordered_group_ids[i - 1]]["source_text"])
        for i in range(1, len(ordered_group_ids))
    }
    # Up to 3 preceding groups' real English (in order), so clause ranking can
    # recognize a fact this window restates as already established elsewhere
    # in the discourse -- e.g. an earlier turn already stating a fact this
    # window's own answer redundantly repeats. Real, confirmed case: a phone-
    # call fact stated once, then re-asked about, then restated a third time
    # -- invisible to ranking without this, since only the ONE immediately
    # preceding sentence reaches preceding_english above.
    recent_turns_digest_by_group_id = {
        gid: " ".join(
            str(groups_by_id[ordered_group_ids[i - k]]["source_text"])
            for k in range(_RECENT_TURNS_DIGEST_MAX_LOOKBACK, 0, -1)
            if i - k >= 0
        )
        for i, gid in enumerate(ordered_group_ids)
    }
    # Resolved deterministically in code, not left for the model to infer from
    # context -- context_ledger.speaker_roles is keyed by speaker_id, already
    # populated for every real job (e.g. "Commission witness", "evidence
    # leader", "programme host"), and already reaches every translation call
    # unused until now. A window's own speaker is known exactly (one turn is
    # always one speaker by construction), so this lookup is unambiguous.
    speaker_role_by_speaker_id = {
        str(entry.get("speaker_id")): str(entry.get("role") or "")
        for entry in (context_ledger or {}).get("speaker_roles") or []
        if entry.get("speaker_id")
    }
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    if not groups:
        return [], {}, usage_totals

    def _payload_window(window_id: str, member_ids: Sequence[str]) -> dict[str, Any]:
        members = [groups_by_id[gid] for gid in member_ids]
        combined_source_text = " ".join(str(member["source_text"]) for member in members)
        syllable_budget = _rolling_syllable_budget(
            {
                "start_ms": 0,
                "end_ms": int(members[-1]["source_end_ms"]) - int(members[0]["start_ms"]),
            },
            preferred_raw_speed_percent=preferred_raw_speed_percent,
            content_syllables_per_second=DEFAULT_ZULU_CONTENT_SYLLABLES_PER_SECOND,
            azure_utterance_overhead_ms=DEFAULT_AZURE_UTTERANCE_OVERHEAD_MS,
            tolerance_percent=DEFAULT_SYLLABLE_TOLERANCE_PERCENT,
        )
        return {
            "window_id": window_id,
            "members": [
                {"index": index + 1, "english_text": natural_english_by_group[gid]}
                for index, gid in enumerate(member_ids)
            ],
            "preceding_english": preceding_english_by_group_id.get(member_ids[0], ""),
            "recent_turns_digest": recent_turns_digest_by_group_id.get(member_ids[0], ""),
            "speaker_role": speaker_role_by_speaker_id.get(
                str(groups_by_id[member_ids[0]]["speaker_id"]), "",
            ),
            "reference_vocabulary": zulu_lexicon.relevant_entries(combined_source_text),
            "target_syllables": syllable_budget["target_syllables"],
            "min_syllables": syllable_budget["min_syllables"],
            "max_syllables": syllable_budget["max_syllables"],
        }

    def _run_translate_batches(
        payload_windows_by_batch: Sequence[Sequence[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Run one or more batches of already-built payload windows through
        _request_turn_block_translation_batch, accumulating usage_totals and
        emitting a fallback-worthy log line per failed batch. Returns raw,
        not-yet-validated segments keyed by window_id -- a window whose batch
        outright failed simply has no entry (caller treats that exactly like
        a validation failure).
        """
        by_window_id: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(payload_windows_by_batch)))
        ) as pool:
            futures: dict[Any, Sequence[dict[str, Any]]] = {}
            for payload_windows in payload_windows_by_batch:
                future = pool.submit(
                    _request_turn_block_translation_batch, provider=provider, windows=payload_windows,
                    glossary=glossary, context_ledger=context_ledger,
                )
                futures[future] = payload_windows
            for future in as_completed(futures):
                payload_windows = futures[future]
                try:
                    result, usage = future.result()
                except ValueError as exc:
                    _emit_progress(progress, f"[native candidate pool] Turn block translate batch failed: {exc}")
                    continue
                for key in usage_totals:
                    usage_totals[key] += usage.get(key, 0)
                by_window_id.update(result)
        return by_window_id

    all_windows: list[dict[str, Any]] = []
    for turn in turns:
        member_ids = [str(gid) for gid in turn["member_group_ids"]]
        for chunk in _chunk_sequence(member_ids, max_members_per_window):
            all_windows.append({"window_id": chunk[0], "member_ids": chunk})

    validated_segments_by_window_id: dict[str, list[dict[str, Any]]] = {}
    fallback_member_ids: list[str] = []

    initial_batches = [
        [_payload_window(w["window_id"], w["member_ids"]) for w in batch]
        for batch in _chunk_sequence(all_windows, batch_size)
    ]
    by_window_id = _run_translate_batches(initial_batches)
    for window in all_windows:
        window_id = window["window_id"]
        member_ids = window["member_ids"]
        raw_segments = by_window_id.get(window_id)
        if raw_segments is None:
            _emit_progress(
                progress,
                f"[native candidate pool] Window {window_id} {member_ids}: no response from the "
                "turn-block translate batch (request/parse failure, see above) -- falling back to "
                "independent per-sentence translation.",
            )
            fallback_member_ids.extend(member_ids)
            continue
        validated = _validate_turn_block_segments(member_count=len(member_ids), segments=raw_segments)
        if validated is None:
            _emit_progress(
                progress,
                f"[native candidate pool] Window {window_id} {member_ids}: response failed structural "
                "validation (not exactly one segment covering the whole window, or a missing/malformed "
                "clause breakdown) -- falling back to independent per-sentence translation.",
            )
            fallback_member_ids.extend(member_ids)
            continue
        members = [groups_by_id[gid] for gid in member_ids]
        combined_source_text = " ".join(str(member["source_text"]) for member in members)
        combined_candidate_text = " ".join(segment["isizulu_text"] for segment in validated)
        missing = _missing_required_literals(
            source_text=combined_source_text, candidate_text=combined_candidate_text,
        )
        if missing:
            missing_desc = ", ".join(
                f"{item['literal']!r} (needed {item['required_count']}x)" for item in missing
            )
            _emit_progress(
                progress,
                f"[native candidate pool] Window {window_id} {member_ids}: merged translation dropped "
                f"required literal(s) [{missing_desc}] -- falling back to independent per-sentence "
                "translation.",
            )
            fallback_member_ids.extend(member_ids)
            continue
        validated_segments_by_window_id[window_id] = validated

    fallback_candidates: dict[str, dict[str, Any]] = {}
    if fallback_member_ids:
        _emit_progress(
            progress,
            f"[native candidate pool] {len(fallback_member_ids)} sentence(s) fell back to independent "
            "per-sentence translation (their window failed merge or literal-preservation validation).",
        )
        fallback_groups = [groups_by_id[gid] for gid in fallback_member_ids]
        fallback_candidates, fallback_usage = _translate_natural_via_grok(
            provider=provider, groups=fallback_groups, natural_english_by_group=natural_english_by_group,
            glossary=glossary, context_ledger=context_ledger, workers=workers, progress=progress,
            preceding_english_by_group_id=preceding_english_by_group_id,
        )
        for key in usage_totals:
            usage_totals[key] += fallback_usage.get(key, 0)

    block_groups: list[dict[str, Any]] = []
    candidate_by_block_group_id: dict[str, dict[str, Any]] = {}
    for window in all_windows:
        window_id = window["window_id"]
        member_ids = window["member_ids"]
        members = [groups_by_id[gid] for gid in member_ids]
        validated = validated_segments_by_window_id.get(window_id)
        if validated is not None:
            # Exactly one segment, always -- Phase 23 removed multi-segment
            # tiling entirely, so a validated window is always kept as ONE
            # whole block covering every one of its members.
            segment = validated[0]
            block_group_id = str(members[0]["group_id"])
            block_groups.append(_build_turn_block_group(
                group_id=block_group_id, members=members,
                shortened_candidates=segment.get("shortened_candidates") or [],
                clauses=segment.get("clauses"),
                communicative_goal=segment.get("communicative_goal"),
                register_notes=segment.get("register_notes"),
            ))
            candidate_by_block_group_id[block_group_id] = {
                "candidate_id": "grok_natural", "variant_id": "natural", "translator": "grok",
                "spoken_text": segment["isizulu_text"],
            }
            continue
        for gid in member_ids:
            if gid in fallback_candidates:
                block_groups.append(_build_turn_block_group(
                    group_id=gid, members=[groups_by_id[gid]],
                    shortened_candidates=fallback_candidates[gid].get("shortened_candidates") or [],
                    clauses=fallback_candidates[gid].get("clauses"),
                    register_notes=fallback_candidates[gid].get("register_notes"),
                ))
                candidate_by_block_group_id[gid] = fallback_candidates[gid]

    return block_groups, candidate_by_block_group_id, usage_totals


def _translate_natural_via_grok(
    *,
    provider: FoundryGrokProvider,
    groups: Sequence[Mapping[str, Any]],
    natural_english_by_group: Mapping[str, str],
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
    batch_size: int = DEFAULT_QA_BACK_TRANSLATION_BATCH_SIZE,
    workers: int = DEFAULT_CANDIDATE_POOL_WORKERS,
    progress: Callable[[str], None] | None = None,
    preceding_english_by_group_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Translate each sentence's ONE natural (filler-stripped) English text through
    Grok, in parallel across sentence batches -- no cross-unit dependency (the
    glossary already carries whatever rolling continuity_context used to), same
    ThreadPoolExecutor pattern as _run_qa_verdicts. Exactly one candidate per
    sentence -- there is no length ladder to fan out over anymore (see Phase 9).

    Each unit also carries the immediately preceding sentence's English as
    "preceding_english" -- real grounding for a sentence whose correct meaning
    depends on it (an elliptical denial, an anaphoric reference), computed
    once from groups' own known order (no sequential translation dependency
    reintroduced; every unit's preceding text is already known upfront) -- and
    "reference_vocabulary", a per-sentence lookup against the Autshumato
    English-isiZulu lexicon (zulu_lexicon.py) for this sentence's own content
    words. See CANDIDATE_TRANSLATE_SYSTEM_PROMPT's own rules for how both may
    be used.

    By default, "preceding_english" comes from simple list-adjacency within
    `groups` (today's exact behavior, unchanged). Phase 16's turn-block
    translation fallback path calls this on a SCATTERED subset of groups
    (whichever windows failed validation) where list-adjacency would be
    wrong -- pass `preceding_english_by_group_id` (an explicit, global
    lookup keyed by group_id) to override it for exactly those callers;
    omitting it preserves today's behavior byte-for-byte.
    """
    units = [
        {
            "unit_id": str(group["group_id"]),
            "english_text": natural_english_by_group[str(group["group_id"])],
            "preceding_english": (
                preceding_english_by_group_id.get(str(group["group_id"]), "")
                if preceding_english_by_group_id is not None
                else (str(groups[index - 1]["source_text"]) if index > 0 else "")
            ),
            "reference_vocabulary": zulu_lexicon.relevant_entries(
                natural_english_by_group[str(group["group_id"])],
            ),
        }
        for index, group in enumerate(groups)
    ]

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    candidate_by_group: dict[str, dict[str, Any]] = {}
    if not units:
        return candidate_by_group, usage_totals

    batches = _chunk_sequence(units, batch_size)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(batches)))) as pool:
        futures = {
            pool.submit(
                _request_candidate_translation_batch, provider=provider, units=batch,
                glossary=glossary, context_ledger=context_ledger,
            ): len(batch)
            for batch in batches
        }
        for future in as_completed(futures):
            by_id, usage = future.result()
            for key in usage_totals:
                usage_totals[key] += usage.get(key, 0)
            for group_id, result in by_id.items():
                zulu_text = str(result.get("zulu_text") or "")
                if not zulu_text:
                    continue
                candidate_by_group[group_id] = {
                    "candidate_id": "grok_natural",
                    "variant_id": "natural",
                    "translator": "grok",
                    "spoken_text": zulu_text,
                    "clauses": result.get("clauses"),
                    "shortened_candidates": list(result.get("shortened_candidates") or []),
                    "register_notes": result.get("register_notes") or None,
                }
            completed += 1
            _emit_progress(
                progress,
                f"[native candidate pool] Grok batch {completed}/{len(batches)} complete ({futures[future]} unit(s))",
            )
    return candidate_by_group, usage_totals


def _enrich_measured_candidates(
    *, measured: Sequence[Mapping[str, Any]], original_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """_measure_temporal_candidate returns a fresh dict carrying only measurement
    fields (see its own return statement) -- it knows nothing about candidate-pool
    metadata. Merge translator/variant_id/qa_penalty/qa_record/requires_review back
    in, matched by candidate_id (always preserved through _measure_temporal_batch),
    before this reaches selection or the written artifact. Also carries through
    dropped_clause_ids/omitted_items when present -- the audit trail for a
    fact-priority-trim or last-resort-compression candidate, so a promoted
    trim's committed record always shows exactly what was touched.
    """
    by_id = {str(item["candidate_id"]): item for item in original_candidates}
    enriched: list[dict[str, Any]] = []
    for item in measured:
        original = by_id.get(str(item["candidate_id"]), {})
        merged = dict(item)
        merged["translator"] = original.get("translator")
        merged["variant_id"] = original.get("variant_id")
        merged["qa_penalty"] = int(original.get("qa_penalty", 0))
        merged["qa_record"] = original.get("qa_record")
        merged["requires_review"] = bool(original.get("requires_review", False))
        if original.get("dropped_clause_ids"):
            merged["dropped_clause_ids"] = list(original["dropped_clause_ids"])
        if original.get("omitted_items"):
            merged["omitted_items"] = list(original["omitted_items"])
        enriched.append(merged)
    return enriched


# --- Speaker turns --------------------------------------------------------------------
# Groups consecutive same-speaker, timing-adjacent sentences into the PRIMARY timing
# unit (Phase 9, uncapped since Phase 13) -- not a secondary rebalancing fallback on
# top of per-sentence candidates, but the thing "fits its timing" is judged against in
# the first place. Ported from direct_azure_dub.py's proven "performance regions"
# (build_performance_regions), adapted to native_dub's whole-sentence atomic unit.
# Grouping is deliberately speaker/timing continuity only, never discourse/topic
# structure -- detecting genuine "train of thought" boundaries is a fuzzy,
# actively-researched NLP problem that isn't actually necessary here; what makes it
# safe to share timing budget between two sentences is that the same person said them
# close together, not that they're discussing the same idea. A real turn is
# UNBOUNDED -- no duration or sentence-count cap, and no bridge-forced break: a short
# interjection (see _classify_discourse_unit_kind) now merges into a turn like any
# other sentence when speaker continuity holds. Bridge classification survives only
# for its one unrelated, still-real use: judging a solo short utterance on whether it
# collides with its neighbors (see the bridge timing model in
# build_candidate_pool/_run_batch_adaptive_timing_controller), not on grouping.


def _classify_discourse_unit_kind(group: Mapping[str, Any]) -> str:
    """"bridge" for a short interjection with no real discourse content (e.g. "All
    right.", "Paragraph four."), "normal" otherwise. Both the word-count AND the
    source-span condition must hold -- a short but slow-spoken utterance, or a
    fast-spoken longer one, is still "normal".
    """
    source_text = str(group.get("source_text") or "")
    word_count = len(source_text.split())
    source_span_ms = int(group.get("source_span_ms") or 0)
    if word_count <= _BRIDGE_MAX_WORD_COUNT and source_span_ms <= _BRIDGE_MAX_SOURCE_SPAN_MS:
        return "bridge"
    return "normal"


def _build_speaker_turns(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive groups into real, UNCAPPED speaker turns -- a turn extends
    for as long as the same speaker keeps talking with no real pause, full stop
    (Phase 13). Earlier versions of this grouping (Phase 9's "discourse segments")
    also capped combined duration at 30s, sentence count at 8, and force-broke
    around any "bridge" interjection even mid-turn for the SAME speaker -- three
    artificial fragmentation rules that had nothing to do with speaker or timing
    discontinuity, confirmed by direct read to be a deliberate (and now removed)
    choice inside this function, not something forced by how sentences are
    chunked upstream. `_classify_discourse_unit_kind`/bridge classification is
    unaffected by this change -- it still exists, just decoupled from grouping
    (see BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS's own render-time use).

    Returns [{"member_group_ids": [...], "start_ms": int, "source_end_ms": int},
    ...] covering every input group exactly once, in order. A turn of size 1 is
    just an ordinary sentence -- every earlier phase's existing behavior is
    exactly what happens when a turn never grows past that. Unlike the earlier
    "discourse segment" version, no "kind" field is returned -- bridge/normal
    classification is a per-SENTENCE render-time signal (see
    _classify_discourse_unit_kind's own callers), not a property of a turn as a
    whole, since a bridge sentence can now sit anywhere inside a longer turn.
    """
    ordered = list(groups)
    turns: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        first = ordered[index]
        member_ids = [str(first["group_id"])]
        turn_start = int(first["start_ms"])
        previous_end = int(first["source_end_ms"])
        cursor = index + 1
        while cursor < len(ordered):
            candidate = ordered[cursor]
            gap_ms = int(candidate["start_ms"]) - previous_end
            safe = (
                str(candidate["speaker_id"]) == str(first["speaker_id"])
                and 0 <= gap_ms <= SPEAKER_TURN_MAX_GAP_MS
            )
            if not safe:
                break
            member_ids.append(str(candidate["group_id"]))
            previous_end = int(candidate["source_end_ms"])
            cursor += 1
        turns.append({
            "member_group_ids": member_ids,
            "start_ms": turn_start,
            "source_end_ms": previous_end,
        })
        index = cursor
    return turns


# A large enough spread (percentage points) between members' own required-speed
# values, even when the segment already fits its total envelope, means one member
# is being rushed while a neighbor has spare room going unused -- worth attempting
# a redistribution purely for evenness, not just to fix an outright failure to fit.
# Conservative starting value; the trigger is logged every time it fires so it can
# be tuned against real jobs before it needs adjusting.
_SPEAKER_TURN_EVENNESS_TRIGGER_THRESHOLD = 30.0
# Balances two priorities the user identified directly: (1) naturalness from the
# Zulu segment finishing in sync with the English segment, (2) naturalness from
# never colliding with the next segment. A rush absorbed into REAL silence before
# the next segment ("gap_aware" -- see _speed_fit_mode_label) is inaudible: it is
# the mechanism that makes priority (2) safe, so it stays accepted up to the full
# DEFAULT_MAX_NATURAL_SPEED_PERCENT ceiling. A rush with no such gap to absorb into
# ("sentence_end" -- genuine pitch-preserving compression of the speech itself) is
# audible well before that ceiling; confirmed by the user directly ("rushes above
# between 10% and 20% creates unnatural speech... should only be accepted if we are
# rushing to keep sentence to sentence space with the next discourse segment"). A
# segment that nominally "fits" under the coarse ceiling but is doing so via an
# unabsorbed sentence_end rush above this stricter threshold still triggers a
# rebalance attempt -- redistribution that converts a sentence_end rush into a
# gap_aware one (or removes the need for it) is preferred over accepting it as-is.
_SPEAKER_TURN_SENTENCE_END_RUSH_TRIGGER_THRESHOLD = 10.0


def _speaker_turn_geometry(
    segment: Mapping[str, Any],
    *,
    groups_by_id: Mapping[str, Mapping[str, Any]],
    next_source_start_ms: int,
    preferred_raw_speed_percent: int,
) -> dict[str, Any]:
    """Build one speaker turn's timing envelope by feeding a synthetic,
    turn-shaped mapping into the existing, unmodified _temporal_mask_geometry --
    safe because that function only reads start_ms/source_end_ms/source_span_ms
    off its `group` argument. This is a decision signal only (see the
    module-level scope note above): the actual per-sentence render-time
    speed-fit stays per-sentence.
    """
    members = [groups_by_id[gid] for gid in segment["member_group_ids"]]
    synthetic_segment_group = {
        "start_ms": int(members[0]["start_ms"]),
        "source_end_ms": int(members[-1]["source_end_ms"]),
        "source_span_ms": int(members[-1]["source_end_ms"]) - int(members[0]["start_ms"]),
    }
    return _temporal_mask_geometry(
        synthetic_segment_group,
        next_source_start_ms=next_source_start_ms,
        preferred_raw_speed_percent=preferred_raw_speed_percent,
        max_natural_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        mouth_tolerance_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
        max_protected_gap_overflow_ms=DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS,
        min_protected_pause_ms=DEFAULT_MIN_PROTECTED_PAUSE_MS,
    )


def _measure_speaker_turn_total(
    *,
    segment: Mapping[str, Any],
    member_winners: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Stitch each member's already-synthesized WAV into one continuous turn WAV,
    using AZURE_SENTENCE_BOUNDARY_PAUSE_MS as the inter-sentence silence (floored
    at DEFAULT_MIN_PROTECTED_PAUSE_MS), and return the stitched total duration.

    Previously used the real ASR-measured gap between consecutive members' English
    source instead -- wrong model, confirmed by real measurement (2026-09-05): Phase
    14's real one-shot bookmarked synthesis pays AZURE's own sentence-boundary pause
    (consistently ~1000-1050ms across every real measurement taken), not a pause
    matching the original English speaker's real timing, which is what the ASR gap
    actually measures. Using the real ASR gap here systematically understated a
    multi-member turn's true rendered cost by 5-12 points of required_speed on real
    jobs. A pure measurement device -- the actual rendered audio is unaffected; only
    which wording decisions get made from this measurement changes. Reuses
    speech_islands.py's stitch_islands_to_group_wav, already used for an analogous
    purpose elsewhere in this file (_restitch_island_measurements).
    """
    member_ids = list(segment["member_group_ids"])
    wav_paths = [Path(str(member_winners[gid]["path"])) for gid in member_ids]
    gaps_ms = [
        max(DEFAULT_MIN_PROTECTED_PAUSE_MS, AZURE_SENTENCE_BOUNDARY_PAUSE_MS)
        for _ in range(len(member_ids) - 1)
    ]
    # Bounded regardless of turn size -- joining every member id (as the old,
    # capped-at-8 "discourse segment" version safely did) overflows Windows'
    # 255-character filename component limit once a real, uncapped turn grows
    # past roughly a dozen sentences (confirmed by a real job run: a 15-member
    # turn hit `[Errno 22] Invalid argument`). First/last id plus count is
    # unique enough for this per-job temp-measurement directory.
    output_path = Path(output_root) / f"segment_{member_ids[0]}_to_{member_ids[-1]}_n{len(member_ids)}.stitched.wav"
    return stitch_islands_to_group_wav(wav_paths, output_path, inter_island_silence_ms=gaps_ms)


def _reallocate_turn_windows(
    *,
    turn: Mapping[str, Any],
    groups_by_id: Mapping[str, Mapping[str, Any]],
    winners: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resize each member's window share proportionally to its own real,
    already-measured content duration, keeping the turn's own fixed outer
    START AND END unchanged -- pure arithmetic, no text ever moves and no new
    TTS/Speech-SDK measurement is needed (Phase 13, replacing Phase 10's
    pairwise word-shift). A sentence that was too dense for its own tiny
    native ASR slice borrows window from a neighbor with real spare
    content-to-window ratio; a neighbor with less content gets a smaller
    slice. Lifted verbatim from the "uniform pacing fallback" that used to
    run only after word-shifting failed to fully resolve a segment --
    Phase 10's own real-job run found triggered segments kept the SAME
    aggregate required_speed_percent before/after while spread dropped
    sharply, exactly the signature of this formula already doing the real
    work, not the word-move that preceded it.

    REVERTED (Phase 13.1, 2026-08-31): a version of this function accepted a
    ``trailing_borrow_ms`` that extended the turn's own outer END, before
    splitting shares, into real post-articulation silence -- reasoned to be
    safe because it could never exceed the real gap to the next turn. That
    reasoning covered collision safety but missed a real, confirmed defect:
    the render-time per-sentence machinery (`_run_batch_adaptive_timing_
    controller`, unrelated to this function, always active) INDEPENDENTLY
    grants its OWN gap-borrow on top of whatever `source_end_ms` it's
    handed -- it has no way to know this function already spent some of
    that same real silence. The two compounded: on job 33cd7b46...'s first
    real speaker turn, the committed end was pre-extended by ~400ms here,
    then render borrowed ANOTHER ~400ms on top, and the rendered Zulu audio
    ran audibly past the true mouth-close point (confirmed directly by the
    user listening, then confirmed by silencedetect: real speech continued
    to ~39.5s against a true ASR mouth-close of 38.75s). The turn's own END
    is therefore never extended past the real, raw ASR value -- render's own
    existing per-sentence mechanism is what safely grants gap-borrowing, and
    it already runs unconditionally for every committed group; this function
    doesn't need its own, separate copy of that same allowance.

    Returns one entry per member ({"start_ms", "source_end_ms",
    "source_span_ms"}) -- no "spoken_text" field, since text is never
    touched. The caller always re-scores against the baseline before ever
    promoting this over it.
    """
    member_ids = list(turn["member_group_ids"])
    turn_start_ms = int(groups_by_id[member_ids[0]]["start_ms"])
    turn_end_ms = int(groups_by_id[member_ids[-1]]["source_end_ms"])
    total_span_ms = max(1, turn_end_ms - turn_start_ms)
    min_share_ms = min(200, total_span_ms // max(1, len(member_ids)))
    raw_shares = {gid: max(float(winners[gid].get("measured_ms") or 0.0), 1.0) for gid in member_ids}
    raw_total = sum(raw_shares.values())
    shares_ms = {
        gid: max(min_share_ms, total_span_ms * raw_shares[gid] / raw_total) for gid in member_ids
    }
    scale = total_span_ms / sum(shares_ms.values())
    result: dict[str, dict[str, Any]] = {}
    cursor = turn_start_ms
    for gid in member_ids:
        span = max(1, int(round(shares_ms[gid] * scale)))
        start_ms = cursor
        cursor += span
        result[gid] = {"start_ms": start_ms, "source_end_ms": cursor, "source_span_ms": max(1, cursor - start_ms)}
    last_id = member_ids[-1]
    result[last_id]["source_end_ms"] = turn_end_ms  # absorb rounding drift on the last member
    result[last_id]["source_span_ms"] = max(1, turn_end_ms - result[last_id]["start_ms"])
    return result


def _replace_or_append_candidate(
    candidates: list[dict[str, Any]], winner: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """A rebalanced (or turn-compacted-and-then-rebalanced) winner may keep its
    ORIGINAL candidate_id -- window reallocation never generates new text, only
    effective_start_ms/effective_source_end_ms -- so it must REPLACE the
    matching entry in-place rather than being appended alongside a stale
    duplicate. commit_candidate_pool looks up a winner by candidate_id from
    this exact list. A winner with a genuinely new candidate_id (a promoted
    compaction candidate) is simply appended.
    """
    winner_candidate_id = winner.get("candidate_id")
    for position, candidate in enumerate(candidates):
        if candidate.get("candidate_id") == winner_candidate_id:
            candidates[position] = dict(winner)
            return candidates
    candidates.append(dict(winner))
    return candidates


def _score_speaker_turn(
    *,
    turn: Mapping[str, Any],
    turn_geometry: Mapping[str, Any],
    member_winners: Mapping[str, Mapping[str, Any]],
    member_geometries: Mapping[str, Mapping[str, Any]],
    measurement_audio_root: Path,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Score a turn's real, stitched aggregate fit against its own envelope --
    extracted from _rebalance_speaker_turns's own inline closures (Phase 13) so
    other turn-aware callers can reuse the exact same scoring for their own
    baseline check, not just one member's own number.
    """
    member_ids = list(turn["member_group_ids"])

    def _member_required_speed(gid: str) -> float:
        member_geometry = member_geometries[gid]
        target_ms = int(member_geometry["source_window_ms"]) + int(member_geometry["mouth_close_late_tolerance_ms"])
        measured_ms = max(1, int(member_winners[gid].get("measured_ms") or 0))
        return _temporal_required_rush_percent(measured_ms, target_ms)

    stitched = _measure_speaker_turn_total(
        segment=turn, member_winners=member_winners,
        output_root=Path(measurement_audio_root) / "turns",
    )
    duration_ms = int(stitched["final_duration_ms"])
    direction = _timing_recast_direction(
        duration_ms, int(turn_geometry["source_window_ms"]),
        max_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        mouth_close_early_tolerance_ms=int(turn_geometry["mouth_close_early_tolerance_ms"]),
        mouth_close_late_tolerance_ms=int(turn_geometry["mouth_close_late_tolerance_ms"]),
    )
    speed_fit_target_ms = int(turn_geometry["source_window_ms"]) + int(turn_geometry["mouth_close_late_tolerance_ms"])
    required_speed_percent = _temporal_required_rush_percent(duration_ms, speed_fit_target_ms)
    member_speeds = [_member_required_speed(gid) for gid in member_ids]
    spread = (max(member_speeds) - min(member_speeds)) if member_speeds else 0.0
    total_qa_penalty = sum(int(member_winners[gid].get("qa_penalty", 0)) for gid in member_ids)
    total_syllables = sum(
        _estimate_zulu_syllables(str(member_winners[gid].get("spoken_text") or "")) for gid in member_ids
    )
    fits = direction is None
    speed_fit_mode = _speed_fit_mode_label(int(turn_geometry["mouth_close_late_tolerance_ms"]))
    unabsorbed_rush = bool(
        speed_fit_mode == "sentence_end"
        and required_speed_percent > _SPEAKER_TURN_SENTENCE_END_RUSH_TRIGGER_THRESHOLD
    )
    score = (
        0 if fits else 1,
        1 if unabsorbed_rush else 0,
        round(required_speed_percent, 3), round(spread, 3), total_qa_penalty, -total_syllables,
    )
    info = {
        "fits": fits, "required_speed_percent": required_speed_percent, "spread": spread,
        "speed_fit_mode": speed_fit_mode, "unabsorbed_rush": unabsorbed_rush,
    }
    return score, info


def _rebalance_speaker_turns(
    *,
    groups: Sequence[Mapping[str, Any]],
    winners: Mapping[str, Mapping[str, Any]],
    geometries: Mapping[str, Mapping[str, Any]],
    preferred_raw_speed_percent: int,
    measurement_audio_root: Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Phase 13's orchestrator: group sentences into real, UNCAPPED speaker turns
    (the PRIMARY timing unit -- see _build_speaker_turns), and for every
    multi-member turn that doesn't already fit its envelope evenly and without
    an unabsorbed rush, resize each member's window share proportionally to its
    own real, already-measured content (see _reallocate_turn_windows) -- pure
    arithmetic, no text ever moves, and no new TTS/Speech-SDK call is needed
    since nothing textual changes. Translation and QA never run again here, and
    neither does re-synthesis -- only how much time window each sentence is
    judged against can ever change.

    Never worse than the winners passed in: a turn is only updated if its real
    re-scored result is strictly better than its real baseline.

    Returns (updated_winners, updated_geometries) -- the second dict carries an
    entry for exactly the members of turns that were actually promoted (a
    real, fixed bug: the caller previously never learned a promoted member's
    real reallocated geometry, so anything measuring a NEW candidate against
    that member afterward was silently using the stale, pre-rebalance window).
    """
    groups_by_id = {str(g["group_id"]): g for g in groups}
    group_index_by_id = {str(g["group_id"]): i for i, g in enumerate(groups)}
    turns = _build_speaker_turns(groups)
    updated_winners = dict(winners)
    updated_geometries: dict[str, dict[str, Any]] = {}

    multi_member_turns = [turn for turn in turns if len(turn["member_group_ids"]) > 1]
    if not multi_member_turns:
        return updated_winners, updated_geometries

    turn_geometry_by_index: dict[int, dict[str, Any]] = {}
    for index, turn in enumerate(turns):
        next_start = (
            int(turns[index + 1]["start_ms"]) if index + 1 < len(turns)
            else int(turn["source_end_ms"])
        )
        turn_geometry_by_index[index] = _speaker_turn_geometry(
            turn, groups_by_id=groups_by_id, next_source_start_ms=next_start,
            preferred_raw_speed_percent=preferred_raw_speed_percent,
        )

    for index, turn in enumerate(turns):
        if len(turn["member_group_ids"]) < 2:
            continue
        member_ids = turn["member_group_ids"]
        geometry = turn_geometry_by_index[index]
        # Phase 14 only ever turn-finalizes (one continuous synthesis, ONE
        # shared speed factor for the whole turn) a multi-member turn whose
        # members are ALL "normal" -- any bridge member excludes the whole
        # turn from that path, falling back to today's per-sentence
        # independent rendering. Reporting must match: for an all-normal
        # turn, every member's own individually-split required_speed_percent
        # is fiction (real user feedback, 2026-09-07: "why are we
        # aggregating segments rushes, the turn is suppose to be the unit")
        # -- only the turn's real, stitched aggregate corresponds to what
        # gets rendered. A bridge-containing turn is genuinely rendered
        # per-sentence, so ITS members' own individual numbers stay honest.
        # block_groups themselves never carry a "discourse_unit_kind" key (it
        # only gets attached to the candidate-pool record later, in
        # build_candidate_pool) -- classify directly instead of assuming it's
        # already there.
        has_bridge_member = any(
            _classify_discourse_unit_kind(groups_by_id[gid]) == "bridge" for gid in member_ids
        )

        def _score_state(
            member_winners: Mapping[str, Mapping[str, Any]],
            member_geometries: Mapping[str, Mapping[str, Any]],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return _score_speaker_turn(
                turn=turn, turn_geometry=geometry, member_winners=member_winners,
                member_geometries=member_geometries,
                measurement_audio_root=measurement_audio_root,
            )

        baseline_winners = {gid: updated_winners[gid] for gid in member_ids if gid in updated_winners}
        if len(baseline_winners) < len(member_ids):
            continue  # a member has no winner at all -- shouldn't happen, but nothing to rebalance against
        baseline_geometries = {gid: geometries[gid] for gid in member_ids}
        baseline_score, baseline_info = _score_state(baseline_winners, baseline_geometries)

        if not has_bridge_member:
            # The aggregate is invariant under window reallocation (same
            # stitched duration_ms, same turn-level target window) -- set it
            # now, once, so it's correct even for a turn that never triggers
            # reallocation below (the early-continue "already fits" case).
            for group_id in member_ids:
                winner = dict(updated_winners[group_id])
                winner["required_speed_percent"] = round(baseline_info["required_speed_percent"], 3)
                winner["speed_fit_mode"] = baseline_info["speed_fit_mode"]
                winner["fit"] = bool(baseline_info["fits"])
                updated_winners[group_id] = winner
            baseline_winners = {gid: updated_winners[gid] for gid in member_ids}

        if (
            baseline_info["fits"]
            and baseline_info["spread"] <= _SPEAKER_TURN_EVENNESS_TRIGGER_THRESHOLD
            and not baseline_info["unabsorbed_rush"]
        ):
            continue  # already fits well, evenly, and without an audible unabsorbed rush -- nothing to rebalance

        if not baseline_info["fits"]:
            reason = "doesn't fit its turn envelope"
        elif baseline_info["unabsorbed_rush"]:
            reason = "rushes without absorbing into real neighbor silence (audible compression)"
        else:
            reason = "pacing is uneven"
        _emit_progress(
            progress,
            f"[native candidate pool] Turn {member_ids} {reason} "
            f"(required_speed={baseline_info['required_speed_percent']:.1f}%, spread={baseline_info['spread']:.1f}); "
            "reallocating window shares...",
        )

        reallocated = _reallocate_turn_windows(turn=turn, groups_by_id=groups_by_id, winners=baseline_winners)

        reallocated_groups_by_id = dict(groups_by_id)
        for group_id in member_ids:
            adjustment = reallocated[group_id]
            adjusted_group = dict(groups_by_id[group_id])
            adjusted_group["start_ms"] = adjustment["start_ms"]
            adjusted_group["source_end_ms"] = adjustment["source_end_ms"]
            adjusted_group["source_span_ms"] = adjustment["source_span_ms"]
            reallocated_groups_by_id[group_id] = adjusted_group

        reallocated_geometries: dict[str, dict[str, Any]] = {}
        for position, group_id in enumerate(member_ids):
            global_index = group_index_by_id[group_id]
            next_group_id = (
                member_ids[position + 1] if position + 1 < len(member_ids)
                else (str(groups[global_index + 1]["group_id"]) if global_index + 1 < len(groups) else None)
            )
            next_start = (
                int(reallocated_groups_by_id[next_group_id]["start_ms"]) if next_group_id is not None
                else int(reallocated_groups_by_id[group_id]["source_end_ms"])
            )
            reallocated_geometries[group_id] = _temporal_mask_geometry(
                reallocated_groups_by_id[group_id],
                next_source_start_ms=next_start,
                preferred_raw_speed_percent=int(preferred_raw_speed_percent),
                max_natural_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
                mouth_tolerance_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
                max_protected_gap_overflow_ms=DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS,
                min_protected_pause_ms=DEFAULT_MIN_PROTECTED_PAUSE_MS,
            )

        reallocated_score, reallocated_info = _score_state(baseline_winners, reallocated_geometries)
        if reallocated_score < baseline_score:
            for group_id in member_ids:
                winner = dict(baseline_winners[group_id])
                member_geometry = reallocated_geometries[group_id]
                if has_bridge_member:
                    # Confirmed real gap found on job 33cd7b46...: measured_ms
                    # never changes (no re-synthesis), but required_speed_percent/
                    # speed_fit_mode/direction/fit were computed against the OLD
                    # per-sentence window and, unlike Phase 10's re-synthesized
                    # boundary-shift winners, were never refreshed -- leaving a
                    # stale, misleadingly-high number on a sentence that actually
                    # measures fine against its new, larger effective window.
                    # Recomputed here from the same unchanged measured_ms against
                    # the already-computed reallocated_geometries -- free, no new
                    # measurement. Only done for a bridge-containing turn, which
                    # Phase 14 renders per-sentence -- these individual numbers
                    # are the real, honest truth for that case (see the
                    # has_bridge_member aggregate write-back above for the
                    # opposite, all-normal case).
                    measured_ms = max(1, int(winner.get("measured_ms") or 0))
                    late_tolerance_ms = int(member_geometry["mouth_close_late_tolerance_ms"])
                    speed_fit_target_ms = int(member_geometry["source_window_ms"]) + late_tolerance_ms
                    direction = _timing_recast_direction(
                        measured_ms, int(member_geometry["source_window_ms"]),
                        max_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
                        mouth_close_early_tolerance_ms=int(member_geometry["mouth_close_early_tolerance_ms"]),
                        mouth_close_late_tolerance_ms=late_tolerance_ms,
                    )
                    winner["required_speed_percent"] = round(
                        _temporal_required_rush_percent(measured_ms, speed_fit_target_ms), 3,
                    )
                    winner["speed_fit_mode"] = _speed_fit_mode_label(late_tolerance_ms)
                    winner["direction"] = direction
                    winner["fit"] = bool(direction is None)
                # else: required_speed_percent/speed_fit_mode/fit already hold
                # the turn's real aggregate (set above, before this reallocation
                # branch even ran) -- invariant under window reallocation, so
                # nothing to recompute here for an all-normal turn.
                winner["effective_start_ms"] = int(reallocated_groups_by_id[group_id]["start_ms"])
                winner["effective_source_end_ms"] = int(reallocated_groups_by_id[group_id]["source_end_ms"])
                updated_winners[group_id] = winner
                updated_geometries[group_id] = reallocated_geometries[group_id]
            _emit_progress(
                progress,
                f"[native candidate pool] Turn {member_ids} rebalanced via window reallocation "
                f"(spread {baseline_info['spread']:.1f} -> {reallocated_info['spread']:.1f}).",
            )
        else:
            _emit_progress(
                progress,
                f"[native candidate pool] Window reallocation for turn {member_ids} did not improve on "
                "the original; keeping original winners.",
            )

    return updated_winners, updated_geometries


def _missing_required_literals(
    *, source_text: str, candidate_text: str,
) -> list[dict[str, Any]]:
    """The specific requirements from source_text NOT satisfied by
    candidate_text -- the real, itemized reason a literal-preservation
    check fails, used both by _sentence_preserves_required_literals for its
    pass/fail decision and by callers that want to LOG why a candidate was
    rejected instead of just counting it.

    Real confirmed false-positive (job fb3d08b63fed4d90922b08f7e325b906,
    2026-09-09): a genuinely good turn-block merge rendered "November 1" as
    the natural isiZulu ordinal word "mhla wokuqala kuNovemba" instead of
    keeping a bare digit -- correctly economizing the repeated "Izakhamuzi"
    subject across two sentences exactly as it should -- but this check's
    plain substring search never finds the digit "1" in a word-form
    rendering, so it silently discarded a BETTER candidate and forced a
    fallback to two separately-translated, repetitive sentences.
    _is_high_stakes_numeric_literal already documents this exact case for a
    different caller ("Numbers worth a hard stop if missing... A bare
    single digit 1-9 is exactly the range fluent isiZulu commonly renders
    as a natural number word") -- a bare single-digit NUMBER requirement is
    now exempt here too, for the same reason. This exemption is narrow and
    deliberately does NOT extend to acronyms (an acronym is never
    considered "high-stakes" by that function for an unrelated reason --
    it has its own separate warning path -- but must stay strictly required
    here, unlike a spelled-out number) or to any multi-digit/dated/
    percentage number.
    """
    missing: list[dict[str, Any]] = []
    for requirement in _temporal_mask_source_occurrence_requirements(source_text):
        if requirement["kind"] == "number" and not _is_high_stakes_numeric_literal(
            requirement["kind"], requirement["literal"],
        ):
            continue
        match_literal = str(requirement.get("match_literal", requirement["literal"]))
        required_count = int(requirement["required_count"])
        if candidate_text.count(match_literal) < required_count:
            missing.append(requirement)
    return missing


def _sentence_preserves_required_literals(*, source_text: str, candidate_text: str) -> bool:
    """Free, deterministic check before spending on translation: every
    high-stakes required literal (a name-bearing number/acronym) present
    in the TRUE original source_text must still appear in the candidate's
    text -- see _missing_required_literals for exactly what "high-stakes"
    excludes and why.
    """
    if _missing_required_literals(source_text=source_text, candidate_text=candidate_text):
        return False
    return True


def _zulu_morphology_known_proper_nouns(*, glossary: Mapping[str, Any], source_text: str) -> set[str]:
    """Lowercase tokens the morphology check should never flag: every
    do_not_translate entry, every glossary term's English AND Zulu wording
    (split into words -- covers a name glued to a Zulu prefix/concord, e.g.
    "noShadrack"), and every non-sentence-initial capitalized word in THIS
    sentence's own English source (covers a one-off proper noun the glossary
    never tracked, e.g. "Nyati" mentioned only once). Confirmed necessary
    2026-08-29: without this, a real job run's morphology check was almost
    entirely false positives, drowning its one genuine catch.

    Not all of these are foreign proper nouns despite the name of this function
    -- "Mnu." is a real isiZulu word, an ABBREVIATION of "Mnumzana" (confirmed:
    "mnumzana" parses cleanly, "mnu" alone does not, exactly like "Mr." is short
    for "Mister" in English). It fails the morphology check for the same reason
    "Mr." would fail an English one: abbreviations aren't full word forms, not
    because it's foreign. Both categories -- genuine foreign names and locked
    abbreviated titles -- are correctly excluded here for the same underlying
    reason: this check can only ever recognize complete, unabbreviated isiZulu
    word forms.
    """
    tokens: set[str] = set()
    for item in glossary.get("do_not_translate") or []:
        for word in re.findall(r"[A-Za-z]+", str(item)):
            tokens.add(word.lower())
    for term in glossary.get("terms") or []:
        for field in ("english", "zulu"):
            for word in re.findall(r"[A-Za-z]+", str(term.get(field) or "")):
                tokens.add(word.lower())
    source_words = re.findall(r"[A-Za-z]+", source_text)
    for word in source_words[1:]:  # skip sentence-initial -- always capitalized regardless
        if word[:1].isupper():
            tokens.add(word.lower())
    tokens.discard("")
    return tokens


def _shortened_candidate_required_source_text(
    *, source_text: str, clauses: Sequence[Mapping[str, Any]] | None, dropped_clause_ids: Sequence[str],
) -> str:
    """The real target text a literal-preservation check should hold a
    shortened candidate to: the combined English of every clause it did NOT
    declare dropped -- never the whole segment's original source_text
    unconditionally.

    Real confirmed bug (job fb3d08b63fed4d90922b08f7e325b906, 2026-09-09):
    checking every candidate against the full source_text meant a candidate
    that dropped the reporter sign-off ("Ofentse Setimo, SABC News,
    eMalahleni.") -- exactly the FIRST thing this file's own ranking
    guidance tells the model to drop -- was silently rejected outright,
    because "SABC" (an acronym) lives only in that one clause and the full
    source_text still "required" it. A literal that lives ONLY inside a
    clause the model explicitly, deliberately ranked as droppable (and
    which already survived the generation prompt's own HARD RULE gate) was
    never an accidental omission -- accidental omission is the ONLY failure
    mode this safety net exists to catch (see _filter_safe_shortened_
    candidates' own docstring). Falls back to the full source_text when the
    group carries no clause breakdown at all (should not normally happen
    alongside real shortened_candidates, but conservative-safe if it ever
    does).
    """
    if not clauses:
        return str(source_text or "")
    dropped = {str(value) for value in dropped_clause_ids}
    surviving = [
        str(clause.get("english_text") or "")
        for clause in clauses
        if str(clause.get("clause_id") or "") not in dropped
    ]
    return " ".join(surviving)


def _filter_safe_shortened_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    source_text: str,
    clauses: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The deterministic safety net behind the model's own restraint: keep
    only candidates whose isizulu_text still preserves every required
    literal (name/number/acronym) demanded by that candidate's OWN surviving
    content -- reusing the same _sentence_preserves_required_literals check
    every other fidelity gate in this file relies on. A candidate that fails
    is silently discarded, regardless of how the model itself worded it;
    this is the hard floor for the classes prompt-level guidance alone
    cannot mechanically guarantee (a hallucinated or careless omission is
    still possible even with the prompt's explicit hard rule). Order
    (lightest cut first) is preserved. This is Tier 1's own floor -- the
    separate, later last-resort protected-content mechanism deliberately
    does NOT apply this check, since omitting a literal is exactly what it
    may be asked to do.

    When ``clauses`` is supplied, each candidate is checked against
    _shortened_candidate_required_source_text (its own surviving clauses
    only) rather than the whole segment's source_text -- see that
    function's own docstring for the real bug this fixes. Omitting
    ``clauses`` preserves the original, whole-segment-source behavior.
    """
    safe: list[dict[str, Any]] = []
    for candidate in candidates:
        required_source = (
            _shortened_candidate_required_source_text(
                source_text=source_text, clauses=clauses,
                dropped_clause_ids=candidate.get("dropped_clause_ids") or (),
            )
            if clauses is not None
            else source_text
        )
        if _sentence_preserves_required_literals(
            source_text=required_source, candidate_text=str(candidate.get("isizulu_text") or ""),
        ):
            safe.append(candidate)
    return safe


def _resync_turn_rush_aggregates(
    *,
    groups: Sequence[Mapping[str, Any]],
    winners: Mapping[str, Mapping[str, Any]],
    geometries: Mapping[str, Mapping[str, Any]],
    preferred_raw_speed_percent: int,
    measurement_audio_root: Path,
) -> dict[str, dict[str, Any]]:
    """Re-apply the turn's real, stitched aggregate required_speed_percent/
    speed_fit_mode/fit to every member of every multi-member, all-normal
    speaker turn, using whatever the CURRENT winners are.

    _rebalance_speaker_turns already does this once, but later stages
    (_trim_by_fact_priority, _compress_protected_content_last_resort) can
    each independently shorten ONE member's own content afterward -- a real
    duration change that makes the earlier aggregate stale for the WHOLE
    turn, not just that one member. Real user feedback, 2026-09-07: "why are
    we aggregating segments rushes, the turn is suppose to be the unit" --
    confirmed as a real gap by direct inspection of a real job's committed
    output, where one turn-block-translation member (fact-priority-trimmed)
    ended up with a different reported number than its own turn partner,
    even though Phase 14 still renders them as one shared-speed-factor unit.

    Never touches window/effective_start_ms/effective_source_end_ms (Phase
    13's own concern, untouched by anything after it runs) -- purely a
    rush-number reporting resync, reusing the exact same _score_speaker_turn
    formula _rebalance_speaker_turns itself uses.
    """
    groups_by_id = {str(g["group_id"]): g for g in groups}
    turns = _build_speaker_turns(groups)
    updated_winners = dict(winners)
    for index, turn in enumerate(turns):
        member_ids = turn["member_group_ids"]
        if len(member_ids) < 2:
            continue
        if any(_classify_discourse_unit_kind(groups_by_id[gid]) == "bridge" for gid in member_ids):
            continue
        member_winners = {gid: updated_winners[gid] for gid in member_ids if gid in updated_winners}
        member_geometries = {gid: geometries[gid] for gid in member_ids if gid in geometries}
        if len(member_winners) < len(member_ids) or len(member_geometries) < len(member_ids):
            continue
        next_start = (
            int(turns[index + 1]["start_ms"]) if index + 1 < len(turns)
            else int(turn["source_end_ms"])
        )
        turn_geometry = _speaker_turn_geometry(
            turn, groups_by_id=groups_by_id, next_source_start_ms=next_start,
            preferred_raw_speed_percent=preferred_raw_speed_percent,
        )
        _score, info = _score_speaker_turn(
            turn=turn, turn_geometry=turn_geometry, member_winners=member_winners,
            member_geometries=member_geometries, measurement_audio_root=measurement_audio_root,
        )
        for group_id in member_ids:
            winner = dict(updated_winners[group_id])
            winner["required_speed_percent"] = round(info["required_speed_percent"], 3)
            winner["speed_fit_mode"] = info["speed_fit_mode"]
            winner["fit"] = bool(info["fits"])
            updated_winners[group_id] = winner
    return updated_winners


def _trim_by_fact_priority(
    *,
    groups: Sequence[Mapping[str, Any]],
    winners: Mapping[str, Mapping[str, Any]],
    tts: AzureTTSBackend,
    geometries: Mapping[str, Mapping[str, Any]],
    voice_assignments: Mapping[str, Mapping[str, Any]],
    measurement_audio_root: Path,
    force: bool,
    candidate_pool_tts_workers: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str]]:
    """The sole remaining compaction mechanism: for any block still measuring
    above DEFAULT_MAX_NATURAL_SPEED_PERCENT after turn rebalancing (i.e.
    audibly rushed -- the SAME +12% quality-warning bar render-time atempo
    already uses) AND carrying at least one shortened candidate, measure
    those candidates -- already generated, for free, as part of the same
    call that produced this block's translation (see
    TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT) -- and promote whichever fits best.
    This makes the whole mechanism ZERO-LLM-CALL at trim time: no provider
    parameter, no new generation, no QA/repair round; the only real cost is
    real Azure TTS measurement.

    A higher, dedicated trigger (originally 50%, calibrated from a different,
    now-superseded per-sentence overload check) was tried first and reverted:
    a real job showed blocks at 35-37% required speed -- comfortably "fits"
    by the pipeline's OLD standard, but a genuinely audible rush on real
    playback -- sitting on real, already-available shortened candidates that
    never got used because neither block crossed 50%. Tying the trigger to
    the same bar that already defines "rushed" everywhere else in this file
    closes that gap.

    Each shortened candidate is a COMPLETE, independently-written alternative
    text (never a mechanical substring removal) -- an earlier design had the
    model tag raw "droppable spans" for this function to excise itself via
    string replacement, which two real production failures confirmed is
    fundamentally unsafe: removing a substring can leave a dangling fragment
    (an orphaned vocative with no verb) or a stray leading punctuation mark
    that no amount of regex cleanup can fully repair, because fixing it
    requires understanding the sentence, not just its punctuation. Having
    the model write each shortened alternative directly (still ranked
    lightest-cut-first, still bound by the same hard rule protecting names/
    numbers/dates/quotes/attributions/claims/denials) makes grammaticality
    the model's own responsibility, exactly as it already is for the
    original translation -- this function's only job is to measure real
    candidates and pick the best-fitting one, never to edit text itself.

    All safe candidates are measured via real Azure TTS -- the same
    generate-with-redundancy/select-by-measured-fit discipline this file
    uses everywhere else for timing decisions. Among candidates that already
    fit within natural pace, the LIGHTEST cut (earliest in the model's own
    ranked list) wins (never use a heavier cut than the real, measured
    result shows was actually necessary); if none fit, the one with the
    lowest real required_speed_percent wins.

    Replaces every earlier wording-level compaction mechanism this pipeline
    tried (sentence compaction, discourse/backchannel compaction, turn-
    aggregate compaction, cross-turn exchange arbitration) with this single
    one, per explicit user direction: the objective is a natural retelling
    (already produced upstream) plus fact-priority-ranked dropping when a
    block genuinely doesn't fit -- never re-wording, re-translating, or
    re-QA'ing. Names, numbers, dates, quotes, attributions, and any claim/
    denial can never be dropped: enforced upstream by the generation prompt's
    own hard rule, and independently re-enforced here, deterministically, by
    _filter_safe_shortened_candidates on every candidate before it is ever
    measured.

    Never worse than the winner passed in: a block is only updated if its
    selected candidate measures strictly better than its own original
    required_speed_percent. Whenever a shortened candidate IS promoted,
    requires_review is set so a human always sees it (alongside the original,
    still present in the committed record's own candidate list) before it
    airs. A block with no shortened candidates, or whose candidates all fail
    literal-safety, is left untouched -- it falls through to the existing,
    unmodified Phase 14 render-time atempo compression exactly as it already
    does today.

    Returns (winners, usage_totals, triggered_group_ids) -- the third value
    is every block this function actually had a real ladder to try (whether
    or not a candidate ended up promoted), so a later, DELIBERATELY more
    destructive stage (_compress_protected_content_last_resort) can tell
    "this block already proved nothing safe gets it under budget" apart from
    "this block was never given a real shot" -- see that function's own
    trigger for why the distinction matters.
    """
    groups_by_id = {str(g["group_id"]): g for g in groups}
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    updated_winners = dict(winners)

    triggered_group_ids = [
        gid for gid, winner in updated_winners.items()
        if float(winner.get("required_speed_percent") or 0.0) > DEFAULT_MAX_NATURAL_SPEED_PERCENT
        and groups_by_id.get(gid, {}).get("shortened_candidates")
    ]
    if not triggered_group_ids:
        return updated_winners, usage_totals, []

    _emit_progress(
        progress,
        f"[native candidate pool] {len(triggered_group_ids)} block(s) still measure above "
        f"{DEFAULT_MAX_NATURAL_SPEED_PERCENT:.0f}% required speed with shortened candidates "
        "available; measuring them...",
    )

    candidates_by_group: dict[str, list[dict[str, Any]]] = {}
    for gid in triggered_group_ids:
        group = groups_by_id[gid]
        source_text = str(group["source_text"])
        safe_candidates = _filter_safe_shortened_candidates(
            candidates=group.get("shortened_candidates") or [], source_text=source_text,
            clauses=group.get("clauses"),
        )
        if not safe_candidates:
            continue
        candidates_by_group[gid] = [
            {
                "candidate_id": f"{gid}__fact_priority_trim_{index}", "variant_id": "fact_priority_trim",
                "translator": "grok", "spoken_text": str(candidate["isizulu_text"]),
                "dropped_clause_ids": list(candidate.get("dropped_clause_ids") or []),
                "qa_penalty": 0, "qa_record": None, "requires_review": False,
            }
            for index, candidate in enumerate(safe_candidates, start=1)
        ]

    if not candidates_by_group:
        _emit_progress(progress, "[native candidate pool] No safe shortened candidates were available.")
        return updated_winners, usage_totals, triggered_group_ids

    measured_group_ids = list(candidates_by_group)
    raw_measured = _measure_temporal_batch(
        tts=tts, groups=[groups_by_id[gid] for gid in measured_group_ids],
        candidates_by_group=candidates_by_group,
        geometries={gid: geometries[gid] for gid in measured_group_ids},
        voice_assignments=voice_assignments, output_root=measurement_audio_root,
        round_number=9, force=force, workers=candidate_pool_tts_workers,
    )

    for gid in measured_group_ids:
        measured = raw_measured.get(gid) or []
        if not measured:
            continue
        baseline = updated_winners[gid]
        baseline_required_speed = float(baseline.get("required_speed_percent") or 0.0)
        enriched = _enrich_measured_candidates(
            measured=measured, original_candidates=candidates_by_group[gid],
        )
        # candidates_by_group[gid] is already ordered lightest-cut-first;
        # _measure_temporal_batch restores that original order after
        # concurrent synthesis, so a position lookup is safe and simple.
        rank_by_id = {str(c["candidate_id"]): i for i, c in enumerate(candidates_by_group[gid])}
        fitting = [c for c in enriched if c.get("fit")]
        if fitting:
            candidate = min(fitting, key=lambda c: rank_by_id[str(c["candidate_id"])])
        else:
            # Real confirmed bug (job 8372120960474ef6b1d75af7de51a605,
            # 2026-09-04): picking whichever candidate reports the lowest
            # required_speed_percent treats a candidate that undershoots its
            # window by SECONDS as "the best", since a much-too-short
            # candidate reports a very low (even negative-feeling) required
            # speed -- but a large undershoot means several real seconds of
            # dead air while the video keeps showing the speaker's mouth
            # moving, an audible "the speech just breaks" defect that is
            # worse than the rushing this trim was meant to fix. Use the
            # same real, already-measured timing_candidate_rank tuple every
            # other timing decision in this file relies on -- it already
            # encodes the correct, asymmetric policy (fits < rushed-but-
            # atempo-rescuable < undersized-with-dead-air, worst tier).
            candidate = min(enriched, key=lambda c: tuple(c.get("rank") or (9, 0, 0)))
        candidate_required_speed = float(candidate.get("required_speed_percent") or 0.0)
        candidate_rank = tuple(candidate.get("rank") or (9, 0, 0))
        baseline_rank = tuple(baseline.get("rank") or (9, 0, 0))
        if candidate_rank < baseline_rank:
            candidate["requires_review"] = True
            if baseline.get("effective_start_ms") is not None:
                candidate.setdefault("effective_start_ms", baseline["effective_start_ms"])
            if baseline.get("effective_source_end_ms") is not None:
                candidate.setdefault("effective_source_end_ms", baseline["effective_source_end_ms"])
            updated_winners[gid] = candidate
            _emit_progress(
                progress,
                f"[native candidate pool] {gid} improved via fact-priority trim "
                f"(candidate {rank_by_id[str(candidate['candidate_id'])] + 1}/{len(candidates_by_group[gid])}): "
                f"{baseline_required_speed:.1f}% -> {candidate_required_speed:.1f}%.",
            )
        else:
            _emit_progress(
                progress, f"[native candidate pool] Fact-priority trim for {gid} did not improve on the original.",
            )

    return updated_winners, usage_totals, triggered_group_ids


# --- Last-resort protected-content compression -----------------------------------
# Explicit, deliberate product decision (2026-09-04): Mathula TV is a fixed-runtime
# social broadcast product, not a court interpreter -- a sentence that renders
# audibly rushed risks a viewer disengaging from the REST of the video, not just
# missing this one sentence's content, so an occasional, tightly-ordered compression
# of a normally-protected fact is judged the smaller loss. This is a SEPARATE,
# LATER mechanism from _trim_by_fact_priority above: Tier 1's own HARD RULE (never
# drop a name/number/date/quote/attribution/claim-denial) stays completely intact
# there -- this only ever runs for a block that already exhausted Tier 1's entire
# ladder and still does not fit.
PROTECTED_CONTENT_LAST_RESORT_PROMPT_VERSION = "native-protected-content-last-resort-v2-hedge-generalization"
# Deliberately HIGHER than DEFAULT_MAX_NATURAL_SPEED_PERCENT (12) -- confirmed
# real bug (job 8372120960474ef6b1d75af7de51a605, 2026-09-04): gating this
# tier on the routine 12% "fits naturally" bar fired it on 8 individual
# members of one already-rebalanced turn, each sitting a mere 0.2-0.6 points
# over 12% (turn rebalancing spreads a turn's real excess evenly across
# members, so a small residual over 12% is normal and expected there, not a
# sign anything is "genuinely stuck"). This is the same mistake this
# project's own abandoned Won et al. per-sentence-shorten design already
# made and reverted once (it fired on 55/157 sentences using the same bar).
# 25.0 is grounded in real user feedback from this same job: a sentence
# measured at 25.4% was reported as audibly "rushed"; turn members left at
# 12-16% by ordinary rebalancing were never flagged as a problem.
_PROTECTED_CONTENT_LAST_RESORT_TRIGGER_PERCENT = 25.0

PROTECTED_CONTENT_LAST_RESORT_SYSTEM_PROMPT = r"""Mathula TV is a fixed-runtime social broadcast
product: a sentence that plays back audibly rushed risks losing that viewer's attention for the
REST of the video, not just this one sentence -- so a viewer never hearing one fact is a smaller
loss than a viewer disengaging before every fact in the rest of the video. Each unit you are given
has ALREADY had every safely-droppable clause removed by an earlier stage and STILL does not fit
its real time budget. This is the FINAL fallback: you may now compress content that is normally
permanently protected everywhere else in this pipeline -- a name, a number, a date, or the
substance of a claim/denial.

Follow this STRICT order for each unit, attempting only what is actually needed and no more:

1. NEVER invert meaning. A denial must stay a denial; a claim must stay attributed to the same
   person and the same target. This is the one absolute floor that survives even this fallback --
   nothing below ever licenses changing what is true, false, claimed, or denied, only how much
   detail is spent stating it.
2. COLLAPSE a repeated literal to a single mention first, if it is not already. If the same year,
   name, or number appears more than once, state it once where a natural retelling would (e.g.
   "November or December 2024" instead of "November of 2024 or December of 2024").
3. GENERALIZE a defensive enumeration into its real approximate range, when the unit is a speaker
   listing every specific possibility to avoid committing to one that could be checked against other
   evidence (common for a witness protecting themselves), rather than because each option is
   independently important. This is MORE aggressive than step 2 and should be tried whenever it
   applies, not just as an afterthought. The generalization MUST preserve the real SHAPE of the
   uncertainty (a year boundary, an order of magnitude, a before/after) -- never just vaguely round
   it away. Worked example, a real confirmed case: "I couldn't tell you if it was November of 2024
   or December of 2024 or January of 2025, but I had spoken to him on the phone once" generalizes to
   "I spoke to him on the phone towards the end of 2024, maybe early 2025" -- this keeps the real
   year-spanning uncertainty (the witness cannot commit to which calendar year) while dropping the
   month-by-month enumeration and the already-redundant "I had spoken to him" framing entirely. A
   flatter "at the end of the year" would be UNACCEPTABLE here: it silently narrows the claim to
   December 2024 alone, losing the January 2025 possibility -- generalize the RANGE, never just the
   wording.
4. SHORTEN a unique name to its shortest still-identifying form (surname alone, or a form already
   established elsewhere in this programme) rather than a fuller titled/full-name form, if a fuller
   form is what is currently present.
5. STATE a claim or denial in its barest essential gist -- who claimed/denied what, and whether it
   was true -- dropping elaborating detail even if that detail was previously judged part of the
   core proposition under ordinary rules.
6. Only if the unit still cannot fit after steps 2-5, OMIT a single unique name, number, or date
   entirely, choosing whichever one is least central to the story (see the supporting-cast/
   protagonist distinction used elsewhere in this pipeline: a peripheral figure's own detail before
   a protagonist's). Never omit more than the minimum needed to fit.

For each unit, return isizulu_text (the compressed result) and omitted_items: short plain-English
descriptions of anything touched under steps 2-6 (e.g. "stated the year once instead of twice",
"generalized three named months to 'end of 2024, maybe early 2025'", "shortened 'General Lincoln' to
'Lincoln' on second mention", or, only for step 6, "omitted the specific January 2025 date, kept
November/December 2024"). Return an empty omitted_items list if steps 2-5 alone were enough -- that
is normal and expected, not a failure. No tools or web search. Return JSON only."""


def _protected_content_last_resort_schema(expected_unit_ids: Sequence[str]) -> dict[str, Any]:
    requested_ids = [str(value) for value in expected_unit_ids]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Protected-content last-resort schema received duplicate expected unit IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "minItems": len(requested_ids),
                "maxItems": len(requested_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["unit_id", "isizulu_text", "omitted_items"],
                    "properties": {
                        "unit_id": {"type": "string", "enum": requested_ids},
                        "isizulu_text": {"type": "string", "minLength": 1},
                        "omitted_items": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "maxItems": 8,
                        },
                    },
                },
            },
        },
    }


def _request_protected_content_last_resort_batch(
    *,
    provider: FoundryGrokProvider,
    items: Sequence[Mapping[str, Any]],
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """items: [{"unit_id", "source_text", "current_isizulu_text"}, ...]. Returns
    {unit_id: {"isizulu_text", "omitted_items"}}. Identity mismatch on unit_id
    is fatal (same convention as every other candidate-translate-family
    function in this file).
    """
    expected_ids = [str(item["unit_id"]) for item in items]
    response = provider.complete_json(
        operation="native_protected_content_last_resort_batch",
        system_prompt=PROTECTED_CONTENT_LAST_RESORT_SYSTEM_PROMPT,
        payload={
            "zulu_terminology_glossary": dict(glossary),
            "context_ledger": dict(context_ledger or {}),
            "units": [
                {
                    "unit_id": str(item["unit_id"]),
                    "source_text": str(item["source_text"]),
                    "current_isizulu_text": str(item["current_isizulu_text"]),
                }
                for item in items
            ],
        },
        schema=_protected_content_last_resort_schema(expected_ids),
        max_output_tokens=min(max(1500, 400 * len(items)), 8192),
    )
    returned = list(response.data.get("units") or [])
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        unit_id = str(item.get("unit_id") or "")
        if unit_id in by_id:
            duplicate_ids.append(unit_id)
            continue
        if unit_id not in expected_set:
            unknown_ids.append(unit_id)
            continue
        by_id[unit_id] = {
            "isizulu_text": str(item.get("isizulu_text") or "").strip(),
            "omitted_items": [str(value) for value in (item.get("omitted_items") or [])],
        }

    missing_ids = [unit_id for unit_id in expected_ids if unit_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Protected-content last-resort batch identity mismatch: " + "; ".join(details))

    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return by_id, usage


def _compress_protected_content_last_resort(
    *,
    provider: FoundryGrokProvider,
    groups: Sequence[Mapping[str, Any]],
    winners: Mapping[str, Mapping[str, Any]],
    tts: AzureTTSBackend,
    geometries: Mapping[str, Mapping[str, Any]],
    voice_assignments: Mapping[str, Mapping[str, Any]],
    measurement_audio_root: Path,
    force: bool,
    candidate_pool_tts_workers: int,
    glossary: Mapping[str, Any],
    context_ledger: Mapping[str, Any] | None = None,
    ladder_exhausted_group_ids: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """The FINAL fallback, for a block whose current winner still does not
    fit its immutable time budget. Real user direction, 2026-09-08: "the
    only thing worthy of protecting is the time frame and the story...no
    class should stop that" -- Tier 1's HARD RULE (never drop a name/number/
    date/quote/attribution/claim-denial) is deliberately narrow and stays
    intact there, but it was never meant to be the SYSTEM's last word: this
    tier exists specifically so a fixed category never permanently blocks
    fitting the time frame once every safe option has genuinely been tried.

    Two different populations reach ``not fit`` here, and they need two
    different bars, not one flat threshold:
    - A block in ``ladder_exhausted_group_ids`` (passed through from
      _trim_by_fact_priority) already had a REAL ladder of safe cuts and
      used the best one available -- it has already, concretely, proven
      that nothing safe gets it under budget. For this population, ANY
      remaining overrun (``not fit``, no percentage floor) is enough to
      trigger this tier: the "story" is already being told as tightly as
      Tier 1 allows, and the time frame is still the thing that must give.
    - A block NOT in that set never had a real ladder to try (every clause
      it ranked was rank 1 -- "nothing safe to drop" -- or turn-rebalancing
      alone already left it close to fitting). For this population, keep
      requiring _PROTECTED_CONTENT_LAST_RESORT_TRIGGER_PERCENT (25%,
      deliberately higher than the routine 10% "fits naturally" bar -- see
      that constant's own comment for the real, confirmed bug this guards
      against: ordinary turn-rebalancing residual of 12-16%, with nothing
      ever tried against it, wrongly triggering this destructive tier).

    One batched Grok call covers every triggered block in the whole job at
    once (same batching discipline as every other compaction mechanism in
    this file). Deliberately does NOT run _filter_safe_shortened_candidates/
    _sentence_preserves_required_literals on the result -- omitting a
    literal is exactly what step 5 of PROTECTED_CONTENT_LAST_RESORT_SYSTEM_
    PROMPT may legitimately do; that check exists to protect Tier 1's
    different, absolute guarantee.

    Never worse than the winner passed in: promoted only if the compressed
    candidate measures strictly better than the current winner, exactly like
    every other mechanism in this file. Always sets requires_review and
    carries omitted_items through to the committed record when promoted, so
    a human always sees exactly what this fallback touched before it airs.
    """
    groups_by_id = {str(g["group_id"]): g for g in groups}
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    updated_winners = dict(winners)
    ladder_exhausted_set = set(ladder_exhausted_group_ids)

    triggered_group_ids = [
        gid for gid, winner in updated_winners.items()
        if not winner.get("fit")
        and (
            gid in ladder_exhausted_set
            or float(winner.get("required_speed_percent") or 0.0) > _PROTECTED_CONTENT_LAST_RESORT_TRIGGER_PERCENT
        )
    ]
    if not triggered_group_ids:
        return updated_winners, usage_totals

    _emit_progress(
        progress,
        f"[native candidate pool] {len(triggered_group_ids)} block(s) still don't fit after the "
        "fact-priority ladder; attempting last-resort protected-content compression...",
    )

    items = [
        {
            "unit_id": gid,
            "source_text": str(groups_by_id[gid]["source_text"]),
            "current_isizulu_text": str(updated_winners[gid]["spoken_text"]),
        }
        for gid in triggered_group_ids
    ]
    try:
        by_id, usage = _request_protected_content_last_resort_batch(
            provider=provider, items=items, glossary=glossary, context_ledger=context_ledger,
        )
    except Exception as exc:
        _emit_progress(progress, f"[native candidate pool] Last-resort compression request failed: {exc}")
        return updated_winners, usage_totals
    for key in usage_totals:
        usage_totals[key] += usage.get(key, 0)

    candidates_by_group: dict[str, list[dict[str, Any]]] = {}
    for gid in triggered_group_ids:
        result = by_id.get(gid)
        if not result:
            continue
        text = str(result.get("isizulu_text") or "").strip()
        if not text or text == str(updated_winners[gid]["spoken_text"]).strip():
            continue
        candidates_by_group[gid] = [{
            "candidate_id": f"{gid}__protected_content_last_resort", "variant_id": "protected_content_last_resort",
            "translator": "grok", "spoken_text": text, "qa_penalty": 0, "qa_record": None,
            "requires_review": True, "omitted_items": list(result.get("omitted_items") or []),
        }]

    if not candidates_by_group:
        _emit_progress(progress, "[native candidate pool] Last-resort compression produced no usable candidates.")
        return updated_winners, usage_totals

    measured_group_ids = list(candidates_by_group)
    raw_measured = _measure_temporal_batch(
        tts=tts, groups=[groups_by_id[gid] for gid in measured_group_ids],
        candidates_by_group=candidates_by_group,
        geometries={gid: geometries[gid] for gid in measured_group_ids},
        voice_assignments=voice_assignments, output_root=measurement_audio_root,
        round_number=10, force=force, workers=candidate_pool_tts_workers,
    )

    for gid in measured_group_ids:
        measured = raw_measured.get(gid) or []
        if not measured:
            continue
        baseline = updated_winners[gid]
        baseline_required_speed = float(baseline.get("required_speed_percent") or 0.0)
        enriched = _enrich_measured_candidates(measured=measured, original_candidates=candidates_by_group[gid])
        candidate = enriched[0]
        candidate_required_speed = float(candidate.get("required_speed_percent") or 0.0)
        # Real confirmed bug (job 8372120960474ef6b1d75af7de51a605, 2026-09-04): comparing raw
        # required_speed_percent lets a massively-undershot ("expand"/tier-2) candidate look
        # artificially "best" (a large undershoot reports a very low/negative-feeling percentage)
        # even though it produces several seconds of dead-air silence at render time, since a
        # rushed (tier-1, atempo-rescuable) candidate is always preferable to an undersized one.
        # Compare the pre-existing, correct _timing_candidate_rank tuple instead.
        candidate_rank = tuple(candidate.get("rank") or (9, 0, 0))
        baseline_rank = tuple(baseline.get("rank") or (9, 0, 0))
        if candidate_rank < baseline_rank:
            candidate["requires_review"] = True
            if baseline.get("effective_start_ms") is not None:
                candidate.setdefault("effective_start_ms", baseline["effective_start_ms"])
            if baseline.get("effective_source_end_ms") is not None:
                candidate.setdefault("effective_source_end_ms", baseline["effective_source_end_ms"])
            updated_winners[gid] = candidate
            _emit_progress(
                progress,
                f"[native candidate pool] {gid} improved via last-resort protected-content compression: "
                f"{baseline_required_speed:.1f}% -> {candidate_required_speed:.1f}% "
                f"(touched: {'; '.join(candidate.get('omitted_items') or []) or 'nothing protected, steps 2-4 sufficed'}).",
            )
        else:
            _emit_progress(
                progress,
                f"[native candidate pool] Last-resort compression for {gid} did not improve on the original.",
            )

    return updated_winners, usage_totals


def _apply_zulu_morphology_check(
    *,
    winners: dict[str, dict[str, Any]],
    groups_by_id: Mapping[str, Mapping[str, Any]],
    measured_by_group: dict[str, list[dict[str, Any]]],
    glossary: Mapping[str, Any],
    progress: Callable[[str], None] | None = None,
) -> None:
    """Real, mechanical isiZulu morphology check (see zulu_morphology.py) applied
    to EVERY sentence's FINAL winner, regardless of which stage produced it
    (natural translation, turn rebalancing, or compaction). Mutates `winners` and
    the matching entry in `measured_by_group` in place -- never a gate, never
    blocks promotion, only sets morphology_warning/requires_review for human
    review.

    Confirmed necessary 2026-08-29: when this check was wired only into
    _compact_overloaded_sentences's own promotion branch, it silently skipped
    the ~85% of sentences that never trigger compaction at all -- a coverage
    gap only found by checking real committed candidate-pool data, not by
    inspection.

    Sets `morphology_warning` (informational) but deliberately does NOT set
    `requires_review` -- confirmed 2026-08-29 by running this check on every
    sentence of a real job, twice: even after fixing a real tokenizer bug
    (digit/hyphen-fused compounds like "ka2024"/"kwe-T" being split into bare,
    unparseable fragments), the flag rate only dropped from 80/157 to 70/157.
    The dominant remaining cause is a genuine ZulMorph lexicon gap on ordinary
    class 9/10/11 nouns and common particles ("into", "indawo", "inqubo",
    "uhlobo", "ngenkathi") -- unambiguously correct, everyday isiZulu that
    happens to fall outside this analyser's real coverage (its actual strength
    is verb morphology). At that false-positive rate, driving requires_review
    from this signal would swamp genuine QA-worthy flags with noise.
    """
    for group_id, winner in winners.items():
        group = groups_by_id.get(group_id)
        if group is None:
            continue
        proper_nouns = _zulu_morphology_known_proper_nouns(
            glossary=glossary, source_text=str(group.get("source_text") or ""),
        )
        bad_words = zulu_morphology.unparsed_words(
            str(winner.get("spoken_text") or ""), known_proper_nouns=proper_nouns,
        )
        if bad_words:
            winner["morphology_warning"] = bad_words
            _emit_progress(
                progress,
                f"[native candidate pool] {group_id}'s {winner.get('variant_id')} winner has "
                f"word(s) that don't parse as isiZulu morphology: {bad_words} -- informational "
                "only, does not set requires_review (see this function's docstring).",
            )
        candidate_id = winner.get("candidate_id")
        measured_by_group[group_id] = [
            winner if c.get("candidate_id") == candidate_id else c
            for c in measured_by_group.get(group_id, [])
        ]


def build_candidate_pool(
    *,
    job_root: Path,
    provider: FoundryGrokProvider,
    tts: AzureTTSBackend,
    candidate_pool_workers: int = DEFAULT_CANDIDATE_POOL_WORKERS,
    candidate_pool_tts_workers: int = DEFAULT_CANDIDATE_POOL_TTS_WORKERS,
    preferred_raw_speed_percent: int = DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
    force: bool = False,
    skip_turn_rebalance: bool = False,
    skip_turn_block_translation: bool = False,
    skip_pronunciation_research: bool = False,
    skip_pronunciation_round_trip: bool = False,
    stt_backend: AzureFastTranscriptionBackend | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build the full candidate pool for every sentence and write it to
    pass2_candidate_pool.json ONLY -- never touches pass2_natural_translation.json.

    No automated QA/repair step runs on the natural translation, per explicit
    user direction: it ships as its own winner unaudited. This was a real,
    informed trade-off, not an oversight -- the removed step (back-translation
    audit + auto-repair) is what previously caught real production errors
    like a fabricated name and a swapped attribution, and nothing in this
    pipeline replaces that catch automatically anymore. Real failures are
    now expected to be diagnosed from actual rendered output after the fact
    (the standing project preference behind every other in-pipeline LLM
    retry/repair loop already removed from this file), not caught upfront.
    The standalone `native-qa` command (`run_native_qa_back_translation`)
    still exists as a separate, deliberate, on-demand audit tool for anyone
    who wants to run one -- it is untouched by this change, only never
    invoked automatically here.

    ``skip_pronunciation_research=True`` is a diagnostic-only mode: skips
    Phase 17's web-search-backed pronunciation research entirely (real
    Azure Responses API call, real cost) -- entities keep whatever
    pronunciation the hand-curated dictionaries already give them.

    ``stt_backend``, when supplied, enables Phase 18's automated TTS+STT
    round-trip verification of every accepted pronunciation correction --
    reuses the already-required ``tts`` backend for synthesis. Omitted (the
    default, e.g. no AZURE_SPEECH_KEY configured) preserves the exact
    pre-Phase-18 behavior: corrections are applied on the research call's own
    confidence alone. ``skip_pronunciation_round_trip=True`` is a diagnostic
    override that disables verification even when ``stt_backend`` is given.

    ``skip_turn_block_translation=True`` is a diagnostic-only mode: skips
    Phase 16's whole-turn-block translation entirely and falls back to the
    older, proven per-sentence _translate_natural_via_grok (Phase 9) for
    EVERY sentence -- real A/B comparison against a job's own baseline. A
    sentence translated this way has no shortened candidates of its own, so
    it is never eligible for fact-priority trimming below.

    ``skip_turn_rebalance=True`` is a diagnostic-only mode: skips Phase 13's
    window-reallocation entirely, so every sentence's `required_speed_percent`
    reflects its own raw natural-translation timing, unaffected by turn
    rebalancing.

    Sentences are grouped into real, uncapped speaker turns (the PRIMARY unit
    -- see _build_speaker_turns) BEFORE translation now (Phase 16): each turn
    is translated as one or more bounded, contiguous windows, each ONE Grok
    call free to merge/reorder content across original sentence boundaries
    within the window for economy, and to write, for each resulting segment,
    a small ranked set of complete SHORTENED alternative texts (see
    _translate_turn_blocks_via_grok, TURN_BLOCK_TRANSLATE_SYSTEM_PROMPT). The
    resulting BLOCKS -- one-or-more merged original sentences, or a lone
    untouched sentence when nothing merged -- replace the raw per-sentence
    groups as the atomic unit for everything from here on: measurement, turn
    rebalancing (Phase 13, resizing each block's window share proportionally
    to its own real content when a turn doesn't fit or paces unevenly --
    never by re-wording or re-synthesizing), and fact-priority trimming (see
    _trim_by_fact_priority -- the sole remaining mechanism for a block that
    still doesn't fit after rebalancing: measuring its own already-ranked,
    already-literal-safe shortened candidates and promoting whichever fits
    best, never requesting new wording). A solo "bridge" block (a short
    interjection like "All right.") is judged only on whether it collides
    with its neighbors, never on fitting its own tiny native window.
    """
    paths = native_dub_paths(job_root)
    if not paths.pass1.is_file():
        raise ValueError("Run native-translate pass 1 first; pass1_restored_transcript.json is missing")
    pass1 = read_json(paths.pass1)
    segments_raw = list(pass1.get("segments") or [])
    if not segments_raw:
        raise ValueError("Pass-1 artifact has no restored segments")

    context_ledger, cached_context_usage = _load_existing_context_ledger(
        job_root=job_root, pass1=pass1, segments=segments_raw, progress=progress,
    )
    if context_ledger is None:
        context_ledger, context_usage = _ensure_context_ledger(
            job_root=job_root, provider=provider, pass1=pass1, segments=segments_raw, progress=progress,
        )
    else:
        context_usage = dict(cached_context_usage)

    glossary, glossary_usage = _ensure_zulu_glossary(
        job_root=job_root, provider=provider, pass1=pass1, segments=segments_raw,
        context_ledger=context_ledger, progress=progress,
    )

    # Phase 17: runs unconditionally (own internal checksum cache), independent
    # of whether the candidate-pool checkpoint below gets reused -- this is what
    # guarantees the job-scoped pronunciation dictionary (in-memory here, and the
    # on-disk artifact render_native_dub reads back in a possibly-separate
    # process) is always populated whenever build_candidate_pool runs at all.
    pronunciation_research_usage = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    if not skip_pronunciation_research:
        _, pronunciation_research_usage = _ensure_native_pronunciation_research(
            job_root=job_root, context_ledger=context_ledger, glossary=glossary,
            force=force, progress=progress, tts=tts, stt_backend=stt_backend,
            skip_round_trip_verification=skip_pronunciation_round_trip,
            grok_provider=provider, pass1_corrections=pass1.get("corrections") or (),
        )
    else:
        _emit_progress(progress, "[native candidate pool] Pronunciation research skipped (diagnostic mode).")

    # Project-level case-state context (working/case_state/master_case_state.json,
    # see master_case_context.py) is deliberately NOT folded into context_ledger
    # here (removed 2026-09-02, real cost incident): case_state_context could run
    # up to _CASE_STATE_CONTEXT_MAX_CHARS (12,000 chars, ~3,000 tokens) and was
    # being resent on every translation/QA/compaction/arbitration call regardless
    # of job size -- for a tiny job this was a large, constant tax unrelated to
    # actual content. The stated priority for this pipeline is isiZulu grammar
    # and comprehension, not case-narrative grounding; _load_case_state_context
    # itself is left intact (unused) rather than deleted, in case this is
    # reconsidered later.
    sentence_records = _build_pass1_sentence_records(segments_raw, job_root=job_root)
    groups = _build_temporal_mask_source_groups(sentence_records)
    if not groups:
        raise ValueError("No candidate-pool source groups were created")

    natural_english_by_group = {
        str(group["group_id"]): _strip_confirmed_filler_phrases(str(group.get("source_text") or ""))
        for group in groups
    }
    # Phase 16: turns are built BEFORE translation now -- translation itself
    # needs to know turn membership to decide which sentences may be merged.
    turns = _build_speaker_turns(groups)

    input_hash = _hash_payload({
        "pass1_sha256": checksum(paths.pass1),
        "prompt_version": CANDIDATE_TRANSLATE_PROMPT_VERSION,
        "turn_block_translate_prompt_version": TURN_BLOCK_TRANSLATE_PROMPT_VERSION,
        "turn_block_translate_max_members": _TURN_BLOCK_TRANSLATE_MAX_MEMBERS,
        "provider_version": FOUNDRY_GROK_PROVIDER_VERSION,
        "preferred_raw_speed_percent": int(preferred_raw_speed_percent),
        "bridge_max_word_count": _BRIDGE_MAX_WORD_COUNT,
        "bridge_max_source_span_ms": _BRIDGE_MAX_SOURCE_SPAN_MS,
        "max_shortened_candidates_per_segment": _MAX_SHORTENED_CANDIDATES_PER_SEGMENT,
        "skip_turn_rebalance": bool(skip_turn_rebalance),
        "skip_turn_block_translation": bool(skip_turn_block_translation),
        "skip_pronunciation_research": bool(skip_pronunciation_research),
        # Deliberately excludes the glossary's own content -- see the plan's
        # Checkpointing section: a corrected glossary entry shouldn't discard
        # translation work. Phase 17's pronunciation research artifact content
        # is excluded for the identical reason -- correcting a pronunciation
        # entry (or a fresh web-search verdict) should never discard
        # translation/timing work, and it never affects any timing this
        # function measures (only render_native_dub's later synthesis).
    })
    if paths.candidate_pool.is_file() and not force:
        existing = read_json(paths.candidate_pool)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native candidate pool] Reusing matching candidate-pool checkpoint")
            return existing

    if skip_turn_block_translation:
        grok_candidate_by_group, grok_usage = _translate_natural_via_grok(
            provider=provider, groups=groups, natural_english_by_group=natural_english_by_group,
            glossary=glossary, context_ledger=context_ledger, workers=candidate_pool_workers, progress=progress,
        )
        # A copy, not a mutation of the caller's own group dicts -- carries
        # shortened_candidates/clauses through to _trim_by_fact_priority
        # exactly like _build_turn_block_group already does for the primary
        # path, so this diagnostic flag doesn't also silently disable
        # fact-priority trim's own clause-scoped literal-safety check.
        block_groups = [
            {
                **group,
                "shortened_candidates": list(
                    grok_candidate_by_group.get(str(group["group_id"]), {}).get("shortened_candidates") or []
                ),
                "clauses": grok_candidate_by_group.get(str(group["group_id"]), {}).get("clauses"),
                "register_notes": grok_candidate_by_group.get(str(group["group_id"]), {}).get("register_notes"),
            }
            for group in groups
        ]
    else:
        block_groups, grok_candidate_by_group, grok_usage = _translate_turn_blocks_via_grok(
            provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english_by_group,
            glossary=glossary, context_ledger=context_ledger,
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
            workers=candidate_pool_workers, progress=progress,
        )

    groups_by_id = {str(g["group_id"]): g for g in block_groups}
    discourse_unit_kind_by_group = {
        str(g["group_id"]): _classify_discourse_unit_kind(g) for g in block_groups
    }

    # No automated QA/repair step: per explicit user direction, the natural
    # translation ships as its own selected winner with no back-translation
    # audit or auto-repair round -- real failures are diagnosed from actual
    # rendered output afterward (the same "generate with redundancy, verify
    # by real measurement, never re-check with another LLM call" discipline
    # this file already applies to fact-priority trim), not caught in-pipeline.
    winners: dict[str, dict[str, Any]] = {
        group_id: {**candidate, "qa_penalty": 0, "qa_record": None, "qa_records": [], "requires_review": False}
        for group_id, candidate in grok_candidate_by_group.items()
    }

    geometries: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(block_groups):
        group_id = str(group["group_id"])
        next_start = (
            int(block_groups[index + 1]["start_ms"]) if index + 1 < len(block_groups)
            else int(group["source_end_ms"])
        )
        is_bridge = discourse_unit_kind_by_group[group_id] == "bridge"
        geometries[group_id] = _temporal_mask_geometry(
            group,
            next_source_start_ms=next_start,
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
            max_natural_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
            mouth_tolerance_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
            max_protected_gap_overflow_ms=(
                BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS if is_bridge else DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS
            ),
            min_protected_pause_ms=DEFAULT_MIN_PROTECTED_PAUSE_MS,
        )

    measurement_voice_info = {"selected_voice": str(tts.default_voice), "base_prosody": {}}
    voice_assignments = {str(g["speaker_id"]): measurement_voice_info for g in block_groups}
    measurement_audio_root = paths.root / "pass2_candidate_pool_measurements"
    candidates_by_group = {group_id: [winner] for group_id, winner in winners.items()}
    raw_measured_by_group = _measure_temporal_batch(
        tts=tts, groups=block_groups, candidates_by_group=candidates_by_group, geometries=geometries,
        voice_assignments=voice_assignments, output_root=measurement_audio_root,
        round_number=0, force=force, workers=candidate_pool_tts_workers,
    )
    measured_by_group = {
        group_id: _enrich_measured_candidates(
            measured=measured, original_candidates=candidates_by_group.get(group_id, [])
        )
        for group_id, measured in raw_measured_by_group.items()
    }
    winners = {group_id: measured[0] for group_id, measured in measured_by_group.items() if measured}

    def _merge_updated_winners(current_winners: Mapping[str, Mapping[str, Any]]) -> None:
        for group_id, winner in current_winners.items():
            measured_by_group[group_id] = _replace_or_append_candidate(
                list(measured_by_group.get(group_id, [])), winner,
            )

    if not skip_turn_rebalance:
        winners, rebalanced_geometries = _rebalance_speaker_turns(
            groups=block_groups, winners=winners, geometries=geometries,
            preferred_raw_speed_percent=int(preferred_raw_speed_percent),
            measurement_audio_root=measurement_audio_root, progress=progress,
        )
        # Without this, anything measuring a NEW candidate against a
        # rebalanced member afterward (fact-priority trimming below) would
        # silently use the stale, pre-rebalance window instead of the
        # member's real, current effective one.
        geometries.update(rebalanced_geometries)
        _merge_updated_winners(winners)
    else:
        _emit_progress(progress, "[native candidate pool] Turn rebalancing skipped (diagnostic mode).")

    winners, trim_usage, fact_priority_ladder_exhausted_group_ids = _trim_by_fact_priority(
        groups=block_groups, winners=winners, tts=tts, geometries=geometries,
        voice_assignments=voice_assignments, measurement_audio_root=measurement_audio_root,
        force=force, candidate_pool_tts_workers=candidate_pool_tts_workers, progress=progress,
    )
    _merge_updated_winners(winners)

    winners, last_resort_usage = _compress_protected_content_last_resort(
        provider=provider, groups=block_groups, winners=winners, tts=tts, geometries=geometries,
        voice_assignments=voice_assignments, measurement_audio_root=measurement_audio_root,
        force=force, candidate_pool_tts_workers=candidate_pool_tts_workers,
        glossary=glossary, context_ledger=context_ledger,
        ladder_exhausted_group_ids=fact_priority_ladder_exhausted_group_ids, progress=progress,
    )
    _merge_updated_winners(winners)

    # Either compaction stage above can independently shorten one member of a
    # multi-member turn without the other members changing at all -- re-sync
    # the turn's real aggregate one final time so every member reports the
    # SAME honest number that Phase 14 will actually render as one shared
    # speed factor, rather than a stale pre-trim aggregate on some members
    # and a fresh post-trim individual number on whichever one got trimmed.
    winners = _resync_turn_rush_aggregates(
        groups=block_groups, winners=winners, geometries=geometries,
        preferred_raw_speed_percent=int(preferred_raw_speed_percent),
        measurement_audio_root=measurement_audio_root,
    )
    _merge_updated_winners(winners)

    _apply_zulu_morphology_check(
        winners=winners, groups_by_id=groups_by_id, measured_by_group=measured_by_group,
        glossary=glossary, progress=progress,
    )

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    for usage in (
        context_usage, glossary_usage, pronunciation_research_usage, grok_usage, trim_usage,
        last_resort_usage,
    ):
        for key in usage_totals:
            usage_totals[key] += int(usage.get(key) or 0)

    candidate_pool_records = [
        {
            "group_id": group_id,
            "speaker_id": str(groups_by_id[group_id]["speaker_id"]),
            "source_text": str(groups_by_id[group_id]["source_text"]),
            # New (Phase 16): a committed block can no longer be reconstructed
            # from Pass-1 alone (Pass-1 doesn't know about merges), so the pool
            # artifact must be self-describing -- commit_candidate_pool reads
            # these straight off each record instead of a freshly-rebuilt
            # one-per-sentence group.
            "segment_ids": [str(value) for value in groups_by_id[group_id]["segment_ids"]],
            "start_ms": int(groups_by_id[group_id]["start_ms"]),
            "source_end_ms": int(groups_by_id[group_id]["source_end_ms"]),
            "member_group_ids": list(groups_by_id[group_id].get("member_group_ids") or [group_id]),
            "discourse_unit_kind": discourse_unit_kind_by_group[group_id],
            # Persisted for auditability (2026-09-09) -- previously computed
            # and used correctly in-memory (it drives shortened_candidates
            # and _trim_by_fact_priority's clause-scoped literal-safety
            # check) but silently never written to this artifact, so there
            # was no way to tell "genuinely nothing droppable" apart from
            # "no clause breakdown was ever produced" without re-deriving it
            # from member-merge shape, as this session just had to do.
            "clauses": groups_by_id[group_id].get("clauses"),
            # The model's own stated reasoning for this block's translation/
            # ranking choices (2026-09-09, real user direction: "isn't there
            # a way to also log what the model is thinking") -- both fields
            # were already required by the generation schema and used to
            # steer the model's own behavior, but silently discarded rather
            # than surfaced anywhere inspectable. communicative_goal only
            # exists for a block that went through the turn-block path (a
            # per-sentence fallback unit never identifies one).
            "communicative_goal": groups_by_id[group_id].get("communicative_goal"),
            "register_notes": groups_by_id[group_id].get("register_notes"),
            "shortened_candidates": list(groups_by_id[group_id].get("shortened_candidates") or []),
            "candidates": measured_by_group.get(group_id, []),
            "selected_candidate_id": winners[group_id]["candidate_id"] if group_id in winners else None,
            "requires_review": bool(winners.get(group_id, {}).get("requires_review")),
            "effective_start_ms": winners.get(group_id, {}).get("effective_start_ms"),
            "effective_source_end_ms": winners.get(group_id, {}).get("effective_source_end_ms"),
        }
        for group_id in groups_by_id
    ]

    artifact = {
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "input_sha256": input_hash,
        "candidate_pool_mode": True,
        "candidate_pool": candidate_pool_records,
        "requires_review_count": sum(1 for record in candidate_pool_records if record["requires_review"]),
        "usage": usage_totals,
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.candidate_pool, artifact)
    _emit_progress(
        progress,
        f"[native candidate pool] Complete: {len(winners)}/{len(groups)} sentence(s) have a selected "
        f"winner, {artifact['requires_review_count']} require review",
    )
    return artifact


def compare_candidate_pool_to_committed_pass2(*, job_root: Path) -> dict[str, Any]:
    """Compare each sentence's candidate-pool winner against what is already
    committed in pass2_natural_translation.json, at zero additional spend -- both
    artifacts are read from disk, nothing is translated or measured again. A
    reporting helper for the Phase 5 rollout checklist; never writes anything.
    """
    paths = native_dub_paths(job_root)
    if not paths.candidate_pool.is_file():
        raise ValueError("Run the candidate-pool builder first; pass2_candidate_pool.json is missing")
    if not paths.pass2.is_file():
        raise ValueError("No committed pass2_natural_translation.json to compare against")
    pool_artifact = read_json(paths.candidate_pool)
    committed = read_json(paths.pass2)
    committed_by_group = {
        str(item["group_id"]): item for item in committed.get("temporal_mask_groups") or []
    }
    rows: list[dict[str, Any]] = []
    for record in pool_artifact.get("candidate_pool") or []:
        group_id = str(record["group_id"])
        committed_group = committed_by_group.get(group_id)
        committed_text = str(committed_group.get("spoken_text") or "") if committed_group else ""
        selected_id = record.get("selected_candidate_id")
        selected_candidate = next(
            (c for c in record.get("candidates") or [] if str(c.get("candidate_id")) == selected_id),
            None,
        )
        selected_text = str(selected_candidate.get("spoken_text") or "") if selected_candidate else ""
        selected_ms = int(selected_candidate.get("measured_ms") or 0) if selected_candidate else 0
        rows.append({
            "group_id": group_id,
            "committed_text": committed_text,
            "candidate_pool_translator": (selected_candidate or {}).get("translator"),
            "candidate_pool_candidate_id": selected_id,
            "candidate_pool_text": selected_text,
            "same_text": committed_text.strip().casefold() == selected_text.strip().casefold(),
            "candidate_pool_measured_ms": selected_ms,
            "requires_review": bool(record.get("requires_review")),
        })
    changed = [row for row in rows if not row["same_text"]]
    return {
        "sentence_count": len(rows),
        "changed_count": len(changed),
        "requires_review_count": sum(1 for row in rows if row["requires_review"]),
        "rows": rows,
    }


CANDIDATE_POOL_COMMIT_SCHEMA_VERSION = "mathula-native-candidate-pool-commit-v2-turn-blocks"


def commit_candidate_pool(
    *, job_root: Path, force: bool = False, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Commit pass2_candidate_pool.json's per-sentence winners into
    pass2_natural_translation.json, in the SAME temporal_mask_groups shape the
    existing --temporal-mask path writes, so every downstream consumer
    (referent_audit, native-qa, render_native_dub) works completely unchanged
    -- see the approved candidate-pool plan's framing: additive, not a rewrite.

    This is Phase 6 of that plan: the first point where the candidate pool's
    output becomes the REAL, committed translation rather than a side
    artifact. Refuses to commit an incomplete pool (any sentence missing a
    selected winner) or one whose sentences no longer match the current
    Pass-1 transcript, rather than silently committing a partial result.
    """
    paths = native_dub_paths(job_root)
    if not paths.candidate_pool.is_file():
        raise ValueError(
            "Run native-translate --candidate-pool first; pass2_candidate_pool.json is missing"
        )
    pool_artifact = read_json(paths.candidate_pool)
    pool_order = list(pool_artifact.get("candidate_pool") or [])
    if not pool_order:
        raise ValueError("pass2_candidate_pool.json has no candidate_pool records")

    missing_winners = sorted(
        str(record["group_id"]) for record in pool_order if not record.get("selected_candidate_id")
    )
    if missing_winners:
        raise ValueError(
            f"{len(missing_winners)} sentence(s) have no selected candidate-pool winner "
            f"(e.g. {missing_winners[0]}); cannot commit an incomplete pool"
        )

    pass1 = read_json(paths.pass1)
    segments = list(pass1.get("segments") or [])
    if not segments:
        raise ValueError("Pass-1 artifact has no restored segments")
    pass1_sha256 = checksum(paths.pass1)
    sentence_records = _build_pass1_sentence_records(segments, job_root=job_root)
    # render_native_dub's temporal-mask path requires pass1_sentences.json to
    # already exist; the candidate-pool builder never wrote it (unlike the
    # --temporal-mask generator), so ensure it exists here too.
    atomic_write_json(paths.pass1_sentences, {
        "schema_version": "mathula-native-pass1-sentences-v1",
        "source_pass1_sha256": pass1_sha256,
        "sentence_count": len(sentence_records),
        "records": sentence_records,
        "completed_at": utcnow(),
    })

    # Phase 16: a committed record may now represent MULTIPLE merged original
    # sentences (Turn-Block Translation), so a block's group_id is one of its
    # own constituent original ids, not a freshly-rebuilt one-per-sentence id
    # -- a bare SET-equality check no longer means anything. Verify instead
    # that every pool record's segment_ids, flattened in commit order,
    # reconstruct Pass-1's true chronological segment order -- the exact
    # coverage/order pattern _temporal_mask_groups_for_render already uses
    # successfully for this same "possibly-merged groups" case.
    expected_segment_ids = [str(record["segment_id"]) for record in sentence_records]
    seen_segment_ids: list[str] = []
    for record in pool_order:
        record_segment_ids = [str(value) for value in record.get("segment_ids") or []]
        if not record_segment_ids:
            raise ValueError(f"Candidate-pool record {record.get('group_id')} has no segment_ids")
        seen_segment_ids.extend(record_segment_ids)
    if seen_segment_ids != expected_segment_ids:
        raise ValueError(
            "pass2_candidate_pool.json's sentences no longer reconstruct the current Pass-1 "
            "transcript's chronological segment order "
            "(re-run native-translate --candidate-pool --force before committing)"
        )

    input_hash = _hash_payload({
        "pass1_sha256": pass1_sha256,
        "candidate_pool_sha256": checksum(paths.candidate_pool),
        "commit_schema_version": CANDIDATE_POOL_COMMIT_SCHEMA_VERSION,
    })
    if paths.pass2.is_file() and not force:
        existing = read_json(paths.pass2)
        if existing.get("input_sha256") == input_hash:
            _emit_progress(progress, "[native candidate pool] Reusing matching committed pass2 checkpoint")
            return existing

    temporal_mask_groups: list[dict[str, Any]] = []
    translations: list[dict[str, Any]] = []
    # Iterate the pool's OWN record array directly (already chronological, by
    # construction -- build_candidate_pool writes it from block_groups in
    # order) instead of a freshly-rebuilt one-per-sentence list; every field
    # a merged block needs comes straight from the record itself, since
    # Pass-1 alone cannot reconstruct which original sentences were merged.
    for record in pool_order:
        group_id = str(record["group_id"])
        winner_id = str(record["selected_candidate_id"])
        winner = next(c for c in record["candidates"] if c["candidate_id"] == winner_id)
        spoken_text = str(winner["spoken_text"])
        # Phase 13: a speaker turn rebalanced via window reallocation carries an
        # adjusted window here -- everything downstream (render's own per-sentence
        # synthesis/speed-fit) reads start_ms/source_end_ms/source_span_ms straight
        # off this committed group dict with no idea whether a value is the raw
        # Pass-1 one or a reallocated one, so this is the only place that override
        # needs to be applied.
        effective_start_ms = (
            int(record["effective_start_ms"]) if record.get("effective_start_ms") is not None
            else int(record["start_ms"])
        )
        effective_source_end_ms = (
            int(record["effective_source_end_ms"]) if record.get("effective_source_end_ms") is not None
            else int(record["source_end_ms"])
        )
        effective_source_span_ms = max(1, effective_source_end_ms - effective_start_ms)
        required_speed_percent = float(winner.get("required_speed_percent") or 0.0)
        rush_warning = None
        if required_speed_percent > DEFAULT_MAX_NATURAL_SPEED_PERCENT:
            rush_warning = {
                "type": "measured_rush_warning",
                "measured_ms": int(winner.get("measured_ms") or 0),
                "source_window_ms": effective_source_span_ms,
                "estimated_required_rush_percent": required_speed_percent,
                "speed_fit_mode": winner.get("speed_fit_mode") or "sentence_end",
                "fatal": False,
                "tts_policy": "rushed_speed",
            }
        semantic_guard_warnings = [
            f"{item['kind']}:{item['literal']} x{int(item['missing_count'])}"
            for item in winner.get("semantic_guard_requirements") or []
            if not item.get("blocking", True)
        ]
        segment_ids = [str(value) for value in record["segment_ids"]]
        temporal_mask_groups.append({
            "group_id": group_id,
            "speaker_id": str(record["speaker_id"]),
            "segment_ids": segment_ids,
            "member_group_ids": list(record.get("member_group_ids") or [group_id]),
            "start_ms": effective_start_ms,
            "source_end_ms": effective_source_end_ms,
            "source_span_ms": effective_source_span_ms,
            "source_text": str(record.get("source_text") or ""),
            "discourse_unit_kind": record.get("discourse_unit_kind"),
            "spoken_text": spoken_text,
            "undersized": bool(winner.get("direction") == "expand"),
            "semantic_guard_warnings": semantic_guard_warnings,
            "selected_candidate_id": winner_id,
            "selected_candidate_translator": winner.get("translator"),
            "selected_candidate_variant": winner.get("variant_id"),
            "selected_candidate_path": winner.get("path"),
            "speed_fit_mode": winner.get("speed_fit_mode"),
            "measured_ms": int(winner.get("measured_ms") or 0),
            "azure_timing_deferred": False,
            "rush_warning": rush_warning,
            "requires_review": bool(record.get("requires_review")),
            "morphology_warning": winner.get("morphology_warning"),
            "candidate_pool_source": True,
            "dropped_clause_ids": winner.get("dropped_clause_ids"),
            "omitted_items": winner.get("omitted_items"),
        })
        # First segment carries the full text, the rest are blank -- the SAME
        # convention _temporal_mask_groups_for_render already relies on for a
        # merged group (it concatenates a group's own constituent English and
        # assigns the group's FULL combined Zulu to every constituent
        # segment_id), unchanged and correct here too.
        for position, segment_id in enumerate(segment_ids):
            translations.append({
                "segment_id": segment_id,
                "spoken_text": spoken_text if position == 0 else "",
                "temporal_mask_group_id": group_id,
            })

    artifact = {
        "schema_version": CANDIDATE_POOL_COMMIT_SCHEMA_VERSION,
        "candidate_pool_committed": True,
        "candidate_pool_input_sha256": pool_artifact.get("input_sha256"),
        "provider": "azure-foundry-grok",
        "input_sha256": input_hash,
        "source_pass1_path": str(paths.pass1),
        "source_pass1_sha256": pass1_sha256,
        "temporal_mask_mode": True,
        "rolling_syllable_mode": False,
        "timing_optimized": False,
        "tts_rate_percent": 0,
        "max_natural_speed_percent": DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        "temporal_mask_groups": temporal_mask_groups,
        "translations": translations,
        "summary": {
            "mask_count": len(temporal_mask_groups),
            "source_segment_count": len(segments),
            "residual_calls": 0,
            "rollback_mask_count": 0,
            "undersized_syllable_count": sum(1 for g in temporal_mask_groups if g["undersized"]),
            "oversized_syllable_count": sum(1 for g in temporal_mask_groups if g["rush_warning"]),
            "requires_review_count": sum(1 for g in temporal_mask_groups if g["requires_review"]),
        },
        "usage": pool_artifact.get("usage") or {},
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.pass2, artifact)
    _emit_progress(
        progress,
        f"[native candidate pool] Committed {len(temporal_mask_groups)} sentence(s) to "
        f"pass2_natural_translation.json — {paths.pass2}",
    )
    return artifact


def _estimate_text_output_tokens(text: str, *, overhead: int = 10) -> int:
    """Content-length-based output-token estimate for one piece of translated text.

    Conservative for agglutinative isiZulu plus JSON/id overhead. A flat "N tokens per
    item" guess badly underestimates a batch containing one unusually long, dense sentence
    (real testimony passages run long) -- budgeting from the actual character length of
    what needs to be reproduced is the same fix referent_audit.py's own
    _estimate_output_tokens already applies to its repair batches.
    """
    chars = len(str(text or ""))
    return max(18, math.ceil(chars / 3.0) + overhead)


def _estimate_segment_output_tokens(item: Mapping[str, Any]) -> int:
    return _estimate_text_output_tokens(str(item.get("source_text") or ""))


def _estimate_pass2_output_tokens(segments: Sequence[Mapping[str, Any]]) -> int:
    return 80 + sum(_estimate_segment_output_tokens(item) for item in segments)


def _estimate_temporal_translate_window_tokens(batch: Sequence[Mapping[str, Any]]) -> int:
    # _TEMPORAL_MASK_OUTPUT_CALIBRATION corrects _estimate_text_output_tokens's raw
    # character-based guess (calibrated for a different, sparser output shape) toward the
    # real observed density of a full-sentence translation response -- see its definition
    # for the confirmed production numbers this closes a real ~4x undershoot on.
    return 80 + sum(
        int(_estimate_text_output_tokens(str(group.get("source_text") or "")) * _TEMPORAL_MASK_OUTPUT_CALIBRATION)
        for group in batch
    )


def _estimate_temporal_compact_chunk_tokens(repairs: Sequence[Mapping[str, Any]]) -> int:
    # Budgeted off the CURRENT Zulu length, not the English -- a compacted result is at
    # most as long as what it compacts (same reasoning referent_audit.py's own repair
    # estimate uses: "corrected_zulu is roughly the same length as current_zulu"). Same
    # calibration correction as the translate-window estimate above.
    return 60 + sum(
        int(_estimate_text_output_tokens(str(item.get("current_zulu") or "")) * _TEMPORAL_MASK_OUTPUT_CALIBRATION)
        for item in repairs
    )


def _shrink_batch_to_budget(
    candidates: Sequence[Any],
    *,
    budget: int,
    base_overhead: int,
    estimate_fn: Callable[[Any], int],
) -> list[Any]:
    """Take a leading prefix of ``candidates`` whose estimated output stays under ``budget``.

    Always includes at least the first candidate -- progress must never stall even if a
    single item alone would exceed budget; the truncation-retry safety net
    (_retry_batch_call) handles that rare case instead of this proactive shrink. A
    window/chunk of normal-length content passes through unchanged; one containing
    unusually long items naturally shrinks, so the caller sends more, smaller calls for
    that specific stretch instead of risking (or over-paying for) one oversized call.
    """
    batch: list[Any] = []
    running = int(base_overhead)
    for item in candidates:
        estimate = int(estimate_fn(item))
        if batch and running + estimate > budget:
            break
        batch.append(item)
        running += estimate
    return batch


def _retry_batch_call(
    call: Callable[[Sequence[Any]], tuple[dict[str, Any], dict[str, int]]],
    items: Sequence[Any],
    *,
    operation: str,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Retry a batched Grok call with a halved batch when output is truncated.

    Mirrors referent_audit.py's _call_with_truncation_retry: a batch's token budget is an
    estimate, and one unusually long item -- or several moderately long ones together --
    can still exceed it even after _shrink_batch_to_budget's proactive sizing. Rather than
    losing the whole batch (and everything already completed before it), split the batch
    in half and retry down to single items. A single item that still truncates propagates
    the error rather than being silently dropped -- unlike referent_audit's repair batches,
    a temporal-mask window/chunk has no "leave it unresolved and continue" fallback path.
    """
    try:
        return call(items)
    except GrokOutputTruncated:
        if len(items) <= 1:
            _emit_progress(
                progress,
                f"[native pass 2] WARNING: {operation} truncated even for a single item; "
                "cannot shrink further.",
            )
            raise
        _emit_progress(
            progress,
            f"[native pass 2] {operation} truncated for a batch of {len(items)}; "
            "retrying as two smaller batches",
        )
        midpoint = len(items) // 2
        left_result, left_usage = _retry_batch_call(
            call, items[:midpoint], operation=operation, progress=progress
        )
        right_result, right_usage = _retry_batch_call(
            call, items[midpoint:], operation=operation, progress=progress
        )
        merged_usage = {
            key: left_usage.get(key, 0) + right_usage.get(key, 0)
            for key in set(left_usage) | set(right_usage)
        }
        return {**left_result, **right_result}, merged_usage


def _batch_neighbour_context(
    all_segments: Sequence[Mapping[str, Any]],
    batch: Sequence[Mapping[str, Any]],
    *,
    index_by_id: Mapping[str, int],
    radius: int,
) -> list[dict[str, Any]]:
    if not batch:
        return []
    first = index_by_id[str(batch[0]["segment_id"])]
    last = index_by_id[str(batch[-1]["segment_id"])]
    positions = list(range(max(0, first - radius), first)) + list(
        range(last + 1, min(len(all_segments), last + 1 + radius))
    )
    return _compact_wire_segments([all_segments[pos] for pos in positions], text_key="source_text")


def _normalize_pass1(
    data: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Validate correction identity while keeping local STT text authoritative.

    LLM echo fields are useful for review but are not safe identity keys. Segment ID
    is the immutable mapping contract, so an echo mismatch is repaired from the
    local transcript rather than aborting an otherwise valid whole-program pass.
    """
    source = {item["segment_id"]: item["source_text"] for item in segments}
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for correction in data["corrections"]:
        segment_id = correction["segment_id"]
        if segment_id not in source:
            raise ValueError(f"Pass 1 returned unknown segment_id: {segment_id}")
        if segment_id in seen:
            raise ValueError(f"Pass 1 returned duplicate correction: {segment_id}")
        seen.add(segment_id)
        canonical = source[segment_id]
        echoed = str(correction.get("original_text") or "")
        if "original_text" in correction and echoed != canonical:
            _emit_progress(
                progress,
                f"[native pass 1] WARNING {segment_id}: Grok source echo differed from local STT; "
                "using the local transcript as canonical",
            )
        corrected_text = str(correction.get("corrected_text") or "").strip()
        if not corrected_text:
            raise ValueError(f"Pass 1 returned empty corrected_text for {segment_id}")
        if corrected_text == canonical:
            _emit_progress(
                progress,
                f"[native pass 1] WARNING {segment_id}: dropping no-op correction",
            )
            continue
        normalized.append({
            "segment_id": segment_id,
            "original_text": canonical,
            "corrected_text": corrected_text,
            "reason_code": correction["reason_code"],
        })
    return normalized


def _build_pass2_batches(
    segments: Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Pre-budget batches by estimated target output, never splitting a segment."""
    target = max(700, int(max_output_tokens * 0.52))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_estimate = 80
    for raw in segments:
        item = dict(raw)
        estimate = _estimate_segment_output_tokens(item)
        if current and current_estimate + estimate > target:
            batches.append(current)
            current = []
            current_estimate = 80
        current.append(item)
        current_estimate += estimate
    if current:
        batches.append(current)
    return batches


def _normalize_pass2(
    data: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    expected = [item["segment_id"] for item in segments]
    actual = [item["segment_id"] for item in data["translations"]]
    if actual != expected:
        raise ValueError("Pass 2 must return every segment exactly once and in source order")
    source = {item["segment_id"]: item["source_text"] for item in segments}
    normalized: list[dict[str, Any]] = []
    for item in data["translations"]:
        segment_id = item["segment_id"]
        canonical = source[segment_id]
        if "source_text" in item:
            echoed = str(item.get("source_text") or "")
            if echoed != canonical:
                _emit_progress(
                    progress,
                    f"[native pass 2] WARNING {segment_id}: Grok source echo differed; "
                    "using the restored local English as canonical",
                )
        spoken = str(item.get("spoken_text") or "").strip()
        if not spoken:
            raise ValueError(f"Pass 2 returned empty spoken_text for {segment_id}")
        normalized.append({
            "segment_id": segment_id,
            "source_text": canonical,
            "spoken_text": spoken,
        })
    return normalized


def _whole_transcript_preflight(segments: Sequence[Mapping[str, Any]], *, stage: str) -> None:
    # Grok-4.1-fast has a large context window, but the provider deployment is
    # operator-selectable. Refuse silent chunking because this architecture's
    # defining property is whole-program discourse context.
    characters = sum(len(str(item.get("source_text") or item.get("restored_text") or "")) for item in segments)
    approximate_tokens = math.ceil(characters / 3.2)
    if approximate_tokens > 100_000:
        raise ValueError(
            f"{stage} requires whole-transcript context but this transcript is approximately "
            f"{approximate_tokens:,} text tokens before prompt overhead. Split the programme at an editorial boundary "
            "or use a Grok deployment with a verified larger context; Mathula will not silently chunk this pass."
        )


def _hash_payload(value: Mapping[str, Any]) -> str:
    import hashlib
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "MANUAL_WEB_OVERRIDE_SCHEMA_VERSION",
    "NATIVE_DUB_SCHEMA_VERSION",
    "NATURAL_TRANSLATION_SCHEMA_VERSION",
    "PASS1_PROMPT_VERSION",
    "PASS2_PROMPT_VERSION",
    "STT_RESTORATION_SCHEMA_VERSION",
    "build_phrase_groups",
    "install_manual_web_overrides",
    "ensure_native_voice_assignments",
    "load_native_voice_assignments",
    "native_dub_paths",
    "render_native_dub",
    "resolve_job_source_video",
    "run_native_pass1",
    "run_native_pass2",
    "run_native_translation",
]
