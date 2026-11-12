"""A dubbing agent, written from first principles: no agent framework.

An agent is a language model called in a loop with tools. Each turn the model
either asks to call tools (we run them and feed the results back as
observations) or answers in plain text (it believes it is done). The loop is the
whole idea; everything else here is what makes it safe and inspectable:

  * tools        hand-written JSON schemas -> Workbench methods (the environment)
  * observations a tool failure is returned to the model as data, never raised,
                 so the agent can recover (e.g. TTS budget exhausted -> submit now)
  * stop rules   the model decides it is done, OR a hard ceiling (turns, tokens)
  * completion   an agent that stops without a scored submission gets one nudge
  * trace        every model call and tool call is appended to trace.jsonl, so a
                 run can be read turn by turn -- that is where the learning is

``run_one_shot`` is the no-tools baseline (one prompt, one answer) so the value
the tools add is measurable on the same scoreboard.

Run:  python -m mathula_tv.dub_agent --clip fb3d08b6 --run-id agent-v1 [--mode oneshot]
"""
from __future__ import annotations

import argparse
import json
import re
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

from .dub_workbench import BudgetExhausted, Workbench

DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT = 2.0
DEFAULT_MAX_PUSHBACKS = 2
DEAD_AIR_SLACK_PCT = 0.5          # the run counts as having met the bar this far above the stated residual dead air: the last half point is not audible and costs dozens of turns
WRAPUP_TURNS = 6                  # once the bar is met the agent may use this many more turns for a plainly wrong unit, then the run ends (prod-69: bar met at turn 21, run ended at turn 80)
MET_STOP_REASONS = frozenset({"agent_finished", "bar_met"})
MAX_TOOL_RESULT_CHARS = 60000   # a 28-segment clip with ledger + glossary is ~19k; this is a runaway ceiling

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert English-to-isiZulu news dubbing agent working for Mathula TV.

Goal: dub ONE complete news clip. Produce isiZulu speech that a native speaker would find
natural, faithful to the English, and that fits the time the English speaker occupies on screen.

You work through tools and decide your own method. You are scored on:

1. Hard gates (a failure is a defect): every name, number and date survives; no invented names;
   the same entity is not stated twice for one English mention; nothing left untranslated.

2. Timing. Each unit's Zulu is played at its natural speed -- it is NEVER sped up or slowed down -- so the
   words you choose are the only thing that controls timing. The speech should fill its source window:
   too short = dead air while the speaker is still on screen; too long = an overrun that spills past its
   window into the next speaker.
   FIT FIRST. Fitting every unit into its time frame is the first job, because that is what keeps speakers
   from talking over each other and lets a multi-speaker clip flow from start to finish. The harness locks
   timing as you go: once a segment has fitted in an accepted submission, a later submission that breaks
   it is rejected unless it is clearly better overall.
   Priority when timing trades off: an overrun is WORSE than dead air. Speech that spills past its window
   collides with the next speaker and breaks sync, while a little silence is only a pause. So first make
   sure no unit overruns (overrun_pct must be 0), and only then shorten the dead air that is left. Never
   accept an overrun to remove a shortfall.
   Silence padding: the renderer closes a shortfall of up to 150 ms with trailing silence, and for a
   speaker turn of several consecutive units it can spread a larger shortfall over the real gaps between
   them (each gap widened by at most 1.5 s; nothing if the units are back to back). Prefer wording that
   fills the time, but only wording the speaker's meaning carries. Never add a word, a phrase or an
   English-derived piece of Zulu just to fill a window: a short line that leaves a fraction of a second
   is correct and the silence is harmless. A filler a Zulu speaker would not say is a worse defect than a
   short unit. Understanding and natural speech come before the last 300 ms of fit.
   Length guide: open_clip gives each segment fits_about_chars, the Zulu length that fills its window at this voice's natural speed (about 12
   characters per second). Write your first draft to that length, with what the speaker said: a compact first draft has landed 30% short and
   cost most of the run to close unit by unit. Every measurement tells you the unit's chars and, when it is more than 6% off its window,
   add_about_chars or cut_about_chars: the size of the edit at the rate that very text measured. Edit by that amount (fuller phrasing of what
   the speaker said, never a restatement) so one more measurement closes the unit, and do the units that are furthest off first. When the
   measurement says leave_short, the gap is under half a second: add nothing.

   Figures. Write numbers and amounts as digits with the isiZulu prefix the sentence needs ("kuka-31%", "abantu abangu-11",
   "eminyakeni engu-20", "ngo-2006"); never spell a small figure out in isiZulu words. A figure of one or two digits (with its
   decimals and %) is spoken as code-switched English after its prefix ("angu-thethi waan"), so the prefix and the unit word are
   your choice: ask figure_style("31 percent") (or "20 years", "11 municipalities", "in 2006") for how translators wrote that kind of
   figure before you write a percentage, an age, a duration, a head count or a year. Dates and ordinals are spoken as code-switched
   English too. Larger amounts ("u-4,500", "abantu abangu-500") are spoken in native isiZulu (izinkulungwane ezine namakhulu
   amahlanu zamaRandi), which takes longer than English. When a unit does not fit its window, you may make such an amount shorter by
   writing ~en straight after it ("u-4,500~en"): it is then spoken in English ("four thousand fayiv hundred rand"). The ~en is
   never spoken, and the digits stay for the number checks. Use it only where the time requires it; native is the default there.

   Dates. Write a day of a month the way the isiZulu newspaper Isolezwe writes it: the MONTH FIRST with the isiZulu preposition glued on, then the day
   as plain digits: "Kusukela ngoNovemba 4, ..." for "From the 4th of November, ...", "kuze kube uDisemba 1", "ngoJanuwari 5, 2027". Never keep the English
   "of" or the day-first order ("ngo-4th of November" is wrong), never an ordinal suffix ("4th"), and never the "ngomhla wesine kuNovemba" construction.
   Use the everyday borrowed month names (Novemba, Mashi, Disemba), not the old calendar names (Lwezi, Ndasa, Zibandlela). The renderer reads the day in English.

   Loanwords first. When Zulu has an everyday word for an English word, a native word or an established loanword (isayensi for science,
   ubuchwepheshe for technology, ibhiliyoni for billion), write that Zulu word, with the prefix the sentence needs, instead of keeping the
   English: a Zulu voice says a Zulu word naturally, a Zulu listener hears ordinary Zulu, and the dub stops fighting the voice's rule that every
   written vowel is spoken in full. Keep an English word only for a name, a title, an acronym, or a word with no everyday Zulu word. A word that
   translators themselves write in English behind a particle (i-microphone) has its everyday Zulu form already: that borrowed form is the one to write.
   When you are not sure what to do with an English word, ask what real Zulu does: attested_zulu(word, zulu_forms) shows the Zulu forms that
   translated text pairs with it, whether the English word stays English there, how Zulu news sites write it, and how often a form you are thinking
   of has been written. The ladder:
   first a borrowed form that real text writes (a loanword such as igranti or imeya, or the English word kept with a hyphenated particle, i-microphone: attested_zulu
   shows how translators, news and speakers write a word); then an attested native Zulu word; otherwise code-switch, which means keeping the English word
   as the speaker says it (same number, plural -s included), with a hyphenated particle if the grammar needs one (ne-, ye-). A formal native coinage never
   outranks a borrowed form that translators, news and speakers write (i-microphone before umbhobho, although both are used: the borrowed form is the product's choice for widest reach). Never invent a form
   that real text does not write. Only when real text itself adapts such a word, create_loanword builds Zulu forms of it by the loan rules from the
   dictionary's sounds (a vowel inserted where Zulu has no cluster and at the end, p t k b written ph th kh bh) and shows what each changes; nothing is
   listened to or spoken. Write the one you choose in the text with the prefix the sentence needs, and call use_loanword to remember it.

   A kind of thing is not a name: "medical aid", "pension fund", "provident fund" are common nouns, so each noun takes its everyday Zulu
   word (or one you create), even though the speaker says them in English.

   English words you keep in the Zulu (names, acronyms, titles, a word with no everyday Zulu word) are spoken by the Zulu voice from a spelling the
   harness writes for them by rules, from a dictionary's English sounds. You do not choose or check how they sound: keep the word itself in the text, as
   the speaker says it. Never glue a Zulu prefix onto an English word the speaker says as it is (a hyphenated particle such as ne- or ye- before it is
   ordinary Zulu).

3. Naturalness: retell the way a person would say it informally. This is not a word-for-word transfer.
   Use the whole clip's context.
   Comprehension and fluidity are what the listener gets, so they are part of fitting the time, not a casualty of it. Hear each unit as a
   listener who hears it once: short common words, one idea per sentence, the verb early, the sentence ending on its stress. Fit by choosing
   among natural ways to say what the speaker said (a longer or shorter synonym, a different construction), never by padding or by cutting
   a fact. Where English has a set phrase, say what a Zulu speaker says in that moment, not a translation of the English words.
   A unit that sounds stiff, glued or word-for-word is a defect even when it fits to the millisecond.
   Length by grammar. You know isiZulu's grammar and registers more widely than any one speaker does, so make it your length control: the
   language says one meaning at many lengths, and choosing among those constructions is how a unit fits its window and sounds natural at once.
   {grammar_pointer}To shorten without losing a fact: let a verb extension carry what a phrase would (-el-
   for "for/to/at", -is- for "make do", -w- for "by" clauses, -an- for "each other"), leave out an emphatic pronoun the subject concord
   already carries unless the speaker stresses it, use the conjunctive verb form where a complement follows, turn a relative clause into a
   participial or an infinitive, take the shorter of two equally common words. To lengthen with content the English does carry: the tense and
   aspect forms its "still", "already", "just", "used to", "was about to" imply, an emphatic pronoun where the speaker leans on it, a full
   relative clause or demonstrative where it makes the sentence plainer. When a unit is off, draft two or three renderings that differ in
   construction, measure them together in one measure_tts_batch call, and keep the one that is clearest and most natural of those that fit.
   Never lengthen with an intensifier, a repeated word or a word the English lacks, and never shorten by dropping a fact, a qualifier or a negation.
   This is a SOCIAL MEDIA product: simplicity and maximum reach are the point. A viewer gives a clip a second or
   two, so use plain, short, everyday wording that lands immediately for as many people as possible.
   Widest understanding beats formality. Choose the word that the most everyday isiZulu listeners understand the
   moment they hear it -- urban and rural, older and younger, first- and second-language speakers -- even if it is
   a loanword or an informal form, and even if a more formal or purist word exists ("igranti", not "isibonelelo";
   "i-VAT"; "imeya"). The glossary fixes names and terms for consistency; where one of its terms is a stiffer
   or less familiar word than a common loanword for the same thing, use the familiar word.
   Language switches: the speaker moves between English, isiZulu and sometimes Sepedi, Sesotho or another language. Where the English
   recogniser wrote nonsense for such a stretch, the transcript has already been restored from YouTube's own captions (the restoration pass),
   so a segment's English may hold his own isiZulu, Sepedi or Sesotho words. Politicians speak like this on purpose, so you read it as a
   bilingual listener would: keep his isiZulu as isiZulu, say other languages' meaning in plain isiZulu, and do not skip the stretch or keep
   only the clear English (a lost 'unga-' turns "do not confuse" into "confuse").
   Delivery: the clip carries a `delivery` sheet measured from the audio. A speaker people know (a politician above all) should be
   recognisable in the dub, so write the isiZulu with their shape: keep their pace (a quick run stays short and quick, a slow weighted
   line stays weighted), keep the pauses they take for effect by ending the sentence there, keep the phrases they repeat repeated, and
   let the words they lean on (stressed_words) land at the end of a phrase or right after a pause. This shapes wording only: never
   lengthen a unit past its window for it, never invent a pause, and the meaning and plain wording above still come first.
   Where the sheet shows the speaker holding a pause of 600 ms or more (pauses_inside), you may write ~pause1500 (milliseconds, 200-2000)
   at the matching spot in your isiZulu: "...ngentela yethu. ~pause1500 Uma uya esitolo...". It is never spoken. ~pauseN is the TOTAL silence at that spot, punctuation
   included: a full stop already leaves about 0.9 s and a comma about 0.15 s by itself, so "ngentela yethu. ~pause1500" is the stop's 0.9 s plus the rest, 1.5 s in all, and
   "ngentela yethu. ~pause700" adds nothing (the stop alone is longer than 0.7 s). To match a speaker's pause shorter than about 1.1 s, do not end the phrase with a full stop:
   end it with a comma or no mark and write ~pause (the speaker's pause in ms) after it. The pause counts toward the unit's duration like any other part, and it is also the
   cheapest way to fill a window honestly. Use it only for pauses the speaker really takes (open_clip's phrasing gives each pause's length), at most two per unit. After
   the voice has spoken, the renderer also moves the silence between your sentences so each one starts near where the source speaker's sentence starts (the speech and the unit's
   length are never touched), so write the sentences in the order the speaker made them and let the pauses be the speaker's.

   Pacing and punctuation. Punctuation is how these voices breathe, and what each mark costs was measured (the silence left between two
   clauses, the same words each time): no mark leaves none, so the clauses run on; an ellipsis ("...") is ignored and also leaves none; a
   comma, semicolon, colon or dash all leave the same short breath, about 0.12-0.18 s, and two marks together do not add up; a full stop,
   question mark, exclamation mark or new line leaves about 0.9 s, whatever follows it. So punctuate as a speaker breathes: a comma where a
   speaker would take a breath between phrases, nothing inside a phrase that belongs together, a full stop only where the thought is over. A
   full stop in the middle of a thought costs nearly a second and makes the line stilted; a comma the speaker did not take costs 0.15 s. Do
   not use punctuation or ~pause to fill time that the wording should fill, and match the speaker's own pauses (pauses_inside) before any
   of your own. A hyphen joins the pieces of one word (i-microphone, ne-, the parts of a figure) and adds no silence of its own; spacing
   inside a number or a name does not add silence either, so join figures and names as written and do not split them with commas. But a figure is what the viewer has to take in: after each percentage or amount that is followed by more words, a
   comma gives the listener the breath to digest it ("ku-75.8%, ngo-2006, laya ku-44.05%, kulo nyaka"); never run two figures together with no
   break between them. The renderer adds that comma after a percentage if you leave it out, so write it yourself where you want the break.

   Learn from the listener. recall_lessons also returns ear_lessons: what the person who listens to these dubs has heard wrong before, by item class (a title read raw, digits read in a
   Zulu accent, a name run together, a figure with no breath), with what fixed it, and the spellings they chose over the rules'. Read them before your first draft and write so they do
   not recur. They are the same gap, seen again: between what you wrote and what the listener heard.

   See what the voice gets. What you write is not what the voice reads: a dictionary of rules turns your digits, titles, abbreviations and English names into the spoken form,
   and a listener found its gaps one by one ("Dkt" read raw, "Section 79" read in digits, a name said as separate words). Before you submit, call say_it on any unit that holds a
   figure, a year, a title, an abbreviation or an English name, and read the spoken form the way a listener would hear it. 'still_as_written' lists what no rule converted: write
   that another way (spell the title out, put the figure after its prefix) or say in approach_notes what you could not fix. It only shows; the spelling of a word is still decided by
   the rules and by the listener's ear.

   Listening load. The viewer hears each line once and cannot re-read it, so write for what the ear can take in, not only for fit. The
   words between two breath marks are one breath group: it should carry one new fact and last about two to three seconds of speech,
   rarely more than about six (a run past about eight is flagged). A figure that carries the point gets the breath after it, and when isiZulu word order allows, put the figure
   at the end of its phrase so the breath lands right after it. Never run two figures, or a figure and a name, into each other with no
   break. The delivery sheet shows how the speaker spaced the line for the listener (breaths_inside, figures_spoken, pauses_inside): give
   the viewer the same room. check_text and every submit_dub result measure this (figures_run_together, long_run_without_breath, listening_load); the flags are evidence,
   you decide. When a sentence is too dense for a listener, split it or reorder it, with the facts and names unchanged, instead of
   cutting it; a sentence end costs about 0.9 s, so spend it where the thought really ends.

   A conversation, not a text. Most clips are people talking to each other: a presenter, a reporter, a guest, each turn answering the one before and handing over by name
   ("Yeah, Mfundo, ...", "Kenny, ...", "Indeed, Ayanda, ..."). open_clip says so (conversation) and shows, for each segment, where its speaker really paused (phrasing: |0.7s| is a
   pause of 0.7 s). Take your punctuation from that, not from habit: the speakers ran a name of address into the words beside it with no pause at all
   ("Indeed, Ayanda, hopefully" was said without a gap on either side of Ayanda). A comma is about 0.15 s of silence in your Zulu and a full stop 0.9 s, so a comma on both sides of
   every name of address makes people who are talking to each other sound like a script being read, and a listener hears it at once. Write a name of address in the same breath as the words round it, put a pause where the
   speaker's thought changes, and answer the speaker before you in the way a person answers a colleague. submit_dub lists the commas of your text that no pause of the speaker's matches
   (comma_check): read it as the listener's view of your punctuation. It also lists the pauses of 0.8 s or more that the speaker held (phrasing |1.4s|) where your Zulu has
   none that long: a speaker who stops for 1.4 s mid-sentence ("as well as ... Masemola") and a dub that runs straight on get out of step with the picture, so put a
   ~pauseN (the total silence) at that place.

Words the voice cannot say: a listener heard the voice stutter on "imininingwane" (and its forms, "emininingwaneni"): never write it. Where the English says details write the English word "details" with the Zulu prefix the sentence needs, hyphenated: you choose the concord ("anikeze ama-details", "angene kuma-details",
"nama-details", "ngama-details", "ze-details"), as everyday speakers say it; where it says information write "ulwazi". check_text and submit_dub name any such word in your text.

Tools give you real measurements. Azure TTS timing is not predictable from word counts, and with no speed
control you must get the wording right. The scorer re-measures your final text with the same function, so
what you measure is what you are graded on. Measurements are budgeted.

Every tool call re-sends your whole history, so each extra turn costs more than the last. After you
submit, your earlier working steps (drafts, measurements, checks) are trimmed from your history: the clip,
your latest submit_dub call and its result are always kept.

submit_dub returns the current Zulu text of the units it flags (and of the claims_to_check), so read it there; call get_submission
only for units whose text you do not have.

submit_dub takes the full set of units, or, with partial=true, only the units you changed: each replaces the
current unit with the same item_id (ids in remove are deleted) and the rest carry over. Sending only what
changed is much cheaper than rewriting the whole clip. get_submission shows your current units
(all, or just the ones you ask for), so you never have to rewrite from memory.

The bar for "done" (checked by the harness against your last submit_dub result, not by your say-so): no unit
overruns or runs into another unit's speech (collisions), no hard-gate failures, no unit
whose meaning an independent reader finds materially changed from the English (submit_dub reports these as
meaning_loss, with what your Zulu reads back as; a lost qualifier, a changed time or negation, a dropped clause that
carried a fact), and residual_dead_air_pct at or below {max_dead_air}. submit_dub also lists claims_to_check:
units where the same reader found something your Zulu says that the English does not (an addition or a changed
detail), with what it reads back as. That is not a bar and the reader is sometimes wrong: look at the English of
each, take out what it does not say (a wider word than the speaker used, an intensifier, a figure or a reason of your
own), and leave it alone when the English does say it. Window length is never a reason to add a claim: a shorter
faithful unit and a pause are better than a longer unit that says more than the speaker did. If you stop while a submission is still below that bar and you have budget left,
you will be sent back to the specific units to fix.

When you are satisfied, call submit_dub (it returns your score and which units are flagged; you may revise
and resubmit). Then reply with a short plain-text summary of your approach and stop.
"""

# Said only when the grammar reference follows: with it switched off the sentence would point at nothing (the first grammar-off runs of the comparison had that flaw).
GRAMMAR_POINTER = "The grammar reference below (six modules, from real news and a checked corpus) is the rule system to apply. "

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT}%").replace("{grammar_pointer}", "")


def _grammar_version_in(system_prompt: str) -> str | None:
    """Which grammar reference the agent was given (None when the prompt carries none): written into agent_summary.json so a result can be tied to the text it saw."""
    from . import zulu_grammar

    return zulu_grammar.VERSION if zulu_grammar.HEADER[:40] in system_prompt else None


def build_system_prompt(playbook: dict | None = None, *, max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, grammar: bool | None = None) -> str:
    """The rules (always), the isiZulu grammar reference (unless switched off: MATHULA_AGENT_GRAMMAR=0 or grammar=False), and the learned strategy notes (if a playbook is given).
    The reference replaces the pass-2 grammar block this prompt used to carry: its claims were not checked against data and the new modules cover and correct them."""
    from . import zulu_grammar
    from .playbook import render_section

    use_grammar = zulu_grammar.enabled() if grammar is None else grammar
    base = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{max_dead_air_pct}%").replace("{grammar_pointer}", GRAMMAR_POINTER if use_grammar else "")
    return base + ("\n\n" + zulu_grammar.reference() + "\n" if use_grammar else "\n") + render_section(playbook)

_UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "start_ms": {"type": "integer"},
        "end_ms": {"type": "integer"},
        "speaker_id": {"type": "string"},
        "zulu_text": {"type": "string"},
        "source_segment_ids": {"type": "array", "items": {"type": "string"}},
        "source_text": {"type": "string", "description": "Only for units that share a source segment: the exact English excerpt this unit covers, copied from the segment (checked; a paraphrase is rejected). Ignored for other units."},
    },
    "required": ["item_id", "start_ms", "end_ms", "speaker_id", "zulu_text", "source_segment_ids"],
}
_MEASURE_PROPS = {
    "text": {"type": "string", "description": "The isiZulu text to synthesize."},
    "start_ms": {"type": "integer"},
    "end_ms": {"type": "integer"},
    "next_start_ms": {"type": ["integer", "null"], "description": "When the next speech starts."},
    "speaker_id": {"type": "string"},
    "source_text": {"type": "string", "description": "The English this text renders."},
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SPECS = [
    _tool("open_clip", "Open the clip: returns every English segment (id, speaker, start/end ms), the whole-clip "
          "context ledger, glossary and speaker modes. Call this first. If source_corrections is present, the English had "
          "words the speaker's audio showed were misheard; the english already carries the fix, so translate that meaning.",
          {"clip_id": {"type": "string"}}, ["clip_id"]),
    _tool("recall_lessons", "Cross-clip memory. Mistakes native-speaker reviewers marked 'bad' in earlier clips, with corrections where given; and (ear_lessons) what the listener has heard wrong "
          "in earlier dubs, by item class (title, figure, year, name, pacing, spelling), with what was written, what was heard and what fixed it. Call it first and read both.",
          {"limit": {"type": "integer"}}, []),
    _tool("measure_tts", "Synthesize text with Azure TTS at natural speed and report how it sits in "
          "[start_ms, end_ms]: measured_ms, shortfall_pct (speech missing), tail_pad_ms (trailing silence the "
          "renderer adds itself when the gap is tiny), residual_shortfall_pct (what is left as dead air), "
          "overrun_pct (how far past its allowance it runs), direction ('expand' too short, 'compress' too "
          "long, null fits), fits_naturally. Costs 1 measurement.",
          _MEASURE_PROPS, ["text", "start_ms", "end_ms"]),
    _tool("measure_tts_batch", "Measure up to 40 texts at once; same fields as measure_tts. Costs 1 per text.",
          {"requests": {"type": "array", "items": {"type": "object", "properties": _MEASURE_PROPS,
                                                    "required": ["text", "start_ms", "end_ms"]}}}, ["requests"]),
    _tool("preview_turn_padding", "Free arithmetic (no TTS). If a speaker turn made of these units finishes early, "
          "how much silence does the renderer add, and how much remains as dead air? Mechanisms in order: a "
          "tail pad of up to 150 ms when it fits the free gap, else the shortfall is spread over the turn's real "
          "inter-sentence gaps (each gap widened at most 1500 ms). members: {start_ms, end_ms, measured_ms}.",
          {"members": {"type": "array", "items": {"type": "object", "properties": {
              "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"}, "measured_ms": {"type": "integer"}},
              "required": ["start_ms", "end_ms", "measured_ms"]}},
           "last_free_gap_ms": {"type": "integer", "description": "free_gap_after_ms of the turn's last unit."}},
          ["members"]),
    _tool("check_text", "Free deterministic checks on one unit (no TTS): required literals, restated references, "
          "invented names, doubling, English leakage, English weekday names.",
          {"zulu_text": {"type": "string"}, "source_text": {"type": "string"}}, ["zulu_text", "source_text"]),
    _tool("submit_dub", "Submit the dub as ordered, non-overlapping units covering every English segment. "
          "Returns the official score and flagged units. You may resubmit. With partial=true, send ONLY the units you "
          "changed: each replaces the current unit with the same item_id, ids listed in remove are deleted, and every "
          "other current unit carries over (needs a previous accepted submission). To re-split units, send the new "
          "units and list the old ids in remove.",
          {"items": {"type": "array", "items": _UNIT_SCHEMA}, "approach_notes": {"type": "string"},
           "partial": {"type": "boolean", "description": "true = update only the units you send."},
           "remove": {"type": "array", "items": {"type": "string"},
                      "description": "partial only: ids of current units to delete (e.g. units you re-split)."}}, ["items"]),
    _tool("create_loanword", "For an English noun the speaker says that has NO everyday Zulu word: make a Zulu word for it the way Zulu adapts loans "
          "(science -> isayensi, bus -> ibhasi, spoon -> isipunu), by rule, from the dictionary's sounds for the word (p t k b written ph th kh bh, vowels "
          "reduced to a e i o u, a vowel inserted where Zulu has no cluster and at the end of the word). The forms come back with what each changes from the "
          "most typical. Nothing is listened to, spoken or scored. Not for names, acronyms or titles.",
          {"word": {"type": "string", "description": "the English word as written in your Zulu, e.g. provident"}}, ["word"]),
    _tool("use_loanword", "Remember the Zulu word you chose from create_loanword so every later clip uses the same one. You still write it in the Zulu text yourself, "
          "with the noun prefix the sentence needs.", {"word": {"type": "string"}, "zulu_word": {"type": "string"}}, ["word", "zulu_word"]),
    _tool("attested_zulu", "What real written isiZulu does with an English word, from 233,691 aligned sentence pairs of translated text (2.9 million Zulu words), the "
          "NCHLT lexicon, and 56.8 million words of Zulu web text (2013-2024) counted apart for Zulu news sites: the Zulu forms that real text pairs with the word "
          "(shared sentences, score), how often the English word itself stays English in the Zulu sentence, how Zulu news sites write the English word (bare, "
          "with a particle: web_forms), and, for forms you give, how often each has really been written (corpus_count, in_lexicon, web_news, web_other; a phrase or "
          "a form with a hyphenated particle is counted exactly and word by word). Free: no try, no model. Use it before you adapt or create a word: a form that "
          "no source writes (all counts 0) is not to be invented; one that Zulu news sites write at least twice is attested. The caution says how far the evidence reaches.",
          {"word": {"type": "string", "description": "the English word, e.g. pension"},
           "zulu_forms": {"type": "array", "items": {"type": "string"}, "description": "forms you are thinking of writing, e.g. [\"impesheni\"]"}}, ["word"]),
    _tool("human_translations", "How professional translators wrote an English word or phrase in isiZulu: the aligned pairs whose English contains it (about 270,000 pairs: 10,671 news "
          "pairs by professional translators from Umsuka, and government text from Autshumato), shortest English first, each with its source, and how many pairs there are in all. Use it "
          "for a stock phrase (a greeting, thanks, a title) or a term, to see how human translators handled it: a segment that translates a short phrase shows what they put and what they "
          "left out. The pairs are sometimes shifted by a line and the government text is older and formal: it is evidence to weigh, not an answer. Free: no try, no model.",
          {"phrase": {"type": "string", "description": "the English phrase, e.g. good afternoon, press briefing, microphone"},
           "limit": {"type": "integer", "description": "pairs to return (default 5, at most 8)"}}, ["phrase"]),
    _tool("say_it", "See your own line as the voice will receive it: the spoken form (figures and years in English after their Zulu prefix, titles expanded, names respelled by the rules), "
          "each change the dictionary made (was -> now), and what stayed as written (the field still_as_written: digits, abbreviations, capitals no rule converted, which the voice then reads by itself and often wrongly). "
          "It is the same dictionary measure_tts uses. Call it on a unit's text before you submit: if something is wrong for a listener (a title left raw, a figure still in digits, a name that will "
          "run together), write it another way. Free: no try, no speech.",
          {"text": {"type": "string", "description": "the isiZulu line, as you would submit it"},
           "speaker_id": {"type": "string", "description": "the unit's speaker (default SPEAKER_00); the voice can have its own spellings"}}, ["text"]),
    _tool("figure_style", "How professional translators wrote a FIGURE in isiZulu: give the English phrase with its figure in digits (\"31 percent\", \"20 years\", \"in 2006\", \"11 municipalities\", "
          "\"at 7am\") and you get the pairs whose English has that shape, with how the isiZulu writes the number: the prefix before the digits (abangu-, engama-, ngo-, ka-, none), whether the % "
          "sign stays, a real example of each, and how often the number is spelled out instead of written in digits. The renderer says any figure of one or two digits in English after its "
          "prefix, so the prefix and the unit word are yours to choose: ask before you write a percentage, an age, a duration, a year, a head count or a time. Several prefixes can be right "
          "(angu-/angama-): this shows usage, not a rule. Free: no try, no model.",
          {"phrase": {"type": "string", "description": "the English phrase with one figure in digits, e.g. 31 percent, 11 municipalities, in 2006"}}, ["phrase"]),
    _tool("get_submission", "The units currently accepted (all, or only unit_ids), with their text and any flags from "
          "the last score. Use it to see your own current state, e.g. after your history was trimmed.",
          {"unit_ids": {"type": "array", "items": {"type": "string"}}}, []),
]


REQUEST_TIMEOUT_SECONDS = 420.0      # one model turn; the provider default (900 s) lets a dropped connection stall a run for 15 minutes


def post_with_retry(url: str, headers: dict[str, str], body: dict[str, Any], *, timeout: float, retries: int,
                    sleep: Callable[[float], None]) -> dict[str, Any]:
    """POST once, retrying transient failures a bounded number of times.

    Transient = HTTP 429/5xx, a read timeout or a dropped connection. Each is retried with backoff up to
    ``retries`` times; anything else (a 4xx, bad JSON) is raised at once. A model call that hangs must cost a
    bounded delay, never a whole batch.
    """
    for attempt in range(1, retries + 2):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt <= retries:
                sleep(min(2 ** attempt, 20))
                continue
            raise RuntimeError(f"model call failed after {attempt} attempts: {type(exc).__name__}") from exc
        if response.status_code in (429, 500, 502, 503, 504) and attempt <= retries:
            sleep(min(2 ** attempt, 20))
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"model HTTP {response.status_code}: {response.text[:600]}")
        return response.json()
    raise RuntimeError("model call failed")


class ChatClient:
    """One chat-completions call with tools, over plain HTTP. No SDK, no hidden retries beyond transport."""

    def __init__(self, config: Any, *, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.retries = retries
        self.sleep = sleep
        self.url = f"{config.endpoint}/openai/v1/chat/completions"

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 json_mode: bool = False, max_output_tokens: int = 16000) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
            "max_completion_tokens": max_output_tokens,
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        payload = post_with_retry(self.url, headers, body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS),
                                  retries=self.retries, sleep=self.sleep)
        usage = payload.get("usage") or {}
        usage["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return payload


class ResponsesClient:
    """The /v1/responses API, which reasoning models need in order to use function tools.

    The loop keeps a neutral chat-style history; this client translates it on every call. The
    model's own output items (including its reasoning) ride along on the assistant message under
    ``_raw`` and are replayed verbatim, because a reasoning model plans across tool calls and
    loses that plan if only the visible text is sent back.
    """

    def __init__(self, config: Any, *, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.retries = retries
        self.sleep = sleep
        self.url = f"{config.endpoint}/openai/v1/responses"

    @staticmethod
    def to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": message["content"]})
            elif role == "assistant":
                if message.get("_raw"):
                    items.extend(message["_raw"])
                elif message.get("content"):
                    items.append({"role": "assistant", "content": message["content"]})
            else:
                items.append({"role": role, "content": message["content"]})
        return items

    @staticmethod
    def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [{"type": "function", **t["function"]} for t in tools or []]

    @staticmethod
    def to_chat_shape(response: dict[str, Any]) -> dict[str, Any]:
        output = response.get("output") or []
        text = "".join(
            part.get("text", "") for item in output if item.get("type") == "message"
            for part in item.get("content") or [] if part.get("type") == "output_text"
        )
        calls = [
            {"id": item["call_id"], "type": "function",
             "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}}
            for item in output if item.get("type") == "function_call"
        ]
        message: dict[str, Any] = {"role": "assistant", "content": text or None, "_raw": output}
        if calls:
            message["tool_calls"] = calls
        usage = response.get("usage") or {}
        return {
            "choices": [{"message": message, "finish_reason": "tool_calls" if calls else response.get("status")}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
            },
        }

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 json_mode: bool = False, max_output_tokens: int = 16000) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.deployment, "input": self.to_input(messages),
            "max_output_tokens": max_output_tokens,
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        if tools:
            body["tools"] = self.to_tools(tools)
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        return self.to_chat_shape(post_with_retry(
            self.url, headers, body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS),
            retries=self.retries, sleep=self.sleep))


class Trace:
    """Append-only JSONL log of everything the agent saw and did."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._started = time.monotonic()

    def log(self, kind: str, **fields: Any) -> None:
        record = {"t": round(time.monotonic() - self._started, 2), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]"


COMPACT_AFTER_TOKENS = 15_000
_KEEP_STEP_TOOLS = frozenset({"open_clip", "recall_lessons"})
COMPACTION_NOTE = (
    "[Earlier working steps (drafts, measurements, checks, past feedback) were removed to save tokens. The harness "
    "holds your accepted submission: call get_submission to see the current units. Your latest submit_dub call and "
    "its result are below; the opened clip above is unchanged.]"
)


def _step_tools(messages: list[dict[str, Any]], index: int) -> set[str]:
    return {c["function"]["name"] for c in messages[index].get("tool_calls") or []}


def compact_history(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop the working steps a newer submission has superseded. Returns (messages, dropped_count).

    A step is an assistant message plus the tool results that answer it; dropping whole steps keeps
    every remaining tool call paired with its result. What stays: the system prompt, the task, the
    steps that opened the clip and recalled lessons, the LATEST submit_dub step, and everything after
    it. What goes: every step and every user note between them (earlier drafts, measurements, checks,
    old pushbacks). Idempotent: a second pass over a compacted history changes nothing.
    """
    last_submit = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and "submit_dub" in _step_tools(messages, index):
            last_submit = index
    if last_submit is None:
        return messages, 0
    keep = {0, 1}
    index = 2
    while index < last_submit:
        message = messages[index]
        if message.get("role") == "assistant":
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            if _step_tools(messages, index) & _KEEP_STEP_TOOLS:
                keep.update(range(index, end))
            index = end
        else:
            index += 1
    dropped = [i for i in range(2, last_submit) if i not in keep and messages[i].get("content") != COMPACTION_NOTE]
    if not dropped:
        return messages, 0
    kept = [m for i, m in enumerate(messages) if i not in dropped and m.get("content") != COMPACTION_NOTE]
    position = next(i for i, m in enumerate(kept) if m is messages[last_submit])
    kept.insert(position, {"role": "user", "content": COMPACTION_NOTE})
    return kept, sum(1 for i in dropped if messages[i].get("role") == "assistant")


_SLIM_HEADLINE = (
    "group_count", "dead_air_pct", "turn_shortfall_pct", "silence_padded_pct", "residual_dead_air_pct",
    "groups_dead_air", "groups_rushed", "gate_failures", "worst_turn_shortfall_pct",
    "collisions", "cross_speaker_collisions", "worst_collision_ms", "meaning_loss_units",
)


CLAIMS_NOTE = (
    "Taking a claim out shortens the unit, and a unit left short is dead air: a submission that breaks a unit that had fitted is rejected. "
    "Compare chars_now with fits_about_chars, put back only what the English does say (or a pause) to keep the length, measure the edited "
    "units, and send all of them in one partial submit.")


def slim_submit(result: dict[str, Any]) -> dict[str, Any]:
    """The model-facing view of a submit_dub result (the harness keeps the full one)."""
    if not result.get("accepted"):
        return result
    slim: dict[str, Any] = {
        "accepted": True,
        "headline": {k: result["headline"][k] for k in _SLIM_HEADLINE if k in result["headline"]},
    }
    if result.get("merged"):
        slim["merged"] = result["merged"]
    if result.get("timing_lock"):
        slim["timing_lock"] = result["timing_lock"]
    if result.get("regressions_accepted"):
        slim["regressions_accepted"] = result["regressions_accepted"]
        slim["note"] = result.get("note")
    gates = {k: v for k, v in (result.get("gates") or {}).items() if v}
    if gates:
        slim["gates"] = gates
    padding = [t for t in result.get("turn_padding", []) if t.get("residual_ms", 0) > 0]
    if padding:
        slim["turns_ending_early"] = padding
    if result.get("flagged_units"):
        slim["flagged_units"] = result["flagged_units"]
    if result.get("spoken_form_gaps"):
        slim["spoken_form_gaps"] = result["spoken_form_gaps"]
        slim["spoken_form_gaps_note"] = ("Soft evidence, not a gate: these units still hold digits, an abbreviation or capitals that no rule converted, so the voice will read them by itself, "
                                         "often wrongly. Call say_it on the unit to see the whole spoken form; write the item another way, or leave it if it is right as it stands (a time, a fraction).")
    if result.get("listening_load"):
        slim["listening_load"] = result["listening_load"]
        slim["listening_load_note"] = ("Soft evidence, not a gate: a figure run into the next with no breath, or a stretch of about eight seconds with no breath, is hard for a listener to follow. "
                                       "Give a figure that carries the point its breath, or split the sentence at the end of a thought; leave it if the speaker ran on the same way.")
    if result.get("comma_check"):
        slim["comma_check"] = result["comma_check"]
        slim["comma_check_note"] = ("Soft evidence, not a gate, measured on the source audio: a comma costs about 0.15 s of silence, and these units have commas beside a name of address "
                                    "that the speaker said in the same breath as the words round it (or far more commas than the speaker had pauses). Take the comma out where the speaker ran "
                                    "the words together; keep it where the phrasing of the segment shows a pause. pauses_the_speaker_held_that_the_dub_does_not is the other direction: the speaker "
                                    "held a pause of 0.8 s or more (phrasing |1.4s|) where your Zulu has only a comma or nothing, so your sentence runs ahead of the speaker and then waits; "
                                    "write ~pauseN (the total silence, in ms) after the word that ends the thought, or end the sentence there.")
    if result.get("voice_limit_words"):
        slim["voice_limit_words"] = result["voice_limit_words"]
        slim["voice_limit_words_note"] = ("Soft evidence, not a gate: a word the voice stutters on (a listener heard it). Swap it for the word named, and rephrase the clause around the new word; "
                                          "do not respell it.")
    if (result.get("meaning") or {}).get("loss"):
        slim["meaning_loss"] = result["meaning"]["loss"]
    if (result.get("meaning") or {}).get("claims"):
        slim["claims_to_check"] = result["meaning"]["claims"]
        slim["claims_note"] = CLAIMS_NOTE
    if result.get("warnings"):
        slim["warnings"] = result["warnings"]
    return slim


def build_dispatch(
    workbench: Workbench, *, clip_id: str, run_id: str, tts_budget: int, speed_policy: str = "natural",
    repeat: int = 1,
) -> dict[str, Callable[..., Any]]:
    """Tool name -> callable. The harness, not the model, fixes the run id and the measurement budget."""

    episode_clip = clip_id

    def open_clip(clip_id: str = episode_clip) -> Any:
        if clip_id != episode_clip:
            raise ValueError(f"this episode is for clip {episode_clip!r}")
        return workbench.open_clip(
            episode_clip, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy,
        )

    tools: dict[str, Callable[..., Any]] = {
        "open_clip": open_clip,
        "recall_lessons": workbench.recall_lessons,
        "measure_tts": workbench.measure_tts,
        "measure_tts_batch": workbench.measure_tts_batch,
        "check_text": workbench.check_text,
        "preview_turn_padding": workbench.preview_turn_padding,
        "submit_dub": lambda **kw: slim_submit(workbench.submit_dub(**kw)),
        "get_submission": workbench.get_submission,
    }
    for name in ("create_loanword", "use_loanword", "attested_zulu", "human_translations", "figure_style", "say_it"):       # vocabulary tools (the loan rules on the dictionary's sounds, a corpus lookup), where the workbench has them; the agent has no recogniser and no pronunciation tool
        if hasattr(workbench, name):
            tools[name] = getattr(workbench, name)
    return tools


def execute_tool(dispatch: dict[str, Callable[..., Any]], name: str, raw_arguments: str) -> tuple[str, bool]:
    """Run one tool call. Returns (observation text, ok). Failures become observations, not exceptions."""
    function = dispatch.get(name)
    if function is None:
        return json.dumps({"error": f"unknown tool {name!r}", "available": sorted(dispatch)}), False
    try:
        arguments = json.loads(raw_arguments or "{}")
        result = function(**arguments)
        return _clip(json.dumps(result, ensure_ascii=False)), True
    except BudgetExhausted as exc:
        return json.dumps({"error": "budget_exhausted", "detail": str(exc)}), False
    except Exception as exc:  # the model must be able to see and recover from any tool failure
        return json.dumps({"error": type(exc).__name__, "detail": str(exc)[:600]}), False


def unmet_targets(result: dict[str, Any] | None, *, max_dead_air_pct: float) -> list[str]:
    """What is still wrong with the last accepted submission, judged by the harness from its own score."""
    headline = (result or {}).get("headline")
    if not headline:
        return []
    unmet = []
    if headline.get("cross_speaker_collisions", 0) > 0:
        unmet.append(f"{headline['cross_speaker_collisions']} unit(s) run into ANOTHER speaker's speech")
    elif headline.get("collisions", 0) > 0:
        unmet.append(f"{headline['collisions']} unit(s) run into the next unit's speech")
    if headline.get("groups_rushed", 0) > 0:
        unmet.append(f"{headline['groups_rushed']} unit(s) overrun their window")
    if headline.get("gate_failures", 0) > 0:
        unmet.append(f"{headline['gate_failures']} hard-gate failure(s)")
    if headline.get("meaning_loss_units", 0) > 0:
        unmet.append(f"{headline['meaning_loss_units']} unit(s) lose material meaning (see meaning_loss)")
    residual = headline.get("residual_dead_air_pct")
    if residual is not None and residual > max_dead_air_pct:
        unmet.append(f"residual dead air is {residual}% (bar: {max_dead_air_pct}% or less)")
    return unmet


LISTENER_POINTS_SHOWN = 8              # the wrap-up names at most this many open listener points
_LISTENER_POINT_KEYS = ("pauses_the_speaker_held_that_the_dub_does_not", "commas_where_the_speaker_did_not_pause")


def open_listener_points(result: dict[str, Any] | None) -> list[str]:
    """The soft listener evidence of the last accepted submission that is still open (comma_check): pauses the source speaker held and the Zulu does not, commas the speaker
    did not take.  Not gates; the wrap-up shows them while the agent can still act, because at the final submit they arrived after the stop rule had already said 'finish' (prod-84, prod-85)."""
    points: list[str] = []
    for item in (result or {}).get("comma_check") or []:
        if not isinstance(item, dict):
            continue
        for key in _LISTENER_POINT_KEYS:
            for line in item.get(key) or []:
                points.append(f"{item.get('item_id')}: {line}")
    for item in (result or {}).get("voice_limit_words") or []:
        if isinstance(item, dict):
            points.extend(f"{item.get('item_id')}: {line}" for line in item.get("words") or [])
    return points[:LISTENER_POINTS_SHOWN]


def _listener_points_message(points: list[str]) -> str:
    return (", but the listener's view of the same submission has open points (soft evidence, not gates; the speaker's pauses are measured on the source audio):\n- "
            + "\n- ".join(points)
            + "\nYou have a few turns: for the points that are right, change only those units' text (write ~pauseN, the total silence in ms, after the word that ends the thought; "
              "or take out a comma where the speaker ran the words together), check them with ONE measure_tts_batch, send only those units with submit_dub partial=true, then reply "
              "with one short sentence and no tool call. Leave a point alone if acting on it would cost a name, a number or the meaning, or the unit has no room for the silence. "
              "Do not touch any other unit; this is a single round, not a loop.")


def pushback_message(result: dict[str, Any], unmet: list[str], *, tts_left: int) -> str:
    lines = ["Not done yet. Your last submission misses the bar: " + "; ".join(unmet) + "."]
    short = sorted((t for t in result.get("turn_padding", []) if t.get("residual_ms", 0) > 0),
                   key=lambda t: -t["residual_ms"])[:6]
    if short:
        lines.append("Turns still ending early after silence padding: " + "; ".join(
            f"{t['turn_id']} ({t['residual_ms'] / 1000:.1f} s of dead air, units {', '.join(t['units'])})" for t in short) + ".")
    flagged = result.get("flagged_units", [])[:10]
    if flagged:
        lines.append("Flagged units: " + "; ".join(
            f"{f['item_id']} [{', '.join(f['gates'] + f['flags'])}]" for f in flagged) + ".")
    lost = (result.get("meaning") or {}).get("loss", [])[:6]
    if lost:
        lines.append("Meaning lost (an independent reader of your Zulu disagrees with the English): " + "; ".join(
            f"{m['item_id']}: {'; '.join(m['issues']) or 'unclear'} (your Zulu reads back as: {m['reads_back_as']!r})"
            for m in lost) + ".")
    lock = result.get("timing_lock")
    if lock and lock.get("fitting_segments"):
        lines.append(
            f"Timing is locked for {lock['fitting_segments']} of {lock['total_segments']} segments: a submission that "
            "breaks one of them is rejected, so fix the failing units without touching the ones that fit."
        )
    lines.append(
        f"You still have {tts_left} measurements. Fix these units and resubmit: where speech is short, restore faithful "
        "content from the English you compressed or dropped (never filler); where it overruns, tighten. "
        "Overruns come first. If you truly cannot meet the bar, resubmit your best and say exactly which units and why."
    )
    return " ".join(lines)


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _source_correction_note(workbench: Any, seed_items: list[dict[str, Any]]) -> str:
    """In a repair run: what changed since the earlier submission (source corrections, glossary terms) and the units it touches."""
    note = ""
    corrections = getattr(workbench, "source_corrections", None) or []
    seed_units = getattr(workbench, "seed_unit_texts", None)
    current = {str(s["segment_id"]): str(s.get("restored_text") or s.get("source_text") or "") for s in getattr(workbench, "segments", None) or []}
    changed_units: dict[str, tuple[str, str]] = {}
    if seed_units and current:        # a unit whose English now reads differently from what the earlier submission was written against
        for item in seed_items:
            ids = [str(s) for s in item.get("source_segment_ids") or []]
            before, now = str(seed_units.get(str(item.get("item_id")), "")), " ".join(current.get(i, "") for i in ids)
            if before and _norm_words(before) != _norm_words(now):
                changed_units[str(item.get("item_id"))] = (before, now)
        # a correction is only news when it touches such a unit
        corrections = [c for c in corrections if any(str(c["segment_id"]) in [str(s) for s in item.get("source_segment_ids") or []]
                                                     and str(item.get("item_id")) in changed_units for item in seed_items)]
    if corrections:
        parts = []
        for correction in corrections:
            units = [str(item.get("item_id") or "?") for item in seed_items
                     if correction["segment_id"] in [str(s) for s in item.get("source_segment_ids") or []]]
            said = (f"the speaker said {correction['replacement']!r}" if correction.get("replacement")
                    else "the speaker broke that sentence off and started again, so it is not part of the speech: leave it out")
            parts.append(f"segment {correction['segment_id']}: the transcript had {correction['token']!r} but {said}"
                         + (f" (units: {', '.join(units)})" if units else ""))
        note += ("The English source was corrected after that submission, against the speaker's audio: " + "; ".join(parts) +
                 ". Re-translate the units that carry these segments so the Zulu says what the speaker meant, and treat them "
                 "as units that need change. ")
    if changed_units:
        shown = "; ".join(f"{unit}: the English now ends {now[-130:]!r} (it ended {before[-130:]!r})" for unit, (before, now) in changed_units.items())
        note += ("The English source of these units changed after that submission, so the Zulu may say something the speaker did not: "
                 + shown + ". Make each of them say only what the English now says, and treat them as units that need change. ")
    changes = getattr(workbench, "glossary_changes", None) or []
    if changes:
        parts = []
        for change in changes:
            old_forms = [f.strip() for f in change["was"].split("/") if f.strip()]
            units = [str(item.get("item_id") or "?") for item in seed_items
                     if any(form.casefold() in str(item.get("zulu_text") or "").casefold() for form in old_forms)]
            if units:
                parts.append(f"{change['english']!r}: now {change['now']!r} (was {change['was']!r}) in units {', '.join(units)}")
        if parts:
            note += ("The glossary was revised for simpler wording with the widest reach. These units use a term it replaced; "
                     "change that term in them and treat them as units that need change: " + "; ".join(parts) + ". ")
    return note


def loanwords_note(workbench: Any) -> str:
    """English common nouns the Zulu keeps although Zulu has an everyday word for them: the loanword comes first."""
    ask = getattr(workbench, "loanwords", None)
    try:
        found = ask() if callable(ask) else []
    except Exception:
        found = []
    if not found:
        return ""
    with_word = [item for item in found if item.get("zulu")]
    without = [item for item in found if not item.get("zulu")]
    note = ""
    if with_word:
        note += (" Loanwords first: your Zulu keeps these English words although Zulu has an everyday word for them: "
                 + "; ".join(f"{item['written']} -> {item['zulu']}" for item in with_word[:10])
                 + ". Write the Zulu word (with the prefix the sentence needs) unless it would change what the speaker said, then re-measure and resubmit the "
                   "units you changed.")
    if without:
        note += (" No everyday Zulu word is known for: " + ", ".join(item["written"] for item in without[:10])
                 + ". Look each up with attested_zulu: use an attested form if real text has one; if none is attested, code-switch (keep the English word as the "
                   "speaker says it; the harness sets how the voice says it). Create a word with create_loanword (and remember it with use_loanword) only "
                   "when real text itself adapts that word.")
    return note


def glued_prefix_note(workbench: Any) -> str:
    """English words the Zulu text carries with a noun-class prefix glued on although the speaker says them as they are: the text should carry the
    word as spoken (the prompt already says so), and a prefix makes it a different word from the one the speaker says."""
    ask = getattr(workbench, "glued_class_prefix_words", None)
    try:
        found = ask() if callable(ask) else []
    except Exception:
        found = []
    if not found:
        return ""
    return (" A Zulu noun prefix is glued straight onto an English word the speaker says as it is: "
            + "; ".join(f"{item['written']} (the speaker says {item['speaker_says']!r})" for item in found[:8])
            + ". Write the word as the speaker says it, without the prefix (a hyphenated particle such as ne- or ye- before it is ordinary Zulu), change it "
              "where it stands, then re-measure and resubmit the units you changed.")


def run_agent(client: ChatClient, workbench: Workbench, **kwargs: Any) -> dict[str, Any]:
    """Run the agent (see ``_run_agent``), then remove this run's scratch TTS audio. Records (submission, scorecard, trace) stay.

    Only this run's folder is touched: autopilot batches run several repeats at once, so nothing here may clean another run's."""
    from .housekeeping import clean_measure, keep_measure

    try:
        return _run_agent(client, workbench, **kwargs)
    finally:
        episode = getattr(workbench, "episode", None)
        if not keep_measure() and episode and episode.get("dir"):
            try:
                clean_measure(workbench.store.root, Path(episode["dir"]))
            except Exception:        # cleaning is housekeeping: it must never turn a finished run into a failure
                pass


def _run_agent(
    client: ChatClient, workbench: Workbench, *, clip_id: str, run_id: str, repeat: int = 1,
    max_turns: int = 40, max_total_tokens: int = 500_000, tts_budget: int = 300,
    system_prompt: str = SYSTEM_PROMPT, speed_policy: str = "natural",
    max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, max_pushbacks: int = DEFAULT_MAX_PUSHBACKS,
    task_message: str | None = None, trace_dir: Path | None = None,
    compact_after_tokens: int = COMPACT_AFTER_TOKENS, playbook_version: str | None = None,
    seed_items: list[dict[str, Any]] | None = None, wrapup_turns: int = WRAPUP_TURNS, dead_air_slack_pct: float = DEAD_AIR_SLACK_PCT,
) -> dict[str, Any]:
    """``wrapup_turns``: once a submission meets the bar (within ``dead_air_slack_pct`` on residual dead air) the agent is told to finish and has this many turns left;
    0 turns the rule off.  ``seed_items`` starts a REPAIR run: the units of an earlier accepted submission are re-scored under today's rules
    and the agent is told to change only what misses the bar, leaving every other unit exactly as it was."""
    directory = Path(trace_dir) if trace_dir else workbench.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "trace.jsonl")
    dispatch = build_dispatch(
        workbench, clip_id=clip_id, run_id=run_id, tts_budget=tts_budget, speed_policy=speed_policy,
        repeat=repeat,
    )
    if seed_items:
        workbench.open_clip(clip_id, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy)
        seeded = workbench.submit_dub(list(seed_items), approach_notes="seeded from an earlier accepted submission")
        trace.log("seed", accepted=seeded.get("accepted"), errors=seeded.get("errors"), headline=seeded.get("headline"))
        if seeded.get("accepted"):
            seed_unmet = unmet_targets(seeded, max_dead_air_pct=max_dead_air_pct)
            tts_now = int((workbench.episode or {}).get("tts_budget", tts_budget)) - int((workbench.episode or {}).get("tts_used", 0))
            task_message = task_message or (
                f"Repair, do not redo. Clip {clip_id!r} already has an accepted submission (get_submission shows its units). "
                f"Re-scored under the current timing rules it "
                + ("misses the bar. " + pushback_message(seeded, seed_unmet, tts_left=tts_now) if seed_unmet else "meets the bar. ")
                + _source_correction_note(workbench, seed_items)
                + "Change only the units that need it, send just those with submit_dub partial=true, leave every other unit "
                  "exactly as it is, and stop once the bar is met.")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_message or f"Dub clip {clip_id!r}. Start by opening it."},
    ]
    tokens = {"in": 0, "out": 0, "cached": 0}
    last_prompt_tokens = 0
    compactions = 0
    tool_counts: dict[str, int] = {}
    stop_reason = "max_turns"
    nudged = False
    loanwords_noted = False
    pushbacks = 0
    unmet: list[str] = []
    turn = 0
    good_enough_pct = max_dead_air_pct + dead_air_slack_pct
    seen_submit: Any = None
    wrapup_at: int | None = None

    def accepted() -> bool:
        return bool(workbench.episode and workbench.episode.get("submissions"))

    for turn in range(1, max_turns + 1):
        if compact_after_tokens and last_prompt_tokens >= compact_after_tokens:
            compacted, dropped_steps = compact_history(messages)
            if dropped_steps:
                trace.log("compact", turn=turn, dropped_steps=dropped_steps, messages_before=len(messages),
                          messages_after=len(compacted), prompt_tokens_before=last_prompt_tokens)
                messages[:] = compacted
                compactions += 1
        text_exists = True
        try:                                    # a repair run is accepted from its first turn, before its seeded text has been written out
            text_exists = (Path(workbench.episode["dir"]) / "submission.json").is_file()
        except Exception:
            pass
        if not loanwords_noted and accepted() and text_exists:
            # The Zulu draft exists now, so the English words it keeps can be named while the agent is still fitting units: at the end a swap means
            # fitting again, and prod-41 kept "provident fund", "pension fund" and "medical aid" for that reason.
            loanwords_noted = True
            early = loanwords_note(workbench) + glued_prefix_note(workbench)
            if early:
                early_text = "Your first accepted submission keeps English words that may have a Zulu word." + early
                messages.append({"role": "user", "content": early_text})
                trace.log("loanwords_early", turn=turn, text=early_text)
        response = client.complete(messages, tools=TOOL_SPECS)
        usage = response.get("usage") or {}
        last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        tokens["in"] += last_prompt_tokens
        tokens["out"] += int(usage.get("completion_tokens") or 0)
        tokens["cached"] += int(usage.get("cached_tokens") or 0)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        trace.log("model", turn=turn, finish=choice.get("finish_reason"), text=message.get("content"),
                  tool_calls=[{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
                  usage=usage)
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "_raw") and v is not None})

        if not calls:
            if not accepted() and not nudged:
                nudged = True
                nudge = "You have not submitted a dub yet. Call submit_dub with your best units now."
                messages.append({"role": "user", "content": nudge})
                trace.log("nudge", turn=turn, text=nudge)
                continue
            if not accepted():
                stop_reason = "finished_without_submission"
                break
            # The agent says it is done. The harness checks that against its own score instead of
            # taking its word for it -- bounded, and only while budget remains to act on it.
            unmet = unmet_targets(getattr(workbench, "last_submit", None), max_dead_air_pct=good_enough_pct)
            tts_left = int((workbench.episode or {}).get("tts_budget", tts_budget)) - int((workbench.episode or {}).get("tts_used", 0))
            if (unmet and pushbacks < max_pushbacks and tts_left > 0
                    and tokens["in"] + tokens["out"] < 0.8 * max_total_tokens):
                pushbacks += 1
                text = pushback_message(workbench.last_submit, unmet, tts_left=tts_left)
                messages.append({"role": "user", "content": text})
                trace.log("pushback", turn=turn, number=pushbacks, unmet=unmet, text=text)
                continue
            stop_reason = "finished_below_target" if unmet else "agent_finished"
            break

        for call in calls:
            name = call["function"]["name"]
            started = time.monotonic()
            observation, ok = execute_tool(dispatch, name, call["function"]["arguments"])
            tool_counts[name] = tool_counts.get(name, 0) + 1
            trace.log("tool", turn=turn, name=name, ok=ok, seconds=round(time.monotonic() - started, 2),
                      result=observation[:3000])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})

        latest = getattr(workbench, "last_submit", None)
        if wrapup_turns and latest is not None and latest is not seen_submit:       # a new accepted submission: has it met the bar?
            seen_submit = latest
            if wrapup_at is None and not unmet_targets(latest, max_dead_air_pct=good_enough_pct):
                wrapup_at = turn + wrapup_turns
                residual = (latest.get("headline") or {}).get("residual_dead_air_pct")
                points = open_listener_points(latest)
                text = ("Your last submission meets the bar by the harness's own score: no hard-gate failures, no collisions, nothing overruns, no meaning lost"
                        + (f", residual dead air {residual}%" if residual is not None else "")
                        + (_listener_points_message(points) if points else
                           ". Finish now: reply with one short sentence and no tool call. Only if one unit is plainly wrong (a name, a number, a date or its meaning) fix that "
                           "one thing first; timing is locked for the units that fit, and every further turn costs tokens."))
                if points:
                    trace.log("listener_points", turn=turn, points=points)
                messages.append({"role": "user", "content": text})
                trace.log("wrapup", turn=turn, turns_left=wrapup_turns, residual=residual, text=text)
        if wrapup_at is not None and turn >= wrapup_at:
            stop_reason = "bar_met"
            break
        if tokens["in"] + tokens["out"] > max_total_tokens:
            stop_reason = "token_budget"
            break

    summary = {
        "mode": "agent", "clip_id": clip_id, "run_id": run_id, "stop_reason": stop_reason, "turns": turn,
        "tokens": tokens, "tool_calls": tool_counts, "speed_policy": speed_policy,
        "pushbacks": pushbacks, "unmet_at_end": unmet, "compactions": compactions,
        "playbook": playbook_version,
        "grammar": _grammar_version_in(system_prompt), "system_prompt_chars": len(system_prompt),
        **({"judge_tokens": dict(workbench.judge.usage)} if getattr(workbench, "judge", None) else {}),
        "tts_used": (workbench.episode or {}).get("tts_used"), "submissions": (workbench.episode or {}).get("submissions"),
        "model": client.config.deployment,
    }
    trace.log("end", **summary)
    (directory / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


ONE_SHOT_PROMPT = """\
Dub this complete English news clip into isiZulu. Reply with JSON only:
{"items":[{"item_id":"u1","start_ms":0,"end_ms":0,"speaker_id":"","zulu_text":"","source_segment_ids":["..."]}]}
Units must be ordered, non-overlapping, inside the clip, and cover every English segment.
Retell naturally and informally (not word for word), keep every name, number and date, invent no names,
and aim for speech that fills each unit's time window at natural speaking speed (audio is never sped up).

CLIP:
"""


def run_one_shot(
    client: ChatClient, workbench: Workbench, *, clip_id: str, run_id: str, repeat: int = 1, tts_budget: int = 300,
    speed_policy: str = "natural",
) -> dict[str, Any]:
    """Baseline: no tools, no loop. One model call, then the same official scoring."""
    directory = workbench.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "trace.jsonl")
    clip = workbench.open_clip(clip_id, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy)
    response = client.complete(
        [{"role": "system", "content": "You are an expert English-to-isiZulu news dubbing translator."},
         {"role": "user", "content": ONE_SHOT_PROMPT + json.dumps(clip, ensure_ascii=False)}],
        json_mode=True,
    )
    usage = response.get("usage") or {}
    text = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    trace.log("model", turn=1, text=text[:6000], usage=usage)
    items = json.loads(text).get("items", [])
    result = workbench.submit_dub(items, approach_notes="one-shot baseline: single call, no tools")
    trace.log("tool", turn=1, name="submit_dub", ok=bool(result.get("accepted")), result=json.dumps(result, ensure_ascii=False)[:3000])
    summary = {
        "mode": "oneshot", "clip_id": clip_id, "run_id": run_id, "speed_policy": speed_policy,
        "stop_reason": "agent_finished" if result.get("accepted") else "submission_rejected", "turns": 1,
        "tokens": {"in": int(usage.get("prompt_tokens") or 0), "out": int(usage.get("completion_tokens") or 0)},
        "tool_calls": {"submit_dub": 1}, "tts_used": (workbench.episode or {}).get("tts_used"),
        "submissions": (workbench.episode or {}).get("submissions"), "model": client.config.deployment,
        "errors": result.get("errors"),
    }
    trace.log("end", **summary)
    (directory / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_agent")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("agent", "oneshot"), default="agent")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses",
                        help="model API for agent mode (reasoning models need 'responses' to use tools)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--speed-policy", choices=("natural", "pipeline"), default="natural",
                        help="natural: audio is never sped up, any overrun is a failure; "
                             "pipeline: tolerate the renderer's bounded speed-up")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--tts-budget", type=int, default=300)
    parser.add_argument("--max-pushbacks", type=int, default=DEFAULT_MAX_PUSHBACKS,
                        help="times the harness sends the agent back when its last score misses the bar")
    parser.add_argument("--max-dead-air", type=float, default=DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT)
    parser.add_argument("--playbook", default=None,
                        help="playbook version to load into the system prompt ('current' = the adopted one, "
                             "'none' = rules only; default: current)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the independent meaning check (it is part of the bar by default)")
    parser.add_argument("--compact-after", type=int, default=COMPACT_AFTER_TOKENS,
                        help="prompt-token size above which superseded working steps are dropped (0 = never)")
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    config = native_gpt_config_from_environment(dict(os.environ))
    judge = None
    if not args.no_judge and args.mode == "agent":
        from .dub_judge import UnitJudge

        judge = UnitJudge(ChatClient(config))
    workbench = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings), judge=judge)
    if args.mode == "oneshot":
        client: Any = ChatClient(config)
        summary = run_one_shot(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                               tts_budget=args.tts_budget, speed_policy=args.speed_policy)
    else:
        client = ResponsesClient(config) if args.api == "responses" else ChatClient(config)
        from .playbook import current_playbook, load_playbook

        choice = args.playbook or "current"
        playbook = None if choice == "none" else (
            current_playbook(workbench.store) if choice == "current" else load_playbook(workbench.store, choice))
        system_prompt = build_system_prompt(playbook, max_dead_air_pct=args.max_dead_air)
        print(f"playbook: {playbook['version'] if playbook else 'none'}")
        summary = run_agent(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                            max_turns=args.max_turns, max_total_tokens=args.max_tokens, tts_budget=args.tts_budget,
                            speed_policy=args.speed_policy, max_dead_air_pct=args.max_dead_air,
                            max_pushbacks=args.max_pushbacks, compact_after_tokens=args.compact_after,
                            system_prompt=system_prompt, playbook_version=playbook["version"] if playbook else None)
    print(json.dumps(summary, indent=2))
    scorecard = workbench.store.repeat_dir(args.run_id, args.clip, args.repeat) / "scorecard.json"
    if scorecard.is_file():
        card = json.loads(scorecard.read_text(encoding="utf-8"))
        print(json.dumps({"headline": card["headline"], "gates": {k: v for k, v in card["gates"].items() if v}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
