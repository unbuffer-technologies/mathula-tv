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
from .budget_governor import DRAFT_FRACTION, DRAFT_GRACE_TURNS, FINAL_TOOLS
from .cache_watch import CacheWatch, request_fingerprint
from .warm_start import current_voice_provider

DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT = 2.0
DEFAULT_MAX_PUSHBACKS = 2
DEAD_AIR_SLACK_PCT = 0.5          # the run counts as having met the bar this far above the stated residual dead air: the last half point is not audible and costs dozens of turns
WRAPUP_TURNS = 6                  # once the bar is met the agent may use this many more turns for a plainly wrong unit, then the run ends (prod-69: bar met at turn 21, run ended at turn 80)
MET_STOP_REASONS = frozenset({"agent_finished", "bar_met"})
MAX_TOOL_RESULT_CHARS = 150000  # a runaway ceiling, not a budget. The opened clip of a 15-minute, 21-unit clip is about 40-45k characters and grows with the clip (its segments, delivery sheet and
                                # dialogue); at 60k a clip of 25 minutes or more lost its tail (the delivery sheet, then the dialogue) without the agent being told why. 150k holds about an hour.

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert English-to-isiZulu news dubbing agent working for Mathula TV.

Goal: dub ONE complete news clip. Produce isiZulu speech that a native speaker would find natural, faithful to the English, and that fits the time the English speaker occupies on screen.
You work through tools and decide your own method. You are scored on three things.

1. Hard gates (a failure is a defect): every name, number and date survives; no invented names; the same entity is not stated twice for one English mention; nothing left untranslated.

2. Timing. Each unit's Zulu is played at its natural speed -- it is NEVER sped up or slowed down -- so the words you choose are the only thing that controls timing. The speech should fill its
   source window: too short is dead air while the speaker is still on screen, too long is an overrun into the next speaker.
   FIT FIRST. Fitting every unit into its time frame is the first job: it keeps speakers from talking over each other. The harness locks timing as you go: once a segment has fitted in an accepted
   submission, a later submission that breaks it is rejected unless it is clearly better overall. An overrun is WORSE than dead air (it collides with the next speaker, while a little silence is only
   a pause): first make sure no unit overruns (overrun_pct must be 0), then shorten the dead air that is left.
   Zulu takes about twice the syllables of English for the same words and the voice says them at its own pace, so a dub that fills every millisecond of a slow, deliberate speaker's window is a dub that
   rushes: say less before you pack more. What must hold is that your Zulu starts with the speaker and stops with the speaker; a few seconds of rest inside the window is fine
   (speaker_pauses_pct is the part of a shortfall that is silence the speaker took too, and it is not counted as dead air).
   Never add a word, a phrase or an English-derived piece of Zulu just to fill a window: a short line that leaves a fraction of a second is correct and the silence is harmless. A filler a Zulu
   speaker would not say is a worse defect than a short unit.
   Length guide. open_clip gives each segment fits_about_chars, the Zulu length that fills its window at that speaker's pace. Write your first draft to that length, with what the speaker said (a
   compact draft has landed 30% short and cost most of the run to close). Every measurement gives the unit's chars and, when it is more than 6% off its window, add_about_chars or cut_about_chars,
   the size of the edit at the rate that very text measured: edit by that amount (fuller phrasing of what the speaker said, never a restatement), units furthest off first. When the measurement says
   leave_short the gap is under half a second: add nothing. The measurement is repeatable, so a text that fits measured once fits in the render.
   Where the time goes. A measurement carries sentence_timing: each sentence's characters, the seconds the voice took, where it starts and the source speaker's matching sentence. A unit can fit its
   window and still run behind the speaker in the middle: the renderer moves only silence between sentences, never speech, so a sentence longer than the English one pushes every later one late.
   submit_dub lists those (behind_the_mouth, with the lag and how many characters to take out); fix them in the words of the sentences before the late one, never in the money or the facts. timeline_fit
   (free) shows each sentence's estimated start against the speaker's and where lateness BEGINS: when a sentence starts late, go back to the sentence that lost the time and shorten THAT one, then
   look again. estimate_fit (free) is a rough screen of many candidates (nine in ten fall in its band); it must never be the reason you pick a less natural wording or leave a natural one unmeasured.

3. Naturalness. Retell the way a person would say it informally; this is not a word-for-word transfer. Use the whole clip's context. Comprehension and fluidity are what the listener gets, so they are
   part of fitting the time, not a casualty of it: a unit that sounds stiff, glued or word-for-word is a defect even when it fits to the millisecond. Where English has a set phrase, say what a Zulu
   speaker says in that moment. Fit by choosing among natural ways to say what the speaker said, never by padding or by cutting a fact.
   When Zulu cannot carry everything at a natural pace in the window, do not pack it: take the time from the least important content first (a line the speaker repeats or restates, a filler, an
   emphasis, a connective, a hedge that qualifies no claim), then from wording that is shorter in syllables. Never a name, a number, a claim, a hedge that qualifies a claim, or who did what to whom.
   Say in approach_notes what you left out and why.

   Write for the ear. This is a SOCIAL MEDIA product: simplicity and maximum reach are the point, and a viewer gives a clip a second or two. The reader is a TTS model and the audience is general
   listeners, not expert Zulu speakers: plain, short, everyday words; one idea to a clause; the verb early; the sentence ending on its stress; a verb where you would otherwise stack nouns; no dense
   chains the voice must improvise (a nominalisation with an adverb and an adjective stacked on it is better as two plain phrases, even at a few more words). A phrase an ordinary speaker would have to
   slow down for is a phrase the voice will rush. This holds for every sentence you write, not only repairs.
   Widest understanding beats formality. Choose the word the most everyday isiZulu listeners understand the moment they hear it (urban and rural, older and younger), even if it is a loanword or informal
   and a more formal word exists ("igranti", not "isibonelelo"; "i-VAT"; "imeya"). The glossary fixes names and terms for consistency; where one of its terms is stiffer than a common loanword, use the
   familiar word.
   Listening load. The viewer hears each line once and cannot re-read it. Give them one new fact at a time, in the order a speaker would say them, connected the way speech connects them. Put a figure that carries
   the point at the end of its phrase and leave it room after it; never run two figures, or a figure and a name, into each other.
   check_text and every submit_dub result measure this (figures_run_together, long_run_without_breath, listening_load); the flags are evidence, you decide. A clause too dense to hear is split or
   reordered with the facts unchanged, not cut.
   Natural speech. The voice already carries the speaker's own cadence, pauses and breathing, so short sentences are not needed to imitate them (that was only a way to copy pauses with the old Azure voice).
   Write flowing speech: thoughts that follow each other are connected (a comma, futhi, kanti, bese, a relative clause, a participle) and not stacked as separate short sentences. A run of three- and
   four-word sentences sounds like a list being read aloud, and listeners hear it at once. End a sentence where the thought really turns. Every submit also shows naturalness evidence measured against real
   isiZulu: word pairs that never meet in the corpora although two common words would (usually a calque from the English: say it the way a speaker would), and units chopped into very short sentences.
   Sentences and commas. What matters most is that your Zulu starts and stops with the English speaker: the unit's window; inside it you are free to write the most fluent, best understood Zulu. Inside it, where the speaker's sentences end is a guide and a best effort, never
   a requirement: follow it when the Zulu runs naturally that way, and put a boundary somewhere else, or merge or split, whenever following it would break the flow of the Zulu. Fluent Zulu comes first, then
   the speaker's boundaries. A comma marks something in the speech: a clause boundary Zulu writers mark (one clause ends and another with its own verb
   begins, often before a connecting word such as kodwa, futhi, kanti), a name of address, an item in a list, an aside. It is never there to give the voice a breath: the voice takes its breaths from
   the structure of the sentence. When you are unsure whether a clause boundary takes a comma, ask punctuation_examples(word): real news
   sentences in which the word follows a comma and others in which it does not, so you see what real writers do. They are examples to read, not a rule and not a count.
   Length by grammar. You know isiZulu's grammar and registers more widely than any one speaker does, so make that your length control: the language says one meaning at many lengths. {grammar_pointer}
   To shorten without losing a fact: let a verb extension carry what a phrase would (-el- "for/to/at", -is- "make do", -w- "by", -an- "each other"), leave out an emphatic pronoun the subject concord already carries unless
   the speaker stresses it, use the conjunctive verb form where a complement follows, turn a relative clause into a participial or an infinitive, take the shorter of two equally common words. To lengthen
   with content the English does carry: the tense and aspect forms its "still", "already", "just", "used to" imply, an emphatic pronoun where the speaker leans on it, a full relative clause or demonstrative
   where it makes the sentence plainer. When a unit is off, draft two or three renderings that differ in construction, measure them together in one measure_tts_batch call, and keep the clearest and most
   natural of those that fit. Calls that do not depend on each other go in the SAME turn (check_text, say_it, human_translations, attested_zulu, figure_style, estimate_fit, a measurement of the same text, or several of one kind): a turn is a round trip of several seconds and these lookups cost nothing, so ask for everything you will need for a wording together. Measure freely: the voice's own audio is the only judge of how long a wording takes and how natural it sounds. Never lengthen with an intensifier, a repeated word or a
   word the English lacks, and never shorten by dropping a fact, a qualifier or a negation.

   Loanwords first. When Zulu has an everyday word for an English word, a native word or an established loanword (isayensi for science, ubuchwepheshe for technology, ibhiliyoni for billion), write it,
   with the prefix the sentence needs, instead of keeping the English. Keep English only for a name, a title, an acronym, or a word with no everyday Zulu word. A kind of thing is not a name: "medical
   aid" and "pension fund" are common nouns, so each noun takes its everyday Zulu word (or one you create). When unsure, ask what real Zulu does: attested_zulu(word, zulu_forms) shows the Zulu forms
   translated text pairs with the word, whether it stays English there, how news sites write it, and how often a form you have in mind has been written. The ladder: first a borrowed form that real text
   writes (igranti, or the English word with a hyphenated particle, i-microphone); then an attested native word; otherwise code-switch, which means keeping the English word as the speaker says it (same
   number, plural -s included), with a hyphenated particle where the grammar needs one (ne-, ye-). A formal native coinage never outranks a borrowed form that translators, news and speakers write. Never
   invent a form real text does not write; only when real text itself adapts such a word, create_loanword builds Zulu forms of it by the loan rules from the dictionary's sounds (nothing is
   listened to or spoken), and use_loanword remembers the one you choose.
   English words you keep (names, acronyms, titles) are spoken from a spelling the harness writes for them by rules, from a dictionary's English sounds. You do not choose or check how they sound: keep the
   word itself in the text, as the speaker says it. Never glue a Zulu prefix onto an English word the speaker says as it is (a hyphenated particle such as ne- or ye- before it is ordinary Zulu).
   check_text and submit_dub name any word the voice is known to stumble on, with the form to write instead: use it.

   Figures. Write numbers and amounts as digits with the isiZulu prefix the sentence needs ("kuka-31%", "abantu abangu-11", "eminyakeni engu-20", "ngo-2006"). A COUNT of 1 to 9 (people, things, days, years of age) is the exception: write it as the isiZulu counting word that agrees with the counted
   noun ("abahlanu", "iminyaka eyisithupha", "izinsuku ezintathu"; the numerals table of the grammar reference gives every form), as the newspaper Isolezwe writes it (two: amabili, ezimbili, ababili thousands of times, against a few
   digits) and as people say it aloud (in 478 hours of transcribed speech 2, 3, 6, 7, 8 and 9 are said in isiZulu far more often than in English). A figure of 10 or more (with its decimals and %), a year, a clock time and a day of the month
   ("ngo-9", "ngo-2006") stay digits and are spoken as code-switched English after its prefix ("angu-thethi waan"; a year is "nineteen ninety-four"): ask figure_style("31 percent") (or "20 years", "in 2006") for how
   translators wrote that kind of figure. Dates and ordinals are spoken as code-switched English too. Larger amounts ("u-4,500") are spoken in native isiZulu, which takes longer; when a unit does not
   fit, you may shorten one by writing ~en straight after it ("u-4,500~en"): it is then spoken in English (the ~en is never spoken, the digits stay for the number checks). Where a unit allows it,
   native is the default; use ~en only where the time requires it, and say so in a choice point. Viewers weigh the money most, above all in politics and corruption stories, so an amount is the last thing to give
   up: first take the time from words the viewer will not miss (a connective, a title, a repeated modifier, an idiom you can say shorter, a loanword that can stay English).
   Dates. Write a day of a month the way the newspaper Isolezwe does: the MONTH FIRST with the isiZulu preposition glued on, then the day as plain digits ("Kusukela ngoNovemba 4, ...", "kuze kube uDisemba
   1", "ngoJanuwari 5, 2027"), with the everyday borrowed month names (Novemba, Mashi, Disemba); never the English "of", the day-first order or an ordinal suffix. A time of day takes a colon, never a point:
   ngo-9:30, not ngo-9.30.

   Pacing and punctuation. Punctuation tells the voice how the sentence is built. A comma marks a clause boundary, a name of address, a list item or an aside, and the voice takes a short breath there; a
   full stop ends a thought, the voice falls, pauses and starts the next sentence fresh; an ellipsis ("...") is ignored. So punctuate by the structure of what you say (see Sentences and commas): nothing
   inside a phrase that belongs together, a comma at a clause boundary, a full stop only where the thought turns. A full stop in the middle of a thought makes the line stilted. A hyphen joins the pieces
   of one word and spacing inside a number or a name adds no silence, so do not split figures and names with commas; but after each percentage or amount followed by more words, a comma sets it off from
   what follows.

   A conversation, not a text. Most clips are people talking to each other, each turn answering the one before and often handing over by name. open_clip says so (conversation). When a turn says the name
   of the other person in the conversation (`says_the_name_of`), the speaker is addressing that colleague: a name of address, not a second person in the story. Write it as Zulu writers write a name of
   address, and answer the person before you the way a colleague answers a colleague.
   A clean dialogue for people who never heard the original. The viewers hear only your Zulu and cannot tell what was broken in the English (a false start, a fragment, a word said twice by accident, a name
   said wrongly and corrected, a stretch the recogniser garbled), so a stumble repeated in the Zulu just sounds like a broken dub: write what the speaker meant as a clear, well-formed turn. The limits:
   the facts stay exactly (claims and hedges such as "allegedly", numbers, names, who did what to whom), nothing is added the speaker did not mean, and where a repair would mean guessing a fact keep only
   what the evidence shares and say what you needed in missing_evidence. Say what you cleaned in a choice point (evidence: "cleaned", with what the speaker actually said). A repeat for weight (a deliberate
   "very, very") is delivery and stays unless the window cannot hold it at a natural pace; a repeat that is a stumble goes. People misspeak, above all in informal talk ("legal census" for "legal consensus"): when the sentence and the setting show the word
   the speaker meant, dub that word and note it in a choice point.
   Language switches. A speaker may move between English, isiZulu and sometimes Sepedi, Sesotho or another language. Where the English recogniser wrote nonsense for such a stretch, the transcript has been
   restored from YouTube's own captions, so a segment's English may hold their own isiZulu, Sepedi or Sesotho words. Read it as a bilingual listener would: keep their isiZulu as isiZulu, say other
   languages' meaning in plain isiZulu, and do not skip the stretch (a lost 'unga-' turns "do not confuse" into "confuse").
   When the English of a segment looks odd, ambiguous or wrong (a double negative, a name, a number, a sentence that does not make sense), call source_readings before you decide what it meant: it shows what
   the recogniser heard, how sure it was word by word, what YouTube's caption says for the same moment, and where the two differ. It cannot play the audio; two readings that agree are strong evidence.

Tools give you real measurements. Voice timing is not predictable from word counts, and with no speed control you must get the wording right. The scorer re-measures your final text with the same function, so
what you measure is what you are graded on. Measurements are budgeted.

Learn from the listener. recall_lessons also returns ear_lessons: what the person who listens to these dubs has heard wrong before, by item class (a title read raw, digits read in a Zulu accent, a name run
together, a figure with no breath), with what fixed it, and the spellings they chose over the rules'. Read them before your first draft and write so they do not recur.

See what the voice gets. What you write is not what the voice reads: a dictionary of rules turns your digits, titles, abbreviations and English names into the spoken form. Before you submit, call say_it on any
unit that holds a figure, a year, a title, an abbreviation or an English name, and read the spoken form the way a listener would hear it. 'still_as_written' lists what no rule converted: write that another
way (spell the title out, put the figure after its prefix) or say in approach_notes what you could not fix. It only shows; the spelling of a word is still decided by the rules and by the listener's ear.

Decisions so far. Your earlier drafts, measurements and checks are cleared from the conversation as you work (only the latest submit and its result stay). When you settle how a term or phrase that recurs in
this clip is said in Zulu, pass it in submit_dub decisions as {english, zulu, note}, at most twelve a call. Every submit result then returns decisions_so_far (each with the units that still hold it),
settled_units (no open flag: leave them alone unless a point names them) and speaker_rates_chars_per_s (the measured speech rate, to reckon a length edit at). Use them to keep one word for one idea across
the clip; a repair run starts with the earlier run's decisions.

Every tool call re-sends your whole history, so each extra turn costs more than the last. After you
submit, your earlier working steps (drafts, measurements, checks) are trimmed from your history: the clip,
your latest submit_dub call and its result are always kept.

submit_dub returns the current Zulu text of the units it flags (and of the claims_to_check), so read it there; call get_submission only for units whose text you do not have. It takes the full set of
units, or, with partial=true, only the units you changed: each replaces the
current unit with the same item_id (ids in remove are deleted) and the rest carry over. Sending only what
changed is much cheaper than rewriting the whole clip. get_submission shows your current units (all, or just the ones you ask for), so you never have to rewrite from memory.

The bar for "done" (checked by the harness against your last submit_dub result, not by your say-so): no unit overruns or runs into another unit's speech (collisions), no hard-gate failures, no unit
whose meaning an independent reader finds materially changed from the English (submit_dub reports these as meaning_loss, with what your Zulu reads back as), and residual_dead_air_pct at or below
{max_dead_air}. submit_dub also lists claims_to_check: units where the same reader found something your Zulu says that the English does not. That is not a bar and the reader is sometimes wrong: look at the
English of each, take out what it does not say (a wider word than the speaker used, an intensifier, a figure or a reason of your own), and leave it alone when the English does say it. Window length is never a
reason to add a claim: a shorter faithful unit and a pause are better than a longer unit that says more than the speaker did. If you stop while a submission is still below that bar and you have budget left,
you will be sent back to the specific units to fix.

When you are satisfied, call submit_dub (it returns your score and which units are flagged; you may revise and resubmit). Then reply with a short plain-text summary of your approach and stop.
"""

# Said only when the grammar reference follows: with it switched off the sentence would point at nothing (the first grammar-off runs of the comparison had that flaw).
GRAMMAR_POINTER = "The grammar reference below (six modules, from real news and a checked corpus) is the rule system to apply. "

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT}%").replace("{grammar_pointer}", "")


def _grammar_version_in(system_prompt: str) -> str | None:
    """Which grammar reference the agent was given (None when the prompt carries none): written into agent_summary.json so a result can be tied to the text it saw."""
    from . import zulu_grammar

    return zulu_grammar.VERSION if zulu_grammar.HEADER[:40] in system_prompt else None


STATE_NAME = "agent_state.json"
PAUSE_NAME = "PAUSE"
STATE_SCHEMA = "agent-state-v1"


def state_path(directory: Path) -> Path:
    return Path(directory) / STATE_NAME


def request_pause(directory: Path) -> Path:
    """Ask a running agent to stop at the end of its current turn: it saves its checkpoint and returns ``paused``.  ``dub_production --pause`` does this for a run id."""
    path = Path(directory) / PAUSE_NAME
    path.write_text("pause requested\n", encoding="utf-8")
    return path


def save_state(directory: Path, state: dict[str, Any]) -> None:
    """The agent's whole position, atomically: the conversation (with the model's own reasoning items, which are replayed verbatim), the counters and flags, and the workbench's in-memory
    results (its last accepted units, its last submit result, the segments that fitted).  Written at the start of every turn, so a stop or a crash loses at most the call in flight."""
    from .atomic_io import atomic_write_json

    atomic_write_json(state_path(directory), {"schema_version": STATE_SCHEMA, **state})


def load_state(directory: Path) -> dict[str, Any] | None:
    path = state_path(directory)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and state.get("schema_version") == STATE_SCHEMA and isinstance(state.get("messages"), list) else None


def workbench_snapshot(workbench: Any) -> dict[str, Any]:
    return {"last_items": getattr(workbench, "last_items", None), "last_submit": getattr(workbench, "last_submit", None),
            "locked_segments": sorted(getattr(workbench, "locked_segments", None) or [])}


def restore_workbench(workbench: Any, snapshot: Mapping[str, Any]) -> None:
    """Put a checkpoint's results back into a freshly opened workbench (open_clip has already reloaded the episode counters, the decisions and the voices from disk)."""
    if snapshot.get("last_items") is not None:
        workbench.last_items = [dict(i) for i in snapshot["last_items"]]
    if snapshot.get("last_submit") is not None:
        workbench.last_submit = snapshot["last_submit"]
    if hasattr(workbench, "locked_segments"):
        workbench.locked_segments = set(snapshot.get("locked_segments") or [])


BUDGET_POLICIES = ("free", "lean")


def agent_budget(value: str | None = None) -> str:
    """``free`` (the default for now: quality first, no pressure to economise: generous limits, the bar is a minimum and the agent decides when it is done, more of its own history kept) or
    ``lean`` (token-lean: tight limits, the bar ends the run).  From the argument, else MATHULA_AGENT_BUDGET.  The owner chose free on 2026-11-13: spend now, so the agent builds the
    experience (decisions, lessons, tools) that lets it spend less later."""
    policy = str(value or os.environ.get("MATHULA_AGENT_BUDGET") or "free").strip().lower()
    return policy if policy in BUDGET_POLICIES else "free"


BUDGET_LIMITS: dict[str, dict[str, Any]] = {
    "free": {"max_turns": 200, "max_total_tokens": 8_000_000, "tts_budget": 3000, "compact_after_tokens": 60_000, "dead_air_slack_pct": 0.0, "focus_growth_tokens": 10_000},
    "lean": {"max_turns": 80, "max_total_tokens": 2_000_000, "tts_budget": 600, "compact_after_tokens": 15_000, "dead_air_slack_pct": DEAD_AIR_SLACK_PCT, "focus_growth_tokens": 0}  # lean: the old COMPACT_AFTER_TOKENS,
}

_COST_RE_MEASURE = "Measurements are budgeted."
_COST_PARA = """Every tool call re-sends your whole history, so each extra turn costs more than the last. After you
submit, your earlier working steps (drafts, measurements, checks) are trimmed from your history: the clip,
your latest submit_dub call and its result are always kept."""
_COST_PARTIAL = "Sending only what changed is much cheaper than rewriting the whole clip."
FREE_PROMPT = """
QUALITY FIRST. There is no token, turn or measurement pressure on this run: the people running it want the highest quality you can reach and are prepared to spend for it, so that what you
learn here makes later runs better. Take as many turns, measurements, checks and rewrites as quality needs. The bar below is the minimum, not the finish line: when a submission meets it, keep
going while any unit could sound more like a person talking, say the speaker's meaning more exactly, or sit better in its window; finish when you judge that no unit would be better, not when
the bar is first met. Re-read your own text as a listener would, compare alternatives, and use every tool and every piece of evidence the harness shows you. If you ever need information that
you were not given, say so in approach_notes.
"""


def free_prompt(base: str) -> str:
    """The system prompt with the cost framing taken out and the quality-first paragraph in (the lean prompt is unchanged).  Matches across the prompt's own line breaks."""
    base = base.replace(_COST_RE_MEASURE, "Measure as much as you need.", 1)
    base = base.replace(_COST_PARA, "After you submit, your earlier working steps (drafts, measurements, checks) are trimmed from your history to keep it short: the clip, your latest submit_dub call "
                                    "and its result (with decisions_so_far, settled_units, units_overview) are always kept.", 1)
    base = re.sub(r"Sending only what\s+changed is much cheaper than rewriting the whole clip\.", "Sending only what changed keeps every other unit exactly as it is.", base, count=1)
    base = re.sub(r"If you stop while a submission is still below that bar and you have budget left,\s+you will be sent back", "If you stop while a submission is still below that bar, you will be sent back", base, count=1)
    return base + FREE_PROMPT


def free_bar_message(residual: Any, points: list[str], *, first: bool) -> str:
    """Quality-first runs are not stopped when the bar is met: the agent is told it is met, shown the open points, and left to decide when it is done."""
    head = ("Your last submission meets the bar by the harness's own score: no hard-gate failures, no collisions, nothing overruns, no meaning lost"
            + (f", residual dead air {residual}%" if residual is not None else "") + ". " if first else "New open points on your latest submission. ")
    body = (("The listener's view of the same submission has open points (soft evidence, not gates):\n- " + "\n- ".join(points) + "\nAct on the ones that are right; leave one alone if acting on it would cost a name, "
             "a number or the meaning. ") if points else "")
    tail = ("This is the end of the work unless something is plainly wrong (a name, a number, a date, the meaning). If an open point above is right, act on ALL the right ones in ONE round: change every unit "
            "together and send one submit_dub with partial=true. Then reply with one short sentence and no tool call; further passes over soft points are not wanted.")
    return head + body + tail


FOCUS_NOTE = ("THE SCOPE OF THIS RUN IS THESE UNITS ONLY: {units}. The listener asked for them to be re-fitted: the voice speaks each unit at its own pace (the length estimates you get from "
              "estimate_fit and check_texts already include it) and these units do not fill their windows the way their text was written for. The target for each is its window to within about half a second, "
              "never over: a unit that stops early leaves a silence the listener hears, however small the total dead air is. Keep the meaning, the names and every number. Do NOT change any other unit, "
              "even if it is flagged: leave it exactly as it is. ")


def focus_budget_note(workbench: Any, seed_items: list[dict[str, Any]], focus_units: list[str]) -> str:
    """What each unit in scope needs, in letters, and the speaker's own padding words from the English: the work the agent would otherwise do in turns (estimate, reword, estimate again) done once, free.
    Nothing here picks the agent's words; a failure to build it leaves the task message as it was."""
    try:
        from .speaker_fillers import budget_note, budget_rows, fillers_by_speaker

        rows = budget_rows(seed_items, focus_units, lambda text, speaker, start, end: workbench._text_ms(text, speaker, start, end)[0])
        from .attestation import resources_dir
        from .config import load_settings
        from .speaker_fillers import conversational_fillers

        return budget_note(rows, fillers_by_speaker(getattr(workbench, "segments", None) or []), conversational_fillers(resources_dir(Path(load_settings().work_dir)).parent))          # the corpora sit beside the attestation folder, not in it
    except Exception as error:  # noqa: BLE001 - a budget is an improvement, never a reason a repair cannot start; but never silent: a note that did not arrive looks like an agent that ignored it
        print(f"[budget] the fit budget and filler note could not be built: {type(error).__name__}: {str(error)[:200]}", flush=True)
        return ""


MAX_QUALITY_ROUNDS = 1          # how many times a quality-first run is sent back to its open points when it stops (bounded: a loop needs a ceiling; prod-y1 spent 21 turns after the bar on soft points, with three rounds)
QUALITY_POINTS_SHOWN = 12


def quality_pushback_message(points: list[str], number: int) -> str:
    """Sent when a quality-first agent stops while its last submission still has open listener points: act on each, or give a reason you measured.  In prod-95 and prod-96 the agent was told the
    bar was met and given the points, and finished on the next turn: once with 'further pause changes would break the current collision-free timing', which it had not measured."""
    lines = [f"You stopped, but your last submission still has {len(points)} open point(s) (soft evidence, not gates; measured on the source audio and against real isiZulu):"]
    lines += [f"- {point}" for point in points]
    lines.append(
        "Before you finish, take each one. Either change the unit (check the result with "
        "measure_tts or measure_tts_batch, and shorten words elsewhere in the unit if it would overrun; send only the changed units with submit_dub partial=true), or, if you decide a point "
        "should stay, say why in one line of your reply. A reason you measured counts; 'it would break the timing' without a measurement does not. "
        f"Round {number} of {MAX_QUALITY_ROUNDS}: when you have acted on what is right, reply with no tool call.")
    return "\n".join(lines)


AGENT_MODES = ("live", "training")


def agent_mode(value: str | None = None) -> str:
    """``live`` (the default: token-lean, what runs in production) or ``training`` (the agent also records why it chose what it chose; see TRAINING_PROMPT).  From the argument, else MATHULA_AGENT_MODE."""
    mode = str(value or os.environ.get("MATHULA_AGENT_MODE") or "live").strip().lower()
    return mode if mode in AGENT_MODES else "live"


TRAINING_PROMPT = """
TRAINING MODE. This run is studied, not published: the people reading it want to understand how you decide. Work exactly as you would otherwise (the same tools, the same bar, no extra
turns); the only difference is what you write down. Everything you record is read by someone who does NOT read Zulu, so write it in English (Zulu only for the Zulu words themselves).
- submit_dub takes choice_points: every place where you weighed two or more ways of saying something, or chose against a flag, a lesson, a tool's suggestion or the corpus. For each:
  {unit, english (the chunk), zulu (the form you chose, exactly as written in the unit), alternatives (the other forms you considered, with a few words on each), why (the reason, in English),
  evidence (what you relied on or set aside: a tool result, a lesson, a measurement, the glossary, the speaker's pauses, your own sense of the Zulu)}. A word with no real alternative is not a
  choice point. Send them in the same submit_dub call as the units they belong to; at most 40 a call.
- submit_dub also takes unit_strategies: for each unit you rewrote or built, {unit, strategy}: one line on how you approached it (close to the English or retold, what you restructured and why).
- submit_dub also takes missing_evidence: anything you wanted to know while deciding and were not given (a count, an example, a check, a piece of context, a way to hear the result): {unit
  (optional), wanted, why}. This is how the people running this learn what to show you next, so say it plainly whenever you notice the gap; an empty list is a fine answer when you had what you needed.
- When you decline to act on evidence the harness showed you (a flag, a listener point, a suggestion), say so in a choice point with evidence "set aside" and the reason.
"""


def tool_specs(mode: str = "live") -> list[dict[str, Any]]:
    """The tools the agent is given: the same in both modes, except that in training mode submit_dub also takes choice_points and unit_strategies."""
    if mode != "training":
        return TOOL_SPECS
    import copy

    specs = copy.deepcopy(TOOL_SPECS)
    for spec in specs:
        if spec["function"]["name"] == "submit_dub":
            props = spec["function"]["parameters"]["properties"]
            props["choice_points"] = {"type": "array", "maxItems": 40, "description": "Training mode. Where you weighed alternatives or went against evidence (English).",
                                      "items": {"type": "object", "properties": {"unit": {"type": "string"}, "english": {"type": "string"}, "zulu": {"type": "string"},
                                                                                  "alternatives": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"},
                                                                                  "evidence": {"type": "string"}}, "required": ["unit", "english", "zulu", "why"]}}
            props["missing_evidence"] = {"type": "array", "maxItems": 20, "description": "Training mode. Information you wanted while deciding and were not given (English).",
                                         "items": {"type": "object", "properties": {"unit": {"type": "string"}, "wanted": {"type": "string"}, "why": {"type": "string"}}, "required": ["wanted", "why"]}}
            props["unit_strategies"] = {"type": "array", "maxItems": 40, "description": "Training mode. One line per unit you built or rewrote (English).",
                                        "items": {"type": "object", "properties": {"unit": {"type": "string"}, "strategy": {"type": "string"}}, "required": ["unit", "strategy"]}}
    return specs


WEB_PER_TURN = 4          # web lookups the model may make in one turn
WEB_PROMPT = """

WEB SEARCH. You have a web search tool, for a limited number of lookups in this run. Use it to settle a NAME the speaker said: a player, coach, official, team, place or organisation that the
recogniser and YouTube's captions garbled or disagree on, or that the glossary holds in a form nobody can vouch for. Search with the context (the video title, the two teams, the competition,
the date) and never for facts to add to the dub. A result is evidence only if it is about THIS event: check the date, the competition and the teams (a line-up from another year is not this match), and
say in a choice point what you looked up, what you found and which page it was. When the results do not settle it, keep what the speaker's audio supports and record it in missing_evidence.
"""


def build_system_prompt(playbook: dict | None = None, *, max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, grammar: bool | None = None, mode: str = "live", budget: str = "lean", web: bool = False) -> str:
    """The rules (always), the isiZulu grammar reference (unless switched off: MATHULA_AGENT_GRAMMAR=0 or grammar=False), and the learned strategy notes (if a playbook is given).
    The reference replaces the pass-2 grammar block this prompt used to carry: its claims were not checked against data and the new modules cover and correct them."""
    from . import zulu_grammar
    from .playbook import render_section

    use_grammar = zulu_grammar.enabled() if grammar is None else grammar
    base = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{max_dead_air_pct}%").replace("{grammar_pointer}", GRAMMAR_POINTER if use_grammar else "")
    prompt = base + ("\n\n" + zulu_grammar.reference() + "\n" if use_grammar else "\n") + render_section(playbook) + (TRAINING_PROMPT if mode == "training" else "") + (WEB_PROMPT if web else "")
    return free_prompt(prompt) if budget == "free" else prompt

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
          "words the speaker's audio showed were misheard; the english already carries the fix, so translate that meaning. `video` (when present) is the uploader's own title, "
          "channel and description: a human-written source for the names and the setting of the story.",
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
    _tool("estimate_fit", "Free (no TTS, costs no measurement): for up to 40 candidate texts, how long the voice is estimated to take against the source window, with the "
          "error band of the estimate (nine in ten estimates fall inside it). The estimate is learned from every duration measured so far and is corrected by this run's own "
          "measurements. It cannot confirm a fit; it tells you which candidates are far from the window, so you measure the ones that could fit.",
          {"requests": {"type": "array", "items": {"type": "object", "properties": {
              "text": {"type": "string"}, "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"}, "speaker_id": {"type": "string"}},
              "required": ["text", "start_ms", "end_ms"]}}}, ["requests"]),
    _tool("grammar_review", "Free (no TTS, costs no measurement): a second reader of the isiZulu in your current submission. For each unit it lists the phrases it finds ungrammatical ('grammar') or "
          "too fused to say in one breath ('fused'), each with a suggested rewrite. It is an opinion, not a gate: apply a fix only where you agree it is better isiZulu and says the same. "
          "item_ids limits it to those units.", {"item_ids": {"type": "array", "items": {"type": "string"}}}, []),
    _tool("timeline_fit", "Free (no TTS, no measurement): lay a draft out on the timeline sentence by sentence, as a person watching the speaker would. Give one item per source segment, "
          "{segment_id, text}, or per unit of your own, {start_ms, end_ms, text} (optionally speaker_id and item_id), in any order. Each sentence starts where its source starts, or after the previous one ends when that one ran long, so lateness is inherited by every later "
          "sentence. You get, per sentence, its estimated start and end against the source's, how late it is, whether it runs into another speaker's speech, and, where the lateness BEGINS, "
          "which earlier sentences are over their own windows and could give the time back. Use it to decide where to cut before you measure: jump back to the sentence that lost the time, "
          "not to the one that shows it. It is an estimate; measure_tts confirms.",
          {"items": {"type": "array", "items": {"type": "object", "properties": {"segment_id": {"type": "string"}, "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"},
                                                                                  "speaker_id": {"type": "string"}, "item_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}}}, ["items"]),
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
    _tool("check_texts", "check_text for up to 40 units in ONE call: [{item_id, zulu_text, source_text}]. Use this, not check_text one unit at a time: every call is a turn. "
          "Every submit_dub also returns the same checks for all its units (text_checks).",
          {"units": {"type": "array", "items": {"type": "object", "properties": {"item_id": {"type": "string"}, "zulu_text": {"type": "string"}, "source_text": {"type": "string"}},
                                                  "required": ["zulu_text", "source_text"]}}}, ["units"]),
    _tool("submit_dub", "Submit the dub as ordered, non-overlapping units covering every English segment. "
          "Returns the official score and flagged units. You may resubmit. With partial=true, send ONLY the units you "
          "changed: each replaces the current unit with the same item_id, ids listed in remove are deleted, and every "
          "other current unit carries over (needs a previous accepted submission). To re-split units, send the new "
          "units and list the old ids in remove.",
          {"items": {"type": "array", "items": _UNIT_SCHEMA}, "approach_notes": {"type": "string"},
           "partial": {"type": "boolean", "description": "true = update only the units you send."},
           "remove": {"type": "array", "items": {"type": "string"},
                      "description": "partial only: ids of current units to delete (e.g. units you re-split)."},
           "decisions": {"type": "array", "maxItems": 12, "description": "Optional. Terms or phrases that recur in this clip and that you have settled how to say in Zulu; they come back with every submit. english: the term; zulu: ONE form exactly as written in your units (a word or a phrase, no alternatives, no slashes); note: a few words IN ENGLISH (the reader of this log does not read Zulu).",
                         "items": {"type": "object", "properties": {"english": {"type": "string"}, "zulu": {"type": "string"}, "note": {"type": "string"}}, "required": ["english", "zulu"]}}}, ["items"]),
    _tool("create_loanword", "For an English noun the speaker says that has NO everyday Zulu word: make a Zulu word for it the way Zulu adapts loans "
          "(science -> isayensi, bus -> ibhasi, spoon -> isipunu), by rule, from the dictionary's sounds for the word (p t k b written ph th kh bh, vowels "
          "reduced to a e i o u, a vowel inserted where Zulu has no cluster and at the end of the word). The forms come back with what each changes from the "
          "most typical. Nothing is listened to, spoken or scored. Not for names, acronyms or titles.",
          {"word": {"type": "string", "description": "the English word as written in your Zulu, e.g. provident"}}, ["word"]),
    _tool("use_loanword", "Remember the Zulu word you chose from create_loanword so every later clip uses the same one. You still write it in the Zulu text yourself, "
          "with the noun prefix the sentence needs.", {"word": {"type": "string"}, "zulu_word": {"type": "string"}}, ["word", "zulu_word"]),
    _tool("punctuation_examples", "How real isiZulu news punctuates a word or short phrase: real sentences, with their own commas and full stops, in which it follows a comma, and others in which it does not "
          "(from 30,000 Zulu news documents, 2013-2024). Use it when you are unsure whether a clause boundary takes a comma: you see what real writers do with the same word. Examples to read, "
          "not a rule and not a count. Free: no try, no model. Give the word as it is written in Zulu (kanti, bese, ukuthi, kodwa); a form that is rare in the news returns few or no examples.",
          {"phrase": {"type": "string", "description": "A Zulu word or short phrase (up to three words)."},
           "limit": {"type": "integer", "description": "Examples of each kind, 1 to 6 (default 4)."}}, ["phrase"]),
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
    _tool("source_readings", "What the pipeline heard at one moment of the source, for when the English of a segment looks odd, ambiguous or wrong (a double negative, a name, a number, a sentence "
          "that does not make sense): the recogniser's text and what the restoration changed in it, the recogniser's confidence for each word and the words it was unsure of, YouTube's caption for the "
          "same moment (a second recogniser run on the same audio) and the words the two readings differ on, and the sentences either side. Give a segment_id from open_clip, or a few words of the "
          "English. It cannot play the audio or say what the speaker meant: two readings that agree are strong evidence, a low-confidence word is the one to doubt. Free: no try, no model.",
          {"segment_id": {"type": "string", "description": "a segment id from open_clip, e.g. seg-00007"}, "phrase": {"type": "string", "description": "a few words of the English, if you do not have the id"}}, []),
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
        self.reasoning_summary: str | None = None          # "detailed" in training mode: the model's own summary of its reasoning each turn, logged to the trace
        self.web_budget = 0          # web lookups (searches and pages opened) the agent may make in a run; 0 = the tool is not offered
        self.web_used = 0

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
    def web_items(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """What the model looked up this turn, for the trace: each search with its queries, each page it opened, and the pages its answer cites."""
        found: list[dict[str, Any]] = []
        for item in output:
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                row: dict[str, Any] = {"action": action.get("type")}
                if action.get("type") == "search":
                    row["queries"] = [str(q)[:200] for q in (action.get("queries") or [action.get("query")]) if q][:4]
                if action.get("url"):
                    row["url"] = str(action.get("url"))[:300]
                found.append(row)
            elif item.get("type") == "message":
                cited = [{"title": str(a.get("title") or "")[:120], "url": str(a.get("url") or "")[:300]}
                         for part in item.get("content") or [] for a in part.get("annotations") or [] if a.get("type") == "url_citation"]
                if cited:
                    found.append({"cites": cited[:6]})
        return found

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
        summary = "\n".join(part.get("text", "") for item in output if item.get("type") == "reasoning"
                            for part in item.get("summary") or [] if isinstance(part, dict) and part.get("text"))
        if summary:
            message["_reasoning"] = summary          # only for the trace: it is not sent back (the raw reasoning items are)
        web = ResponsesClient.web_items(output)
        if web:
            message["_web"] = web          # only for the trace, like the reasoning
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
        if self.reasoning_summary:
            body.setdefault("reasoning", {})["summary"] = self.reasoning_summary
        if tools:
            body["tools"] = self.to_tools(tools)
        left = self.web_budget - self.web_used
        if left > 0:          # the model's own web search, bounded: at most WEB_PER_TURN lookups in a turn and the run's budget in all
            body.setdefault("tools", []).append({"type": "web_search"})
            body["max_tool_calls"] = min(WEB_PER_TURN, left)
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        raw = post_with_retry(self.url, headers, body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS), retries=self.retries, sleep=self.sleep)
        self.web_used += sum(1 for item in raw.get("output") or [] if item.get("type") == "web_search_call")
        return self.to_chat_shape(raw)


def drop_empty_replies(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A model reply with no text and no tool call says nothing: a checkpoint saved before cut-off replies were handled can hold some (a run that was cut off twice), and a model turn with no
    content can be refused when the history is sent again.  They are dropped when a checkpoint is loaded."""
    return [m for m in messages if not (m.get("role") == "assistant" and not str(m.get("content") or "").strip() and not m.get("tool_calls"))]


MEASURE_TOOLS = frozenset({"measure_tts", "measure_tts_batch"})          # the per-unit real syntheses; submit_dub synthesizes and scores EVERY unit, so a first submission gives all of them at once
DRAFT_PHASE_NOTE = ("Work breadth first. Until you have submitted a complete dub, measure_tts and measure_tts_batch are not available (a call is refused and costs nothing): submit_dub synthesizes and scores every unit, so your first "
                    "submission gives you real timing, gates and collisions for the whole clip at once, and flags the units that need work. Draft EVERY unit now, sized with fits_about_chars "
                    "(estimate_fit and timeline_fit are free: send ALL the units of the draft in ONE call to each, up to 40 and 60, because every call is a turn; submit_dub also returns check_text for every unit), then call submit_dub. "
                    "After that, spend your effort where the score says it matters. "
                    f"The first submission is due by about {int(DRAFT_FRACTION * 100)}% of the token budget: after that only submit_dub is available until it is in.")
REFINE_PHASE_NOTE = ("Your first complete submission is in, so measure_tts and measure_tts_batch are available now. Work from its score: the flagged units and the dead air they leave, in the order "
                     "that matters (gates, collisions, then the dead air), and send only the units you change with submit_dub partial=true. Every flagged unit and every measurement now says how far it is from its window and what that is in characters at its own pace (to_fit): change the words by about that much in one go, measure two or three renderings together, and when to_fit says within_noise the unit is done, because a further edit moves it within the noise.")
DRAFT_REFUSAL = ("measure_tts is not available until a complete submission exists: draft every unit (fits_about_chars, estimate_fit and timeline_fit are free) and call submit_dub, which "
                 "synthesizes and scores all of them at once. Nothing was spent.")
DRAFT_DEADLINE_REFUSAL = ("The draft deadline has passed and there is no submission yet, so only submit_dub and get_submission are available until there is one. Call submit_dub now with complete text for every unit; "
                          "the other tools come back as soon as it is in. Nothing was spent.")
FINAL_REFUSAL = "The budget is nearly spent, so only submit_dub and get_submission are available now. Nothing was spent."
TRUNCATED_FINISHES = ("length", "incomplete")          # a reply that ran into the output limit: the Gemini client reports MAX_TOKENS as 'length', the Azure Responses API as 'incomplete'
MAX_CONSECUTIVE_TRUNCATIONS = 2
TRUNCATED_NOTE = ("Your last reply was cut off by the output limit before it finished, so nothing in it was done (a tool call that is cut off is no call). Make the next reply smaller: "
                  "think less before acting, and send less in one call (fewer choice points and unit strategies, shorter notes); the text of the units is what matters.")


def native_client_config(environ: dict[str, str] | None = None) -> Any:
    """The model config the agent and its judges run on: Gemini when ``MATHULA_AI_PROVIDER=gemini``, else the Azure OpenAI GPT config."""
    environ = dict(os.environ if environ is None else environ)
    if (environ.get("MATHULA_AI_PROVIDER") or "azure").strip().lower() == "gemini":
        from .gemini_client import gemini_config_from_environment

        return gemini_config_from_environment(environ)
    from .foundry_grok import native_gpt_config_from_environment

    return native_gpt_config_from_environment(environ)


def new_agent_client(config: Any, **kwargs: Any) -> Any:
    """The agent loop's client for this config: Gemini for a Gemini config, else the Azure Responses client (reasoning models need it for function tools)."""
    from .gemini_client import GeminiClient, GeminiConfig

    return GeminiClient(config, **kwargs) if isinstance(config, GeminiConfig) else ResponsesClient(config, **kwargs)


def new_chat_client(config: Any, **kwargs: Any) -> Any:
    """The judges' client for this config: Gemini for a Gemini config, else the Azure chat-completions client."""
    from .gemini_client import GeminiClient, GeminiConfig

    return GeminiClient(config, **kwargs) if isinstance(config, GeminiConfig) else ChatClient(config, **kwargs)


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
    "[Earlier working steps (drafts, measurements, checks, past feedback) were removed to keep the conversation short. Your latest submit_dub call and its result are below, and the result "
    "already shows where every unit stands: units_overview (each unit's length and fit), flagged_units (the text of the units that need work), settled_units, decisions_so_far. "
    "Work from those; call get_submission only to read the text of a unit that is not listed. The opened clip above is unchanged.]"
)


def _step_tools(messages: list[dict[str, Any]], index: int) -> set[str]:
    return {c["function"]["name"] for c in messages[index].get("tool_calls") or []}


FOCUS_FLOOR_TOKENS = 36_000       # below this a request is mostly the system prompt, the opened clip and the lessons: there is nothing to fold away
FOCUS_KEEP_STEPS = 4              # the working steps since the last submission that stay verbatim when the older ones are folded into the ledger
FOCUS_NOTE_PREFIX = "[Working steps since your last submission, folded"
KEEP_RECENT_STEPS = 4      # with no submission to stand on, the last few working steps stay verbatim for continuity: the agent is mid-thought in them


def _frozen_head_end(messages: list[dict[str, Any]]) -> int:
    """Where the head of the conversation ends: the system prompt, the task, the notes placed before any work, and the steps that opened the clip and recalled lessons.

    It is the front of every request, and the opened clip in it is most of the context, so it is never removed or reordered: a provider's cache is a prefix match, and a message dropped from the
    front would make everything after it (the whole clip) be billed in full again.  The head ends at the first working step."""
    index = 2
    while index < len(messages):
        if messages[index].get("role") == "assistant":
            if not (_step_tools(messages, index) & _KEEP_STEP_TOOLS):
                break
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                index += 1
        else:
            index += 1
    return min(index, len(messages))


def _compact_on_ledger(messages: list[dict[str, Any]], ledger: Any) -> tuple[list[dict[str, Any]], int]:
    """No submission exists, so the agent's drafts live only in its tool calls: put the harness's draft ledger (see draft_ledger) where the old steps were and keep the last few steps.

    What stays: the system prompt, the task, the steps that opened the clip and recalled lessons, and everything from the KEEP_RECENT_STEPS-th working step from the end on.  What
    goes: the working steps and notes between.  The history is bounded whatever the agent does and whenever it submits.  Idempotent."""
    from .draft_ledger import LEDGER_NOTE_PREFIX

    body = [m for m in messages if not str(m.get("content") or "").startswith(LEDGER_NOTE_PREFIX)]
    steps: list[tuple[int, int]] = []
    index = 2
    while index < len(body):
        if body[index].get("role") == "assistant":
            end = index + 1
            while end < len(body) and body[end].get("role") == "tool":
                end += 1
            steps.append((index, end))
            index = end
        else:
            index += 1
    opening = [s for s in steps if _step_tools(body, s[0]) & _KEEP_STEP_TOOLS]
    working = [s for s in steps if s not in opening]
    if len(working) <= KEEP_RECENT_STEPS:
        return messages, 0
    recent_from = working[-KEEP_RECENT_STEPS][0]
    keep = {0, 1} | set(range(_frozen_head_end(body))) | {i for s in opening for i in range(*s)} | set(range(recent_from, len(body)))
    dropped = [i for i in range(2, len(body)) if i not in keep]
    if not dropped:
        return messages, 0
    note = {"role": "user", "content": (f"{LEDGER_NOTE_PREFIX} (the harness keeps the latest text you tried for each window, from your own tool calls). Earlier working steps were removed to keep the conversation "
                                        "short; nothing you wrote is lost: it is here, with how long it measured or was estimated to be against its window. Nothing below has been submitted. "
                                        "Continue from it: the clip above is unchanged, and the last few steps follow.]\n" + ledger.render())}
    kept = [m for i, m in enumerate(body) if i not in dropped]
    position = next(i for i, m in enumerate(kept) if m is body[recent_from])
    kept.insert(position, note)
    return kept, sum(1 for i in dropped if body[i].get("role") == "assistant")


def _fold_since_submit(messages: list[dict[str, Any]], last_submit: int, ledger: Any) -> tuple[list[dict[str, Any]], int]:
    """Fold the working steps after the latest submit_dub that are older than the last FOCUS_KEEP_STEPS into the harness's draft ledger.

    Every measurement, estimate and check since the submission stays in the request for the rest of the run, so the request grows by a step a turn for as long as the agent works
    without submitting (prod-121: 41k to 63k tokens in 17 turns).  Those steps are in the ledger already, built by the harness from the agent's own tool calls: the latest text tried
    for each window and how long it measured, which is what the agent needs from them.  The submit step and its result, the opened clip and the last few steps are untouched, so the
    cached front of the request stays the same and only what follows the fold is sent again.  Whole steps only (a tool call stays paired with its result).  Idempotent."""
    if ledger is None or not getattr(ledger, "entries", None):
        return messages, 0
    after = last_submit + 1
    while after < len(messages) and messages[after].get("role") == "tool":
        after += 1                                                # the submit step's own results stay
    body = [m for m in messages[after:] if not str(m.get("content") or "").startswith(FOCUS_NOTE_PREFIX)]
    steps: list[tuple[int, int]] = []
    index = 0
    while index < len(body):
        if body[index].get("role") == "assistant":
            end = index + 1
            while end < len(body) and body[end].get("role") == "tool":
                end += 1
            steps.append((index, end))
            index = end
        else:
            index += 1
    if len(steps) <= FOCUS_KEEP_STEPS:
        return messages, 0
    recent_from = steps[-FOCUS_KEEP_STEPS][0]
    older = body[:recent_from]
    folded = sum(1 for m in older if m.get("role") == "assistant")
    if not folded:
        return messages, 0
    note = {"role": "user", "content": (f"{FOCUS_NOTE_PREFIX} into what the harness kept of them: the latest text you tried for each window and how long it measured. Nothing in your last submission changed; "
                                        "the units it settled stay as they are unless a point names them. Continue from here: the last few steps follow.]\n" + ledger.render())}
    return messages[:after] + [note] + body[recent_from:], folded


def compact_history(messages: list[dict[str, Any]], ledger: Any = None, *, focus: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Drop the working steps a newer submission has superseded. Returns (messages, dropped_count).

    A step is an assistant message plus the tool results that answer it; dropping whole steps keeps
    every remaining tool call paired with its result. What stays: the system prompt, the task, the
    steps that opened the clip and recalled lessons, the LATEST submit_dub step, and everything after
    it. What goes: every step and every user note between them (earlier drafts, measurements, checks,
    old pushbacks). Idempotent: a second pass over a compacted history changes nothing.

    With no submission yet, the same is done from the harness's draft ledger (``_compact_on_ledger``): the history never depends on the agent having submitted.
    """
    last_submit = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and "submit_dub" in _step_tools(messages, index):
            last_submit = index
    if last_submit is None:
        if ledger is not None and ledger.entries:
            return _compact_on_ledger(messages, ledger)
        return messages, 0
    keep = {0, 1} | set(range(min(_frozen_head_end(messages), last_submit)))
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
    dropped_steps = 0
    if dropped:
        kept = [m for i, m in enumerate(messages) if i not in dropped and m.get("content") != COMPACTION_NOTE]
        position = next(i for i, m in enumerate(kept) if m is messages[last_submit])
        kept.insert(position, {"role": "user", "content": COMPACTION_NOTE})
        dropped_steps = sum(1 for i in dropped if messages[i].get("role") == "assistant")
        messages = kept
    if focus:                                                       # and the working steps since that submission, which the size trigger alone leaves in until the next submit
        latest = max((i for i, m in enumerate(messages) if m.get("role") == "assistant" and "submit_dub" in _step_tools(messages, i)), default=None)
        if latest is not None:
            messages, folded = _fold_since_submit(messages, latest, ledger)
            dropped_steps += folded
    return messages, dropped_steps


_SLIM_HEADLINE = (
    "group_count", "dead_air_pct", "turn_shortfall_pct", "silence_padded_pct", "speaker_pauses_pct", "residual_dead_air_pct",
    "groups_dead_air", "groups_rushed", "gate_failures", "worst_turn_shortfall_pct",
    "collisions", "cross_speaker_collisions", "worst_collision_ms", "meaning_loss_units",
)


CLAIMS_NOTE = (
    "Taking a claim out shortens the unit, and a unit left short is dead air: a submission that breaks a unit that had fitted is rejected. "
    "Compare chars_now with fits_about_chars, put back only what the English does say (or a pause) to keep the length, measure the edited "
    "units, and send all of them in one partial submit.")


SENTENCE_TIMING_NOTE = (
    "Soft evidence, not a gate: the renderer moves only SILENCE between sentences, never speech, so a sentence that starts well after the speaker's matching one (behind_the_mouth, in seconds) "
    "can only be fixed in the words. take_out_about_chars is that lag at this speaker's rate: take about that much speech out of the sentences BEFORE it (a connective, a repeated modifier, a "
    "shorter idiom, a loanword left in English), keep the money and the facts, and put it back after it if the unit would then end short. ahead_of_the_mouth is the other way, usually harmless. "
    "mostly_unchecked means most of the unit's sentences could not be matched to the speaker's: read its sentence lengths against the English yourself.")

SUBMIT_KEY_ROUTES: dict[str, tuple[str, ...]] = {
    # every key Workbench.submit_dub puts in its result -> the key(s) of slim_submit's result that carry it to the agent.  A key with no route would be computed and never shown: the guard
    # in tests/test_dub_agent.py reads the workbench source and fails when a new key is added without a route here (slim_submit is an allow-list).
    "accepted": ("accepted",), "warnings": ("warnings",), "headline": ("headline",), "gates": ("gates",), "merged": ("merged",),
    "regressions_accepted": ("regressions_accepted",), "note": ("note",), "meaning": ("meaning_loss", "claims_to_check"), "timing_lock": ("timing_lock",),
    "turn_padding": ("turns_ending_early",), "flagged_units": ("flagged_units",), "listening_load": ("listening_load",), "text_checks": ("text_checks",), "spoken_form_gaps": ("spoken_form_gaps",),
    "sentence_timing": ("sentence_timing",), "voice_limit_words": ("voice_limit_words",), "figures_as_spoken": ("figures_as_spoken",), "decisions_so_far": ("decisions_so_far",),
    "settled_units": ("settled_units",), "units_overview": ("units_overview",), "naturalness": ("naturalness",), "speaker_rates_chars_per_s": ("speaker_rates_chars_per_s",),
}


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
    if result.get("text_checks"):
        slim["text_checks"] = result["text_checks"]
        slim["text_checks_note"] = ("check_text run for you on every unit (what check_texts would return): a problem with a gate is a hard failure to fix; a flag is soft evidence to weigh. "
                                    "You do not need to call check_text again for these units.")
    if result.get("listening_load"):
        slim["listening_load"] = result["listening_load"]
        slim["listening_load_note"] = ("Soft evidence, not a gate: a figure run into the next with no breath, or a stretch of about eight seconds with no breath, is hard for a listener to follow. "
                                       "Give a figure that carries the point its breath, or split the sentence at the end of a thought; leave it if the speaker ran on the same way.")
    if result.get("sentence_timing"):
        slim["sentence_timing"] = result["sentence_timing"]
        slim["sentence_timing_note"] = SENTENCE_TIMING_NOTE
    if result.get("decisions_so_far"):
        slim["decisions_so_far"] = result["decisions_so_far"]
    if result.get("settled_units"):
        slim["settled_units"] = result["settled_units"]
    if result.get("units_overview"):
        slim["units_overview"] = result["units_overview"]
    if result.get("naturalness"):
        slim["naturalness"] = result["naturalness"]
        slim["naturalness_note"] = ("Soft evidence, not a gate, measured against real isiZulu (a 2-million-word spoken corpus and 11 million words of written text): pairs_that_never_meet_in_real_zulu are two common words side by side that never occur together although they should have by chance, which often means a calque or a wrong collocation; reads_like_translated_government_zulu is a unit whose sentences score like translated government text rather than native speech or news (the contrast is translated against native, so political and legal wording is not penalised for being formal); cadence is a unit chopped into sentences shorter than almost any real speaker uses. Reword the pair the way a speaker would say it, or leave it if it is right; join short sentences where the speaker kept going.")
    if result.get("speaker_rates_chars_per_s"):
        slim["speaker_rates_chars_per_s"] = result["speaker_rates_chars_per_s"]
    if result.get("decisions_so_far") or result.get("settled_units"):
        slim["working_state_note"] = ("Kept here because your earlier steps are cleared from the conversation: decisions_so_far are the terms you settled (use the same Zulu for the same idea; a unit "
                                      "that no longer holds one drifted), settled_units had no open flag at this submit (leave them unless a point names them), speaker_rates_chars_per_s is the "
                                      "measured speech rate to reckon a length edit at.")
    if result.get("figures_as_spoken"):
        slim["figures_as_spoken"] = result["figures_as_spoken"]
        slim["figures_as_spoken_note"] = ("Soft evidence, not a gate: each number you wrote and the words the voice is given for it. Read them as a listener will hear them: a time must say "
                                          "a time (nine thirty), a year a year, a figure its figure; a reading that is wrong means writing it another way (a time with a colon, 9:30), or say in approach_notes what you could not fix.")
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
        **({"check_texts": workbench.check_texts} if hasattr(workbench, "check_texts") else {}),
        **({"estimate_fit": workbench.estimate_fit} if hasattr(workbench, "estimate_fit") else {}),
        **({"timeline_fit": workbench.timeline_fit} if hasattr(workbench, "timeline_fit") else {}),
        **({"grammar_review": workbench.grammar_review} if hasattr(workbench, "grammar_review") else {}),
        "preview_turn_padding": workbench.preview_turn_padding,
        "submit_dub": lambda **kw: slim_submit(workbench.submit_dub(**kw)),
        "get_submission": workbench.get_submission,
    }
    for name in ("create_loanword", "use_loanword", "attested_zulu", "punctuation_examples", "human_translations", "figure_style", "source_readings", "say_it"):       # vocabulary tools (the loan rules on the dictionary's sounds, a corpus lookup), where the workbench has them; the agent has no recogniser and no pronunciation tool
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


DEAD_AIR_UNIT_MS = 2000                # a unit that ends this long before its window is a listener point in the wrap-up (see open_listener_points)
LISTENER_POINTS_SHOWN = 8             # the wrap-up names at most this many open listener points


def open_listener_points(result: dict[str, Any] | None, *, limit: int = LISTENER_POINTS_SHOWN) -> list[str]:
    """The soft listener evidence of the last accepted submission that is still open (comma_check): pauses the source speaker held and the Zulu does not, commas the speaker
    did not take.  Not gates; the wrap-up shows them while the agent can still act, because at the final submit they arrived after the stop rule had already said 'finish' (prod-84, prod-85)."""
    points: list[str] = []
    for item in (result or {}).get("voice_limit_words") or []:
        if isinstance(item, dict):
            points.extend(f"{item.get('item_id')}: {line}" for line in item.get("words") or [])
    for item in (result or {}).get("naturalness") or []:          # a unit chopped into very short sentences: said aloud it is mostly full stops
        if isinstance(item, dict) and item.get("cadence"):
            points.append(f"{item.get('item_id')}: {item['cadence']}")
    for item in (result or {}).get("flagged_units") or []:          # a unit that ends well before its window leaves the speaker talking over silence, however short the whole run's dead air is
        early = int(item.get("ends_early_ms") or 0) if isinstance(item, dict) else 0
        if early >= DEAD_AIR_UNIT_MS:
            points.append(f"{item.get('item_id')}: the Zulu ends {early / 1000:.1f} s before the speaker stops (about {round(early / 1000 * 12)} more characters would fill it): the viewer sees them talking in silence")
    return points[:limit]


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


def _unit_english(workbench: Any, items: list[dict[str, Any]]) -> dict[str, str]:
    """The English each unit of a submission is written against (its segments' text now), kept with the submission."""
    current = {str(s["segment_id"]): str(s.get("restored_text") or s.get("source_text") or "") for s in getattr(workbench, "segments", None) or []}
    return {str(item.get("item_id")): " ".join(current.get(str(i), "") for i in item.get("source_segment_ids") or []) for item in items if item.get("item_id") is not None}


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


def prefix_corpus_note(workbench: Any, seed_items: list[dict[str, Any]]) -> str:
    """In a repair run: the units whose text writes a prefix on a word or name the speaker says in a form the Zulu corpora never write, with the forms they do write.  The check runs on every unit of the
    seeded submission at the start, because the agent only checks the units it changes: unit-02's "kwabe-Hawks" (real text writes amaHawks, kumaHawks) was never looked at in two repair runs and the voice
    said it as "kwe ... hwaks" (prod-129 0:56).  A reminder to compare, not a verdict: a form missing from small corpora can be right, and the agent decides."""
    try:
        resources = workbench._resources()
        if resources is None:
            return ""
        from .english_prefix import find_unattested_prefixed

        current = {str(seg["segment_id"]): str(seg.get("restored_text") or seg.get("source_text") or "") for seg in getattr(workbench, "segments", None) or []}
        lines = []
        for item in seed_items:
            english = " ".join(current.get(str(i), "") for i in item.get("source_segment_ids") or [])
            found = [line.split(". Zulu vowels merge")[0] for line in find_unattested_prefixed(str(item.get("zulu_text") or ""), english, resources)]
            if found:
                lines.append(f"{item.get('item_id')}: " + "; ".join(found))
    except Exception:                                       # noqa: BLE001 - a lookup that cannot run is no reason a repair cannot start
        return ""
    if not lines:
        return ""
    return ("The Zulu corpora were looked up for every prefix written in front of a word or name the speaker says, in all the units. These forms are written 0 times there, and the forms real text writes are shown: "
            + " | ".join(lines) + ". Zulu vowels merge (na+u = no, wa+i = we) and the corpora are small, so a form missing from them can still be right; compare each with the attested forms, and if you are unsure "
            "follow the attested one, because the voice has not heard a form nobody writes. Treat the units where you change the form as units that need change. ")


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

    paused = False
    try:
        summary = _run_agent(client, workbench, **kwargs)
        paused = summary.get("stop_reason") == "paused"
        return summary
    finally:
        episode = getattr(workbench, "episode", None)
        if not paused and not keep_measure() and episode and episode.get("dir"):
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
    compact_after_tokens: int = COMPACT_AFTER_TOKENS, focus_growth_tokens: int = 0, playbook_version: str | None = None,
    seed_items: list[dict[str, Any]] | None = None, wrapup_turns: int = WRAPUP_TURNS, dead_air_slack_pct: float = DEAD_AIR_SLACK_PCT, mode: str = "live", free_finish: bool = False, resume: bool = False,
    breadth_first: bool | None = None, stable_tools: bool | None = None, draft_deadline: bool | None = None, focus_units: list[str] | None = None,
) -> dict[str, Any]:
    """``draft_deadline`` (default on; MATHULA_AGENT_DRAFT_DEADLINE=0 turns it off): a complete first submission is due by a share of the token ceiling (see budget_governor), after which only submitting is available and the harness submits the agent's own drafts if it still has not.  ``stable_tools`` (default on; MATHULA_AGENT_STABLE_TOOLS=0 turns it off): the tool declarations are the same on every turn and a phase (draft, final) restricts which may be called, because a
    changed declaration set discards the provider's cache of everything after it.  ``breadth_first`` (default on; MATHULA_AGENT_BREADTH_FIRST=0 turns it off): until a complete submission exists the per-unit measurement tools are not available, so the work is
    draft-everything-then-score by construction (see DRAFT_PHASE_NOTE).  ``resume``: continue from the checkpoint this run saved (agent_state.json: every turn writes one) instead of starting; a run is paused with a PAUSE file in its folder (request_pause) and
    returns stop_reason "paused".  ``mode``: "live" or "training" (the agent also records its choice points; the reasoning summaries and a thinking report are kept).  ``wrapup_turns``: once a submission meets the bar (within ``dead_air_slack_pct`` on residual dead air) the agent is told to finish and has this many turns left;
    0 turns the rule off.  ``seed_items`` starts a REPAIR run: the units of an earlier accepted submission are re-scored under today's rules
    and the agent is told to change only what misses the bar, leaving every other unit exactly as it was."""
    mode = agent_mode(mode)
    specs = tool_specs(mode)
    if mode == "training" and hasattr(client, "reasoning_summary"):
        client.reasoning_summary = "detailed"
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
                + prefix_corpus_note(workbench, seed_items)
                + (FOCUS_NOTE.format(units=", ".join(focus_units)) if focus_units else "")
                + (focus_budget_note(workbench, list(seed_items), focus_units) if focus_units else "")
                + "Change only the units that need it, send just those with submit_dub partial=true, leave every other unit "
                  "exactly as it is, and stop once the bar is met.")
    trace.log("task_message", text=str(task_message or "")[:8000])          # what the agent was told, so a run can be read afterwards (a note that did not arrive looks like an agent that ignored it)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_message or f"Dub clip {clip_id!r}. Start by opening it."},
    ]
    tokens = {"in": 0, "out": 0, "cached": 0}
    last_prompt_tokens = 0
    focus_base: int | None = None              # the request size at the first turn after the last compaction or submission: what the working steps since then are measured against
    compactions = 0
    tool_counts: dict[str, int] = {}
    stop_reason = "max_turns"
    nudged = False
    from .draft_ledger import DraftLedger

    ledger = DraftLedger()                    # the harness's own record of the agent's working drafts: what lets the history be shortened whether or not the agent has submitted
    from .budget_governor import BudgetGovernor

    governor = BudgetGovernor(max_total_tokens, max_output=int(getattr(getattr(client, "config", None), "max_output_tokens", 0) or 16_000))          # the ceiling as phases the harness enforces, so a run cannot end having produced nothing (see budget_governor); the reserve is sized from the client's real reply cap
    truncations = 0                           # consecutive replies cut off by the output limit
    if breadth_first is None:
        breadth_first = (os.environ.get("MATHULA_AGENT_BREADTH_FIRST") or "1").strip().lower() not in {"0", "false", "no", "off"}
    if stable_tools is None:
        stable_tools = (os.environ.get("MATHULA_AGENT_STABLE_TOOLS") or "1").strip().lower() not in {"0", "false", "no", "off"}
    if draft_deadline is None:
        draft_deadline = (os.environ.get("MATHULA_AGENT_DRAFT_DEADLINE") or "1").strip().lower() not in {"0", "false", "no", "off"}
    cache_watch = CacheWatch()                # per turn: what the provider could reuse of the request and, when it did not, why (see cache_watch)
    draft_noted = refine_noted = False        # the agent has been told it is in the draft phase / that the measurement tools are now offered
    from .submission_rank import BestSubmission

    best = BestSubmission()                   # every accepted submission, ranked by the scorer's own order (gates, collisions, overruns, dead air): the run's best one is never lost
    recorded_submit: Any = None
    loanwords_noted = False
    pushbacks = 0
    unmet: list[str] = []
    turn = 0
    good_enough_pct = max_dead_air_pct + dead_air_slack_pct
    seen_submit: Any = None
    wrapup_at: int | None = None
    told_bar = False                          # free_finish: the agent has been told the bar is met
    quality_rounds = 0                        # free_finish: times the agent was sent back to its open points
    quality_seen: Any = None                  # the submission the last round was about
    points_sent: set[str] = set()

    def accepted() -> bool:
        return bool(workbench.episode and workbench.episode.get("submissions"))

    start_turn = 1
    resumed_from = None
    if resume:
        state = load_state(directory)
        if state is None:
            raise FileNotFoundError(f"nothing to resume: {state_path(directory)} is missing or unreadable")
        messages[:] = drop_empty_replies(state["messages"])
        if messages and messages[0].get("role") == "system":          # the current instructions replace the saved ones: a run is resumed after the prompt or a gate was fixed
            prompt_changed = messages[0].get("content") != system_prompt
            messages[0] = {**messages[0], "content": system_prompt}
        else:
            prompt_changed = False
        tokens.update({k: int(v) for k, v in (state.get("tokens") or {}).items()})
        last_prompt_tokens = int(state.get("last_prompt_tokens") or 0)
        compactions = int(state.get("compactions") or 0)
        tool_counts.update({k: int(v) for k, v in (state.get("tool_counts") or {}).items()})
        flags = state.get("flags") or {}
        nudged, loanwords_noted, pushbacks = bool(flags.get("nudged")), bool(flags.get("loanwords_noted")), int(flags.get("pushbacks") or 0)
        ledger.load(state.get("ledger"))
        governor.load(state.get("governor"))
        draft_noted, refine_noted = bool(flags.get("draft_noted")), bool(flags.get("refine_noted"))
        best.load(state.get("best"))
        cache_watch.load(state.get("cache_watch"))
        told_bar, points_sent, wrapup_at = bool(flags.get("told_bar")), set(flags.get("points_sent") or []), flags.get("wrapup_at")
        quality_rounds = int(flags.get("quality_rounds") or 0)
        unmet = list(state.get("unmet") or [])
        workbench.open_clip(clip_id, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy)
        restore_workbench(workbench, state.get("workbench") or {})
        if not state.get("ledger"):                    # a checkpoint saved before the draft ledger existed: rebuild it from the conversation, so the old history can be compacted too
            ledger.set_segments(getattr(workbench, "segments", None))
            ledger.replay(messages)
        seen_submit = getattr(workbench, "last_submit", None)          # the bar message for that submission was already sent (or not needed): do not send it again
        start_turn = int(state.get("turn") or 0) + 1
        resumed_from = start_turn - 1
        try:
            (directory / PAUSE_NAME).unlink()
        except OSError:
            pass
        trace.log("resume", turn=start_turn, messages=len(messages), tokens=dict(tokens), system_prompt_changed=prompt_changed, submissions=(workbench.episode or {}).get("submissions"))
    turn = start_turn - 1

    def checkpoint(completed: int, stop: str = "running") -> None:
        try:
            save_state(directory, {
                "clip_id": clip_id, "run_id": run_id, "turn": completed, "stop_reason": stop, "mode": mode, "free_finish": free_finish, "messages": messages, "tokens": dict(tokens),
                "last_prompt_tokens": last_prompt_tokens, "compactions": compactions, "tool_counts": dict(tool_counts),
                "ledger": ledger.to_dict(), "governor": governor.to_dict(), "best": best.to_dict(), "cache_watch": cache_watch.to_dict(),
                "flags": {"nudged": nudged, "draft_noted": draft_noted, "refine_noted": refine_noted, "loanwords_noted": loanwords_noted, "pushbacks": pushbacks, "told_bar": told_bar, "points_sent": sorted(points_sent), "wrapup_at": wrapup_at, "quality_rounds": quality_rounds},
                "unmet": list(unmet), "workbench": workbench_snapshot(workbench)})
        except Exception as exc:                                             # noqa: BLE001 - a checkpoint that cannot be written must never end a run
            trace.log("checkpoint_failed", turn=completed, error=f"{type(exc).__name__}: {str(exc)[:200]}")

    for turn in range(start_turn, max_turns + 1):
        if turn > start_turn:
            checkpoint(turn - 1)                                                          # the position after the previous turn: a stop or a crash costs at most the call in flight
        if (directory / PAUSE_NAME).exists():
            stop_reason = "paused"
            turn -= 1
            break
        focus_due = bool(focus_growth_tokens and focus_base is not None and last_prompt_tokens >= FOCUS_FLOOR_TOKENS and last_prompt_tokens - focus_base >= focus_growth_tokens)
        if (compact_after_tokens and last_prompt_tokens >= compact_after_tokens) or focus_due:
            compacted, dropped_steps = compact_history(messages, ledger, focus=bool(focus_growth_tokens))
            if dropped_steps:
                trace.log("compact", turn=turn, dropped_steps=dropped_steps, messages_before=len(messages),
                          messages_after=len(compacted), prompt_tokens_before=last_prompt_tokens, trigger="focus" if focus_due and not (compact_after_tokens and last_prompt_tokens >= compact_after_tokens) else "size")
                messages[:] = compacted
                compactions += 1
                focus_base = None
        used = tokens["in"] + tokens["out"]
        phase = governor.phase(used, last_prompt_tokens)
        if phase == "over":
            stop_reason = "token_budget"
            break
        if governor.due(phase):
            budget_note = governor.note(phase, used, submitted=accepted(), drafts=ledger.render() if phase == "final" else "")
            messages.append({"role": "user", "content": budget_note})
            governor.sent(phase)
            trace.log(f"budget_{phase}", turn=turn, used=used, ceiling=max_total_tokens, text=budget_note[:800])
        if phase == "final" and not governor.final_turn_allowed():
            stop_reason = "token_budget"
            break
        if draft_deadline and phase in ("normal", "soft"):            # no submission yet and the deadline is here: say so once, then act if it still is not in (see budget_governor)
            draft_state = governor.draft_state(used, last_prompt_tokens, submitted=accepted())
            if draft_state == "due":
                draft_text = governor.draft_note(used, ledger.render())
                messages.append({"role": "user", "content": draft_text})
                governor.draft_told = True
                trace.log("draft_deadline", turn=turn, used=used, ceiling=max_total_tokens, text=draft_text[:800])
            elif draft_state == "overdue":
                outcome = governor.auto_submit(workbench, ledger)
                governor.draft_acted = True
                trace.log("draft_auto_submit", turn=turn, **outcome)
                if outcome.get("submitted"):
                    headline = (getattr(workbench, "last_submit", None) or {}).get("headline")
                    messages.append({"role": "user", "content": "You did not submit in time, so the harness submitted your working drafts as they stood. Its score: " + json.dumps(headline, ensure_ascii=False)
                                     + ". Call get_submission to read which units it flags, and work from there: measure_tts and the other tools are available again."})
                else:
                    messages.append({"role": "user", "content": governor.draft_uncovered_note(outcome)})
        drafting = bool(breadth_first) and not accepted()
        if drafting and not draft_noted:
            draft_noted = True
            messages.append({"role": "user", "content": DRAFT_PHASE_NOTE})
            trace.log("draft_phase", turn=turn, text=DRAFT_PHASE_NOTE)
        elif not drafting and draft_noted and not refine_noted:
            refine_noted = True
            messages.append({"role": "user", "content": REFINE_PHASE_NOTE})
            trace.log("refine_phase", turn=turn, text=REFINE_PHASE_NOTE)
        # The tool declarations are the front of every request, so a different set from one turn to the next means nothing after them is reused (the whole opened clip is billed again):
        # with stable_tools the set never changes and a phase restricts what may be CALLED instead (a call outside it gets a free refusal, see below); the legacy mode swaps the set.
        if stable_tools:
            turn_specs = specs
        elif phase == "final":
            turn_specs = governor.final_specs(specs)
        elif drafting:
            turn_specs = [s for s in specs if s["function"]["name"] not in MEASURE_TOOLS]
        else:
            turn_specs = specs
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
        response = client.complete(messages, tools=turn_specs)
        usage = response.get("usage") or {}
        if governor.draft_told and not accepted():
            governor.draft_turns += 1                    # a turn taken since the deadline note with nothing submitted
        try:                                    # what the provider could and could not reuse of this request, and why (cache_watch); it records, it never decides
            cache_row = cache_watch.observe(turn, request_fingerprint(client, messages, turn_specs), usage, getattr(client, "last_explicit", None))
            trace.log("cache", **{k: v for k, v in cache_row.items() if k != "turn"}, turn=turn)
            for cache_event in getattr(client, "drain_cache_events", lambda: [])():          # the client's explicit cache (gemini_cache): made, dropped, failed
                trace.log("explicit_cache", turn=turn, **cache_event)
        except Exception as exc:                # noqa: BLE001 - a measurement must never stop a run
            trace.log("cache_failed", turn=turn, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if focus_base is None:
            focus_base = last_prompt_tokens
        tokens["in"] += last_prompt_tokens
        tokens["out"] += int(usage.get("completion_tokens") or 0)
        tokens["cached"] += int(usage.get("cached_tokens") or 0)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        if any(c["function"]["name"] == "submit_dub" for c in calls):
            focus_base = None
        trace.log("model", turn=turn, finish=choice.get("finish_reason"), text=message.get("content"),
                  tool_calls=[{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
                  usage=usage, **({"reasoning": message["_reasoning"]} if message.get("_reasoning") else {}), **({"web": message["_web"]} if message.get("_web") else {}))
        if choice.get("finish_reason") in TRUNCATED_FINISHES and not calls:
            # The reply was cut off by the output limit: nothing in it was done (a cut-off tool call is no call).  That is not the agent stopping, so it must not be read as "finished": say so
            # once per cut and let it try again smaller; after MAX_CONSECUTIVE_TRUNCATIONS in a row the run ends with the reason (and the drafts are still submitted or kept below).
            truncations += 1
            trace.log("truncated", turn=turn, consecutive=truncations, completion_tokens=int(usage.get("completion_tokens") or 0))
            if truncations > MAX_CONSECUTIVE_TRUNCATIONS:
                stop_reason = "output_limit"
                break
            messages.append({"role": "user", "content": TRUNCATED_NOTE})
            continue
        truncations = 0
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
            if free_finish and not unmet and quality_rounds < MAX_QUALITY_ROUNDS and tokens["in"] + tokens["out"] < 0.9 * max_total_tokens:
                submitted = getattr(workbench, "last_submit", None)
                points = open_listener_points(submitted, limit=QUALITY_POINTS_SHOWN)
                if points and submitted is not quality_seen:          # a round that changed nothing is the agent's answer: it is not asked again
                    quality_rounds += 1
                    quality_seen = submitted
                    text = quality_pushback_message(points, quality_rounds)
                    messages.append({"role": "user", "content": text})
                    trace.log("quality_pushback", turn=turn, round=quality_rounds, points=points, agent_said=str(message.get("content") or "")[:600])
                    continue
            stop_reason = "finished_below_target" if unmet else "agent_finished"
            break

        for call in calls:
            name = call["function"]["name"]
            started = time.monotonic()
            if bool(breadth_first) and name in MEASURE_TOOLS and not accepted():          # not available in the draft phase; a model that calls it anyway is told why, and nothing is spent
                observation, ok = json.dumps({"error": "not_available_yet", "detail": DRAFT_REFUSAL}, ensure_ascii=False), False
            elif phase == "final" and name not in FINAL_TOOLS:                              # the final phase only submits (final_specs when the set is swapped; a refusal when it is kept stable)
                observation, ok = json.dumps({"error": "not_available_now", "detail": FINAL_REFUSAL}, ensure_ascii=False), False
            elif draft_deadline and governor.draft_restricted(submitted=accepted()) and name not in FINAL_TOOLS:          # the draft deadline note was sent and there is still no submission: only submitting
                observation, ok = json.dumps({"error": "not_available_yet", "detail": DRAFT_DEADLINE_REFUSAL}, ensure_ascii=False), False
            else:
                observation, ok = execute_tool(dispatch, name, call["function"]["arguments"])
            tool_counts[name] = tool_counts.get(name, 0) + 1
            ledger.set_segments(getattr(workbench, "segments", None))
            ledger.observe(name, call["function"]["arguments"], observation)          # also a failed call: the draft it carried is still the agent's latest
            trace.log("tool", turn=turn, name=name, ok=ok, seconds=round(time.monotonic() - started, 2),
                      result=observation[:3000])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})

        latest = getattr(workbench, "last_submit", None)
        if latest is not None and latest is not recorded_submit:
            recorded_submit = latest
            last_items = list(getattr(workbench, "last_items", None) or [])
            if best.observe(turn, latest, last_items, _unit_english(workbench, last_items)):
                trace.log("best_submission", turn=turn, key=best.best["key"])
        if free_finish and latest is not None and latest is not seen_submit:       # quality first: tell the agent the bar is met and what is open; never stop it for it
            seen_submit = latest
            if not unmet_targets(latest, max_dead_air_pct=good_enough_pct):
                new_points = [x for x in open_listener_points(latest) if x not in points_sent]
                if not told_bar:                                                    # told once: the soft points are not a reason to talk to the agent again after every submit
                    residual = (latest.get("headline") or {}).get("residual_dead_air_pct")
                    text = free_bar_message(residual, new_points, first=not told_bar)
                    points_sent.update(new_points)
                    messages.append({"role": "user", "content": text})
                    trace.log("bar_reached", turn=turn, first=not told_bar, points=new_points, text=text)
                    told_bar = True
        if wrapup_turns and not free_finish and latest is not None and latest is not seen_submit:       # a new accepted submission: has it met the bar?
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

    if not accepted() and stop_reason in ("token_budget", "max_turns", "finished_without_submission", "output_limit"):
        outcome = governor.auto_submit(workbench, ledger)          # the run is ending with nothing submitted: the agent's own drafts are submitted when they cover the clip
        trace.log("auto_submit", **outcome)
        if not outcome.get("submitted"):
            try:                                                   # not lost either way: the drafts are kept beside the trace
                (directory / "unsubmitted_drafts.json").write_text(json.dumps({"why": outcome, "drafts": ledger.to_dict()}, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError:
                pass

    explicit_cache = None
    try:                                                           # release the client's explicit cache and say what it cost (creation is billed once, outside the token ceiling)
        explicit_cache = getattr(client, "cache_report", lambda: None)()
        getattr(client, "close", lambda: None)()
    except Exception:                                              # noqa: BLE001 - the cache expires by itself
        pass
    summary = {
        **({"explicit_cache": explicit_cache} if explicit_cache else {}),
        "mode": "agent", "clip_id": clip_id, "run_id": run_id, "stop_reason": stop_reason, "turns": turn,
        "budget": governor.report(), "best_submission": best.report(), "cache": cache_watch.report(), "stable_tools": bool(stable_tools), "draft_deadline": bool(draft_deadline),
        "tokens": tokens, "tool_calls": tool_counts, "speed_policy": speed_policy,
        "pushbacks": pushbacks, "unmet_at_end": unmet, "compactions": compactions,
        "playbook": playbook_version,
        "grammar": _grammar_version_in(system_prompt), "system_prompt_chars": len(system_prompt),
        **({"judge_tokens": dict(workbench.judge.usage)} if getattr(workbench, "judge", None) else {}),
        "tts_used": (workbench.episode or {}).get("tts_used"), "submissions": (workbench.episode or {}).get("submissions"),
        "model": client.config.deployment, "voice_provider": current_voice_provider(),
    }
    if best.best and not summary["best_submission"].get("final_is_best", True):          # the run ended on a submission that ranks below an earlier one: keep that one beside the trace
        try:
            (directory / "best_submission.json").write_text(json.dumps({"turn": best.best["turn"], "key": best.best["key"], "headline": best.best["headline"], "items": best.best["items"],
                                                                         "unit_texts": best.best.get("unit_texts") or {}},
                                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
        trace.log("best_not_final", best_turn=best.best["turn"], final_turn=summary["best_submission"]["final_turn"])
    checkpoint(turn, "finished" if stop_reason != "paused" else "paused")
    summary["agent_mode"] = mode
    summary["free_finish"] = free_finish
    if resumed_from is not None:
        summary["resumed_from_turn"] = resumed_from
    if stop_reason == "paused":
        summary["resume"] = f"python -m mathula_tv.dub_production --job <job> --run-id {run_id} --resume"
    trace.log("end", **summary)
    (directory / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if mode == "training":
        try:
            from .thinking_report import write_report

            write_report(directory)
        except Exception as exc:                                             # noqa: BLE001 - the report must never turn a finished run into a failure
            trace.log("thinking_report_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
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
    settings = load_settings()
    config = native_client_config()
    judge = None
    if not args.no_judge and args.mode == "agent":
        from .dub_judge import UnitJudge

        judge = UnitJudge(new_chat_client(config))
    workbench = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings), judge=judge)
    if args.mode == "oneshot":
        client: Any = new_chat_client(config)
        summary = run_one_shot(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                               tts_budget=args.tts_budget, speed_policy=args.speed_policy)
    else:
        client = new_agent_client(config) if args.api == "responses" else new_chat_client(config)
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
