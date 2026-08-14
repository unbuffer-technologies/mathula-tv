# Mathula TV v13.18.29 — Collision-Free Final Tail and Calibrated Pauses

This cumulative code patch includes every upgrade through v13.18.29.

## Exact-job diagnosis

- Job `6f410d500d9e4e7ca07653550e3c883e` reached 96.53/100 under
  v13.18.28 with four production blockers.
- Blocks 0049 and 0056 were natural clause/date breaths of 1.52 s and 1.36 s.
  They exceeded Azure's punctuation alignment by only 220 ms and 460 ms, but
  the production gate treated them like genuine unplanned stoppages.
- Blocks 0061 and 0062 had 288 ms and 1,000 ms of real cross-speaker overlap.
  The overlap solver used that compromise because the last speech block ended
  at 805.320 s, while the actual source video continued to 807.427 s.
- The unused 2.107-second source tail was never offered to the timeline solver.

## v13.18.29 recovery

- Exposes up to five seconds of unused source-video tail as timing capacity for
  the last speech block without extending or re-encoding the source video.
- Replaying the supplied final triplet yields sequential windows
  `797810–803390`, `803390–805220`, and `805220–807427`: zero overlap and
  736 ms spare under the natural base-rate edge measurement.
- Keeps small pause-boundary disagreements as quality warnings, while a
  production pause now requires both more than 1.5 seconds and at least 250 ms
  beyond its context-specific allowance. A genuine unplanned two-second pause
  still blocks production.
- Carries safe v5 prepared-variant overrides into policy v6, so block 0059's
  successful concise identity repair is retained. Renderer-local AI variants
  are never carried forward.

# Mathula TV v13.18.28 — Stable Final-Three Mastering Recovery

This cumulative code patch includes every upgrade through v13.18.28.

## Exact-job diagnosis

- Job `6f410d500d9e4e7ca07653550e3c883e` reached 96.47/100 in round 1
  with only three blockers, then regressed to 95.03/100 in round 2.
- The round-2 planner treated renderer-local `ai_balanced_r1` as though it were
  a prepared variant. A request for shorter speech therefore selected
  `natural`, the richest prepared variant, and destabilised downstream STT.
- Blocks 0016 and 0059 were slowed from their natural rates to -12% merely to
  consume spare timeline room. That created both remaining same-speaker tempo
  jumps.
- The controlled-overlap solver correctly expanded block 0061 from 980 ms to
  1,830 ms, but the source-word alignment pass then collapsed both overlap
  boundaries and removed the measured capacity it had just created.

## v13.18.28 recovery

- Never guesses the duration rank of an AI-finalized local variant. If the
  current variant is not on the approved natural/concise/compact ladder, the
  mastering loop retains the best-known audio instead of making a speculative
  replacement.
- When base-rate TTS already fits, rejects Azure slowdown greater than five
  percentage points. Legitimate transition room remains silence instead of
  dragging the speaker's voice to fill the entire window.
- Preserves controlled source-like overlap up to one second when every affected
  block retains its Azure-measured required capacity and boundary movement stays
  inside the existing two-second safety bound.
- Rebases prior mastering overrides to policy v5, archives them under
  `direct_dub/mastering/policy_migrations`, and reruns from the unchanged
  approved translation baseline.
- Adds exact regressions for the supplied block 0059/0060 cadence measurements,
  the block 0060 AI-variant planner failure, and block 0061's 288 ms/1,000 ms
  controlled-overlap geometry.

# Mathula TV v13.18.27 — Five-Block Capacity and Final-Conductor Recovery

This cumulative code patch includes every upgrade through v13.18.27.

## Exact-job diagnosis

- Round 1 of job `6f410d500d9e4e7ca07653550e3c883e` reached 95.83/100 but
  retained five tempo blockers.
- The five-block solver handled only the current overflow. When an adjacent
  turn also overflowed, its candidate was rejected for leaving that neighbour
  below its measured minimum—even though the five-block window had ample
  capacity.
- The overlap solver then replaced some already-resolved non-overlapping
  geometry, and the source-word clamp restored the original narrow boundaries.
- The final conductor rejected meaning-preserving compression notes because
  their free-form descriptions did not exactly equal optional-detail labels.
- A safe-silence expansion was discarded when the code entered final fallback.

## v13.18.27 recovery

- Resolves every adjacent measured deficit in one five-block allocation.
- Routes only still-unresolved interjections into the controlled-overlap solver.
- Preserves a non-overlapping measured fit when its boundary movement is within
  two seconds of the source-word geometry and every affected block retains its
  required Azure-measured window.
- Keeps collision-free safe-silence expansion through AI finalization and
  emergency fitting.
- Accepts audited delivery compression while retaining hard rejection for lost
  protected identities or uncertified numbers, dates, and negation.
- Replaying the supplied round-1 geometry gives all five blockers windows at
  or above their 1.08× production limits, including the final 920 ms reply,
  which receives 1,745 ms without speaker overlap.
- Rebases v3 mastering choices to policy v4 so the repaired renderer runs once
  without manual deletion or `--force`.

# Mathula TV v13.18.26 — Mastering Repair Execution and No-Change Guard

This cumulative patch includes every upgrade through v13.18.26.

## Exact-job diagnosis

- Job `6f410d500d9e4e7ca07653550e3c883e` produced the same final-audio SHA-256
  in rounds 0, 1, and 2 because a mastering override was treated as an initial
  selection hint. The ordinary predicted-fit selector immediately chose the
  former variant again.
- The loop then retranscribed the unchanged checksum from cache and proposed
  the same repair in the next round. No TTS or final audio changed.
- `block_0055` also offered natural/concise/compact variants whose hidden
  `tts_text` was identical, so changing its display variant could never change
  the synthesized audio.

## v13.18.26 recovery

- Marks a mastering variant as an explicit render lock. The renderer must test
  that approved variant while still retaining the full variant set for future
  measured rounds.
- Records the post-repair audio checksum. If a repair render is byte-identical,
  the loop stops before another Azure STT request, records
  `repair_render_produced_identical_audio`, and restores the best override set.
- Never repeats an already-installed identical override.
- Stores per-variant TTS fingerprints in the audit and skips variants whose
  actual TTS input is audio-equivalent to the selected variant.
- Keeps exact WER for diagnostics but scores intelligibility with a bounded
  token-boundary-tolerant WER. Native isiZulu forms such as
  `mhla ziyishumi`/`mhlaziyishumi` and close Azure orthography no longer create
  false failures.
- Replaying the supplied baseline STT reduces aggregate WER from 17.8% to
  10.7%, raises the score from 89.55 to 92.74, removes the false block-0055
  failure, and leaves seven genuine local blockers visible.
- Automatically archives v2 mastering overrides and starts from an empty v3
  policy baseline. Translation, speaker identity, source timing, cached TTS,
  and diagnostic history are preserved.

# Mathula TV v13.18.25 — Calibrated STT Mastering and Reversible Repairs

This cumulative patch includes every upgrade through v13.18.25.

## Exact-job diagnosis

- Replayed all three Azure STT rounds supplied for job
  `6f410d500d9e4e7ca07653550e3c883e`.
- The best v1 round had aggregate WER 14.7% and zero collisions; it was not a
  43%-intelligible dub.
- Its 43.9 score came from double-counting maximum penalties for 25 supposedly
  missing identities, 13 pause observations, and nine tempo warnings.
- The exact identity matcher rejected `Adams` because the canonical protected
  value was `Fadiel Adams`; it also expected the translated target speech to say
  `Madlanga Commission of Inquiry` literally.
- Those false failures selected richer variants. Block 1 moved to natural and
  then required 1.238x time compression, turning a scoring error into an audible
  cadence regression.

## v13.18.25 calibration and recovery

- Uses the better of the immutable spoken reference and actual hidden TTS
  reference for WER, with both values retained in every block audit.
- Protected identities are assessed only through surface tokens present in the
  attempted speech. Matching is Unicode-normalized and conservatively fuzzy for
  Azure spellings such as Chaskalson/Chascarsen.
- Identity values with no attempted canonical surface are recorded as
  `protected_entities_not_assessable`, never counted as missing.
- Global monotonic edit alignment partitions the one Azure STT stream across
  ordered blocks. A 600 ms timing sanity pass corrects repeated-word alignments
  that land several seconds outside their rendered block.
- Pause analysis is suppressed when STT coverage is too weak to distinguish a
  real silence from unrecognized words. Authored cadence boundaries receive
  separate clause and sentence allowances.
- The score is now a transparent weighted combination of intelligibility,
  protected-identity recognition, pause naturalness, tempo naturalness, and
  collision safety. Severe local blockers remain a separate hard production
  gate.
- Exact replay of the supplied best render yields 92.5/100: intelligibility
  86.4, identity recognition 100, pause naturalness 99.5, tempo naturalness
  93.5, and collision safety 100. It remains `best_effort_completed` because
  eight concrete local defects still require repair.
- Mastering selections retain the full approved variant set and use one stable
  source block ID, so later rounds can reverse a bad choice without duplicated
  `_variant_` suffixes.
- Policy-v1 overrides are archived automatically and rebased to an empty v2
  baseline on the first non-audit run. Translation, speaker assignments, source
  boundaries, TTS caches, and prior diagnostic evidence remain intact.

# Mathula TV v13.18.24 — Azure STT Dub Mastering Loop

This cumulative patch includes every upgrade through v13.18.24.

## v13.18.24 production feedback and bounded repair

- Adds `optimize-dub`, a post-render production loop that transcribes the final
  dubbed mix with Azure Speech and aligns word-level output against the exact
  approved block text.
- Scores aggregate and per-block word error, protected-identity recognition,
  unexpected internal pauses, local speaking-rate variation, adjacent
  same-speaker tempo jumps, time-stretch/rate bounds, and cross-speaker overlap.
- Distinguishes intended punctuation pauses from unnatural silence. Clause
  punctuation receives a wider allowance and sentence punctuation a wider one
  again, avoiding false repairs of normal South African speech cadence.
- A repair can select only a prepared meaning-preserving natural, concise, or
  compact variant. It cannot invent dialogue, alter source timestamps, change
  speakers, or mutate the authoritative translation.
- Only one repair is allowed in each five-block neighbourhood per round. This
  gives the existing five-block conductor room to rebalance boundaries without
  creating an alternating fast/slow cadence across neighbouring blocks.
- Azure TTS remeasurement, bounded rate control, the five-block conductor,
  validated AI local finalization, and FFmpeg rendering remain the execution
  tools. The mastering policy chooses when to use them; it does not bypass
  their safety checks.
- Azure STT results are content-addressed by final-mix SHA-256. Unchanged reruns
  reuse the transcript. Large masters are downmixed to timeline-preserving mono
  16 kHz PCM for one full-program request.
- Every round stores its audit and repair plan. If a later iteration scores
  worse, the best override set is restored and rerendered.
- The loop is nonblocking: it reports `production_ready` when all thresholds
  pass or `best_effort_completed` when the bounded safe options are exhausted.
  It never opens the manual collision panel.
- Publication editing is deliberately outside the loop. After mastering,
  ordinary `dub-azure` reuses the clean mastered output and performs the
  publication render once.

# Mathula TV v13.18.23 — Native isiZulu Honorific Flow

This cumulative patch includes every upgrade through v13.18.23.

## v13.18.23 fused-honorific pronunciation repair

- Replayed job `6f410d500d9e4e7ca07653550e3c883e` and traced publication
  time 01:24 to `unit_0022` in `block_0007`.
- The approved line is `Ngakho sikhulumile noMnu. Adams...`. There is no comma,
  authored pause, speaker boundary, or unit boundary after `no`.
- Azure was interpreting the full stop in the fused isiZulu abbreviation
  `noMnu.` as a sentence boundary, producing the audible hesitation.
- A deterministic isiZulu-only hidden-TTS rule expands fused honorifics before
  SSML generation: `noMnu.` becomes `noMnumzane`, `uMnu.` becomes
  `uMnumzane`, `uNkk.` becomes `uNkosikazi`, `uNksz.` becomes `uNkosazana`,
  and `uDkt.` becomes `uDokotela`.
- Approved `spoken_text`, visible captions, protected terms, and translation
  artifacts remain unchanged. Every replacement is auditable as an
  application-default pronunciation substitution.
- Explicit reviewed job pronunciation entries remain authoritative. The rule
  does not run for non-isiZulu locales.
- The direct-dub schema advances to v17 so an old completed report cannot hide
  the fix. Changed TTS text creates a new Azure request hash only for affected
  blocks; unaffected content-addressed TTS remains reusable.
- Exact-job replay confirms that all variants of `unit_0022` and the coalesced
  `block_0007` now send `noMnumzane Adams` to Azure while retaining `noMnu.
  Adams` in the caption.

# Mathula TV v13.18.22 — Native isiZulu Calendar Pronunciation

This cumulative patch includes every upgrade through v13.18.22.

## v13.18.22 application-controlled date readings

- Diagnosed the exact variants in job
  `6f410d500d9e4e7ca07653550e3c883e`. Translation produced hybrid calendar
  forms including `ngomhla ka-19 Agasti`, `ngumhla ka-19 Agasti`, bare
  `19 Agasti`, and `ngomhla ka-25 Okthoba`.
- The pronunciation dictionary previously exposed date APIs but supplied no
  default date rules, so every hybrid and bare digit reached Azure unchanged.
- Government isiZulu usage establishes forms such as `mhla ziyi-19 kuNcwaba`.
  Azure documentation limits general date `say-as` support to a language list
  that excludes isiZulu, so unsupported SSML is not emitted.
- A deterministic locale-specific recognizer now handles days 1–31, optional
  years, English months, AI-transliterated months, and indigenous isiZulu month
  names.
- Hidden TTS expands the day with either the `mhla` subject concord or the
  `umhla` ordinal construction and uses the indigenous month. Approved spoken
  text, subtitles, translation JSON, protected spans, and source meaning remain
  unchanged.
- The rule is limited to complete day-plus-month spans. Month-only references
  and all non-isiZulu target locales are untouched.
- Explicit reviewed dictionary entries remain more authoritative than the
  fallback. Every automatic change is audited as `kind=date`, source
  `application_default`, with a versioned policy marker.
- Direct dubbing always reconstructs all prepared variants from immutable
  spoken text, so legacy model-supplied `tts_text` cannot bypass the rule.
- The direct-dub schema/fingerprint advances to v16. Completed old reports are
  not silently reused, while unaffected content-addressed Azure TTS blocks can
  still reuse cache.
- Exact-job replay certifies native readings for 19 August and 25 October
  across natural, concise, and compact variants.

# Mathula TV v13.18.21 — Contextual Commission Turn Repair

This cumulative patch includes every upgrade through v13.18.21.

## v13.18.21 role-aware short-turn correction

- Diagnosed job `6f410d500d9e4e7ca07653550e3c883e` from its exact
  transcript, translations, and collision package.
- Azure attributed `19 August, Chair.` and `Thank you, Chair.` to Madlanga's
  raw speaker even though both are Chaskalson's replies. The former was also
  split into separate `19` and `August Chair` translation units.
- Reconciliation now groups words by Azure source-phrase provenance, learns
  the chair from repeated clean adjacent vocatives, and repairs only short
  phrases where the inferred chair would otherwise address themself.
- A replacement respondent must already have clean `Chair`-vocative evidence
  and appear within a bounded context window. One isolated vocative is never
  enough to override Azure diarization.
- The complete phrase is reassigned before semantic unit construction. A
  bounded 2.5-second within-phrase pause allowance keeps `19 August, Chair`
  together while still preserving normal long-pause boundaries elsewhere.
- Every repair records version, source phrase, text, original and effective
  speaker, evidence counts, distance, and reason. Azure's raw speaker remains
  immutable provenance on the segment and words.
- Acoustic mapping ignores context-repaired exceptions so one overridden
  phrase cannot redefine the job-wide raw Azure speaker identity.
- New command `repair-speaker-turns` rebuilds an existing job from saved Azure
  transcription without retranscription, reruns authoritative autocorrection,
  and reports every changed phrase before retranslation.
- Exact-job replay certifies Madlanga → Chaskalson → Madlanga turn order and
  separates Chaskalson's closing thanks from Madlanga's adjournment.

# Mathula TV v13.18.20 — isiZulu “Coloured” Code-Switch Pronunciation

This cumulative patch includes every upgrade through v13.18.20.

## v13.18.20 calibrated hidden TTS pronunciation

- Fixed the protected phrase `National Coloured Congress`, where Azure's
  standard isiZulu voice pronounced the English word `Coloured` using inferred
  isiZulu phonetics.
- Microsoft documents `<lang xml:lang>` as a multilingual-voice feature and
  explicitly says non-multilingual voices do not support it. Themba and Thando
  are listed as standard isiZulu voices, so the patch does not emit unsupported
  SSML or silently switch speaker timbre to a separate English voice.
- The application pronunciation dictionary now keeps the approved, captioned,
  and identity-validated word `Coloured` while sending hidden TTS alias
  `Khalad` to the isiZulu voice.
- The entry matches `coloured`, `COLOURED`, title case, and the US alias
  `colored`, and is audited as `english_code_switch`.
- Existing code-switch place-name entries now carry explicit kinds and aliases
  through one typed registry shape; their prior output remains unchanged.
- Existing saved job dictionaries are augmented deterministically. The direct
  dub rebuilds every variant's TTS text from its immutable spoken text, so old
  model-supplied TTS text cannot bypass the update.
- Azure candidate paths are content-addressed from the changed request. Only
  blocks containing the term are synthesized again; unaffected cached blocks
  and paid AI checkpoints remain reusable.
- Tests certify visible/protected-text immutability, hidden TTS output, casing,
  spelling aliases, saved-dictionary upgrades, direct-dub integration, and
  isolation from non-isiZulu locales.

# Mathula TV v13.18.19 — Reporter Attribution and Source-Word Alignment

## v13.18.19 transcript-backed voice resolution

- Diagnosed the exact failure from job `7f4d9024e2444cc385b913048494a3c9`.
  `SPEAKER_01` measured 50.6985% masculine and 49.3015% feminine, so lowering
  the 70% acoustic threshold would turn a coin flip into a false certainty.
- The reporter narration immediately before the quoted speaker says
  `Katalia says ...`. The audited contextual registry already contains
  `Firoz Cachalia`, male. The surname similarity is exactly 0.666667.
- The resolver now recognizes common news attribution verbs (`says`, `said`,
  `adds`, `explains`, and `states`) only when one surname directly precedes the
  verb and exactly one audited registry identity matches at 0.65 or better.
- Registry confidence must still be at least 0.95, prior evidence must exist,
  and gender must already be audited. General mentions, distant names,
  one-word identities, unaudited candidates, and ambiguous surnames do not bind.
- The audit records the observed and canonical surnames, attribution verb,
  similarity, registry gender, boundary direction, and identity source.

## v13.18.19 source-word speaker alignment

- Cross-speaker boundaries are derived from canonical transcript segment word
  IDs, not raw Azure speaker labels. Each handoff records the outgoing
  speaker's final source word and incoming speaker's first source word.
- A 250 ms envelope protects both word edges. Close handoffs share one boundary;
  real silence remains silence; genuine source overlap remains representable.
- The five-block conductor retains broad freedom at same-speaker boundaries and
  in source silence. A final clamp removes any post-fit drift into another
  speaker's source-word activity before paid block finalization.
- Dynamic handoff repair is also capped by the active word anchor. Sandwiched
  interjection expansion cannot bypass a canonical word boundary.
- Capacity reclaimed by alignment routes to the existing validated local AI
  finalizer and, if necessary, the nonblocking final-product cadence fit.
- `direct_dub/source_word_alignment.json`, progress events, block audits, and the
  final report expose anchors, corrections, tolerances, source words, and policy.

# Mathula TV v13.18.18 — Five-Block Final Product Conductor

## v13.18.18 nonblocking five-block collision management

- Diagnosed the exact collision from job `6f410d500d9e4e7ca07653550e3c883e`.
  `block_0051_variant_compact` measured 21,352 ms in a 17,840 ms source
  window and needed 20,731 ms under the ordinary 1.03 cadence policy.
- The old solver inspected only `block_0050`, `block_0051`, and `block_0052`.
  It could not see 1,580 ms of safe source silence after `block_0049`, so it
  incorrectly escalated a solvable case to the manual panel.
- The new preflight conductor evaluates five blocks: two preceding, the
  collision, and two following. It keeps the outer boundaries fixed, consumes
  source silence first, then borrows only Azure-measured surplus above each
  neighbour's minimum duration.
- On the supplied job it releases 1,580 ms of silence, 144 ms from
  `block_0050`, and 1,167 ms from `block_0052`. The failed block receives the
  exact 20,731 ms it needs, while all five blocks retain their required
  measured windows and speaker order.
- Every five-block decision records original and adjusted boundaries, required
  windows, capacity sources, affected block IDs, invariant checks, and whether
  human review was required.
- The finalization AI's three shorter candidates were not actually unsafe.
  They retained the approved isiZulu identity `IKhomishani`, but the local
  finalizer required the literal English `Madlanga Commission of Inquiry`.
  Candidate validation now reuses the translation identity/alias contract and
  recognizes normal isiZulu attached prefixes such as `eKhomishani` and
  `we-AfriForum`.
- Existing v13.18.13 finalization checkpoints are migrated in place, preserving
  paid Claude responses and Azure measurements.
- A final-product conductor now guarantees unattended completion after every
  higher-quality path is exhausted. It applies the exact measured FFmpeg
  cadence needed to keep immutable text within its block, records the quality
  compromise, and continues. It never calls the blocking collision panel.
- The report exposes five-block fit counts, emergency-fit counts, full audits,
  solver versions, the final cadence ceiling, and
  `manual_review_blocking_enabled: false`.

# Mathula TV v13.18.17 — Audited Registry Speaker Rebinding

This is a cumulative patch. It includes the AI-consumption telemetry, unified
AI/FFmpeg progress, phase timing, targeted validation repair, Azure autocorrect
schema resilience, and checkpoint-resume work from v13.16.4 through v13.17.1.

## v13.18.17 audited registry speaker rebinding

- Fixed the exact `dub-azure` voice-resolution failure from job
  `ccc4add3e5844ba4b067f69ae3b103e8` for `SPEAKER_01` / `AZURE_2`.
- The acoustic classifier returned only 66.2496% masculine, below the existing
  70% safety threshold. The threshold remains unchanged and that uncertain
  acoustic result is not promoted into a voice assignment.
- The authoritative transcript contains three complementary identity signals:
  the studio says “The minister is now speaking,” the speaker says “as a police
  minister,” and the closing presenter identifies the acting police minister as
  the STT-damaged `Feroz Kucalia`.
- The previously audited cross-job identity registry already contains
  `Firoz Cachalia`, male, with 0.99 confidence from an earlier explicit
  broadcast handoff. The resolver formerly used this registry only to recover
  gender after an exact identity match; it did not use audited identities as
  candidates for a new contextual rebinding.
- Strong registry identities are now eligible candidates only when confidence
  is at least 0.95, at least one prior audit exists, and gender is already
  explicit. Weak or incomplete registry records remain excluded.
- An STT-damaged registry name may bind only beside an official title, inside
  an explicit presenter introduction/outro, with every name token clearing a
  0.65 similarity floor, the complete name clearing 0.72, and exactly one
  registry identity matching. General story mentions still cannot bind a
  speaker.
- Presenter context can span up to six non-target chatter/thanks segments, but
  remains bounded to 30 seconds and stops if the target speaker returns. This
  captures the supplied closing identification without searching arbitrary
  transcript content.
- The outro grammar now recognizes “you've just been listening to.”
- The exact supplied job resolves `Feroz Kucalia` to audited canonical identity
  `Firoz Cachalia`, masculine voice family, and `zu-ZA-ThembaNeural` without a
  manual voice map.
- The audit records the observed STT name, canonical identity, official title,
  whole-name and per-token similarity, speaker role evidence, extended boundary
  segments, and the registry-backed gender source.

## v13.18.16 compact punctuated-key recovery

- Fixed the exact Chunk 11/15 failure:
  `compact translation unit unit_0191 variant natural contains unexpected fields: ,ms`.
- `ms` is already a legal compact variant property. The error therefore means
  Claude emitted the valid duration value under the punctuation-damaged JSON
  key `",ms"`.
- The compact-wire parser now normalizes leading/trailing punctuation or
  whitespace only when the damaged key maps exactly to one of the five compact
  data keys: `t`, `ms`, `p`, `o`, or `ok`.
- If both damaged and canonical keys are present with identical values, the
  duplicate is removed and audited. If their values differ, validation still
  fails closed instead of guessing which value is authoritative.
- All unrelated or substantive unexpected fields remain forbidden. This is not
  a general `additionalProperties` bypass.
- Every local repair is hoisted into the top-level warnings ledger before the
  strict compact schema is applied.
- The failed provider response already persisted in the chunk's `failure.json`
  or SSE partial-response checkpoint can be reparsed and validated locally.
  Resuming therefore reuses the paid Chunk 11 response with zero new provider
  calls when that saved response is complete.
- The 10 completed chunk checkpoints remain untouched and reusable.

## v13.18.15 contextual titled-guest voice resolution

- Fixed the exact `dub-azure` failure from job
  `ce9bbbb623b34872abae0428f004ae2e`: voice assignment remained unresolved for
  `SPEAKER_02` / `AZURE_3`.
- The acoustic classifier was only 69.7502% masculine, just below the existing
  70% safety threshold. The threshold remains unchanged and the uncertain
  acoustic result is not promoted into a voice assignment.
- The adjacent authoritative transcript explicitly says, “we are joined now
  by the uncle of Ayabonga Gaka, Mr. Bongile Gaka,” immediately before
  `SPEAKER_02` begins. The resolver now recognizes the common “joined now by”
  presenter cue.
- When one presenter handoff mentions multiple researched people, a uniquely
  gender-titled person (`Mr`, `Ms`, `Mrs`, `Miss`) is selected as the guest.
  This prevents the deceased person mentioned in the same sentence from being
  mistaken for the interviewee.
- Rejected correction records may contribute only their original multiword
  transcript aliases to this uniquely titled handoff rule. The rejected
  canonical proposal `Bongile Gaqa` is never applied; the transcript identity
  remains `Bongile Gaka`.
- A rejected-correction transcript alias cannot bind from a general story
  mention, cannot bind without an explicit presenter cue, and is never written
  to the cross-job contextual identity registry.
- The exact supplied job now resolves `SPEAKER_02` to `Bongile Gaka`, masculine
  voice family, and `zu-ZA-ThembaNeural` without another acoustic classifier
  pass and without a manual `--voice-map`.
- `analysis/contextual_speaker_identities.json` records candidate provenance,
  the rejected canonical proposal, the unique titled-handoff binding mode, and
  registry eligibility for auditability.

## v13.18.14 rendered editorial block ID recovery

- Fixed the confirmed publication failure where Claude selected
  `block_0005`, while the direct-dub report contained the same source block as
  `block_0005_variant_natural`.
- Direct Azure dubbing appends `_variant_natural`, `_variant_concise`,
  `_variant_compact`, or an adaptive/AI delivery suffix after selecting the
  rendered wording. Claude can correctly identify the underlying timeline
  block but omit this mechanical renderer suffix.
- Editorial validation now canonicalizes a source-style block ID to the exact
  rendered ID only when one and only one rendered block owns that source ID.
- The recovery covers `opening_boundary_block_id`,
  `opening_payoff_block_id`, every candidate `selected_block_id`, and every
  `evidence_block_ids` entry before grounding, question-anchor, and payoff
  validation.
- Exact rendered IDs remain unchanged. Invented IDs and ambiguous mappings are
  never guessed and continue to fail the existing strict validation.
- The selection artifact preserves the original AI-requested opening and
  payoff IDs and records every deterministic rewrite in
  `rendered_block_id_canonicalization`.
- The editorial prompt and request schema are unchanged. The response already
  saved at `direct_dub/editorial_response_checkpoint.json` is reusable; the
  resumed job performs local validation without another Claude request.

## v13.18.13 non-blocking finalization conductor

## v13.18.13 non-blocking finalization conductor

- Fixed the exact final collision from job
  `de4edf2950444f5ca0b277b5a6954e27`. The renderer measured all three prepared
  variants correctly but retained the last failed variant (`natural`, 7,789ms)
  instead of the shortest measured variant (`compact`, 6,289ms). That inflated
  the apparent shortfall and incorrectly opened the review panel.
- Exhausted variant selection now retains the shortest Azure-measured candidate,
  independent of semantic candidate iteration order.
- Timeline-edge planning now uses the same bounded 1.08 cadence ceiling that
  final edge synthesis is already permitted to use. The supplied final block
  therefore needs `ceil(6289/1.08) = 5,824ms`, only 397ms more than its 5,427ms
  source window.
- The preceding block has enough Azure-measured surplus. The deterministic edge
  solver moves the shared boundary 397ms earlier, leaving block 14 with 33,683ms
  and giving block 15 exactly 5,824ms. Text, identities, speaker order, source
  start, and source end remain unchanged. This exact job requires no AI call.
- A failed geometric preflight no longer pauses immediately. It is passed to the
  final block loop, where every deterministic fit remains available before AI.
- If those fits are genuinely exhausted, a lazy finalization conductor sends
  only the failing speech block to Claude. It requests exactly three distinct
  duration-aware candidates and never resends the full transcript.
- The server rejects candidates that change the block ID, lose protected names
  or institutions, fail numbers/dates/negation certification, contain blank
  speech, or declare omissions outside the translation's approved optional
  details. The archival translation JSON is never rewritten.
- Every accepted AI candidate is synthesized and measured with the exact Azure
  voice and prosody. A candidate must fit the hard window and pass cadence QC.
  If all three overflow, one final round receives the real Azure measurements
  and proportionally compresses the shortest attempt.
- Claude responses and Azure candidate measurements are checkpointed in
  `direct_dub/finalization_conductor.json`. A retry reuses the paid response.
- `MATHULA_TV_FINALIZATION_AI_ROUNDS` controls the bounded AI budget (default 2,
  accepted range 0-3). Normal runs and the supplied collision do not initialize
  or call the AI provider.
- Progress now exposes a `finalization-ai` phase only when the fallback is
  actually needed. The final report records the selected delivery overlay,
  attempts, checkpoint, and AI timing-repair count.

## v13.18.12 contextual speaker identity resolution

## v13.18.12 contextual speaker identity resolution

- Fixed the exact `dub-azure` voice-resolution failure from job
  `7ff71918e76f4046beb169f06eb2bde3`: `Voice family is unresolved for
  SPEAKER_01 (raw: AZURE_2)`.
- The acoustic model was genuinely inconclusive for `SPEAKER_01`: 52.21%
  masculine versus 47.79% feminine. The confidence threshold is deliberately
  unchanged; lowering it would turn a near coin flip into a false certainty.
- The authoritative transcript provides decisive broadcast context. The
  presenter introduces the next feed as acting police minister Firoz Cachalia
  and says he is briefing the media; after `SPEAKER_01` finishes, the presenter
  says, “There you have it. That's the acting police minister, Firoz Cachalia.”
  Both adjacent references use an explicit masculine pronoun.
- Existing name research independently canonicalized the person to
  `Firoz Cachalia` with 0.99 confidence and grounded same-story sources. The
  new resolver combines that canonical person with the two-sided presenter
  handoff and local pronoun evidence.
- `SPEAKER_01` therefore resolves to `Firoz Cachalia`, masculine voice family,
  and `zu-ZA-ThembaNeural`. This is an identity-backed choice, not the removed
  unknown-to-Themba default.
- Contextual resolution runs before an inconclusive acoustic retry, avoiding
  another identical classifier pass. Verified known-speaker voiceprints and
  explicit voice maps remain higher-priority evidence.
- General mentions of a person's name elsewhere in a story never bind that
  person to a diarized speaker. Resolution requires a researched canonical
  person plus an adjacent presenter introduction/outro or the same person on
  both sides of one contiguous speaker run.
- Gender is never guessed from a name. It requires an explicit nearby
  pronoun/title or a previously audited contextual identity for the exact same
  canonical person. Conflicts and ambiguity continue to fail closed.
- Every job writes `analysis/contextual_speaker_identities.json`. Strong
  explicit-pronoun resolutions also update the separate audited
  `working/contextual_speaker_identities/registry.json`, allowing later clips
  to reuse gender only after they independently rebind the same person to the
  new diarized speaker through an explicit broadcast handoff.
- Voice-resolution progress now prints the contextual identity name, and the
  resolved assignment remains cached for normal retries. No AI or web request
  is added to `dub-azure`.

## v13.18.11 final translation tail recovery

- Fixed the exact `dub-azure` preflight failure from job
  `de4edf2950444f5ca0b277b5a6954e27`: `unit_0074 ends beyond the source
  video`.
- The diagnostic bundle proves `unit_0074` is the final speakable unit. Its
  Azure transcript boundary is 445,320 ms and the authoritative source-media
  boundary is 445,057 ms: only 263 ms of drift, or 4.62% of the 5,690 ms
  phrase window.
- The former fixed guard accepted 250 ms, so this harmless case terminated the
  dub by only 13 ms. The loader now permits bounded drift on the final
  speakable unit when the overrun is at most 500 ms and no more than 10% of the
  phrase window.
- Interior units, starts outside the media, overruns above 500 ms, and
  truncations above 10% still fail closed. Trailing punctuation-only Azure
  pseudo-units do not prevent the preceding speech unit from being recognized
  as the final speakable unit.
- Accepted drift is clamped to the exact source boundary before coalescing,
  Azure TTS, mixing, or rendering. The approved translation JSON and its text
  remain immutable.
- `dub-azure` prints a warning containing the unit ID, overrun, and normalized
  boundary. The same structured audit is saved in the final direct-dub report.
- Translation for the reported job is already complete and remains reusable.
  The failed attempt stopped before voice resolution and Azure TTS, so resume
  normally without `--force`.

## v13.18.10 publication identity alias recovery

- Fixed the exact completed-publication failure from job
  `37a232da2f2346ba95b6a89d5e4c1c09`: `Publication title still contains
  rejected autocorrection form(s): Sibiya`.
- Name research correctly expanded the valid surname `Sibiya` to
  `Lieutenant-General Sibiya` in the transcript. The former publication guard
  incorrectly treated every correction input as a rejected spelling, then
  rejected the valid surname contained inside its own canonical expansion.
- Canonical expansions that retain a whole reviewed identity form now register
  that shorter form as an accepted public-title alias. `USibiya` therefore
  remains valid while actual STT corruptions such as `USabir` still normalize
  to `ULieutenant-General Sibiya`.
- Reviewed title replacement now uses one substitution pass. Newly inserted
  canonical text cannot be matched and expanded a second time.
- Canonical spans are masked before the final unresolved-form scan, preventing
  canonical replacements from being mistaken for leftover raw errors while
  still detecting a separate unresolved occurrence.
- The title-authority payload and editorial prompt remain unchanged, preserving
  the existing paid editorial-response checkpoint. Rerunning the job performs
  local revalidation and the final publication encode without another Claude
  call.

## v13.18.9 measured edge-surplus timeline fit

- Fixed the exact final-block collision from job
  `88c279a25bd64e069cbb0e58d01a5e2c`. `block_0009` contains only
  `Siyaqhubeka.` and needs 1,402 ms after bounded time compression, but its
  original window is 1,090 ms.
- The preceding `block_0008` has a 26,100 ms timeline window and an
  Azure-measured compact duration of 22,768 ms. The former solver ignored this
  3,395+ ms surplus because the last block has no following neighbour.
- A new measured edge solver runs before block finalization. It consumes the
  existing 180 ms source gap and moves the shared boundary only 132 ms, giving
  the final block exactly 1,402 ms while leaving block 8 with 25,968 ms.
- Both approved isiZulu texts, protected entities, speaker order, source start,
  and source end remain unchanged. No AI call, human decision, overlap, or
  forced truncation is needed for this collision.
- The solver works symmetrically for first-block overflows and fails closed
  when the adjacent Azure-measured sentence cannot spare the required time or
  the boundary movement exceeds the configured limit.
- AI wording repair remains a later fallback for genuine semantic compression
  cases. It is intentionally not called when measured timeline capacity alone
  resolves the shortfall more safely and cheaply.

## v13.18.8 non-terminating editorial classifications

- Fixed the exact completed-dub failure where Claude returned `direct_quote`
  in `hook_strategy`. `direct_quote` is a valid neighbouring
  `editorial_angle`, but it is not a strategy enum member.
- Advisory `editorial_angle` and `hook_strategy` values are now canonicalized
  before provider schema validation. Recognized cross-field aliases map to the
  closest valid label; unknown or missing labels become `other`.
- Every local recovery is added to the package-level `human_review_flags`
  ledger. The title text, cited evidence, grounding, attribution, safety
  assertions, and candidate scores are not changed by this recovery.
- The JSON schema remains strict after normalization, so malformed substantive
  fields still fail closed. Classification vocabulary drift alone can no
  longer discard an otherwise valid editorial package.
- The editorial prompt remains v12, so this local recovery does not invalidate
  any matching editorial-response checkpoint. Because the reported failure
  occurred inside provider schema validation, an older server may not yet have
  written that application checkpoint; in that case the first resumed run
  makes one new editorial call, then checkpoints the accepted response.

## v13.18.7 non-terminating title acronyms

- Fixed the exact completed-dub failure where both hook forms contained
  `IPID`. IPID is now a reviewed South African public-title acronym.
- Removed the split between discovery-acronym expansions and title-overlay
  fallbacks. Reviewed institutional acronyms and full names now share the same
  local publication knowledge.
- A title candidate containing an unresolved acronym no longer terminates the
  dub. If another grounded candidate is accessible, that safe candidate wins.
- If every candidate contains an unknown acronym, the strongest grounded title
  is retained with `unexplained_title_acronym_retained_for_review` instead of
  raising an exception.
- The prompt remains v12, so the paid response checkpoint from the reported
  IPID failure is reusable without another Claude request.

## v13.18.6 featured-people discovery

- The editorial package now separates the witness/speaker from up to two
  people who are the main characters of the story being discussed.
- When a witness materially discusses two high-profile people, both discussed
  people receive first priority in the English caption and story hashtags. The
  witness no longer displaces one of them merely because the witness is
  speaking.
- The model returns `featured_person_names` as canonical full names. A bounded
  local fallback can derive this advisory ledger from ordered canonical search
  keywords if Claude omits it.
- Personal hashtags use full names without titles, ranks, or isiZulu prefixes.
  `#AdvocateJohnson`, `#GeneralKhumalo`, and `#GenKhumalo` are normalized to
  `#AndreaJohnson` and `#DumisaniKhumalo` when those canonical people are
  featured.
- The first two story-tag positions are reserved for the two featured people.
  The final two dynamic positions select the strongest witness, commission,
  case, institution, acronym, or issue.
- The English caption must lead with both featured names and preserve
  truth-constrained witness attribution. If a model caption omits either name,
  the server restores a two-name lead without changing the factual claim.
- The editorial prompt advances to v12. Existing speech, translation, audio,
  mix, and clean-video artifacts remain reusable; an older editorial response
  is intentionally not reused because it lacks the featured-people contract.

## v13.18.5 English discovery caption and five hashtags

- Separates publication display language from discovery language. Approved
  speech and the footer hook remain isiZulu; the TikTok caption and search
  keywords are generated in natural English.
- The final publication contract is exactly one configured community hashtag
  plus four distinct story hashtags. For the isiZulu account this produces
  `#ZuluTikTok` plus the central person, institution/acronym, commission/case,
  and a second relevant entity or issue.
- Account-brand hashtags are no longer added. The account name already carries
  the brand, so those slots are assigned to searchable story entities.
- Legacy language and geographic padding tags such as `#NgesiZulu`,
  `#IsiZulu`, `#Mzansi`, and `#SouthAfrica` are filtered even when the existing
  server account configuration still contains them.
- Established acronym hashtags are preferred deterministically. In particular,
  `#PoliticalKillingsTaskTeam` is canonicalized to `#PKTT`; equivalent rules
  cover SAPS, NPA, IDAC, IPID, and EMPD.
- When a recognized acronym is selected, its full expansion and acronym are
  guaranteed in the English caption for searchability, for example
  `Political Killings Task Team (PKTT)`.
- Claude is asked for exactly four grounded story tags. A local normalizer
  removes duplicates and padding and can fill omitted slots from canonical
  search keywords, title leads, and the primary story angle without a repair
  call.
- The publication sidecar assembler ignores obsolete configured value/shared
  tags and cannot exceed five total hashtags.
- The editorial prompt version is advanced to v11. An older editorial response
  is intentionally not reused because it was generated under the old
  target-language-caption/two-topic-tag contract. Azure speech, the clean
  dubbed master, mix, and translation checkpoints remain reusable.

## v13.18.4 public title identifier recovery

- Fixed the exact publication failure where `R14` was classified as an
  unexplained title acronym. A one-letter-plus-digits token is a public
  reference, not an organisation initialism.
- South African road references and rand values such as `R14`, `R21`, `N1`,
  and `R350`, plus familiar identifier forms such as `G20`, now bypass the
  unexplained-acronym guard. Their title text is preserved unchanged.
- Multi-letter alphanumeric labels such as `XYZQ2` still pass through the
  safeguard. Reviewed common acronyms such as `CCTV` remain allowlisted and
  unknown organisation labels are not silently approved.
- Every successful paid editorial provider response is now checkpointed
  immediately, before local opening, grounding, title, length, and acronym
  checks. A later local rejection therefore cannot discard the paid result.
- A rerun validates the checkpoint against a deterministic fingerprint of the
  exact editorial request and schema. If it matches, the response is
  revalidated locally without another Claude call. Damaged, stale, forced, or
  mismatched checkpoints are ignored safely.
- Progress distinguishes a reused saved response from a new provider request,
  and the final selection records the checkpoint path, request hash, and reuse
  status for audit.
- This particular already-failed run predates the checkpoint feature, so its
  next publication attempt will make one final editorial call. That response
  is saved before any local rule runs; the clean dubbed master and completed
  Azure speech artifacts remain reusable.

## v13.18.3 context-bound identity recovery

- Fixed the exact `dRK Tactical` / `Witness K` production failure. The entity
  registry's title normalizer previously treated the leading mixed-case `dR`
  in `dRK` as the title `Dr`, reducing `dRK` to the unsafe one-letter surface
  `K`. That allowed `Witness K` to bind the unrelated organisation.
- Title removal now requires a complete title token followed by whitespace or
  end-of-string. `dRK` remains `dRK` during matching.
- Canonical names, aliases and STT phrases shorter than three characters are
  no longer eligible for automatic source matching. Multi-token identities
  such as `Witness K` remain valid.
- A protected identity contributed by metadata but absent from the exact
  source unit under every reviewed source form is now classified as
  `context_bound_identity`. It remains auditable context but is not injected
  into translated speech.
- Successive genuine protected-identity omissions in one paid candidate are
  repaired locally in a bounded loop. In the supplied artifact, the false
  `dRK` requirement is ignored and the next real omission, `EMPD` in
  `unit_0013`, is repaired without another Claude request.
- The old recovery incorrectly treated the next unit's validation error as
  failure of the first repair, then sent the already-fixed first unit to Claude
  and finally repeated the stale original error. The new metadata records all
  repaired unit IDs and the actual sequence of validator issues.
- If a provider fallback is genuinely necessary, its request no longer carries
  the complete-story context spine, neighbouring units or prior terminology.
  It sends only the failed source unit and reviewed identity contract.
- Existing failed chunk candidates remain reusable. The upgrade does not force
  entity-binding regeneration or invalidate the paid Chunk 1 candidate.

## v13.18.2 single video encode

- The ordinary `dub-azure` workflow no longer performs two full H.264 video
  encodes. Its clean dubbed master is now a fast video-stream-copy remux, and
  only the final publication pass runs `libx264`.
- Standalone uses of `DirectAzureDubRenderer` retain the existing smooth-CFR
  clean-master default. The optimization is selected explicitly by the CLI
  only because its publication render immediately follows. The diagnostic
  override `MATHULA_TV_CLEAN_MASTER_VIDEO_MODE=smooth_cfr` restores the former
  intermediate encode when explicitly requested.
- The publication encoder reads the lossless `direct_dub/final_mix.wav`
  directly instead of decoding and re-encoding the intermediate master's AAC
  audio. The final publication file therefore has one AAC generation.
- The final publication pass still applies frame-rate normalization, H.264 High
  profile, `yuv420p`, the opening cut, the persistent title panel, 48 kHz AAC,
  and `faststart`; TikTok/device compatibility is unchanged.
- Terminal timing now labels the cheap intermediate operation `video-remux`
  and reserves `publication-ffmpeg` for the actual video encode.
- Publication reuse now includes the final-mix checksum. An unchanged job
  reuses the publication output; changed audio invalidates it safely.
- A compatibility fallback still reads audio from older clean masters when a
  historical report does not contain `outputs.final_mix`.

## v13.18.1 automatic boundary time-fit

- Small measured overflows at only the first or final speech block may now be
  fitted automatically with bounded post-synthesis time compression up to
  `1.08×`.
- The supplied production case is a first-class regression fixture:
  `block_0015_variant_compact`, measured `24,654 ms` in a `23,130 ms` window,
  requiring `1.065888×`. It is now eligible without panel intervention.
- The approved isiZulu wording, protected names, source timing boundary, and
  speaker order are unchanged. The renderer adds no overlap, truncates no
  speech, and makes no AI request.
- Every automatic fit records the measured duration, target duration, required
  ratio, configured maximum, boundary side, and unchanged-translation/speech
  invariants in `timing_adjustment`.
- Interior collisions remain limited to the existing `1.03×` policy. A
  boundary requiring more than `1.08×` still pauses for review.
- Terminal progress explicitly reports the bounded boundary fit before block
  finalization continues.

## v13.18.0 boundary-aware timeline review

- A missing previous or following neighbour is now rendered as an explicit
  `start of timeline` or `end of timeline` boundary. The panel no longer
  fabricates an `unknown` block with blank speaker, window, source, and text.
- Legitimate zero-valued timestamps such as `0 ms` are preserved instead of
  being displayed as blank.
- Final-block and first-block cases expose only strategies the renderer can
  actually apply. In particular, a boundary case no longer offers an
  impossible two-sided overlap decision that would reopen the same review.
- Prepared-variant selectors are shown only for speech blocks that exist.
- A malformed historical review artifact uses the explicit label
  `block data unavailable`; it never invents a block identity.
- Real measured timing collisions remain fail-closed. The renderer still
  refuses to truncate speech or extend dialogue beyond the source silently.

## v13.17.9 reviewed CCTV title policy

- `CCTV` is now part of the reviewed public-title acronym allowlist. It is a
  widely understood evidence term for ordinary South African news viewers and
  may remain concise on a title card.
- The selected title is preserved unchanged when it uses `CCTV`; no second AI
  call, invented expansion, or unnecessary fallback is required.
- The acronym policy version is advanced to
  `south-african-public-title-acronyms-v4` for auditability.
- The obscure-acronym safeguard is unchanged for unreviewed labels. Regression
  coverage proves that a fabricated `XYZQ` title still fails when neither the
  hook nor accessible alternative explains it.

## v13.17.8 editorial advisory-field recovery

- A Claude editorial package no longer fails merely because an empty
  `human_review_flags` array is omitted from the package or one of its three
  candidates.
- The publication-specific normalizer runs after JSON parsing and before the
  authoritative local schema. This fixes the validation-order bug that made
  later `setdefault` compatibility code unreachable.
- `accessible_hook_text` may be derived from an existing nonblank `hook_text`,
  and an omitted `withheld_answer` may be restored as an empty string. These
  fields add no new factual or editorial claim.
- Every restored field is recorded in the package-level
  `human_review_flags` ledger with a `wire_defaulted:` or `wire_derived:`
  marker.
- Terminal progress reports that safely derivable metadata was restored before
  strict local validation.
- Required evidence IDs, title-lead classification, opening IDs and
  relationship, story angle, caption, keywords, hashtags, scores, confidence,
  grounding, attribution, sensationalism guard, and all generated hook text
  remain mandatory. Invalid values are not coerced.
- Recovery happens locally with `max_repairs=0`; it never spends a second
  Claude call on this omission.

## v13.17.7 punctuation-only non-speech handling

- Azure pseudo-phrases containing only Unicode punctuation, such as
  `source_text: "......."`, are no longer sent to translation or Azure TTS.
- Existing installed translations containing such a unit are handled locally,
  so `unit_0100` can be skipped without rerunning or rewriting the completed
  translation.
- The source video/background interval is preserved. Only nonexistent dialogue
  is omitted.
- Units containing any letter or number remain mandatory. Symbol-only content
  such as currency is not silently treated as punctuation.
- Ordinary lexical speech containing ellipses remains speakable and keeps the
  existing natural boundary cleanup.
- Terminal prepare progress and the direct-dub report record the skipped count
  and exact unit IDs.
- The voice-family classifier is imported only when uncached voice analysis is
  actually required. Cached Azure voice assignments do not need optional
  PyTorch classifier imports.

## v13.17.6 compact metadata recovery

- Missing compact unit `f` and `o` ledgers are restored from the exact
  source-bound chunk request before provider-schema validation.
- Missing per-variant `p` and `o` annotation arrays default to empty ledgers.
  These fields are descriptive metadata; protected identities and omissions are
  still checked independently against the translated speech.
- Every local restoration is written into the top-level warnings ledger for
  audit.
- Saved paid responses that failed only because one of these ledgers was absent
  are recovered during checkpoint resume without another Claude request.
- Core generated fields remain mandatory: unit ID, all three variants, spoken
  text, duration estimate, and meaning-preserved flag.
- Unknown structures, missing translations, identity loss, required-fact loss,
  and undeclared semantic omissions continue to fail closed.

## v13.17.5 publication opening-boundary recovery

- A structurally valid editorial response no longer aborts merely because
  Claude chooses a real retained block from `full_rendered_transcript` instead
  of the narrower `candidate_boundaries` list.
- The server deterministically moves the opening backward to the nearest
  preceding eligible candidate. This can only retain more approved speech; it
  never removes additional dialogue or changes the approved translation.
- The original AI-selected ID, effective candidate ID, reason, and number of
  additionally retained blocks are stored in
  `opening_candidate_boundary_guard`.
- The publication SEO and selection receive the human-review flag
  `opening_boundary_remapped_to_preceding_eligible_candidate`.
- Question/answer anchoring, payoff ordering, title grounding, title quality,
  and all existing publication safeguards run after the remap.
- Invented block IDs and real IDs with no provably safe preceding candidate
  continue to fail closed.
- Recovery is local and does not issue an AI schema-repair request.

## v13.17.4 category-based translation recovery

- Saved Claude responses are inspected locally before any new provider call.
- Compact translation objects are recovered through bounded nested wrappers:
  `result`, `response`, `translation`, `output`, `data`, and `payload`.
- Top-level compact unit arrays and `translation_units`/`dubbing_units` aliases
  are canonicalized into the existing compact contract.
- A complete older full-envelope multivariant response is deterministically
  converted into the compact wire shape. Only fields represented by the compact
  schema are copied.
- The saved `raw_output` in `failure.json` is now a first-class recovery source,
  so a parser upgrade can checkpoint an already-paid response without another
  Claude call.
- A genuinely empty, malformed, or unit-less completion is classified as
  `invalid_translation_wire_shape`. Only that failed chunk is split into two
  smaller serial parts; completed chunks remain reusable.
- Saved wire-shape failures resume directly at split recovery rather than
  resending the failed parent chunk.
- Failed parent token usage is included in the merged consumption metadata.
- Required unit IDs, all three variants, facts, identities, ordering, and
  semantic validators remain fail-closed.

## v13.17.3 publication recovery

- Claude properties explicitly forbidden by a closed application schema are
  removed locally before validation. This recovers harmless explanatory keys
  such as `opening_payoff_block_id_note` without another paid AI request.
- The normalization is recursive and applies only where
  `additionalProperties: false` is explicit.
- Removed property paths are emitted in terminal progress and retained in AI
  request metadata for audit.
- Required properties, values, identities, arrays, semantic validators, and
  publication grounding checks are unchanged. An unknown key cannot substitute
  for a missing required key.
- Azure workspace capability messages using `structured_outputs` with an
  underscore now activate the same automatic fallback as `structured outputs`.

## Claude failures handled

- Azure Foundry native Structured Outputs are attempted in `auto` mode. If a
  deployment rejects `output_config.format`, the request is rebuilt once using
  the prompt-schema plus local-validation path. The capability result is
  remembered for the rest of the process.
- Unsupported Foundry `effort`, `thinking`, and prompt-cache controls are
  disabled independently. A deployment rejecting one feature no longer causes
  the application to discard all optional controls.
- HTTP 529 overload responses are retried. `Retry-After` values up to 15 minutes
  are honored instead of being capped at 60 seconds.
- A complete JSON object received before an SSE disconnect, timeout, or stream
  error is validated and salvaged. Incomplete content is never accepted.
- `max_tokens`, context-window, and HTTP 413 failures can split only the failed
  semantic chunk into smaller serial requests. Completed chunks remain reusable,
  split children are checkpointed, and their responses are merged back in the
  original unit order.
- Saved parent transport failures are recognized on resume, so the paid failed
  parent request is not repeated before split recovery.
- Paid token usage from the rejected parent request is retained in the
  consumption aggregate instead of disappearing during recovery.
- Unexpected `pause_turn`, `tool_use`, `stop_sequence`, and context-window stop
  reasons now produce explicit diagnostic failures.

## Schema safeguards

- Provider-only Claude schemas remove unsupported numeric, string, array, and
  object constraints while preserving those rules for local validation.
- Schema complexity is measured before the HTTP request. Schemas exceeding
  Claude's optional-field or union limits fail locally without consuming tokens.
- Fully open object schemas remain on the prompt-schema path instead of being
  converted into empty closed objects.
- Unambiguous case-only enum/const differences are normalized before local
  validation.
- Application schemas remain authoritative. The recovery code does not accept
  missing facts, identities, required translations, or malformed partial JSON.

## Runtime visibility

The cumulative patch retains:

- request bytes, schema bytes, estimated input tokens, configured output limit,
  attempts, cache reads, actual token usage, and cumulative AI consumption;
- chunk progress, validation/identity repair progress, checkpoint reuse, and
  split-child recovery progress;
- FFmpeg frame/time/speed/percentage progress;
- an end-of-command phase-duration table and `TOTAL WALL TIME`.

## Validation

- Python compilation passed.
- 38 focused boundary-fit, timeline-panel, direct-dub, and multivariant tests
  passed before the
  cumulative gate.
- 225 cumulative regression tests passed after explicitly deselecting the same
  two legacy Khumalo identity-candidate recovery failures. The protected
  wordplay punctuation failure remains outside this cumulative file set. A
  clean v13.17.1 baseline reproduces those failures:
  - protected wordplay punctuation;
  - two legacy Khumalo identity-candidate recovery tests.
- Ruff still reports three pre-existing `F402` loop-variable shadow warnings in
  `ai_provider.py`; this patch does not introduce an additional Ruff finding.

## Intentional limits

The process remains fail-closed when recovery cannot prove a valid result.
Semantic omissions, identity loss, or malformed JSON must not be silently
published. The upgrade prevents avoidable termination by retrying, salvaging,
falling back, or splitting only where correctness can still be verified.
