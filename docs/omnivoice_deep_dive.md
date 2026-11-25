# OmniVoice: what it is, how it works, what it is good and bad at

Written 2026-11-24 from four readings: the model card and its community discussions, the GitHub README and every docs page, the paper (arXiv 2604.00688v2, Xiaomi / k2-fsa) and the Python source (`omnivoice/models/omnivoice.py`, `utils/duration.py`, `utils/text.py`, `utils/audio.py`, `cli/infer_batch.py`).
Every claim is tagged: **[doc]** stated in documentation, **[code]** read in the source, **[paper]** stated in the paper, **[ours]** measured on our own runs, **[inferred]** my reasoning from those, to be tested by ear before anyone relies on it.

The idea behind this file: Gemini is strong at language, OmniVoice is strong at imitation. Gemini should write the most fluent, best understood Zulu; OmniVoice should be left to say it in the speaker's voice. This file says what the voice does with what it is given.

---

## 1. The short version

1. OmniVoice is a 0.6B-parameter **non-autoregressive** text-to-speech model: it fills a block of audio tokens of a fixed length, over 32 refinement steps. It does **not** decide how long the speech is; a length is chosen first and the model fits the speech into it. [paper][code]
2. That length is a **formula on the written characters and the reference clip**, not a prediction of how long a person would take: `length = text weight x reference length / reference text weight`. So the pace of the dub is the pace of the reference clip in *weighted characters a second*. [code] This is the single most important fact for us. It matches what we measured (section 3).
3. It was trained on 581k hours in 646 languages. Zulu has **14.83 hours** in the base training set; our service also merges a Zulu LoRA. On FLEURS, Zulu is read back with 2.03% character error (the real recordings: 3.33%), so what it says is intelligible. [paper][ours]
4. Its strength is **voice cloning**: voice, pace, loudness and recording character come from a 3 to 10 second reference clip. Cloning is "the most stable mode" and the one it was mainly trained for. [doc]
5. Its weaknesses that touch us: a **cross-lingual accent** (an English reference gives Zulu with English colouring), **numbers** (not optimised), **no say in length** (so it rushes or stretches to fill what it was given), and a **non-commercial licence** on the weights (section 10, please read).
6. There are controls we do not use yet: decoding knobs (`position_temperature`, `guidance_scale`, `num_step`), saved voice prompts, batching and FlashInfer (2 to 3 times faster), CMU phoneme input for English words, non-verbal and question tags, and voice design. Section 11 ranks them as ear tests.

---

## 2. How it makes speech

- Input is one sequence: `[style][text][reference audio tokens][masked target]`. The text part is the **reference transcript, a space, then the target text**. [code] So the model reads what the reference said right before what it must say.
- Audio is 8 codebooks from the Higgs-audio tokenizer (Boson AI), 24 kHz. A bidirectional Transformer initialised from Qwen3-0.6B predicts all masked tokens at once, then refines: 32 steps by default. [paper][code]
- At each step it unmasks a share of positions chosen by confidence **with temperature 5.0** (`position_temperature`), and takes the single most likely token for each (`class_temperature` 0 = greedy). Classifier-free guidance is 2.0. [code] So the only randomness is **the order in which positions are filled**; that is why our "draws" of one sentence differ, and why one seed repeats exactly. [inferred from code, consistent with [ours]]
- No step ever ends the speech early. The block has a fixed number of tokens, and the model makes speech, hesitation and silence fill it. [inferred from the fixed length, matches [ours]: pinned lengths squeezed words]

## 3. What sets the pace and the length

The estimate, from `utils/duration.py` [code]:

```
weight(text)      = sum of per-character weights
  latin, cyrillic, greek, armenian, georgian : 1.0   (ASCII letters always 1.0)
  CJK 3.0, ethiopic 3.0, yi 3.0, hangul 2.5, kana 2.2, indic 1.8, khmer/myanmar 1.8, thai/lao 1.5, arabic 1.5, hebrew 1.5
  punctuation and symbols 0.5, space 0.2, digit 3.5, combining mark 0.0
length (tokens)   = weight(target) x ref_audio_tokens / weight(reference text)
                    (below 50 tokens it is boosted by a cube-root rule; no upper cap)
speed             : length / speed   (speed > 1 is faster; ignored when duration is given)
duration          : replaces the estimate with duration x frame_rate tokens
fallback          : no reference -> the reference is "Nice to meet you." with 25 tokens
```

What this means:

- **Pace = reference pace in characters, not syllables.** The same characters a second are applied to Zulu. [code] Zulu has far fewer characters per syllable than English (about 2.5 against 4), so at the same characters a second Zulu gets many more syllables a second. [inferred] Measured on the Mbalula clip [ours]: reference 18.2 chars/s gave Zulu at about 7.0 vowel groups a second; reference 14.4 chars/s gave 5.6. The estimate matched the audio within 3%.
- **The reference's silences count.** The reference is stripped of silences over 200 ms inside and over 100/200 ms at the ends (`preprocess_prompt`), then its token count is the denominator. A reference with hesitations looks like a slower speaker. Our primer (names repeated before the clip) changes the ratio too: this is why the gap between the copies set the pace (the 19% faster span, 2026-11-22). [code][ours]
- **Punctuation and spaces are almost free** in the estimate (0.5 and 0.2 characters). A comma or full stop does not buy the sentence time in the estimate; whatever pause appears is the model's own, learned from data. [code]
- **Digits are expensive**: 3.5 characters each, so "230" is given 10.5 characters of time. [code]
- **The model fills the block it is given.** Too short and it compresses (swallowed syllables, flat rushed words); too long and it stretches or inserts silence. [inferred] Pinning a duration (as we did and removed) forces exactly this. 
- **`speed`** only divides the estimate; **`duration`** wins over speed. With `duration`, default post-processing may trim trailing silence, so set `postprocess_output=False` for an exact length. [doc]

## 4. The controls and what we use

| Control | What it does | Default | Ours today |
|---|---|---|---|
| `ref_audio`, `ref_text` | Voice, pace, recording character. `ref_text` optional: Whisper (large-v3-turbo) transcribes it | none | yes, a 3 to 9 s clip of the speaker |
| `language` | `<\|lang_start\|>zu<\|lang_end\|>` token; unknown values fall back to none (language-agnostic mode) | none | per span, `zu` (and `en` for English runs) |
| `speed` | divides the estimated length | 1.0 | 1.0, no change |
| `duration` | fixes output length | none | not used (pinning squeezed words) |
| `num_step` | refinement steps. 64 barely better than 32 (WER 1.28 against 1.30); 16 is about twice as fast, slightly worse (1.50); 8 is clearly worse | 32 | service accepts it, default |
| `guidance_scale` | how hard it follows the text and reference | 2.0 | accepted, default |
| `t_shift` | time-step schedule | 0.1 | accepted, default |
| `position_temperature` | randomness in which positions fill first; 0 = greedy | 5.0 | accepted, default (seeds make draws) |
| `class_temperature` | token sampling; 0 = greedy | 0.0 | accepted, default |
| `layer_penalty_factor` | lower codebook layers fill first | 5.0 | accepted, default |
| `denoise` | `<\|denoise\|>` token: prompt trained with noise so the output is cleaner than a noisy prompt | True | default |
| `preprocess_prompt` | strips silences of the reference, adds a full stop to `ref_text` | True | default |
| `postprocess_output` | removes silences over 500 ms inside the output, 100 ms at the ends, fades 0.1 s, pads 0.1 s | True | default; we cut and join spans ourselves |
| `audio_chunk_duration` / `_threshold` | long text split into 15 s chunks above an estimated 30 s | 15 / 30 | not reached (one sentence per span) |
| `instruct` | voice design: gender, age, pitch, whisper, English accent, Chinese dialect | none | not used |
| `voice_clone_prompt` | a saved, reusable prompt (`create_voice_clone_prompt`, `.save`, `VoiceClonePrompt.load`) | none | not used: we send the clip every call |
| `normalize_text` | numbers to words (zh/en via WeTextProcessing, others via `num2words` for integers) | False | not used (we convert figures ourselves) |
| LoRA | `merge_lora`, 3% of parameters | none | yes, the Zulu adapter (checkpoint-8000, merged in memory) |
| Batch | `omnivoice-infer-batch`, lists of texts, FlashInfer 2 to 2.9 times faster, CUDA graphs for batch 1 | one at a time | no: our service generates one span at a time under a lock |

Seeds are our own addition (`torch.manual_seed` per span), not a model feature. [ours]

## 5. The reference clip

- Use 3 to 10 seconds; over 20 s only logs a warning, and when you pass your own `ref_text` the automatic trimming is skipped. Longer "slows down inference and may degrade cloning quality". [doc][code] Our 28.9 s bogus reference took 12 to 20 minutes per request and then failed. [ours]
- Reference transcript + target are one text stream; `ref_text` gets a full stop appended if it ends without punctuation (`preprocess_prompt`). [code]
- The reference's RMS is remembered so the output is scaled back to it. Clips below 0.1 RMS are boosted. [code]
- **Cross-lingual accent**: "the generated speech will carry an accent from the reference audio's language"; use a same-language reference for standard pronunciation. [doc] Our references are English, so Zulu from our dubs carries English colouring by design. [inferred]
- `ref_audio` and `instruct` together: it follows the reference. [doc]
- A user reported that the last second or two of the reference sometimes comes out before the generated audio, which sounds like an odd full stop. [doc discussion #12] We see and cut this ("stray head"). [ours]

## 6. What it does with the text

- **Almost nothing**: no lowercasing, no digit conversion, no quote, ellipsis or dash handling unless `normalize_text` is on. The characters are tokenised as written. [code]
- **Newlines are deleted with no space**, so "one" newline "two" becomes "onetwo". Never send line breaks. [code]
- Non-verbal tags, written inline, exactly: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`. Trained on affective datasets, mostly Chinese and English; behaviour in Zulu is unknown. [doc][code][paper]
- Pronunciation override: Chinese pinyin in capitals with a tone digit ("ZHE2"), English CMU phones in capitals in square brackets ("[B EY1 S]"). The paper says training swapped words for phones stochastically. The tokenizer path for CMU blocks was **not found** in the code files read; whether `[B EY1 S]` works in a Zulu sentence is **untested**. [doc][paper]
- The paper says numeric patterns are "not yet optimised"; suggested fix: an external text normaliser. [paper] Our own figure handling (digits with the Zulu prefix, English-spoken figures) is that normaliser.
- Long text: above an estimated 30 s it is cut at `. , ; : ! ? 。 ， ； ： ！ ？` (not after Mr., Dr., e.g.), merged by character count to about 15 s, each chunk generated on its own from the original reference, joined with 0.1 s fade, 0.1 s silence, 0.1 s fade. A single sentence is never split. [code] No chunk hears the one before it. So our one-sentence spans are not in conflict with the model; they simply avoid its chunking.

## 7. Languages and Zulu

- 646 languages, 581k hours; language-level resampling (beta 0.8) lifts the small ones. Text goes through the LLM tokenizer with no grapheme-to-phoneme step and no language-specific normalisation. [paper]
- **Zulu 14.83 h, Xhosa 13.35, Pedi 12.64, Shona 9.96, Chichewa 10.8, Tswana 4.24, Swahili 418, Ganda 448, Kinyarwanda 2,022.** Sesotho, siSwati, isiNdebele, Xitsonga and Tshivenda are not in the table. [paper]
- FLEURS character error, ground truth against OmniVoice: Zulu 3.33 / 2.03, Xhosa 3.62 / 3.83, Shona 4.07 / 1.80, Pedi 4.88 / 4.44, Swahili 2.41 / 1.37. It is worse than the recordings on Yoruba (17.97 / 21.37), Amharic, Cantonese, Hebrew and Khmer. [paper] Intelligibility is therefore not Zulu's problem; naturalness and accent are what we judge by ear.
- The paper's 14.83 hours are the base model's Zulu. What `mzansilives/omnivoice-zulu-checkpoints` (our LoRA) was trained on is a separate question that I did not read.
- The base model is a mixed-language model: cross-lingual transfer "is not perfect" (community report). [doc discussion #22]

## 8. What it is good at

- Voice, pace, loudness and microphone character from a few seconds. Speaker similarity and naturalness at or above the commercial systems it was compared with (SIM-o 0.729, UTMOS 4.28 on LibriSpeech-PC; CMOS +0.44 over ground truth in the subjective test). [paper]
- Intelligible Zulu: 2.03% character error on FLEURS. [paper]
- Speed: RTF 0.03 at batch 1 on a modern GPU; 0.012 at batch 8 with FlashInfer. [doc][paper] Our Cloud Run L4 is slower and handles one span at a time.
- Repeatability with a fixed seed. [ours]
- It copes with a noisy prompt (`denoise`).
- Voice design for new voices when no reference exists (for Chinese and English).

## 9. Weaknesses and failure modes

- **Fixed length**: no early stop; wrong estimates become rushed or stretched speech. [code]
- **Characters, not syllables, set the time** (section 3). [code]
- **Cross-lingual accent** from the reference. [doc]
- **Numbers**; the paper says so. [paper]
- **Voice design** trained on Chinese and English only, unstable elsewhere; some attribute combinations are ignored. [doc]
- Very short outputs (1 to 2 s) without a reference are unreliable. [doc]
- **No streaming** in the official code (community projects: vLLM-Omni, MLX-Audio, RealtimeTTS, omnivoice-server). [doc]
- Quality varies per draw (the position order is random), which is why we offer three draws. [ours]
- Reference problems: over 20 s, silence-heavy, hesitant (slow, with joined-clip pauses), or containing someone else's voice. [ours][doc]
- No documented VRAM figure; the paper's own limitations are data quality, instruction diversity, numbers and step reduction (no distillation for discrete NAR). [paper]
- Sometimes inserts an unrelated word at the end (the "commissioner" of 2026-11-23). Cause not found; draws differ. [ours]
- The code kills the whole process group with SIGKILL on any batch error (`infer_batch.py`); relevant only if we adopt that script. [code]

## 10. Licence: please read before this is a business

- **Code: Apache 2.0.** [doc]
- **Weights: CC-BY-NC**, "due to constraints from its training data (e.g., Emilia)". A maintainer wrote that the weights "are non-commercial and can't be used commercially"; Emilia is partly CC-BY-NC and other training sets (WenetSpeech-Yue) are non-commercial too. The licence was changed from a more permissive one on Jul 3. [doc discussions #29, #37]
- **The audio tokenizer** (bosonai/higgs-audio-v2-tokenizer) has its own licence; a community member quotes it as needing extra permission above 100k annual users. [doc discussion #37]
- Our Zulu LoRA is a derivative of the base weights, so it very likely inherits the restriction. [inferred]
- This is not legal advice. If Mathula TV is monetised, someone who can read the licences should decide this soon; the answer decides whether OmniVoice is a long-term engine or a prototyping one. There is a request for a commercial checkpoint on the model's discussion page with no answer visible.

## 11. Ideas we have not tried, ranked, to test by ear

Each is a Studio comparison: same text, same reference, one change, three draws.

1. **`position_temperature` lower (for example 1 to 2, or 0)**. This is the only source of randomness, and bad draws come from it. If lower values give more consistent, less clipped words, the three-draw lottery shrinks. At 0 every draw is the same, so seeds stop mattering.
2. **`guidance_scale` higher (3 to 4)**: follows the written text harder; may reduce swallowed syllables and invented words. Risk: harsher sound.
3. **`num_step` 64 for the final render only**: tiny measured gain, cheap to hear.
4. **Reference choice by speaker syllable pace**, not characters. We already rank by the speaker's overall pace (done 2026-11-24). A further step: scale the reference's character rate so the Zulu syllable rate is natural for the speaker. This is a systemic fix for section 3's mismatch. It must not become syllable counting to detect errors (the earlier lesson); here it only picks a reference.
5. **English names with CMU phones in brackets** ("[B EY1 S]") instead of respelling. Untested in Zulu text; if it works it replaces the respelling layer for English words and removes a source of error. Test on two or three names.
6. **`[question-en]` / `[surprise-*]` tags** for interview questions or exclamations; an unknown quantity in Zulu.
7. **Saved voice prompts** (`create_voice_clone_prompt`): same voice token prefix for every span, less to send, and likely faster. Not a quality idea, a stability and cost one.
8. **Batching and FlashInfer**: lists of spans in one call and the FlashInfer build give 2 to 3 times speed; agent measurements and the Studio would drop from minutes to under a minute. Cost: per-span seeds need a seed per item, because a batch shares one generator.
9. **A Zulu reference clip of the same speaker** (if any exists) to get rid of the cross-lingual accent; otherwise a two-step trick: dub once, then use the first dub's Zulu audio as the reference for the final take. Both are guesses.
10. **`postprocess_output=False` and our own silence handling** are already effectively in place; revisit if a join sounds odd.
11. **Voice design** for a speaker with no clean reference clip (a very short turn). Expect instability in Zulu.
12. **Streaming** through the community servers if a live product is ever wanted.

## 12. What this changes for us now

- The pace and the rushing are properties of the **reference clip chosen** and the **characters written**. The system prompt and the harness should not try to imitate speaker pauses and sentence shapes (the voice's imitation already does that); they should aim at fluent, understandable Zulu.
- The agent's length guide should stay tied to the reference pace (it is: `chars_to_fill`).
- Never send newlines in a span. Keep one sentence a span.
- Do not rely on `normalize_text` for Zulu figures; keep our own.
- Settle the licence question early.

## Sources

- Model card and discussions: <https://huggingface.co/k2-fsa/OmniVoice>
- Repository, README and docs (`generation-parameters.md`, `voice-design.md`, `languages.md`, `tips.md`, `lora_finetuning.md`, `evaluation.md`): <https://github.com/k2-fsa/OmniVoice>
- Paper: <https://arxiv.org/abs/2604.00688> (v2, HTML <https://arxiv.org/html/2604.00688v2>)
- Source files read: `omnivoice/models/omnivoice.py`, `omnivoice/utils/{duration,text,audio}.py`, `omnivoice/cli/infer_batch.py`
- Not read: `omnivoice_flashinfer.py`, `utils/{lang_map,voice_design}.py`, `cli/{infer,demo}.py`, the data and training modules, the demo Space and the Colab notebook, the LoRA training data of `mzansilives/omnivoice-zulu-checkpoints`.

## 13. Experiment log

**2026-11-24: decoding settings on the Mbalula clip** (job `ee91848c`, the text of `prod-4`, the 8.8 s reference, seeds the same in every variant). Each is a run in the Studio picker; a run carries its settings in `experience/runs/<run>/<clip>/voice.json` (`{"generation": {...}}`), made by `Console._run_generation`.

| Run | Setting | Audio | Vowel groups/s | Draw spread (mean / largest) |
|---|---|---|---|---|
| `prod-4-slow` | defaults (position_temperature 5, guidance 2, 32 steps) | 76.6 s | 5.6 | 0.44 s / 1.15 s |
| `prod-4-pt2` | position_temperature 2 | 75.2 s | 5.8 | 0.50 s / 1.52 s |
| `prod-4-pt1` | position_temperature 1 | 73.9 s | 5.9 | 0.39 s / 0.68 s |
| `prod-4-pt0` | position_temperature 0 (greedy) | 76.9 s | 5.6 | 0.00 s / 0.00 s |
| `prod-4-g3` | guidance_scale 3 | 74.9 s | 5.8 | 0.45 s / 1.39 s |
| `prod-4-s64` | num_step 64 | 75.5 s | 5.7 | 0.24 s / 0.77 s (made in 191 s against about 110 s for 32 steps) |

What these numbers can and cannot say:
- At 0 the three draws are **identical** (greedy decoding has no randomness), so the draw mechanism disappears. That confirms where the randomness lives. [ours]
- The length differences between variants (73.9 to 76.9 s) come from silence the model puts inside the fixed block; the block itself is the same. The spreads at 1, 2, 5 are not ordered, and 11 sentences with three draws each is too little to call a trend. The spread is the model's own variability, not a quality score. **Nothing here says which one sounds better; only the ear can.**
- **Bug found:** the service turns every setting into a float (`knobs = {k: float(v) ...}` in `gcp/omnivoice_service/app.py`), and `num_step` must be a whole number, so `num_step 64` returns HTTP 500. Fixed (an `int` for `num_step`) and deployed on 2026-11-24 as Cloud Run revision `omnivoice-00010-sq5`; `num_step 64` then built normally.

**2026-11-24 (later): 64 steps became the default of every request.** The owner's decision: one setting everywhere (the agent's measurements, its submissions, the render and the Studio), so the first measurement is the final audio. It is `DEFAULT_NUM_STEP = 64` on `OmniVoiceClient.default_generation` (override with `OMNIVOICE_NUM_STEP`); a run's `voice.json` generation settings are added on top. Measured on `prod-6` (same text, reference and seeds): the whole unit changed by -0.08 s at 16 steps and -0.97 s at 64 steps against 32, with single sentences moving up to 0.7 s either way; the cost is about twice the time per synthesis (a unit's three-draw Studio: 191 s at 64 steps against about 110 s at 32). Older Studio audio (made at 32 steps) is rebuilt on opening, because the step count is part of its cache key.

**2026-11-24 (later still): the service now scales out.** Cloud Run revision `omnivoice-00011-rhl`: `--max-instances 4 --concurrency 1` (it was 1 and 4: the lock in `app.py` meant a higher concurrency only queued requests on one GPU; the project's L4 quota without zonal redundancy in europe-west1 is 10). Measured with `conc2.py` / `conc3.py` (scratchpad): four short requests sent together take 12.7 s against 16.6 s one after another (1.3x) because a warm copy serves them in turn and a new copy takes a minute or more to load; under sustained load (four streams for 210 s) throughput rose from about 8 to about 18-20 requests per 30 s once a second copy came up (about 90 s in), and only two copies had served by the end. So scale-out helps long bursts (a submit that measures several units), not a quick handful of sentences. `--min-instances 2` would remove the ramp at the price of two idle GPUs. Not yet tested: that the same text and seed give the same audio on two different copies.
