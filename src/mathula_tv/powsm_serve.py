"""The POWSM worker: run by ``powsm_listener`` in its own Python 3.11 environment (ESPnet does not install on the project's Python), never imported.

It loads POWSM once (``espnet/powsm``, a Whisper-style phonetic model) and answers JSON lines on stdin with JSON lines on stdout, each prefixed by ``@@`` so the
model's own log lines on stdout are skipped:

    {"wav": ".../analysis_mono.wav", "start": 600.2, "end": 602.3}   ->   @@{"phones": ["ð", "ə", "w", "ɛ", "s", ..., "h", "a", "ɪ", ...]}

The task is phone recognition, blind: no text is given, so the model cannot echo the dictionary form of what it was told to expect, and what comes out is what the
audio holds.  Given a phrase of a few words around the one of interest it hears "High" in a mumbled "Western Cape High Court" where the other recognisers read "pa".
The model takes a fixed 20 s window, so the stretch is padded with silence.
"""
import json
import sys

import numpy as np
import soundfile as sf

WINDOW_SECONDS = 20
RATE = 16000


def main() -> None:
    from espnet2.bin.s2t_inference import Speech2Text

    model = Speech2Text.from_pretrained("espnet/powsm", device="cpu", lang_sym="<eng>", task_sym="<pr>")
    print("@@" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            info = sf.info(request["wav"])
            first = max(0, int(float(request["start"]) * info.samplerate))
            last = min(info.frames, int(float(request["end"]) * info.samplerate))
            audio, rate = sf.read(request["wav"], start=first, stop=last, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != RATE:        # the Zulu voice's clips are 24 kHz
                import torch
                import torchaudio

                audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, RATE).numpy()
            padded = np.zeros(WINDOW_SECONDS * RATE, dtype="float32")
            padded[: min(len(audio), len(padded))] = audio[: len(padded)]
            prediction = model(padded, text_prev="<na>")[0][0].split("<notimestamps>")[-1].strip()
            phones = [p for p in prediction.replace("/", " ").split() if p]
            print("@@" + json.dumps({"phones": phones}, ensure_ascii=False), flush=True)
        except Exception as exc:                      # one bad request must not end the worker
            print("@@" + json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:300]}"}), flush=True)


if __name__ == "__main__":
    main()
