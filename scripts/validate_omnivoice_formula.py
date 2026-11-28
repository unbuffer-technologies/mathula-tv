"""Check our copy of OmniVoice's length formula against the live service.

Run on the server (it holds the service credentials and the job's clips):

    python scripts/validate_omnivoice_formula.py --job <job id> --run <run id>

Three checks, each against the service's own answer (``/v1/estimate``, which builds a clip's prompt exactly as ``generate`` does):
  1. every reference clip of the job: seconds per unit of text weight, ours against the model's, and the length of one real sentence, ours against the model's;
  2. whole spans of the run's own units spoken with no pinned length: the model's target, our formula, and the length actually spoken;
  3. a summary line for each.

Last run (2026-11-28, prod-y3 of the SABC job, service revision omnivoice-00015): seconds per weight within +0.4% on 48 clips (worst +1.4%); our formula against the model's target
+0.4% mean, +0.6% worst on 16 real spans; the speech comes out 1-3% shorter than the target (the model's own silence trimming), -1.3% on average.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathula_tv import omnivoice_duration as od  # noqa: E402
from mathula_tv.config import load_settings  # noqa: E402
from mathula_tv.console import Console  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--units", default="2,3,5,7,9,12,15,17", help="unit indexes whose spans are spoken for check 2")
    args = parser.parse_args()
    console = Console(Path(load_settings().work_dir))
    submission = json.loads(Path(os.path.expanduser(f"~/mathula-tv/working/experience/runs/{args.run}/{args.job[:8]}/r1/submission.json")).read_text(encoding="utf-8"))["items"]
    corpus, group, spans, gaps, ref = console._draw_unit(args.job, args.run, 0)
    backend = console._omni_with(console._run_voice(args.job, args.run))
    bank = backend.bank_for(corpus)
    client = backend.client
    probe = next((s for s in (submission[i]["zulu_text"] for i in range(len(submission))) if 120 <= len(s) <= 300), submission[0]["zulu_text"])

    print(f"1. every clip against the service (probe sentence of {len(probe)} characters)")
    pace, length = [], []
    for x in sorted(bank._candidates, key=lambda x: (x["speaker"], x["start"])):
        raw = float(x["end"]) - float(x["start"])
        try:
            answer = client.estimate(bank._clip(float(x["start"]), float(x["end"])), str(x["text"]), [probe])
        except Exception as error:  # noqa: BLE001
            print(f"   {x['speaker'][-2:]} {x['start']:6.1f}-{x['end']:6.1f}  the service did not answer: {str(error)[:100]}")
            continue
        ours = raw / od.weight(str(x["text"]))
        theirs = answer["tokens_per_weight"] / answer["frame_rate"]
        mine = od.span_seconds(probe, ours) - od.EDGE_PAD_S
        pace.append(100 * (ours / theirs - 1))
        length.append(100 * (mine / answer["targets_s"][0] - 1))
    if pace:
        print(f"   seconds per weight, ours against the model's: mean {sum(pace) / len(pace):+.1f}%, worst {max(pace, key=abs):+.1f}% over {len(pace)} clips")
        print(f"   length of the probe, ours against the model's: mean {sum(length) / len(length):+.1f}%, worst {max(length, key=abs):+.1f}%")

    print("\n2. whole spans spoken with no pinned length, speed 1.0, the speaker's own reference, no primer")
    rows = []
    for index in [int(i) for i in args.units.split(",") if i.strip()]:
        corpus, group, spans, gaps, ref = console._draw_unit(args.job, args.run, index)
        start = int(group.get("start_ms") or 0)
        end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
        base = bank.ranked(str(group.get("speaker_id")), start, end, per_unit=False)[0]
        reference = bank.as_reference(base)
        wav = bank._clip(float(base["start"]), float(base["end"]))
        for k, span in enumerate(spans[:2]):
            model_s = client.estimate(wav, reference.text, [span])["targets_s"][0]
            mine = od.span_seconds(span, od.seconds_per_weight(reference.text, reference.seconds)) - od.EDGE_PAD_S
            spoken = client.synthesize(spans=[span], gaps_ms=[], reference=reference, speed=1.0, lora=backend.lora, generation=backend.generation)["duration_ms"] / 1000.0
            rows.append((model_s, mine, spoken))
            print(f"   unit {index:2d} span {k} {len(span):4d} characters: model {model_s:6.2f} s, ours {mine:6.2f} s, spoken {spoken:6.2f} s ({100 * (spoken / model_s - 1):+.1f}% of the model's)")
    if rows:
        spoken_vs_model = [100 * (r[2] / r[0] - 1) for r in rows]
        ours_vs_model = [100 * (r[1] / r[0] - 1) for r in rows]
        print(f"\n3. spoken against the model's target: mean {sum(spoken_vs_model) / len(spoken_vs_model):+.1f}%, range {min(spoken_vs_model):+.1f}% to {max(spoken_vs_model):+.1f}% (the 0.2 s of edge padding is in the spoken length)")
        print(f"   our formula against the model's target: mean {sum(ours_vs_model) / len(ours_vs_model):+.1f}%, worst {max(ours_vs_model, key=abs):+.1f}%")


if __name__ == "__main__":
    main()
