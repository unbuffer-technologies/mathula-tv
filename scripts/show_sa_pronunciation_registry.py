#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mathula_tv.pronunciation import get_zulu_public_affairs_acronym_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the curated isiZulu South African public-affairs acronym registry."
    )
    parser.add_argument("--json", action="store_true", help="Print the complete registry as JSON")
    args = parser.parse_args()

    registry = get_zulu_public_affairs_acronym_registry()
    if args.json:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return 0

    initialisms = sum(item["kind"] == "initials" for item in registry)
    word_acronyms = sum(item["kind"] == "acronym" for item in registry)
    print(f"South African public-affairs pronunciations: {len(registry)}")
    print(f"Letter-name initialisms: {initialisms}")
    print(f"Word acronyms: {word_acronyms}")
    print("Examples:")
    for token in ("PKTT", "NPA", "IDAC", "IPID", "SAPS", "SARS", "NSFAS", "SANRAL"):
        item = next(value for value in registry if value["token"] == token)
        print(f"  {token}: {item['tts_text']} ({item['kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
