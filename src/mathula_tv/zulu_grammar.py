"""The isiZulu grammar reference the dub agent reads: six modules, each written from counts in a linguist-annotated corpus and in 8.4 million words of real news,
with builders tested against ZulMorph.  ``reference()`` is what ``dub_agent.build_system_prompt`` adds after the rules.

OFF BY DEFAULT until the grammar comparison on the Mbalula clip has been read: set MATHULA_AGENT_GRAMMAR=1 (or pass grammar=True) to include it.  The version string goes
into agent_summary.json so a result can be tied to the text it saw.
"""
from __future__ import annotations

import os

from . import zulu_agreement, zulu_extensions, zulu_numerals, zulu_object, zulu_relative, zulu_tam

VERSION = "zulu-grammar-1 (agreement, tense/aspect/mood/negation, relative clauses, verb extensions, object concords and reflexive, numerals)"
MODULES = (
    ("agreement", zulu_agreement.render),
    ("tense", zulu_tam.render),
    ("relative", zulu_relative.render),
    ("extensions", zulu_extensions.render),
    ("object", zulu_object.render),
    ("numerals", zulu_numerals.render),
)
HEADER = """ISIZULU GRAMMAR REFERENCE -- six modules. Every form below was counted in a linguist-annotated corpus or in 8.4 million words of real isiZulu news, and the
rules behind them are tested; you may rely on a form given here, and where a module says it has no checked form, rephrase instead of guessing. Use the modules as your
length control and your fluency check: choose the construction a native writer of news would use for what the English MEANS (a relative verb for 'who / which', one
extension for a phrase, an object concord for a pronoun, the tense that says 'still' or 'already'), never pad with a word the speaker did not say, and never cut a fact."""


def enabled() -> bool:
    return os.environ.get("MATHULA_AGENT_GRAMMAR", "0").strip().lower() in {"1", "on", "true", "yes"}


def reference() -> str:
    """The six modules, in order, under one header."""
    return HEADER + "\n\n" + "\n\n".join(render() for _, render in MODULES)
