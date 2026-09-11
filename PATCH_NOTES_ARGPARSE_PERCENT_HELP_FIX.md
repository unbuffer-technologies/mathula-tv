# Argparse percent help fix

Fixes `native-translate -h` crashing on Python 3.14 because `argparse` performs percent interpolation on help strings.

The rolling syllable tolerance help text now escapes the literal percent sign as `%%` in source, which renders as `12%` to the user.

No runtime translation or dubbing behavior changes.
