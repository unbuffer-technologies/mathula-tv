# Protected source-gap overflow

This cumulative patch adds a bounded, no-shift overflow policy for native hard sync.

## Why

After rolling-syllable translation, many remaining timing failures were borderline: the
isiZulu could not hit the exact English mouth-close within the +12% natural speed cap,
but there was real source silence before the next protected onset. Rewriting those
phrases again damages language quality unnecessarily.

## Policy

- Source phrase start remains immutable.
- Exact mouth-close remains preferred whenever it can be reached within the configured
  natural speed bound.
- If exact mouth-close would require excessive speed, Mathula may use existing source
  silence after articulation.
- Default maximum protected overflow: 400 ms.
- Default minimum pause preserved before the next source onset: 120 ms.
- The next source block is never shifted.
- Large mismatches with insufficient free gap still go to Grok timing repair.
- The feature can be disabled with `--no-protected-gap-overflow`.

The effective safe late allowance is the larger of the normal configured mouth-close
allowance and the bounded usable source gap, but can never exceed the next source onset.

## Regression geometries

- A 4.720 s source window with 5.462 s raw TTS and 480 ms source gap gets a 360 ms
  protected late allowance after reserving 120 ms. It becomes speed-rescuable at <=12%.
- A 4.320 s source window with 6.112 s raw TTS and only 80 ms source gap gets no extra
  protected gap allowance and remains a true compression repair.
