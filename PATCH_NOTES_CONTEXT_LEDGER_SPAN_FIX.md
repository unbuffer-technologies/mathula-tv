# Context ledger story-beat span fix

Fixes `native_context_ledger` structured-output failures when one factual story beat legitimately references more than four source segments.

Changes:
- removes the arbitrary `maxItems: 4` ceiling from `story_beats[].segment_ids`;
- tells Grok that a beat may span any number of relevant segments;
- canonicalizes returned segment IDs to known, unique chronological source order;
- still rejects unknown segment IDs;
- retains the existing maximum of eight story beats and all entity/speaker-role bounds.

Targeted regression exercised the observed five-ID failure shape (`seg-00060`, `seg-00064`, `seg-00066`, `seg-00068`, `seg-00072`), reordered/duplicate IDs, and unknown-ID rejection.
