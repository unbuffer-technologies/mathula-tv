# Temporal-mask acronym preservation is warning-only

## Policy
- Missing exact acronym surface forms such as CCTV, KZN, and SAPS no longer terminate a four-block temporal-mask window.
- Natural isiZulu may expand or grammatically integrate an acronym instead of repeating the exact uppercase token.
- Missing numeric literals and numeric occurrence counts remain semantic-fidelity failures.
- Acronym mismatches are emitted as red console warnings and persisted as `semantic_guard_warnings`.
- No Python residual Grok call is introduced.
