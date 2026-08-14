# GPT semantic-fit output budget v13.18.53

This cumulative release fixes the first live Azure GPT semantic-fit failure in
v13.18.52:

```text
GPT batch response: 4,096 output tokens
Azure OpenAI GPT returned an incomplete structured response: max_tokens
```

The request itself imposed the 4,096-token ceiling. GPT-5.6 Sol supports
reasoning tokens and a much larger output window, so the candidate reasoning
and the required nested JSON were competing inside an unnecessarily small
allowance.

## Correction

- Every GPT semantic-fit operation now has a defensive minimum allowance:
  - candidate and batch generation: 12,288 tokens;
  - portfolio generation: 16,384 tokens;
  - semantic review: 8,192 tokens.
- Batch estimates add a separate 8,192-token reasoning reserve to the expected
  visible candidate payload and scale with repair/candidate count.
- Candidate generation uses low reasoning effort because every timing-fit
  winner still receives a separate medium-effort semantic review. This reduces
  hidden-token consumption and latency without weakening the acceptance gate.
- The provider enforces GPT-5.6 Sol's 128,000-token maximum locally.
- Progress now reports output-budget uplifts and separates reasoning-token use
  from total output tokens.
- The semantic-fit strategy version is advanced so the failed v13.18.52
  checkpoint receives a fresh bounded GPT attempt budget.

The larger values are ceilings, not requested response lengths. Billing remains
based on tokens actually consumed.

## Verification

The cumulative release passes 86 release tests plus 45 shared provider-boundary
compatibility checks. New tests reproduce the 4,096-token portfolio request,
verify its pre-request uplift to 16,384, verify low candidate reasoning effort,
and reject output ceilings above the model limit.
