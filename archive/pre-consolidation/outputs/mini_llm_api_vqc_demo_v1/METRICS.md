# Metrics -- mini_llm_api_vqc_demo_v1

## Training MSE
Minibatch gradient updates: `mean((mu_hat - mu)^2)`.

## Validation RMSE
Candidate comparison / architecture selection within each arm (lower is
better): `sqrt(mean((mu_hat - mu)^2))`.

## Protected-test RMSE
Computed exactly once per completed arm, for the architecture selected on
validation RMSE, after that arm's search ended. Never returned to the LLM
or to any search arm.

## Process metrics (per candidate)
- validity / duplicate status (duplicate is **arm-local** -- a circuit is a
  duplicate only relative to earlier proposals of the SAME arm);
- final training MSE, final validation RMSE, training runtime;
- circuit depth, total gate count, two-qubit gate count;
- quantum parameter count (4), total trainable parameter count (51);
- cache hit, actually-trained-vs-reused, unique training id;
- dtype/device verification result (float64 + CPU);
- whether at least one quantum angle changed after training.

## API metrics
- successful logical calls, failed logical calls, outbound request attempts;
- total input / output / total tokens;
- latency per successful call and total successful-call latency.
