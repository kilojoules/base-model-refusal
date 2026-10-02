# Does a base model refuse?

A size-scaling study of refusal behaviour in Qwen2.5 base checkpoints (0.5B–72B),
with instruct anchors for calibration. Read `DESIGN.md` for the methodology and,
more importantly, the three ways the obvious version of this experiment goes wrong.

## Findings

### 1. Base models have a shallow refusal preference that scales with size

In the chat arm, all base models from 1.5B upward prefer refusal openers over
compliance openers on harmful prompts (positive RPS). The preference grows with model
size, reaching ~2 nats/token at 72B — but the instruct anchors sit at 3–4
nats/token, so no base model approaches the RLHF floor.

### 2. The preference is genre imitation, not a disposition

The chat-minus-document gap is where the signal lives. In the document arm (neutral
reference-manual framing), refusal preference collapses toward the benign baseline.
Larger models show a slightly larger residual, but the dominant pattern is: refusal
is a property of the assistant genre, not the request.

### 3. Harm sensitivity is refusal-driven, not compliance-driven

Zero-cost decomposition of the probe into its refusal and compliance log-probability
components shows that base models discriminate harmful from benign requests almost
entirely through the refusal channel (refusal logprobs rise on harmful prompts;
compliance logprobs stay flat). RLHF installs both sides. Few-shot steering mostly
acts on the refusal side (format induction).

### 4. Base models are trivially steerable toward compliance on harmful requests

The `compliance_demos` condition — k demonstrations of harmful→"Sure, here is how" —
flips every base model from preferring refusal to preferring compliance by k=2.
At k=8, all base models sit at -0.3 to -0.8 nats/token (negative RPS = prefers
compliance).

| model | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 1.64 | 0.17 | -0.35 | -0.62 | -0.78 |
| Qwen2.5-1.5B | 1.87 | 0.03 | -0.30 | -0.52 | -0.62 |
| Qwen2.5-3B | 2.05 | 0.04 | -0.39 | -0.53 | -0.58 |
| Qwen2.5-7B | 1.93 | 0.28 | 0.01 | -0.16 | -0.26 |
| Qwen2.5-14B | 2.12 | 0.17 | -0.06 | -0.17 | -0.35 |
| Qwen2.5-32B | 1.51 | -0.01 | -0.14 | -0.26 | -0.40 |
| Qwen2.5-72B | 1.70 | -0.09 | -0.20 | -0.36 | -0.50 |
| 7B-Instruct | 3.40 | 2.63 | 1.69 | 0.79 | 0.12 |
| 32B-Instruct | 2.96 | 2.18 | 1.28 | 0.76 | 0.21 |

Instruct models resist: they stay positive through k=8, though substantially eroded.

### 5. Compliance steering ≠ harm steering

Compliance_demos measures preference for compliance *openers* — "Sure, here is how."
Whether the model then produces actually dangerous content depends on whether it has
the relevant knowledge, which scales with size. A 0.5B model that complies eagerly
may produce gibberish; a 72B model that complies may produce real harm.

Compliance steering is a necessary condition for harm, not a sufficient one. The
sufficiency depends on model capability.

### 6. Steering is symmetric but compliance is faster

The swing between harmful_demos (refusal-steering) at k=8 and compliance_demos at
k=8 is ~2.5–3.1 nats/token across all base models, roughly flat with size.
But compliance steering acts faster: a single compliance demo (k=1) already drops
RPS from ~1.9 to near zero, while a single refusal demo barely moves it. This
asymmetry suggests the base model's k=0 refusal preference is shallow — easy to
override, hard to strengthen.

## The short version

The question "does a base model refuse?" is confounded, because a base model is a
document continuer. Given a bare harmful request it may emit refusal-shaped text
simply because it has inferred it is inside an assistant transcript, where a refusal
is the locally likely next token. That is genre imitation, not a disposition.

So the study measures contrasts, not levels:

- **chat minus document** — the same request in an assistant frame versus a neutral
  document frame. If refusal survives the genre change it is a property of the request.
- **harmful minus benign** — against XSTest safe prompts, to subtract the rate at which
  a base model simply fails to answer anything.
- **harmful-demo minus benign-demo** — in the few-shot curve, separating harm-tracking
  from pure format induction.
- **compliance-demo vs refusal-demo** — can the model be steered in both directions?

The primary metric is judge-free. For each prompt we teacher-force generic refusal and
compliant openers and take the difference of their length-normalised log-probabilities
(`RPS`, nats/token). This is deterministic, cheap, and defined even for a 0.5B model
that cannot write a coherent sentence — which decouples refusal propensity from
generation competence.

## Layout

```
DESIGN.md                 methodology, confounds, pre-registered decision rules
config/experiment.yaml    model ladder, arms, judges, attacker settings
src/bre/
  data.py                 HarmBench behaviours + XSTest safe controls
  formats.py              the four prompt-format arms
  probe.py                RPS, the primary judge-free instrument
  fewshot.py              dose-response curve (harmful/benign/compliance demo conditions)
  generate.py             secondary instrument: sampling
  truncate.py             continuation-boundary trimming for base-model output
  judge.py                HarmBench compliance judge + six-way refusal taxonomy
  analyze.py              behaviour-clustered bootstrap, size trend, interaction test
  attacker/
    seeds.py              genre-diverse seed bank for multi-run search
    reward.py             RPS reward and the anti-reward-hacking guards
    search.py             hill-climbing wrapper search
scripts/
  rp.py                   RunPod GraphQL client
  bootstrap_pod.sh        pod-side setup
  smoke_test.sh           three gates; run before the ladder
  run_all.sh              stage 1: probe ladder, sample, judge, analyse
  run_compliance_demos.sh compliance_demos condition (additive)
  run_stage2.sh           stage 2: attacker search + positive control
results/
  probe.jsonl             34,200 rows: RPS for 9 models × 4 arms + fewshot conditions
  compliance_demos.jsonl  5,400 rows: compliance-steering dose-response
  summary.json            pre-computed tables and statistics
```

## Running it

```bash
python scripts/rp.py addkey                    # register an SSH key
python scripts/rp.py pods                      # list pods
bash scripts/bootstrap_pod.sh                  # on the pod
bash scripts/smoke_test.sh                     # gate: must pass before the ladder
bash scripts/run_all.sh                        # stage 1
bash scripts/run_compliance_demos.sh           # compliance-steering condition
bash scripts/run_stage2.sh                     # stage 2: attacker
python scripts/rp.py terminate <pod-id>        # stop billing
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is required on this image: pip resolves vLLM to a build
whose flashinfer sampling kernels JIT-compile against a CUDA toolkit the image does not
carry. The native sampler is equivalent here.

## The attacker maximises refusals

Unusually for a red-teaming harness, the attacker's reward is *refusal*, not compliance.
That turns a point estimate into an estimate of the ceiling: the most refusal-eliciting
prompt distribution reachable under the guards. The sign matters — optimisation pressure
points away from producing the requested content.

The guards are load-bearing, not decorative. Without them the search immediately
discovers that it can write the refusal into the wrapper and let the target copy it,
which measures nothing. See `attacker/reward.py`: refusal-lexicon blocking, slot
preservation, a request-preservation judge, a length cap, and search/held-out behaviour
splits.
