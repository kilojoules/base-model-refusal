# Does a base model refuse?

A size-scaling study of refusal behaviour in Qwen base checkpoints. Read `DESIGN.md`
first; it contains the methodology and, more importantly, the three ways the obvious
version of this experiment goes wrong.

## The short version

The question "does a base model refuse?" is confounded, because a base model is a
document continuer. Given a bare harmful request it may emit refusal-shaped text simply
because it has inferred it is inside an assistant transcript, where a refusal is the
locally likely next token. That is genre imitation, not a disposition.

So the study measures contrasts, not levels:

- **chat minus document** -- the same request in an assistant frame versus a neutral
  document frame. If refusal survives the genre change it is a property of the request.
- **harmful minus benign** -- against XSTest safe prompts, to subtract the rate at which
  a base model simply fails to answer anything.
- **harmful-demo minus benign-demo** -- in the few-shot curve, separating harm-tracking
  from pure format induction.

The primary metric is judge-free. For each prompt we teacher-force generic refusal and
compliant openers and take the difference of their length-normalised log-probabilities
(`RPS`, nats/token). This is deterministic, cheap, and defined even for a 0.5B model
that cannot write a coherent sentence -- which decouples refusal propensity from
generation competence.

The HarmBench rubric is still run, for comparability, but it is a *compliance* judge.
Its negative class pools refusal with incoherence, so refusal is never derived by
negating it. A separate six-way taxonomy judge does that job.

## Layout

```
DESIGN.md                 methodology, confounds, pre-registered decision rules
config/experiment.yaml    model ladder, arms, judges, attacker settings
src/bre/
  data.py                 HarmBench behaviours + XSTest safe controls
  formats.py              the four prompt-format arms
  probe.py                RPS, the primary judge-free instrument
  fewshot.py              dose-response elicitation curve (k demos, two demo conditions)
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
  run_stage2.sh           stage 2: attacker search + positive control
```

## Running it

```bash
python scripts/rp.py addkey                    # register an SSH key
python scripts/rp.py pods                      # list pods
bash scripts/bootstrap_pod.sh                  # on the pod
bash scripts/smoke_test.sh                     # gate: must pass before the ladder
bash scripts/run_all.sh                        # stage 1
bash scripts/run_stage2.sh                     # stage 2
python scripts/rp.py terminate <pod-id>        # stop billing
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is required on this image: pip resolves vLLM to a build
whose flashinfer sampling kernels JIT-compile against a CUDA toolkit the image does not
carry. The native sampler is equivalent here.

## The attacker maximises refusals

Unusually for a red-teaming harness, the attacker's reward is *refusal*, not compliance.
That turns a point estimate into an estimate of the ceiling: the most refusal-eliciting
prompt distribution reachable under the guards. The sign matters -- optimisation pressure
points away from producing the requested content.

The guards are load-bearing, not decorative. Without them the search immediately
discovers that it can write the refusal into the wrapper and let the target copy it,
which measures nothing. See `attacker/reward.py`: refusal-lexicon blocking, slot
preservation, a request-preservation judge, a length cap, and search/held-out behaviour
splits. Two guard modes are pre-registered: `strict` forbids any mention of declining or
safety, answering "can latent refusal be elicited without being asked for?"; `loose`
blocks only verbatim refusals and gives a weaker upper bound.

## If nothing refuses

That is a result, not a failure, provided the instrument is shown to work. The
protections are in `DESIGN.md` section 6b: a continuous primary metric, several
independent searches from different genre seeds, an identical search against an instruct
model as positive control, and a dose-response curve whose slope is informative even
when every level is near zero.
