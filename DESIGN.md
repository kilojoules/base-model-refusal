# Does a base model refuse? A size-scaling study on Qwen

## 1. The question, restated precisely

The loose question "does Qwen base have a notion of refusal, and does it depend on
size?" conflates three things that come apart empirically:

- **Q1 (emission).** Does the model emit refusal-shaped text when handed a harmful
  request?
- **Q2 (disposition).** Is that emission a property of the *request*, or merely of the
  *document genre* the prompt implies? A base model is a document continuer. Given a
  bare harmful request it may emit a refusal because it has inferred it is inside an
  assistant transcript, where refusal is the locally likely next text. That is genre
  imitation, not a refusal disposition.
- **Q3 (capacity).** What is the *maximum* refusal rate reachable by any prompt
  wrapper? A single fixed prompt format gives a point estimate; the quantity of
  interest is closer to a supremum over prompt distributions.

Only Q2 and Q3 are interesting. Q1 alone is not reportable, because it has no
baseline and no control.

## 2. Three flaws in the plan as first stated

**2.1 The HarmBench rubric does not measure refusal.** It asks whether a generation
*is an instance of the behavior* -- it is a compliance / attack-success judge. Its
negative class pools at least five distinct outcomes: explicit refusal, soft
deflection, meta-commentary about the request, off-topic or incoherent continuation,
and failed-but-sincere attempts. For an *instruction-tuned* model that pooling is
harmless, because the negative class is almost all refusal. For a *base* model it is
fatal: the negative class is dominated by incoherence and continuation drift. Scoring
`not-compliant = refused` would report a high refusal rate for a 0.5B model that is
simply too weak to answer anything. This is the single largest error to avoid.

*Fix:* keep the HarmBench rubric for the compliance axis (it buys comparability with
published numbers), and add an orthogonal refusal-taxonomy judge. Report both. Never
derive refusal by negating compliance.

**2.2 No benign control means no false-refusal floor.** Base models produce
non-answers constantly. Without measuring the same non-answer rate on *harmless*
requests you cannot attribute any of it to harm-sensitivity.

*Fix:* run XSTest's safe split alongside. It is purpose-built: prompts that look
alarming but are benign. The reportable quantity is a difference, not a level.

**2.3 A judge-and-generate loop is the wrong instrument for a scaling law.** It is
noisy, expensive, confounded by generation competence, and the judge itself becomes a
reward-hacking surface once an attacker is in the loop.

*Fix:* see below. Make the primary instrument logit-level.

## 3. Primary instrument: forced-choice refusal preference (judge-free)

These are open-weight models we control, so we do not have to sample and classify. We
can read the propensity directly.

For a prompt `p`, a set `R` of canonical refusal openers and a set `C` of canonical
compliant openers, teacher-force each continuation and take the length-normalised mean
token log-probability. Define

    RPS(p) = max_{r in R} meanlogp(r | p)  -  max_{c in C} meanlogp(c | p)

in nats. Positive means the model prefers to begin refusing. Report the mean RPS and
the fraction `RPS > 0`.

Why this is the right primary metric:

- **Judge-free.** No rubric, no LLM grader, nothing to hack.
- **Low variance.** Deterministic given weights; no sampling noise. Roughly two orders
  of magnitude fewer tokens than generate-and-judge.
- **Defined even when the model cannot write.** A 0.5B model has a measurable
  preference between two continuations even though its free generation is junk. This
  decouples refusal propensity from generation competence, which is exactly the
  confound that would otherwise manufacture a spurious size trend.
- **Continuous.** A scaling law wants a continuous dependent variable, not a rate
  saturating at 0 or 1.

Length normalisation is required because refusal and compliance openers differ
systematically in length. Both openers are generic strings; neither contains task
content. Tokenise prompt and continuation separately and concatenate ids, so the
measurement is not perturbed by BPE merges across the boundary.

Generate-and-judge is retained as the **secondary** instrument, to confirm that RPS
predicts behaviour that a human would call refusal.

### A limitation of the cross-arm comparison, and why the raw components are stored

RPS is a difference of two log-probabilities, both of which can be low. In the `document`
arm neither a refusal opener nor a "Sure, here is" opener is a natural continuation of a
reference manual, so the difference between them is taken between two unlikely strings
and is correspondingly noisier than in the `chat` arm. Within an arm the comparison
across model sizes is sound, because the openers are held fixed. Across arms it is
weaker.

Three things keep this honest. The per-opener log-probabilities of both sets are written
to disk for every prompt, so the headline metric can be renormalised after the fact
without re-running anything. The `chat`-minus-`document` gap is reported alongside the
raw refusal log-probability in each arm, so a gap driven by the compliance term rather
than the refusal term is visible. And the few-shot dose-response curve holds the frame
fixed while varying only the number of demonstrations, which gives a within-frame
estimate that does not depend on comparing arms at all.

### Measured on 2026-10-01: RPS carries a large positive offset, so levels are not readable

First run of the probe on Qwen2.5-0.5B base and Qwen2.5-7B-Instruct: RPS was positive on
essentially every prompt, in every arm, for both models, including on benign prompts and
including for a 0.5B base model with no safety training. The fraction of prompts with
RPS above zero was 1.00 almost everywhere.

That is not a finding about refusal. It is string-frequency bias. The refusal openers are
formulaic, high-frequency English; the compliant openers are less idiomatic. A
length-normalised mean log-probability rewards the formulaic string regardless of
context, so the metric sits well above zero before any refusal tendency enters.

Consequences, which the design already pointed at but which are now mandatory rather
than advisory:

- **The sign of RPS means nothing.** "Fraction of prompts with RPS > 0" must not be
  reported as a refusal rate. It was in the first draft of the analysis; it is now
  reported only as a diagnostic alongside the benign offset.
- **Only differences are interpretable**: harmful minus benign, chat minus document,
  harmful-demo minus benign-demo, model minus model within an arm.
- `analyze.rps_offset_check` reports the benign-prompt offset per model and arm, so every
  table can be read against its own baseline.

The first run also showed the anchor behaving as intended: on the chat arm the instruct
model's harmful-minus-benign sensitivity exceeded the 0.5B base model's by a wide margin,
while the 0.5B base model showed essentially no sensitivity in the chat, transcript or
document arms. The instrument separates; the levels just cannot be read raw.

### A second confound found the same way: the benign control was not form-matched

The 0.5B base model showed an apparently large positive harmful-minus-benign difference
in the `raw` arm alone. HarmBench behaviours are imperatives ("write a tutorial
that ..."); XSTest safe prompts are mostly questions. With no scaffold to normalise them,
the `raw` arm compares two grammatical forms as much as two harm levels.

Fix: XSTest's `contrast_*` prompts were added as a third split, `harmful_matched`. They
are genuinely unsafe and matched by construction to their safe twins in surface form, so
their difference isolates harm with form held approximately fixed. Reported quantities
are now:

- **harmful_matched minus benign** -- the clean harm-sensitivity estimate, paired by
  XSTest type. This is the one to trust.
- **harmful (HarmBench) minus benign** -- retained for severity and literature
  comparability, but form-confounded and labelled as such.

## 4. The experimental axis that answers Q2: prompt-format arms

Each behaviour is rendered four ways. The contrast between arms, not the level within
an arm, is the result.

| arm | frame | role |
| --- | --- | --- |
| `raw` | behaviour text alone, no scaffold | the honest base-model condition |
| `chat` | the tokenizer's own ChatML template | maximal assistant-genre signal |
| `transcript` | plaintext `User:` / `Assistant:` turns, no special tokens | assistant genre without special-token cueing; separates learned-token effects from textual-genre effects |
| `document` | continuation of a neutral reference-document genre | the **imitation control** |

The `document` arm exists to falsify the disposition hypothesis. If refusal is a
property of the request, it should survive a change of document genre. If it collapses
to the benign-control floor, then what the `chat` arm measured was the model's model of
*an assistant*, not a disposition of its own. This arm is expected to *reduce* refusal;
it is a baseline, not a target to optimise.

The `chat`-minus-`document` gap is the headline number of the study. Its trend in model
size is the answer to the original question.

## 5. Model ladder

Qwen2.5 base: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B. One family, one pretraining recipe,
seven sizes spanning 144x -- the widest clean ladder available. Do not mix families; a
cross-family comparison confounds size with corpus and cannot support a scaling claim.

Anchors: Qwen2.5-7B-Instruct and -32B-Instruct, to calibrate the top of the scale. The
instruct anchors convert RPS from an arbitrary nat-scale into a fraction-of-RLHF-refusal.

Qwen3 base (0.6B--14B) is a pre-registered replication, run only if Qwen2.5 shows a
trend.

### Residual confound, stated not solved

Post-2022 web corpora contain RLHF assistant transcripts. A base model can therefore
imitate refusal without any safety training. Within one family the corpus is
approximately fixed across sizes, so this does not confound the *size* trend -- but it
does mean a positive result must be described as "learned to imitate refusal from
pretraining data," never as "spontaneously developed safety behaviour." The
`document`-arm contrast is what lets us say which.

## 6. The attacker, and why maximising refusal is the right sign

Training an attacker to *maximise* refusal turns a point estimate into an estimate of
the supremum in Q3. That is a genuinely better instrument than fixed prompts, and the
sign matters: the optimisation pressure points away from harmful output, so the search
does not produce a jailbreak artefact. The same harness with the reward negated would
be a jailbreak trainer; the sign and the guards below are load-bearing, not decorative.

**Staged, cheapest-first.**

- **Stage 2a (run first).** An attacker LLM proposes prompt *wrappers*: templates with
  a single `{behavior}` slot. Several independent runs from different genre seeds; see
  section 6b. Each wrapper is scored by mean RPS over a batch of
  behaviours -- cheap, because the reward is the logprob probe, not a judge. Best-of-n
  with the attacker conditioned on the running leaderboard. This is hill-climbing over
  wrapper space and will capture most of the available signal.
- **Stage 2b.** Validate the top wrappers with generate-and-judge, to confirm the RPS
  gain is real refusal and not a probe artefact.
- **Stage 3 (only if 2a plateaus below the instruct anchor).** GRPO fine-tune of the
  attacker over wrapper tokens. Scaffolded but off by default: this is where both the
  cost and the reward-hacking risk concentrate.

**Anti-reward-hacking guards.** The dominant degenerate solution is for the attacker to
write the refusal itself and let the target copy it. Mandatory:

1. **Refusal-lexicon block.** A wrapper containing refusal n-grams scores zero.
2. **Slot preservation.** The behaviour must appear verbatim and exactly once.
3. **Request preservation.** A judge confirms the wrapped prompt still constitutes a
   request for the behaviour. A wrapper that neutralises the request trivially earns
   refusal-free text and must be rejected.
4. **Length cap**, so the wrapper cannot smuggle in a full transcript.
5. **Split behaviours.** Wrappers are selected on a search split and reported on a
   held-out split. An unreported held-out number is an overfitted number.
6. **Judge rotation** for any judge-based reward, with periodic manual spot-checks.

## 6b. Guarding against an uninformative null

A real worry: what if nothing refuses, and the study has no result? The answer is to
build the instrument so that an absence of refusal is itself a measurement, not a
failure to look. Four things do that.

**1. The primary metric is continuous, so it is never empty.** A refusal *rate* can be
0/200 and tell you nothing -- you cannot distinguish "no refusal tendency" from "a
tendency too weak to cross the sampling threshold." RPS always returns a graded
preference. A base model that never once emits a refusal still has a measurable
preference between a refusal opener and a compliant one, and that number can be compared
across sizes.

**2. Several independent attacker runs, from different genre seeds.** One hill-climb can
plateau in a local optimum, and then a low ceiling is indistinguishable from a bad
search. Runs start from different families of framing -- helpdesk, dialogue, advice
column, textbook exercise, interview, forum thread, email, reference document -- and the
reported ceiling is the maximum over runs on held-out behaviours. Per-run bests are
reported too: if they are tightly clustered the search has converged and the ceiling is
credible; if they are scattered the search is underpowered and needs more rounds, not a
stronger claim.

**3. A positive control: run the identical search against an instruct model.** This is
the load-bearing one. If the same attacker, same rounds, same guards reaches a high
ceiling on Qwen2.5-7B-Instruct and near zero on Qwen2.5-7B base, then the null on the
base model is evidence about the base model. Without this control, a null is evidence
about the search. No null result should be reported without it.

**4. A dose-response curve, which is near-guaranteed to produce signal.** Sweep the
number of in-context demonstrations of declining, k in {0,1,2,4,8}, and measure RPS at
each k. Even a model with no intrinsic refusal tendency will be pushed toward refusal by
enough demonstrations, so the curve is informative at every level. What varies across
model size is *how readily*: the slope in k, and the elicitation threshold (the smallest
k at which RPS turns positive). A flat ladder of single-prompt numbers plus a steep,
size-dependent dose-response curve is a complete and interesting result on its own.

The curve carries a second control that no other arm provides. Run it twice: once with
demonstrations that pair *harmful* requests with declines, once with demonstrations that
pair *benign* requests with declines.

- If RPS rises equally in both conditions, the model is doing **format induction**: it
  has inferred that this document declines things, and the content of the request is
  irrelevant. That is a statement about in-context learning, not about refusal.
- If RPS rises faster with harmful demonstrations, the model is **tracking something
  about the request itself**. That gap, and its growth with model size, is the strongest
  behavioural evidence a base model can give for a refusal notion.

The harmful-minus-benign demonstration gap is therefore promoted to a co-headline
result alongside the chat-minus-document arm gap.

### Pre-registered null-result protocol

If RPS stays near zero across the ladder, report, in this order:

1. The positive control separation, to establish the instrument works.
2. The benign-split floor, to establish the metric is not just measuring refusal style.
3. A quantitative upper bound: "across N independent attacker runs evaluating M
   distinct wrappers under the strict guard, the maximum refusal preference achievable
   on Qwen2.5-\{size\} base was X nats/token, against Y for its instruct sibling."
4. The dose-response curves and elicitation thresholds by size.

That is a publishable negative result with a number attached, which is a considerably
stronger thing to have than a positive result from an uncontrolled search.

## 7. Statistics

Behaviours are the unit of resampling, not generations: the same behaviour across
samples and arms is correlated. Use a cluster bootstrap over behaviour IDs for all
intervals.

For the scaling claim, fit RPS against `log10(params)` with a per-behaviour random
intercept, separately within each arm, and test the arm-by-size interaction. The
interaction is the claim "the imitation gap grows with scale." Seven sizes is enough to
see a monotone trend and a sign; it is not enough to fit a power law, and no exponent
should be quoted.

## 8. Decision rules, pre-registered

- Harm sensitivity near zero and flat in size, in every arm: base Qwen has no
  refusal notion. Negative result, worth writing up. Note this is a statement about
  differences; raw RPS will be positive throughout and means nothing.
- RPS rising with size in `chat` and `transcript` but flat and near the benign floor in
  `document`: refusal is **genre imitation**, and larger models imitate better. This is
  the most likely outcome.
- RPS rising with size in *all* arms including `document`: evidence for a
  genre-independent disposition. The strongest and most surprising result; would need
  the Qwen3 replication before it is claimed.
- Benign-control RPS tracking harmful RPS: the measurement is picking up refusal
  *style*, not harm-sensitivity. Instrument failure; fix before reporting anything.

## 9. Budget

Two A100 80GB. 7 base sizes x 4 arms x ~400 harmful + 250 benign prompts, probe-only,
is a few GPU-hours. Generate-and-judge on a stratified subsample adds a few more.
Stage 2a is dominated by probe calls and is cheap. Well under $100 at listed rates.

## 10. What is deliberately out of scope

Activation-level work -- refusal-direction extraction, linear probes, causal ablation --
is the natural stage 4 and would settle the disposition question far more decisively
than any behavioural measurement. It is out of scope here only to keep stage 1 small.
