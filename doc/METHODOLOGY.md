# Methodology

This document explains, in detail, how the Orbis reference implementation works
and how it was derived from the source paper. It has two halves:

1. **The source documentation** — how the original paper was referenced,
   obtained, stored, and what it says (§1).
2. **The implementation** — the design of every component, the math, the
   training pipeline, the live-inference runtime, and how each piece maps back
   to a mechanism in the paper (§2–§14).

---

## 1. Source documentation

### 1.1 Canonical reference

> **Visko Orbis 1.0: A Live Model for Real-Time Interactive Long Video Generation.**
> Team Visko. arXiv:2607.26694v1 [cs.CV], 29 Jul 2026.
> PDF: <https://arxiv.org/pdf/2607.26694>

### 1.2 How it was obtained and stored

The paper PDF was provided directly as the primary source for this project (the
nominal arXiv identifier `2607.26694` is future-dated relative to the arXiv
`YYMM.NNNNN` scheme, so the canonical copy is the one archived here rather than a
live arXiv fetch). To make the repository self-contained and the derivation
auditable, the exact source PDF is committed alongside this document:

| Field | Value |
|---|---|
| Stored path | `doc/references/visko-orbis-1.0-arxiv-2607.26694v1.pdf` |
| Size | 5,242,625 bytes (≈5.0 MiB) |
| Pages | 52 |
| SHA-256 | `d23348df4e4db38c0c46d3f52e409065408978f4af9f20c69d58b595ca92cf0f` |

The checksum pins provenance: anyone can verify the archived PDF is byte-identical
to the source this implementation was written against with

```bash
sha256sum doc/references/visko-orbis-1.0-arxiv-2607.26694v1.pdf
# -> d23348df4e4db38c0c46d3f52e409065408978f4af9f20c69d58b595ca92cf0f
```

The PDF is retained for reference and reproducibility only; all rights to the
original text and figures remain with its authors (Team Visko).

### 1.3 Structured summary of the paper

The paper introduces **Visko Orbis 1.0**, positioned as the first realization of
a **Live Model**: a foundation model that *executes as a persistent process*
rather than answering a bounded query. For video, this fixes the system boundary
at *delivered* video — a stateful rollout in which generation history carries
between chunks, prompt updates are timestamped against the output clock, and
evaluation applies without ever reinitializing the rollout.

**Abstract / headline claims.** A distilled chunk-wise streaming generator plus a
streaming super-resolution stage deliver **4K video at 24 FPS in real time**,
with an average visible response to a prompt change **under one second**. It
supports long-form **T2V, I2V, and V2V** with **multilingual prompts** and
**in-generation prompt switching**. A **bounded multi-scale memory** preserves
subjects, scenes and style across chunks over hour-scale rollouts without evident
quality or color drift.

**§1 Introduction — the gap.** Existing work covers *parts* of the problem but
no system covers it end to end: (1) *causal streaming* (CausVid, Self-Forcing,
Rolling/Causal Forcing) streams but is not focused on live user control; (2)
*interaction* (Oasis, LongLive, Matrix-Game 2.0, Krea Realtime, Helios, Vidu S1);
(3) *long-horizon stability* (FramePack, FAR, MemFlow, VideoSSM, FadeMem,
Echo-Infinity); (4) *serving*. A live model must hold all four at once: accept
updates while a rollout is active, preserve visual state across chunks, deliver
frames progressively, and measure response at the *visible* output boundary.

**Key contributions.** (i) A unified live-video formulation as a persistent,
chunk-wise latent-flow process where visual state is carried across chunks and
time-indexed instructions can change without restarting. (ii) Event-aligned data
and progressive streaming training. (iii) Bounded memory for long-horizon
generation (older spans progressively compressed into a fixed-capacity learned
state). (iv) Few-step flow post-training with physics-aware alignment (guidance
distillation, self-forcing distribution matching, GRPO reward alignment, a latent
world-model consistency reward). (v) Live control and inference alignment (rolling
prompt summary, asynchronous prompt encoding, chunk-boundary condition updates,
versioned state invalidation). (vi) Streaming high-resolution system co-design.

**§2 Data.** A multi-stage curation and captioning pipeline: shot-aware clipping
(3–240 s clips), layered safety filtering, cascaded quality filtering,
distribution rebalancing over a coverage ledger, and human-preference curation.
Captioning is dual-mode: single-event holistic captions (≤10 s) and
event-aligned temporal captions (ordered event boundaries + interval-local
captions) for longer clips. These feed progressively narrower training tiers:
broad-coverage pretraining, distribution-balanced mid-training, high-quality
fine-tuning.

**§3 Model.** A conditional latent-video model trained progressively (Figure 3):
short-video bidirectional pretraining → chunk-wise streaming adaptation →
single/multi-event mid-training → human-curated fine-tuning → distillation +
reward alignment, plus a dedicated super-resolution model.
The formulation: for the *k*-th latent chunk `z_k`, with bounded history `H_k`,
active instruction `c_k`, and optional reference `r_k`, the model draws
`z_k ~ p_θ(z_k | H_k, c_k, r_k)`; composing these kernels in generation order
defines a **causal rollout law** (neither the current kernel nor the state update
may access a future instruction or chunk). Both pretraining and streaming use the
**linear rectified-flow objective** (Eq. 1):

```
z̃_σ = (1 − σ) z + σ ε,   ε ~ N(0, I)
L_RF = E[ || v_θ(z̃_σ, σ; q) − (ε − z) ||²₂ ]
```

with σ=0 clean data, σ=1 pure noise. **Bounded multi-scale memory** keeps recent
chunks at native latent granularity while older spans are progressively
compressed under a fixed token budget into a persistent state `M_k`, updated by an
incremental causal constructor so memory and per-chunk cost stay independent of
rollout length. **Long-horizon reliability** is trained by mixing clean history
with augmentation branches (degradation/corruption/statistic-shift/noise and
rollout-calibrated drift statistics) so the model is robust to its own recurrent
errors. **Post-training**: guidance distillation into a conditional student;
endpoint-anchored consistency distillation with an EMA teacher; self-forcing DMD
on student-generated histories; then GRPO with visual/motion/text-alignment
rewards and a latent world-model consistency reward. **Super-resolution**: a
temporally-distilled compact VAE plus a reference-aware single-refinement
transformer with spatial-window attention over the full temporal window inside a
chunk (no cross-chunk autoregression).

**§4 Inference.** Served as a continuously active session around state reuse,
single-stream distributed execution, and progressive delivery. A rolling prompt
summary carries entities/relationships/style; new prompts are encoded
asynchronously, admitted at the next uncommitted chunk, and guarded by session and
prompt versions so stale work cannot overwrite the current condition. Compiled
transformer execution (fused kernels, UniPC solver, W8A8/BF16), full-sequence
multi-GPU (Ulysses-style) parallelism, progressive decode/delivery via bounded
FIFO queues, and conservative content-adaptive drift stabilization.

**§5 Evaluation.** 74 cases, six events each; a *whole event* is the unit of
analysis. Reported: best DOVER aesthetic/technical (0.8101 / 0.5572), best
VideoAlign visual/motion quality (1.5777 / 1.8646); in a long-form Arena study,
highest overall-preference Elo (**1838**) and temporal-stability Elo (**1940**).

**§6 Conclusion.** Reframes video generation from producing bounded clips to
sustaining a controllable visual stream that a user can direct continuously.

---

## 2. Scope and philosophy of this implementation

The production system depends on a GPU cluster, undisclosed trained weights, and a
large curated video corpus — none of which can be reproduced. This repository
therefore reproduces the **mechanisms**, not the weights, at a scale that trains
and streams **on a CPU in ~15 minutes**. Each idea in the paper is a real,
trainable, and tested component.

To keep every claim *observable* rather than a matter of faith, the model is
grounded in a small **prompt-controlled world**: an instruction determines a
subject (shape), attribute (color) and motion (direction); the rollout carries
state (position, velocity) across chunks; and switching the prompt mid-rollout
visibly redirects the future while the delivered past is immutable. This turns
abstract properties (text→video alignment, live switching, bounded memory) into
things you can *see* and *measure*.

---

## 3. The toy world — the data-generating process

`orbis/world.py`, `orbis/vocab.py`

A *scene* is one soft-edged shape (`circle`/`square`/`triangle`) of a color, at a
position, moving with a velocity set by a direction and speed, over a near-black
background. A *rollout* advances the shape frame by frame with wall bouncing and
renders anti-aliased `32×32×3` frames in `[0,1]`. Rendering is deterministic given
`(scene, initial position)`, which gives:

- a learnable **text→video** mapping (the prompt sets shape/color/motion);
- genuine **temporal state** (position/velocity) that must be carried across
  chunks — the reason memory exists;
- a visible response to **mid-rollout prompt switching** via a `control_schedule`
  that changes an attribute at a chosen frame while motion continues.

The prompt grammar is compositional and **multilingual**: a synonym table
(`SYNONYMS`) collapses tokens in several languages onto canonical tokens
(`rojo`, `红`, `rouge` → `red`), including a greedy multi-character pass for CJK.

---

## 4. Latent space — the VAE

`orbis/vae.py`

Orbis generates in a *latent* video space; the toy analogue is a small
convolutional autoencoder that is **spatial** (per-frame) — temporal structure is
modelled by the flow generator, not here. Frames `32×32×3` encode to an `8×8×8`
latent (downsample ×4, `latent_channels=8`, `base_channels=48`) and decode back
through nearest-upsample + conv with a sigmoid head.

**The foreground-collapse fix (a real design decision).** Plain MSE
reconstruction *mean-collapses onto the dark background*: the ~90% of pixels that
are background dominate the loss, and the network learns to output the background
everywhere (empirically MSE plateaued at ≈0.05, which is just the background
baseline; reconstructions were black). The fix is a **foreground-weighted loss**
that up-weights pixels far from the background color:

```
bg = min over H,W of x            # per-image background estimate
w  = 1 + 8 · max_c |x − bg|       # weight map, high on the shape
L  = mean( w · (x̂ − x)² )
```

This drops reconstruction MSE by ~100× (≈0.05 → ≈0.0005) and yields sharp shapes.
After training, `calibrate()` sets a `latent_scale` buffer to the std of raw
encoder outputs so the flow model operates on roughly unit-variance latents;
`decode` multiplies it back, so the scale cancels in a round trip.

---

## 5. Rectified flow

`orbis/flow.py`

The generative objective is the paper's Eq. 1. Straight-line interpolation between
data `z` (σ=0) and noise `ε` (σ=1):

```
z_σ = (1 − σ) z + σ ε
target v* = ε − z          # constant velocity along the straight path
L = mean( (v_θ(z_σ, σ; q) − v*)² )
```

Because `dz_σ/dσ = ε − z = v*`, **sampling** integrates the learned field from
noise back to data with explicit Euler steps `z ← z − Δσ · v_θ(z, σ)` over a
`linspace(1, 0, steps+1)` schedule. `teacher_steps=16` is the accurate trajectory
used in pretraining and as the distillation teacher; `student_steps=4` is the
few-step real-time path.

A closed-form unit test anchors correctness: integrating a *constant* velocity `c`
from σ=1 to σ=0 must yield exactly `noise − c` (`test_flow_sample_constant_velocity`).

---

## 6. The chunk-wise streaming generator

`orbis/model.py`, `orbis/modules.py`

A DiT that realizes `p_θ(z_k | H_k, c_k, r_k, M_k)` by predicting the rectified-flow
velocity for the current *noised* latent chunk.

**Tokenization.** Each latent frame `(8,8,8)` is patchified with `patch_size=2` →
a `4×4=16`-token grid, each token a `2·2·8=32`-dim patch, linearly projected to
`dim=128`. A chunk of `chunk_frames=4` → 64 chunk tokens; history frames and the
reference are embedded the same way. Every token gets **spatial** (learned grid),
**temporal** (learned per-frame index) and **role** (memory/reference/history/chunk)
positional embeddings.

**Conditioning.**
- *Instruction* `c_k`: canonical token ids → an embedding table → text tokens that
  enter each block via **cross-attention**; their mean pools into the global
  conditioning vector.
- *Timestep* σ: a sinusoidal embedding → MLP → added to the pooled text → drives
  **AdaLN-Zero** modulation (shift/scale/gate for self-attn and MLP, plus a gate
  for cross-attn) in every `DiTBlock`. Zero-initialized gates make an untrained
  block an identity map, which stabilizes early training.

**Sequence and causality.** The assembled sequence is
`[memory | reference | history | chunk]`. All context is *clean*; only the chunk
tokens are noised. Attention is full within the assembled sequence (bidirectional),
but only the chunk-token positions are read out and unpatchified to the velocity
prediction. **Causality across chunks** is enforced by the *rollout* — a chunk only
ever sees past history — which is the paper's causal rollout law; within-chunk
bidirectionality matches the paper's full-temporal-context-per-chunk design.

**Versioned state reuse.** Context that is constant across the solver steps of one
chunk (memory, reference, history, text) is encoded once into a `ModelContext` and
reused across all Euler steps — the toy analogue of the paper's compiled/versioned
state reuse. `test_generator_shapes_and_context_reuse` asserts the reused-context
path is bit-equivalent to a one-shot call. The generator is ≈1.48 M parameters.

---

## 7. Bounded multi-scale memory

`orbis/memory.py`

Two scales: a short **recency window** of the last `history_frames=4` committed
latent frames at native granularity, plus a fixed-capacity **persistent state**
`M` of `memory_tokens=16` slots. When frames leave the recency window they are
*consolidated* into `M` rather than recompressed from the full prefix.

The write is a **gated cross-attention**: the persistent slots (queries) attend to
the evicted tokens (keys/values), followed by a gated MLP; both gates start near
zero (`tanh(gate)`), so an untrained bank is a no-op and consolidation is learned.
Reads add slot positional embeddings and expose `M` as an extra context tier.

The point is **boundedness**: `M` has a constant token budget, so read/write cost
and footprint are independent of rollout length — the property the paper relies on
for hour-scale generation. `test_memory_is_bounded` writes 20 chunks and asserts
the state shape never grows; `test_streaming_state_is_bounded` asserts the same
across a live rollout (window ≤ `history_frames`, memory == `memory_tokens`).

> Honest note: in this low-dimensional world the recency window already captures
> most of the state, so memory's long-horizon benefit is primarily *architectural*
> — faithfully implemented, trained jointly, and strictly bounded — rather than a
> large quality win.

---

## 8. Progressive training pipeline

`orbis/train.py`, `orbis/dataset.py` (mirrors Figure 3)

Data is generated on the fly: `RolloutSampler` renders short rollouts and encodes
them to latents with the (frozen, after stage 1) VAE, producing three conditioning
modes so one set of weights supports every inference path:

| mode | context supplied | inference path it trains |
|---|---|---|
| `text_only` | none | first-chunk **T2V** |
| `reference` | 1 clean prefix frame | **I2V / V2V** |
| `history` | recency window + evicted→memory | streaming **continuation** |

**Stage 0 — VAE** (`train_vae`, 900 steps, foreground-weighted loss, then
`calibrate`).

**Stage 1 — bidirectional short-clip prior** (`train_generator` phase 1, 700
steps): mostly `text_only` with 30% `reference`, so the model learns a
text-conditioned appearance/motion prior.

**Stage 2 — streaming adaptation** (phase 2, 1800 steps): a mix of 60% `history`
(with 0–2 evicted chunks written into memory, and 50% of those with
`history_noise=0.15` Gaussian **history augmentation** for exposure-bias
robustness), 20% `text_only`, 20% `reference`. Loss is the rectified-flow
objective; the VAE is frozen. AdamW, grad-clip 1.0.

Both stages update the same weights, so the T2V capability is retained while
streaming continuation is acquired.

---

## 9. Few-step distillation

`orbis/distill.py`

Post-training reduces inference cost. A frozen **teacher** (a deep copy of the
trained generator) produces the accurate `teacher_steps=16` ODE endpoint from a
given noise + context; the **student** (the same weights, fine-tuned) is trained so
its grad-enabled `student_steps=4` Euler rollout matches that endpoint:

```
z_teacher = sample(teacher, noise, steps=16)          # no grad
z_student = few_step(student, noise, steps=4)          # with grad
L = mean( (z_student − z_teacher)² )
```

This is the toy analogue of the paper's endpoint-anchored consistency /
self-forcing distillation. The engine then samples in 4 steps for real-time
delivery (≈22 ms/chunk on CPU here). Distillation drove the student loss to
≈0.07 in the committed checkpoint.

---

## 10. Live inference — the delivered-video runtime

`orbis/session.py`, `orbis/engine.py`

**Session state.** `LiveSession` is a persistent process carrying: the active
condition (committed at a boundary), a **rolling prompt summary**, a pending
versioned update, the recency window `history`, the persistent `memory_state`, an
optional `reference`, and per-chunk counters/log.

**The versioning contract (the centerpiece).**
`set_prompt(text)` merges the new controls onto the rolling summary (so a partial
update like "moving up" keeps the established subject/color), assigns a new
version, and **queues** it. `admit_pending()` applies the latest queued update at
the *start of the next chunk*. Because each chunk is generated and committed
atomically:

- frames from chunks generated **before** the switch are byte-for-byte unchanged
  (delivered ⇒ immutable);
- the switch takes effect from the boundary chunk onward.

This is verified *without training* by
`test_switch_only_affects_uncommitted_chunks` (pre-boundary frames identical
between a switched and an unswitched run; the version log flips exactly at the
boundary) and demonstrated in the HTML console, where a badge compares the two
streams **pixel by pixel**.

**One chunk** (`generate_chunk`): admit pending → assemble `ModelContext`
(reference only seeds the first chunk, i.e. while history is empty) → few-step
rectified-flow sample → VAE-decode → append to the recency window, evicting
overflow into memory → increment counters. T2V/I2V/V2V differ only in how the
first chunk's context is seeded (empty / encoded image / encoded video prefix).

**Progressive delivery.** `stream()` yields each chunk as soon as it is ready
rather than after the whole video — the toy analogue of the paper's bounded-FIFO
progressive delivery.

---

## 11. Streaming super-resolution

`orbis/superres.py`

A reference-aware, single-refinement upsampler: each low-res frame is bilinearly
upsampled and refined by a small conv residual conditioned on the previous
restored frame (temporal reference). Frames within a chunk are refined with the
previous frame as reference, but there is **no autoregressive dependency across
chunks** — the within-chunk design the paper uses so restoration stays compatible
with tiled/parallel execution. Trained (`train_sr`, 300 steps) on `(LR, HR)` pairs
where LR is the anti-aliased area-downsample of an HR render, so the pair is a true
low-pass match. Scale ×4 (`32→128`), the toy analogue of native→4K. It is a
*presentation* stage: it adds detail, it does not invent semantic state.

---

## 12. Evaluation methodology

`orbis/eval.py`

Because the world is known, following is measured directly, and — per the paper —
a *whole rollout* is the unit of analysis. Per-frame analysis detects the
foreground mask, dominant color (nearest palette), centroid, and shape (mask-IoU
against re-rendered prototypes). Rollout metrics:

- **color / shape accuracy** vs the active instruction;
- **temporal stability** — fraction of frames on the modal color (flicker/identity
  drift);
- **motion** and **direction agreement** with the requested heading;
- **switch compliance** — pre-boundary on the old attribute, post-boundary on the
  new.

Reported on the default 8-case benchmark (`orbis eval`): **color 0.91**,
**temporal stability 0.91**, ~22 ms/chunk with the 4-step student. Shape (≈0.45)
and direction (≈0.43) are softer — blobby shapes at this scale and wall-bouncing
that reverses direction — and are reported without inflation. These numbers are on
the toy benchmark and are **not comparable** to the paper's results.

---

## 13. Reproducing

```bash
pip install -e .
python scripts/train_all.py orbis.pt        # ~15 min, 4 CPU cores (checkpoint is committed)
orbis eval
orbis generate --prompt "a red circle moving right" --chunks 6 --out out.gif
orbis live --prompt "a red circle moving right" --switch "3:a blue square moving up" --out live.gif
orbis demo --out demo.html
pytest
```

Key configuration (`orbis/config.py`):

| group | setting | value |
|---|---|---|
| world | frame / fps / SR scale | 32×32×3 / 8 / ×4 |
| vae | latent / downsample / base ch | 8×8×8 / ×4 / 48 |
| model | dim / depth / heads / patch | 128 / 4 / 4 / 2 |
| model | chunk / history / memory / text | 4 / 4 / 16 tokens / 8 |
| flow | teacher / student steps | 16 / 4 |
| training | vae / pretrain / stream / distill / sr | 900 / 700 / 1800 / 500 / 300 |

---

## 14. Paper → code map

| Paper mechanism | Module(s) |
|---|---|
| Unified live formulation `p_θ(z_k \| H_k, c_k, r_k)` | `model.py` |
| Rectified-flow objective (Eq. 1) | `flow.py` |
| Bounded multi-scale memory | `memory.py` |
| Progressive training (Fig. 3) | `train.py`, `dataset.py` |
| Few-step distillation | `distill.py` |
| Live versioned prompt switching / rolling summary | `session.py`, `engine.py` |
| Delivered-video runtime (T2V/I2V/V2V, progressive delivery) | `engine.py` |
| Streaming super-resolution | `superres.py` |
| Latent video model | `vae.py` |
| Multilingual prompting | `text.py`, `vocab.py` |
| Event-based evaluation | `eval.py` |

## 15. Limitations & honest scope

- A reference implementation of the *mechanisms*, not the trained Orbis model — no
  shared weights or data.
- Shape and precise direction fidelity are limited at this scale.
- Bounded memory's long-horizon benefit is architectural in this toy world.
- The GRPO reward-alignment and physics-aware (world-model consistency) stages of
  the paper's post-training are described but not implemented; distillation stands
  in for the few-step post-training path.

## References

The primary source is the archived PDF in `doc/references/`. Works cited in the
summary above (CausVid, Self-Forcing, Rolling/Causal Forcing, Oasis, LongLive,
Matrix-Game 2.0, Krea Realtime, Helios, Vidu S1, FramePack, FAR, MemFlow, VideoSSM,
FadeMem, Echo-Infinity, DOVER, VideoAlign, HPSv3, and others) appear in the
paper's own reference list (pp. 12–14 of the archived PDF).
