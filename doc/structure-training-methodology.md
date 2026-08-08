# Structure-first training methodology

Goal: one compact moving shape (circle/square/triangle) on a dark background,
with live color/shape switches — before scaling to 480×832.

This document is the **gate** for the next training round. Do not start a new
train until `scripts/validate-structure-methodology.py` exits 0.

---

## 1. Problem statement (observed)

| Run | Canvas | Result |
|---|---|---|
| Wan real-scale + FG RF | 480×832 | Color OK; full-frame / speckles |
| Spatial finetunes #1–3 | 480×832 | bright%≈3–4%; `n_cc` hundreds |
| Endpoint finetune #4 | 480×832 | Interrupted (billing); unevaluated |
| Curriculum stage-1 | 128×128 | bright% OK; **`n_cc≈80` FAIL** |

**Working pieces:** synthetic GT world (single object), VAE recon of GT, color /
prompt-switch signal.

**Broken piece:** generator spatial mass placement under few-step sampling.

**Root causes (ranked):**
1. **AdaLN-Zero gate init (critical):** residual gates start at 0 → attn/MLP get
   ~no gradient (observed `1/182` params with grad). Fixed by opening gates to 1
   at init for scratch training (`orbis/modules.py`).
2. Single-σ velocity training ≠ few-step Euler inference.
3. Speckle / wash local minima once color is learned without structure.
4. Large canvas / coarse patches (480p).

---

## 2. Success criteria (hard gates)

### 2.1 Structure gate (`scripts/eval-structure.py`)

On mid-frame of a T2V GIF (threshold bright > 0.35):

| Metric | Pass |
|---|---|
| `bright_pct` | ∈ [0.5, 12] |
| `n_cc` | ≤ 3 |
| `top_frac` | ≥ 0.6 |
| `local_mass` | ≥ 0.4 |

Ideal visual: one filled red disk translating right; live switch → one blue
square moving up. Reference look: `assets/vast-run/vae-gt.gif`.

### 2.2 Stage gates (must all pass before scale-up)

| Stage | Config | Must pass |
|---|---|---|
| **V0** | Methodology validation script | Exit 0 (this doc §5) |
| **S0** | 64×64 structure micro | Structure gate on `out` + `live` |
| **S1** | 128×128 curriculum | Structure gate + live early/late color |
| **S2** | 480×832 real-scale | Structure gate + toy-style `eval` aggregates |

No stage may start until the previous stage’s gate is green. **Do not resume
from speckled checkpoints** as the base for a later stage.

---

## 3. Known parameters (locked)

### 3.1 Synthetic world (ground truth)

| Parameter | Value | Source |
|---|---|---|
| Scene | Single moving shape | `orbis/world.py` |
| Medium radius | `BASE_RADIUS = 0.16` (norm.) | `world.py` |
| Colors | Named palette RGB | `world.py` `COLORS` |
| Prompt | Controls shape/color/direction/size | `orbis/text.py` |

**Known property:** GT frames always satisfy the structure gate (validated in
§5 check `gt_structure`).

### 3.2 Stage S0 — micro canvas (next train)

| Parameter | Value |
|---|---|
| World | 64×64, fps 12, hr_scale 2 |
| VAE | latent_channels 16, downsample 4, base 32 → latent 16×16 |
| Model | dim 128, depth 6, heads 4, patch_size **2**, chunk 4, hist 4, mem 16, text 16 |
| Flow | teacher_steps 8, student_steps 4 |
| Backbone | `wan` stub, LoRA rank 8, `anchor_reference=True`, drift off |
| Batches (24GB) | vae 32, gen 16 |
| Steps | VAE 600 · pretrain 500 · stream 1200 · mid 200 · distill 300 · SR 100 |
| LR | VAE 3e-4 · gen 5e-4 · distill 1e-4 |

### 3.3 Stage S1 — curriculum (already implemented)

`wan_structure_curriculum_config`: 128×128, VAE÷4, dim 256, depth 8, patch 2,
teacher 12 / student 4. Retained only **after** S0 passes (retrain fresh; do
not fine-tune failed `orbis-wan-curr.pt` for structure).

### 3.4 Stage S2 — real-scale

`wan_real_scale_config`: 480×832, VAE÷8, dim 512, depth 12, patch 4,
teacher 16 / student 4. Init from S1 via resolution curriculum or cold start
with S0/S1 objectives; never from speckled 480p lineage.

### 3.5 Loss stack (structure objective)

Applied every generator step (S0/S1/S2):

| Term | Weight (default) | Definition |
|---|---|---|
| RF velocity | 1.0 | FG-weighted MSE; `fg_boost=24`, `bg_weight=1.5`; mean = Σ(w·err)/Σ(w) |
| Endpoint | **1.0** (primary) | 4-step Euler → GT; FG-weighted + 8× BG ‖z‖² + off-mask energy |
| Pixel aux | **0.75** | Decode `z_hat = z_σ − σ·v`; FG/BG MSE + Dice + BG energy kill |
| **Mask / centroid** | **2.5** (geom aux) | Soft Dice + centroid L2 on `z_hat` and endpoint decode |
| σ sampling | `power=2.0` | Bias toward low σ |
| Conditioning | `t2v_first=True` | ≥70% `text_only`, ≤10% history |

Distill (post stream): `0.35·MSE(student, teacher) + 0.65·MSE(student, GT)`,
same FG weights. Skip guidance/DMD/GRPO until structure gate passes.

**Algebraic identities (must hold — validated in §5):**
- If `v = ε − z`, then `z_hat = z_σ − σ·v = z`.
- If velocity is perfect at every σ, `student_steps` Euler from `ε` recovers `z`.

---

## 4. Training procedure (S0)

1. **Cold start** `OrbisSystem.build(s0_config)` — no resume from speckled ckpt.
2. **VAE** — train to recon MSE ≪ 1e-2; verify encode→decode of GT passes
   structure gate (`vae_recon_structure`).
3. **Generator** — pretrain (text_only/reference) + stream with full loss stack
   including mask/centroid; log endpoint loss separately every 100 steps.
4. **Mid-train** — event switches with same loss stack.
5. **Distill** — GT-weighted few-step only.
6. **SR** — light; presentation only.
7. **Eval** — `curr-out` / `curr-live` style GIFs + `eval-structure.py`.
8. **Decision** — PASS → proceed S1 cold start with same losses; FAIL → stop,
   adjust S0 (do not jump to 480p).

### 4.1 Explicitly out of scope until gate green

- Finetuning failed 480p / failed 128p weights for “more steps”
- Raising BG/FG weights alone as the only change
- Full post-train (guidance / DMD / GRPO)
- `--load-hf` (stub path only until structure works)

---

## 5. Validation suite (known parameters)

`scripts/validate-structure-methodology.py`

| Check ID | Known input | Expected |
|---|---|---|
| `gt_structure` | World rollout “red circle” @ 64 and 128 | Structure gate PASS |
| `rf_identity` | Random `z,ε,σ`; `v=ε−z` | `‖z_hat − z‖∞ < 1e-5` |
| `euler_perfect` | Perfect velocity fn; 4 steps | `‖z_end − z‖ / ‖z‖ < 1e-4` |
| `vae_recon_structure` | Curriculum ckpt VAE + GT @ 128 | Structure gate PASS on recon |
| `mask_loss_zero` | Pred = GT pixels | Dice term ≈ 0, centroid err ≈ 0 |
| `mask_loss_uniform` | Pred = uniform noise | Dice term ≫ 0 |
| `curr_gen_baseline` | `orbis-wan-curr.pt` sample | Document FAIL (regression baseline; not a methodology error) |

**Pass rule:** all checks except `curr_gen_baseline` must succeed.
`curr_gen_baseline` must run and report FAIL (proves the gate catches the
known bad model).

---

## 6. Implementation checklist before S0 train

- [x] `wan_structure_micro_config()` (64×64) in `orbis/config.py`
- [x] Mask + centroid aux in `_flow_step` / `geometry_loss.py` (weights §3.5)
- [x] `--micro` flag on `train-live-wan.py` wiring S0 steps/weights
- [x] Validation script exit 0 on GPU host (2026-08-07, RTX 4090)
- [ ] Operator confirms: no resume from `orbis-wan-curr.pt` / `orbis-wan-real.pt`

---

## 7. Relation to `doc/METHODOLOGY.md`

That document describes the full Live Model toy pipeline and Wan post-train
stack. **This document overrides** the Wan path for structure recovery:
endpoint + mask/centroid first; post-train later; staged canvases with hard
gates. Toy path numbers (shape_acc≈0.45) are **not** the structure-gate target
for Wan stages — Wan stages must pass §2.1.
