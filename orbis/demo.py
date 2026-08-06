"""Build a self-contained interactive HTML console from a trained checkpoint.

Everything is embedded (frames as PNG data URIs, metadata as JSON) so the page
is a single file with no external requests -- suitable for publishing as an
Artifact.  The centrepiece is the live prompt-switch demo, which shows that
already-delivered frames are immutable while a switch redirects only the
uncommitted future.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

from . import media


def _frames_to_uris(frames: np.ndarray) -> List[str]:
    return [media.png_data_uri(f, scale=1) for f in frames]


def _collect(ckpt: str) -> dict:
    from .engine import LiveEngine
    from .system import OrbisSystem
    from .eval import evaluate, default_cases
    from .text import parse_controls
    from .world import SceneSpec, rollout

    eng = LiveEngine(OrbisSystem.load(ckpt))
    cf = eng.cfg.model.chunk_frames
    fps = eng.cfg.world.fps
    H, W = eng.cfg.world.height, eng.cfg.world.width

    def scene_meta(session_log, n_frames):
        meta = []
        for i in range(n_frames):
            k = i // cf
            entry = session_log[k] if k < len(session_log) else session_log[-1]
            meta.append({"chunk": k, "version": entry["version"],
                         "prompt": entry["prompt"]})
        return meta

    scenes = []
    t2v_prompts = [
        "a red circle moving right",
        "a blue square moving down",
        "a green triangle moving up",
        "a yellow circle moving left",
        "a magenta triangle moving downleft",
    ]
    latencies = []
    for pid, prompt in enumerate(t2v_prompts):
        res = eng.generate_video(prompt, n_chunks=6, seed=100 + pid)
        scenes.append({
            "id": f"t2v{pid}", "title": prompt, "mode": "T2V",
            "prompt": prompt, "fps": fps, "w": W, "h": H,
            "frames": _frames_to_uris(res["frames"]),
            "meta": scene_meta(res["log"], len(res["frames"])),
            "boundary": None,
        })

    # measure per-chunk latency
    s = eng.start("a red circle moving right")
    for _ in range(4):
        c = eng.generate_chunk(s)
        latencies.append(c.gen_seconds * 1000)
    latency_ms = float(np.median(latencies))

    # --- switch demo: baseline (no switch) vs live switch -------------------
    base_prompt = "a red circle moving right"
    switch_prompt = "a blue square moving up"
    boundary_chunk = 3
    baseline = eng.generate_video(base_prompt, n_chunks=6, seed=7)
    switched = eng.generate_video(base_prompt, n_chunks=6, seed=7,
                                  schedule={boundary_chunk: switch_prompt})
    switch_demo = {
        "prompt": base_prompt, "switch": switch_prompt,
        "boundary_chunk": boundary_chunk, "boundary_frame": boundary_chunk * cf,
        "fps": fps, "w": W, "h": H, "chunk_frames": cf,
        "baseline": _frames_to_uris(baseline["frames"]),
        "switched": _frames_to_uris(switched["frames"]),
        "meta": scene_meta(switched["log"], len(switched["frames"])),
    }

    # --- I2V ----------------------------------------------------------------
    c = parse_controls("a cyan square moving right")
    spec = SceneSpec(**{k: v for k, v in c.items()
                        if k in ("shape", "color", "direction", "speed", "size")})
    ref_frame, _ = rollout(spec, 1, H, W)
    i2v = eng.generate_video("a cyan square moving right", n_chunks=6,
                             mode="i2v", image=ref_frame[0], seed=3)
    i2v_scene = {
        "id": "i2v", "title": "image -> video (I2V)", "mode": "I2V",
        "prompt": "a cyan square moving right", "fps": fps, "w": W, "h": H,
        "ref": media.png_data_uri(ref_frame[0], scale=1),
        "frames": _frames_to_uris(i2v["frames"]),
        "meta": scene_meta(i2v["log"], len(i2v["frames"])),
        "boundary": None,
    }

    # --- super-resolution showcase -----------------------------------------
    sr_res = eng.generate_video("a red circle moving right", n_chunks=3,
                                seed=5, superres=True)
    sr_scene = {
        "lr": _frames_to_uris(sr_res["frames"]),
        "hr": _frames_to_uris(sr_res["hr_frames"]),
        "fps": fps, "scale": eng.cfg.sr.scale,
        "lw": W, "lh": H, "hw": W * eng.cfg.sr.scale, "hh": H * eng.cfg.sr.scale,
    }

    # --- metrics ------------------------------------------------------------
    ev = evaluate(eng, cases=default_cases(), n_chunks=4, seed=0)

    n_params = sum(p.numel() for p in eng.gen.parameters())
    config = {
        "chunk_frames": cf, "fps": fps,
        "student_steps": eng.cfg.flow.student_steps,
        "teacher_steps": eng.cfg.flow.teacher_steps,
        "memory_tokens": eng.cfg.model.memory_tokens,
        "history_frames": eng.cfg.model.history_frames,
        "native": f"{W}x{H}", "sr_native": f"{W*eng.cfg.sr.scale}x{H*eng.cfg.sr.scale}",
        "params": n_params, "latency_ms": round(latency_ms, 1),
        "distilled": eng.system.distilled,
    }
    return {"scenes": scenes, "switch": switch_demo, "i2v": i2v_scene,
            "sr": sr_scene, "metrics": ev, "config": config}


def build_demo(ckpt: str, out_path: str = "demo.html") -> str:
    data = _collect(ckpt)
    html = _HTML_TEMPLATE.replace("/*DATA*/", json.dumps(data))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# The page template lives in demo_template.py to keep this module readable.
from .demo_template import HTML_TEMPLATE as _HTML_TEMPLATE  # noqa: E402
