"""Command-line interface for the Orbis reference implementation.

    orbis train    [--out orbis.pt] [--scale 1.0]
    orbis generate --prompt "a red circle moving right" [--chunks 6] [--out out.gif]
                   [--mode t2v|i2v|v2v] [--image f.png] [--superres] [--steps N]
    orbis live     --prompt "..." [--switch "3:a blue square moving up"] [--chunks 8]
    orbis eval     [--ckpt orbis.pt] [--chunks 4]
    orbis demo     [--out demo.html]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

DEFAULT_CKPT = "orbis.pt"


def _load(ckpt: str):
    from .system import OrbisSystem
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint '{ckpt}' not found -- run `orbis train` first.")
    return OrbisSystem.load(ckpt)


def _engine(ckpt: str, steps=None, seed=0):
    from .engine import LiveEngine
    return LiveEngine(_load(ckpt), steps=steps, seed=seed)


def _parse_switches(items) -> Dict[int, str]:
    out = {}
    for it in items or []:
        k, _, text = it.partition(":")
        out[int(k)] = text.strip()
    return out


# ---------------------------------------------------------------------------
def cmd_train(args):
    from .train import train_all
    train_all(out_path=args.out, scale=args.scale)


def cmd_generate(args):
    from . import media
    from .world import sample_scene, rollout
    eng = _engine(args.ckpt, steps=args.steps, seed=args.seed)
    image = video = None
    if args.mode == "i2v":
        if args.image:
            image = _read_png(args.image)
        else:  # synthesize a reference frame from the prompt
            import numpy as _np
            from .text import parse_controls
            from .world import SceneSpec
            c = parse_controls(args.prompt)
            spec = SceneSpec(**{k: v for k, v in c.items()
                                if k in ("shape", "color", "direction", "speed", "size")})
            image, _ = rollout(spec, 1, eng.cfg.world.height, eng.cfg.world.width)
            image = image[0]
    switches = _parse_switches(args.switch)
    res = eng.generate_video(args.prompt, n_chunks=args.chunks, mode=args.mode,
                             image=image, video=video, schedule=switches,
                             superres=args.superres)
    frames = res["hr_frames"] if args.superres else res["frames"]
    scale = 4 if args.superres else 12
    media.save_gif(args.out, list(frames), fps=eng.cfg.world.fps, scale=scale)
    print(f"wrote {args.out}  ({len(frames)} frames, {frames.shape[1]}x{frames.shape[2]} native)")
    for e in res["events"]:
        print(f"  @chunk {e['chunk']}: switch -> '{e['text']}' (v{e['version']})")


def cmd_live(args):
    from . import media
    eng = _engine(args.ckpt, steps=args.steps, seed=args.seed)
    switches = _parse_switches(args.switch)
    s = eng.start(args.prompt, mode="t2v")
    print(f"live session: '{args.prompt}'  ({eng.steps}-step student, "
          f"chunk={eng.cfg.model.chunk_frames} frames)")
    all_frames = []
    for k in range(args.chunks):
        if k in switches:
            upd = s.set_prompt(switches[k])
            print(f"  >> prompt update queued: '{switches[k]}' (v{upd.version}) "
                  f"-- admitted at next boundary")
        chunk = eng.generate_chunk(s)
        all_frames.append(chunk.frames)
        _print_ascii(chunk.frames[-1])
        print(f"  chunk {chunk.index:2d} | v{chunk.version} | "
              f"'{chunk.prompt}' | {chunk.gen_seconds*1000:.0f} ms")
    if args.out:
        frames = np.concatenate(all_frames, axis=0)
        media.save_gif(args.out, list(frames), fps=eng.cfg.world.fps, scale=12)
        print(f"wrote {args.out}")


def cmd_eval(args):
    from .eval import evaluate
    eng = _engine(args.ckpt, steps=args.steps, seed=args.seed)
    res = evaluate(eng, n_chunks=args.chunks, seed=args.seed)
    print("per-case:")
    for r in res["per_case"]:
        print(f"  {r['prompt']:36s} color {r['color_acc']:.2f} shape {r['shape_acc']:.2f} "
              f"stab {r['stability']:.2f} dir {r['direction_ok']:.2f} motion {r['motion']:.3f}")
    a = res["aggregate"]
    print("\naggregate:", {k: round(v, 3) for k, v in a.items()})


def cmd_demo(args):
    from .demo import build_demo
    path = build_demo(args.ckpt, args.out)
    print(f"wrote interactive demo -> {path}")


# ---------------------------------------------------------------------------
def _read_png(path):
    # minimal: rely on numpy .npy if provided, else error
    if path.endswith(".npy"):
        return np.load(path)
    sys.exit("only .npy reference frames are supported by the CLI reader")


def _print_ascii(frame: np.ndarray, cols: int = 24):
    from .eval import foreground_mask
    h, w = frame.shape[:2]
    step = max(1, w // cols)
    m = foreground_mask(frame)
    lines = []
    for y in range(0, h, step * 2):
        row = "".join("#" if m[y, x] else "." for x in range(0, w, step))
        lines.append("  " + row)
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(prog="orbis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train all stages and save a checkpoint")
    t.add_argument("--out", default=DEFAULT_CKPT)
    t.add_argument("--scale", type=float, default=1.0)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="generate a video to a GIF")
    g.add_argument("--prompt", required=True)
    g.add_argument("--chunks", type=int, default=6)
    g.add_argument("--mode", default="t2v", choices=["t2v", "i2v", "v2v"])
    g.add_argument("--image", default=None)
    g.add_argument("--switch", action="append", help="'chunk:prompt' switch")
    g.add_argument("--superres", action="store_true")
    g.add_argument("--steps", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="out.gif")
    g.add_argument("--ckpt", default=DEFAULT_CKPT)
    g.set_defaults(func=cmd_generate)

    lv = sub.add_parser("live", help="stream chunks with live prompt switching")
    lv.add_argument("--prompt", required=True)
    lv.add_argument("--switch", action="append", help="'chunk:prompt' switch")
    lv.add_argument("--chunks", type=int, default=8)
    lv.add_argument("--steps", type=int, default=None)
    lv.add_argument("--seed", type=int, default=0)
    lv.add_argument("--out", default=None)
    lv.add_argument("--ckpt", default=DEFAULT_CKPT)
    lv.set_defaults(func=cmd_live)

    e = sub.add_parser("eval", help="evaluate prompt alignment and stability")
    e.add_argument("--chunks", type=int, default=4)
    e.add_argument("--steps", type=int, default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--ckpt", default=DEFAULT_CKPT)
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="build a self-contained interactive HTML demo")
    d.add_argument("--out", default="demo.html")
    d.add_argument("--ckpt", default=DEFAULT_CKPT)
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
