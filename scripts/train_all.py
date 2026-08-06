"""Train every Orbis stage and save a checkpoint.

Usage:  python scripts/train_all.py [out_path] [scale] [--backbone toy|wan]

    out_path : checkpoint file (default orbis.pt)
    scale    : multiply all step counts (default 1.0; use 0.1 for a smoke run)
    --backbone toy (default) | wan  — wan uses the smoke Wan stub + distill + SR
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from orbis.config import OrbisConfig, wan_smoke_config
from orbis.device import device_name, get_device
from orbis.train import train_vae, train_generator, train_sr
from orbis.distill import distill
from orbis.system import OrbisSystem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="orbis.pt")
    ap.add_argument("scale", nargs="?", type=float, default=1.0)
    ap.add_argument("--backbone", choices=("toy", "wan"), default="toy")
    args = ap.parse_args()

    device = get_device()
    print(f"[all] device {device_name(device)} backbone={args.backbone}")
    if device.type == "cpu":
        torch.set_num_threads(4)
    s = lambda n: max(20, int(n * args.scale))
    t0 = time.time()

    if args.backbone == "wan":
        cfg = wan_smoke_config()
        system = OrbisSystem.build(cfg).to(device)
        train_vae(system, steps=s(200))
        train_generator(system, pretrain_steps=s(100), stream_steps=s(200))
        distill(system, steps=s(100))
        train_sr(system, steps=s(50))
    else:
        system = OrbisSystem.build(OrbisConfig()).to(device)
        train_vae(system, steps=s(900))
        train_generator(system, pretrain_steps=s(700), stream_steps=s(1800))
        distill(system, steps=s(500))
        train_sr(system, steps=s(300))

    system.save(args.out)
    print(f"[all] saved {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
