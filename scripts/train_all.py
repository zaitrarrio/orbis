"""Train every Orbis stage and save a checkpoint.

Usage:  python scripts/train_all.py [out_path] [scale]
    out_path : checkpoint file (default orbis.pt)
    scale    : multiply all step counts (default 1.0; use 0.1 for a smoke run)
"""

import sys
import time

import torch

sys.path.insert(0, ".")
from orbis.config import OrbisConfig
from orbis.device import device_name, get_device
from orbis.train import train_vae, train_generator, train_sr
from orbis.distill import distill
from orbis.system import OrbisSystem


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "orbis.pt"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    device = get_device()
    print(f"[all] device {device_name(device)}")
    if device.type == "cpu":
        torch.set_num_threads(4)
    s = lambda n: max(20, int(n * scale))
    t0 = time.time()
    system = OrbisSystem.build(OrbisConfig()).to(device)
    train_vae(system, steps=s(900))
    train_generator(system, pretrain_steps=s(700), stream_steps=s(1800))
    distill(system, steps=s(500))
    train_sr(system, steps=s(300))
    system.save(out)
    print(f"[all] saved {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
