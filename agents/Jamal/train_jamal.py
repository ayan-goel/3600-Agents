from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running as script or module
if __package__ is None or __package__ == "":
    # Add project root to path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.append(_ROOT)
    from agents.Jamal.net import ChickenNet  # type: ignore
    from agents.Jamal.data import load_offline_dataset  # type: ignore
else:
    from .net import ChickenNet
    from .data import load_offline_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_supervised(
    net: ChickenNet,
    replay: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    batch_size: int = 256,
    epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> None:
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for epoch in range(epochs):
        idxs = np.random.permutation(len(replay))
        losses = []
        for start in range(0, len(replay), batch_size):
            sel = idxs[start : start + batch_size]
            batch = [replay[i] for i in sel]
            boards_np = np.stack([b[0] for b in batch], axis=0)
            scalars_np = np.stack([b[1] for b in batch], axis=0)
            pis_np = np.stack([b[2] for b in batch], axis=0)
            zs_np = np.array([b[3] for b in batch], dtype=np.float32)

            boards = torch.from_numpy(boards_np).to(DEVICE)
            scalars = torch.from_numpy(scalars_np).to(DEVICE)
            pi_targets = torch.from_numpy(pis_np).to(DEVICE)
            z_targets = torch.from_numpy(zs_np).to(DEVICE)

            optimizer.zero_grad()
            logits, values = net(boards, scalars)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(pi_targets * log_probs).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, z_targets)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Supervised] epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Jamal from offline matches (+optional self-play later).")
    parser.add_argument("--match_dir", type=str, default="agents/matches", help="Directory with JSON matches")
    parser.add_argument("--limit", type=int, default=2000, help="Limit number of JSON files to parse")
    parser.add_argument("--save", type=str, default="jamal_weights.pt", help="Output weights filename")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    print("Loading offline dataset...")
    rows = load_offline_dataset(args.match_dir, board_size=8, limit=args.limit)
    if not rows:
        print("No offline data found; exiting.")
        return
    print(f"Loaded {len(rows)} training rows.")

    # Initialize Jamal network (reuse Dontique's net architecture)
    net = ChickenNet(board_size=8).to(DEVICE)
    _train_supervised(net, rows, batch_size=args.batch, epochs=args.epochs, lr=3e-4, weight_decay=1e-4)

    out_path = os.path.join(os.path.dirname(__file__), args.save)
    torch.save(net.state_dict(), out_path)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()


