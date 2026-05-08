"""Train ResNet18 ReID encoder on MOT GT (CE + batch-hard triplet).

Usage:
    python -m src.reid_train --seqs MOT_02 MOT_03 MOT_04 MOT_05 --epochs 30
    python -m src.reid_train --epochs 50 --P 16 --K 4 --iters_per_epoch 300

Output: weights/reid_resnet18.pth — auto-loaded by SiamEmbedder if present.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import DEVICE, REID_WEIGHTS
from .reid_dataset import MOTReIDDataset, PKSampler
from .reid_model import ReIDNet


def batch_hard_triplet(emb: torch.Tensor, gids: torch.Tensor, margin: float = 0.3):
    """Hermans batch-hard triplet on cosine distances. emb is L2-normalized."""
    sim = emb @ emb.t()
    dist = 1.0 - sim
    same = gids[:, None] == gids[None, :]
    eye = torch.eye(len(gids), dtype=torch.bool, device=emb.device)
    same = same & ~eye
    diff = ~(same | eye)
    # hardest positive (max dist among same id)
    pos = (dist * same.float() - 1e6 * (~same).float()).max(dim=1).values
    # hardest negative (min dist among diff id)
    neg = (dist * diff.float() + 1e6 * (~diff).float()).min(dim=1).values
    return torch.clamp(pos - neg + margin, min=0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+",
                    default=["MOT_02", "MOT_03", "MOT_04", "MOT_05"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=16, help="identities per batch")
    ap.add_argument("--K", type=int, default=4, help="samples per identity")
    ap.add_argument("--iters_per_epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REID_WEIGHTS))
    args = ap.parse_args()

    print(f"[device] {DEVICE}")

    ds = MOTReIDDataset(args.seqs, augment=True)
    sampler = PKSampler(ds, p=args.P, k=args.K, num_iters=args.iters_per_epoch)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=args.workers > 0,
    )

    net = ReIDNet(num_classes=ds.num_classes).to(DEVICE)
    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(args.epochs):
        net.train()
        loss_sum = ce_sum = tri_sum = 0.0
        correct = total = 0
        pbar = tqdm(loader, desc=f"ep {epoch+1:02d}/{args.epochs}")
        for x, gid in pbar:
            x = x.to(DEVICE, non_blocking=True)
            gid = gid.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                emb, logits = net(x)
                l_ce = ce(logits, gid)
                l_tri = batch_hard_triplet(emb, gid, margin=args.margin)
                loss = args.lambda_ce * l_ce + args.lambda_tri * l_tri
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item()
            ce_sum += l_ce.item()
            tri_sum += l_tri.item()
            correct += (logits.argmax(1) == gid).sum().item()
            total += gid.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{l_ce.item():.3f}",
                             tri=f"{l_tri.item():.3f}")
        sched.step()
        n = max(1, len(loader))
        print(f"[ep{epoch+1}] loss={loss_sum/n:.3f} ce={ce_sum/n:.3f} "
              f"tri={tri_sum/n:.3f} acc={correct/max(1,total):.3f} "
              f"lr={sched.get_last_lr()[0]:.2e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    torch.save(net.export_backbone_state_dict(), out_path)
    print(f"[save] backbone weights -> {out_path}")


if __name__ == "__main__":
    main()
