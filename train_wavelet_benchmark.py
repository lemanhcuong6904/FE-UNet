import os
import csv
import gc
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from fe_unet_sam2_full import (
    build_feunet_sam2,
    deep_supervision_loss,
    build_optimizer,
)

DATA_ROOT = "Kvasir-SEG/Kvasir-SEG"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "sam2/checkpoints/sam2.1_hiera_large.pt"

IMAGE_SIZE = 320
BATCH_SIZE = 1
EPOCHS = 30
PATIENCE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WAVELETS = ["haar", "db4", "sym4", "coif2"]

RESULT_CSV = "runs/wavelet_benchmark_results.csv"


class KvasirDataset(Dataset):
    def __init__(self, root):
        self.img_paths = sorted(glob.glob(os.path.join(root, "images", "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, "masks", "*")))

        assert len(self.img_paths) == len(self.mask_paths), "Số ảnh và mask không khớp"
        assert len(self.img_paths) > 0, f"Không tìm thấy ảnh trong {root}"

        self.img_tfm = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
        ])

        self.mask_tfm = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.img_tfm(img)
        mask = self.mask_tfm(mask)
        mask = (mask > 0.5).float()

        return img, mask


def dice_iou_from_logits(logits, mask):
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).float()

    inter = (pred * mask).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    mask_sum = mask.sum(dim=(1, 2, 3))

    dice = (2 * inter + 1e-6) / (pred_sum + mask_sum + 1e-6)

    union = pred_sum + mask_sum - inter
    iou = (inter + 1e-6) / (union + 1e-6)

    return dice.mean().item(), iou.mean().item()


def evaluate(model, loader):
    model.eval()
    dices, ious = [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outs = model(imgs)
            dice, iou = dice_iou_from_logits(outs[0], masks)

            dices.append(dice)
            ious.append(iou)

    return sum(dices) / len(dices), sum(ious) / len(ious)


def train_one_wavelet(wavelet, train_loader, val_loader, test_loader):
    save_dir = f"runs/kvasir_{wavelet}"
    save_path = f"{save_dir}/best_feunet_kvasir_{wavelet}.pth"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f"Training wavelet: {wavelet}")
    print("=" * 80)

    model = build_feunet_sam2(
        model_cfg=SAM2_CFG,
        ckpt_path=SAM2_CKPT,
        device=DEVICE,
        base_ch=64,
        freeze_hiera=True,
        wavelet=wavelet,
    )
    print(">>> Model built")

    optimizer = build_optimizer(model, lr=LR)
    print(">>> Optimizer built")
    best_val_dice = 0.0
    best_val_iou = 0.0
    best_epoch = 0
    no_improve = 0
    print(">>> Start training")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outs = model(imgs)
            loss = deep_supervision_loss(outs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_dice, val_iou = evaluate(model, val_loader)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {best_epoch}, Val Dice={best_val_dice:.4f}")
        else:
            no_improve += 1

        print(
            f"[{wavelet}] Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Best: {best_val_dice:.4f}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping {wavelet} at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    test_dice, test_iou = evaluate(model, test_loader)

    del model, optimizer
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "wavelet": wavelet,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "save_path": save_path,
    }


def main():
    os.makedirs("runs", exist_ok=True)
    torch.backends.cudnn.benchmark = True

    dataset = KvasirDataset(DATA_ROOT)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    results = []

    for wavelet in WAVELETS:
        result = train_one_wavelet(wavelet, train_loader, val_loader, test_loader)
        results.append(result)

        with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for r in results:
        print(
            f"{r['wavelet']:8s} | "
            f"Best Epoch: {r['best_epoch']:2d} | "
            f"Val Dice: {r['best_val_dice']:.4f} | "
            f"Val IoU: {r['best_val_iou']:.4f} | "
            f"Test Dice: {r['test_dice']:.4f} | "
            f"Test IoU: {r['test_iou']:.4f}"
        )

    print(f"Saved CSV to {RESULT_CSV}")


if __name__ == "__main__":
    main()