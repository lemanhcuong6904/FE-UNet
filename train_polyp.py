from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from fe_unet_sam2_full import (
    build_cosine_scheduler,
    build_feunet_sam2,
    build_optimizer,
    build_tiny_debug_feunet,
    deep_supervision_loss,
)
from polyp_dataset import build_polyp_loader


DATA_ROOT = "polyp_data"
OUTPUT_DIR = "runs/polyp_feunet_320"

SAM2_CFG = "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "sam2/checkpoints/sam2.1_hiera_large.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 8
LR = 0.001
WEIGHT_DECAY = 1e-3
IMAGE_SIZE = 320
NUM_WORKERS = 0
USE_AMP = True
AMP_DTYPE = "bf16"
MAX_SKIPPED_BATCHES = 10
RESUME_TRAINING = True
RESUME_CHECKPOINT = r""  # "runs/polyp_feunet_350/last.pt"


def dice_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    dims = (1, 2, 3)
    inter = (preds * masks).sum(dims)
    pred_sum = preds.sum(dims)
    mask_sum = masks.sum(dims)
    union = pred_sum + mask_sum - inter
    dice = ((2 * inter + 1.0) / (pred_sum + mask_sum + 1.0)).mean()
    iou = ((inter + 1.0) / (union + 1.0)).mean()
    return float(dice), float(iou)


def tensors_are_finite(tensors: list[torch.Tensor]) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if AMP_DTYPE == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("BF16 AMP is not supported on this device. Falling back to FP16 AMP.")
    return torch.float16


def build_model(device: torch.device) -> torch.nn.Module:
    if SAM2_CFG and SAM2_CKPT:
        return build_feunet_sam2(
            model_cfg=SAM2_CFG,
            ckpt_path=SAM2_CKPT,
            device=device,
            num_classes=1,
        )
    print("SAM2 cfg/ckpt not provided. Using tiny debug FE-UNet backbone.")
    return build_tiny_debug_feunet(num_classes=1).to(device)


def validate_image_size(image_size: int) -> None:
    if image_size % 4 != 0:
        raise ValueError("IMAGE_SIZE must be divisible by 4 for SAM2/Hiera patch embedding.")
    patch_grid = image_size // 4
    if patch_grid % 8 != 0:
        raise ValueError(
            "IMAGE_SIZE is incompatible with SAM2/Hiera window positional embedding. "
            "Use a multiple of 32, e.g. 256, 288, 320, 352, or 384."
        )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    if not RESUME_TRAINING:
        return 1, -1.0

    checkpoint_path = Path(RESUME_CHECKPOINT)
    if not checkpoint_path.is_file():
        print(f"Resume checkpoint not found: {checkpoint_path}. Starting from epoch 1.")
        return 1, -1.0

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    last_epoch = int(checkpoint.get("epoch", 0))
    best_dice = float(checkpoint.get("best_dice", checkpoint.get("val_dice", -1.0)))
    start_epoch = last_epoch + 1
    print(
        f"Resumed from {checkpoint_path} at epoch {last_epoch}. "
        f"Next epoch: {start_epoch}, best_dice={best_dice:.4f}"
    )
    return start_epoch, best_dice


def gradients_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all().item():
            return False
    return True


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = get_amp_dtype(device)
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    total_loss = 0.0
    total_dice = 0.0
    seen = 0
    skipped = 0

    progress = tqdm(loader, desc=f"[Epoch {epoch:02d}/{EPOCHS}] train", leave=True)
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_list = model(images)

        logits_list = [logits.float() for logits in logits_list]
        masks = masks.float()
        if not tensors_are_finite(logits_list):
            skipped += images.size(0)
            progress.set_postfix(status="bad_logits", skipped=skipped)
            if skipped >= MAX_SKIPPED_BATCHES * loader.batch_size:
                raise RuntimeError("Too many skipped training samples because logits contain NaN/Inf.")
            continue

        loss = deep_supervision_loss(logits_list, masks)
        if not torch.isfinite(loss).item():
            skipped += images.size(0)
            progress.set_postfix(status="bad_loss", skipped=skipped)
            if skipped >= MAX_SKIPPED_BATCHES * loader.batch_size:
                raise RuntimeError("Too many skipped training samples because loss is NaN/Inf.")
            continue

        dice, _ = dice_iou_from_logits(logits_list[0].detach(), masks)
        if scaler_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if not gradients_are_finite(model):
            skipped += images.size(0)
            optimizer.zero_grad(set_to_none=True)
            if scaler_enabled:
                scaler.update()
            progress.set_postfix(status="bad_grad", skipped=skipped)
            if skipped >= MAX_SKIPPED_BATCHES * loader.batch_size:
                raise RuntimeError("Too many skipped training samples because gradients contain NaN/Inf.")
            continue

        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_dice += dice * batch_size
        seen += batch_size
        progress.set_postfix(
            loss=f"{total_loss / seen:.4f}",
            dice=f"{total_dice / seen:.4f}",
            step_loss=f"{float(loss.detach()):.4f}",
            step_dice=f"{dice:.4f}",
            skipped=skipped,
        )

    if seen == 0:
        raise RuntimeError("All training batches were skipped because of non-finite values.")
    return total_loss / seen


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    seen = 0
    skipped = 0

    progress = tqdm(loader, desc=f"[Epoch {epoch:02d}/{EPOCHS}] val", leave=True)
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        logits_list = [logits.float() for logits in model(images)]
        if not tensors_are_finite(logits_list):
            skipped += images.size(0)
            progress.set_postfix(status="bad_logits", skipped=skipped)
            continue

        loss = deep_supervision_loss(logits_list, masks)
        if not torch.isfinite(loss).item():
            skipped += images.size(0)
            progress.set_postfix(status="bad_loss", skipped=skipped)
            continue

        dice, iou = dice_iou_from_logits(logits_list[0], masks)
        batch_size = images.size(0)
        total_loss += float(loss) * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        seen += batch_size
        progress.set_postfix(
            loss=f"{total_loss / seen:.4f}",
            dice=f"{total_dice / seen:.4f}",
            step_loss=f"{float(loss):.4f}",
            step_dice=f"{dice:.4f}",
            skipped=skipped,
        )

    if seen == 0:
        raise RuntimeError("All validation batches were skipped because of non-finite values.")
    return total_loss / seen, total_dice / seen, total_iou / seen


def main() -> None:
    validate_image_size(IMAGE_SIZE)
    device = torch.device(DEVICE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_polyp_loader(
        DATA_ROOT, "train", BATCH_SIZE, IMAGE_SIZE, num_workers=NUM_WORKERS
    )
    val_loader = build_polyp_loader(
        DATA_ROOT, "val", BATCH_SIZE, IMAGE_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model(device)
    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_cosine_scheduler(optimizer, epochs=EPOCHS)

    start_epoch, best_dice = maybe_resume_training(model, optimizer, scheduler, device)
    if start_epoch > EPOCHS:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; EPOCHS={EPOCHS}. Nothing to train.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, USE_AMP, epoch)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device, epoch)
        scheduler.step()

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_dice": val_dice,
            "best_dice": max(best_dice, val_dice),
            "config": {
                "data_root": DATA_ROOT,
                "output_dir": OUTPUT_DIR,
                "sam2_cfg": SAM2_CFG,
                "sam2_ckpt": SAM2_CKPT,
                "device": DEVICE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "image_size": IMAGE_SIZE,
                "num_workers": NUM_WORKERS,
                "use_amp": USE_AMP,
                "amp_dtype": AMP_DTYPE,
                "max_skipped_batches": MAX_SKIPPED_BATCHES,
                "resume_training": RESUME_TRAINING,
                "resume_checkpoint": RESUME_CHECKPOINT,
            },
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(ckpt, output_dir / "best.pt")

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"dice={val_dice:.4f} iou={val_iou:.4f} lr={lr:.6f}"
        )


if __name__ == "__main__":
    main()
