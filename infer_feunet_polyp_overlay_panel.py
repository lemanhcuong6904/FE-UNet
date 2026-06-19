from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from fe_unet_sam2_full import build_feunet_sam2
from polyp_dataset import build_polyp_loader


# =========================
# Config
# =========================

DATA_ROOT = "polyp_data"

# Checkpoint sau khi train. Nên dùng best.pt để inference/evaluate.
CHECKPOINT = "runs/kvasir_feunet/best.pt"

# Nơi lưu kết quả inference + panel ảnh.
OUTPUT_DIR = "runs/kvasir_feunet/inference_test"

SAM2_CFG = "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "sam2/checkpoints/sam2.1_hiera_large.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_WORKERS = 0
SPLIT = "ETIS_test"  # kvasir_test, CVC-ColonDB_test, CVC-300_test, ETIS_test

THRESHOLD = 0.5

SAVE_PANELS = True
MAX_PANELS = 30
PANEL_DPI = 150

# Overlay setting:
# Panel sẽ là: Image | GT overlay | Pred overlay
GT_COLOR = (0.0, 1.0, 0.0)      # green
PRED_COLOR = (1.0, 0.0, 0.0)    # red
OVERLAY_ALPHA = 0.45

USE_AMP = True
AMP_DTYPE = "bf16"


# =========================
# Helpers
# =========================

def validate_image_size(image_size: int) -> None:
    if image_size % 4 != 0:
        raise ValueError("IMAGE_SIZE must be divisible by 4 for SAM2/Hiera patch embedding.")
    patch_grid = image_size // 4
    if patch_grid % 8 != 0:
        raise ValueError(
            "IMAGE_SIZE is incompatible with SAM2/Hiera window positional embedding. "
            "Use a multiple of 32, e.g. 256, 288, 320, 352, or 384."
        )


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if AMP_DTYPE == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("BF16 AMP is not supported on this device. Falling back to FP16 AMP.")
    return torch.float16


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(device: torch.device) -> torch.nn.Module:
    return build_feunet_sam2(
        model_cfg=SAM2_CFG,
        ckpt_path=SAM2_CKPT,
        device=device,
        num_classes=1,
    )


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "best_dice" in checkpoint:
            print(f"Checkpoint best_dice: {float(checkpoint['best_dice']):.4f}")

    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def dice_iou_per_sample_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        dice: [B]
        iou:  [B]

    Mean Dice / IoU được tính theo từng ảnh, rồi lấy trung bình trên toàn dataset.
    """
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    masks = (masks.float() > 0.5).float()
    preds = (torch.sigmoid(logits.float()) >= threshold).float()

    dims = (1, 2, 3)
    inter = (preds * masks).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    mask_sum = masks.sum(dim=dims)
    union = pred_sum + mask_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + mask_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    image: [3,H,W].
    Nếu ảnh không nằm trong [0,1], normalize min-max để hiển thị.
    """
    image = image.detach().cpu().float()

    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(image.shape)}")

    image_np = image.permute(1, 2, 0).numpy()

    img_min = float(image_np.min())
    img_max = float(image_np.max())
    if img_min < 0.0 or img_max > 1.0:
        image_np = (image_np - img_min) / max(img_max - img_min, 1e-7)

    return image_np.clip(0.0, 1.0)


def tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """
    mask: [1,H,W] hoặc [H,W].
    """
    mask = mask.detach().cpu().float()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.numpy()


def make_overlay(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay binary/soft mask lên ảnh RGB.

    Args:
        image_np: [H,W,3], range [0,1].
        mask_np:  [H,W], giá trị 0/1 hoặc soft mask.
        color:    RGB tuple trong [0,1].
        alpha:    độ trong suốt của mask overlay.
    """
    image_np = image_np.astype(np.float32).clip(0.0, 1.0)
    mask_np = mask_np.astype(np.float32)

    if mask_np.ndim != 2:
        raise ValueError(f"Expected mask [H,W], got {mask_np.shape}")

    binary = mask_np > 0.5
    overlay = image_np.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    overlay[binary] = (1.0 - alpha) * overlay[binary] + alpha * color_arr
    return overlay.clip(0.0, 1.0)


def make_safe_name(name, index: int) -> str:
    if isinstance(name, (list, tuple)):
        name = name[0] if len(name) > 0 else f"sample_{index:05d}"
    name = str(name)
    stem = Path(name).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)
    return stem or f"sample_{index:05d}"


def save_prediction_panel(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: Path,
    dice: float,
    iou: float,
    title: str = "",
) -> None:
    image_np = tensor_image_to_numpy(image)
    gt_np = tensor_mask_to_numpy(gt_mask)
    pred_np = tensor_mask_to_numpy(pred_mask)

    gt_overlay = make_overlay(
        image_np=image_np,
        mask_np=gt_np,
        color=GT_COLOR,
        alpha=OVERLAY_ALPHA,
    )
    pred_overlay = make_overlay(
        image_np=image_np,
        mask_np=pred_np,
        color=PRED_COLOR,
        alpha=OVERLAY_ALPHA,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT overlay")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Pred overlay\nDice={dice:.4f}, IoU={iou:.4f}")

    if title:
        fig.suptitle(title)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


# =========================
# Inference / Evaluation
# =========================

@torch.no_grad()
def inference_and_evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
) -> tuple[float, float]:
    model.eval()

    amp_enabled = USE_AMP and device.type == "cuda"
    amp_dtype = get_amp_dtype(device)

    panel_dir = output_dir / "panels"
    pred_dir = output_dir / "pred_masks"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    saved_panels = 0

    progress = tqdm(loader, desc=f"Inference [{SPLIT}]", leave=True)

    for batch_idx, batch in enumerate(progress):
        # Dataset hiện tại trả về: images, masks, names_or_paths
        images, masks, names = batch

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_list = model(images)

        # Dùng output đầu tiên/head có độ phân giải cao nhất.
        logits = logits_list[0].float()
        probs = torch.sigmoid(logits)
        preds = (probs >= THRESHOLD).float()

        dice_vec, iou_vec = dice_iou_per_sample_from_logits(
            logits,
            masks,
            threshold=THRESHOLD,
        )

        batch_size = images.size(0)
        total_dice += float(dice_vec.sum())
        total_iou += float(iou_vec.sum())
        total_samples += batch_size

        mean_dice = total_dice / total_samples
        mean_iou = total_iou / total_samples
        progress.set_postfix(mean_dice=f"{mean_dice:.4f}", mean_iou=f"{mean_iou:.4f}")

        # Lưu panel Image | GT overlay | Pred overlay
        if SAVE_PANELS and saved_panels < MAX_PANELS:
            for i in range(batch_size):
                global_idx = batch_idx * loader.batch_size + i
                safe_name = make_safe_name(names[i] if isinstance(names, Sequence) else names, global_idx)

                pred_mask = preds[i]
                panel_path = panel_dir / f"{global_idx:05d}_{safe_name}.png"
                save_prediction_panel(
                    image=images[i],
                    gt_mask=masks[i],
                    pred_mask=pred_mask,
                    save_path=panel_path,
                    dice=float(dice_vec[i]),
                    iou=float(iou_vec[i]),
                    title=safe_name,
                )

                # Lưu riêng predicted binary mask, tiện cho hậu xử lý.
                pred_save_path = pred_dir / f"{global_idx:05d}_{safe_name}.png"
                plt.imsave(pred_save_path, tensor_mask_to_numpy(pred_mask), cmap="gray", vmin=0, vmax=1)

                saved_panels += 1
                if saved_panels >= MAX_PANELS:
                    break

    if total_samples == 0:
        raise RuntimeError("No samples were evaluated.")

    mean_dice = total_dice / total_samples
    mean_iou = total_iou / total_samples

    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text(
        f"checkpoint={CHECKPOINT}\n"
        f"split={SPLIT}\n"
        f"image_size={IMAGE_SIZE}\n"
        f"threshold={THRESHOLD}\n"
        f"num_samples={total_samples}\n"
        f"mean_dice={mean_dice:.6f}\n"
        f"mean_iou={mean_iou:.6f}\n"
        f"panel_format=Image | GT overlay | Pred overlay\n"
        f"gt_color={GT_COLOR}\n"
        f"pred_color={PRED_COLOR}\n"
        f"overlay_alpha={OVERLAY_ALPHA}\n",
        encoding="utf-8",
    )

    return mean_dice, mean_iou


def main() -> None:
    validate_image_size(IMAGE_SIZE)

    device = torch.device(DEVICE)
    output_dir = Path(OUTPUT_DIR) / SPLIT
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_polyp_loader(
        DATA_ROOT,
        SPLIT,
        BATCH_SIZE,
        IMAGE_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model(device)
    load_model_weights(model, CHECKPOINT, device)

    mean_dice, mean_iou = inference_and_evaluate(
        model=model,
        loader=loader,
        device=device,
        output_dir=output_dir,
    )

    print("=" * 60)
    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Mean IoU : {mean_iou:.6f}")
    print(f"Panels saved to: {output_dir / 'panels'}")
    print(f"Pred masks saved to: {output_dir / 'pred_masks'}")
    print(f"Metrics saved to: {output_dir / 'metrics.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
