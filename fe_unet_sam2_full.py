"""Unofficial one-file PyTorch implementation of FE-UNet + SAM2/Hiera-L extractor.

Install official SAM2 before using the real backbone:
  git clone https://github.com/facebookresearch/sam2.git
  cd sam2 && pip install -e .

Example:
  model = build_feunet_sam2(
      model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml',
      ckpt_path='./checkpoints/sam2.1_hiera_large.pt',
      device='cuda',
  )
  logits_list = model(images)  # three logits maps, upsampled to input size
  loss = deep_supervision_loss(logits_list, masks)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy  # noqa: F401  # Initialize NumPy before PyTorch loads it through torch.storage.
import torch
from torch import Tensor, nn
import torch.nn.functional as F


SAM2_CFG = "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "sam2/checkpoints/sam2.1_hiera_large.pt"
PROFILE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROFILE_IMAGE_SIZE = 320
PROFILE_BATCH_SIZE = 1

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SAM2_REPO = PROJECT_ROOT / "sam2"


# -----------------------------
# Basic blocks
# -----------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch:int, out_ch:int, kernel_size:Union[int,Tuple[int,int]]=3,
                 stride:int=1, padding:Union[int,Tuple[int,int],None]=None,
                 dilation:int=1, groups:int=1, relu:bool=True):
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(((k - 1) // 2) * dilation for k in kernel_size)
            else:
                padding = ((kernel_size - 1) // 2) * dilation
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class DoubleConv(nn.Sequential):
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__(ConvBNReLU(in_ch,out_ch,3), ConvBNReLU(out_ch,out_ch,3))


# -----------------------------
# SAM2 / Hiera adapters + extractor
# -----------------------------

class AdapterBHWC(nn.Module):
    """Pre-Hiera adapter for official SAM2 Hiera blocks, which use [B,H,W,C] tensors."""
    def __init__(self, channels:int, bottleneck_ratio:int=4):
        super().__init__()
        hidden = max(1, channels // bottleneck_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.GELU(),
        )
    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

class HieraBlockWithPreAdapter(nn.Module):
    """x -> x + Adapter(x) -> frozen/original Hiera block."""
    def __init__(self, block:nn.Module, channels:int, bottleneck_ratio:int=4,
                 residual:bool=True, freeze_block:bool=True):
        super().__init__()
        self.adapter = AdapterBHWC(channels, bottleneck_ratio)
        self.block = block
        self.residual = residual
        if freeze_block:
            for p in self.block.parameters():
                p.requires_grad = False
    def forward(self, x:Tensor) -> Tensor:
        ax = self.adapter(x)
        return self.block(x + ax if self.residual else ax)

def _ensure_local_sam2_importable() -> None:
    """Make the bundled sam2/sam2 package importable when running from repo root."""
    package_dir = LOCAL_SAM2_REPO / "sam2"
    if not (package_dir / "build_sam.py").is_file():
        return

    repo_path = str(LOCAL_SAM2_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    loaded_sam2 = sys.modules.get("sam2")
    if loaded_sam2 is not None and getattr(loaded_sam2, "__file__", None) is None:
        del sys.modules["sam2"]

def _normalize_sam2_config_name(model_cfg:str) -> str:
    cfg_path = Path(model_cfg)
    parts = cfg_path.as_posix().split("/")
    if len(parts) >= 4 and parts[0] == "sam2" and parts[1] == "sam2" and parts[2] == "configs":
        return "/".join(parts[2:])
    return model_cfg

def load_official_sam2(model_cfg:str, ckpt_path:str, device:Union[str,torch.device]='cuda',
                       mode:str='eval', apply_postprocessing:bool=False) -> nn.Module:
    _ensure_local_sam2_importable()
    model_cfg = _normalize_sam2_config_name(model_cfg)
    hydra_overrides_extra = [
        "++model.image_encoder.neck.position_encoding.warmup_cache=false",
        "++model.memory_encoder.position_encoding.warmup_cache=false",
    ]
    try:
        from sam2.build_sam import build_sam2
    except Exception as e:
        raise ImportError(
            "Cannot import official SAM2. Install with:\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e .\n"
            "Or keep the local SAM2 repo at D:\\FE-UNet\\sam2.\n"
            f"Original error: {repr(e)}"
        ) from e
    try:
        model = build_sam2(model_cfg, ckpt_path, device=device, mode=mode,
                           hydra_overrides_extra=hydra_overrides_extra,
                           apply_postprocessing=apply_postprocessing)
    except TypeError:
        try:
            model = build_sam2(model_cfg, ckpt_path, device=device, mode=mode,
                               hydra_overrides_extra=hydra_overrides_extra)
        except TypeError:
            model = build_sam2(model_cfg, ckpt_path)
            model.to(device)
    return model.to(device)

def _find_hiera_trunk(sam2_model:nn.Module) -> nn.Module:
    if hasattr(sam2_model, 'image_encoder') and hasattr(sam2_model.image_encoder, 'trunk'):
        return sam2_model.image_encoder.trunk
    if hasattr(sam2_model, 'trunk'):
        return sam2_model.trunk
    raise AttributeError('Cannot find Hiera trunk. Expected sam2_model.image_encoder.trunk.')

def insert_hiera_adapters(trunk:nn.Module, bottleneck_ratio:int=4, residual:bool=True,
                          freeze_blocks:bool=True) -> nn.Module:
    if not hasattr(trunk, 'blocks'):
        raise AttributeError('Hiera trunk has no blocks attribute.')
    for i, block in enumerate(trunk.blocks):
        if isinstance(block, HieraBlockWithPreAdapter):
            continue
        channels = getattr(block, 'dim', None)
        if channels is None:
            channels = getattr(block, 'dim_out', None)
        if channels is None:
            raise AttributeError(f'Cannot infer channel dim for Hiera block {i}.')
        trunk.blocks[i] = HieraBlockWithPreAdapter(block, int(channels), bottleneck_ratio,
                                                   residual=residual, freeze_block=freeze_blocks)
    return trunk

class SAM2HieraFeatureExtractor(nn.Module):
    """Returns four SAM2/Hiera trunk features in high-to-low order.

    Expected channels for SAM2.1 Hiera-L: (144, 288, 576, 1152).
    The official Hiera trunk returns stage features before SAM2's FPN neck.
    """
    def __init__(self, sam2_model:nn.Module, insert_adapters_flag:bool=True,
                 adapter_bottleneck_ratio:int=4, adapter_residual:bool=True,
                 freeze_trunk:bool=True, normalize:bool=True, input_range:str='0_1',
                 expected_channels:Sequence[int]=(144,288,576,1152)):
        super().__init__()
        if input_range not in {'0_1','0_255'}:
            raise ValueError("input_range must be '0_1' or '0_255'")
        self.trunk = _find_hiera_trunk(sam2_model)
        self.normalize = normalize
        self.expected_channels = tuple(int(c) for c in expected_channels)
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False
        if insert_adapters_flag:
            insert_hiera_adapters(self.trunk, adapter_bottleneck_ratio,
                                  residual=adapter_residual, freeze_blocks=freeze_trunk)
        mean255 = torch.tensor([123.675,116.28,103.53]).view(1,3,1,1)
        std255 = torch.tensor([58.395,57.12,57.375]).view(1,3,1,1)
        if input_range == '0_1':
            mean255, std255 = mean255 / 255.0, std255 / 255.0
        self.register_buffer('pixel_mean', mean255, persistent=False)
        self.register_buffer('pixel_std', std255, persistent=False)
    def _norm(self, x:Tensor) -> Tensor:
        if not self.normalize:
            return x
        return (x - self.pixel_mean.to(x.device, x.dtype)) / self.pixel_std.to(x.device, x.dtype)
    def _order(self, feats:Sequence[Tensor]) -> List[Tensor]:
        feats = list(feats)
        if len(feats) != 4:
            raise RuntimeError(f'Expected 4 features from Hiera trunk, got {len(feats)}')
        ch = tuple(int(f.shape[1]) for f in feats)
        if ch == self.expected_channels:
            return feats
        if ch == self.expected_channels[::-1]:
            return feats[::-1]
        areas = [int(f.shape[-2] * f.shape[-1]) for f in feats]
        if areas != sorted(areas, reverse=True):
            feats = [f for _, f in sorted(zip(areas, feats), key=lambda t: t[0], reverse=True)]
        return feats
    def forward(self, x:Tensor) -> List[Tensor]:
        out = self.trunk(self._norm(x))
        if isinstance(out, dict):
            if 'backbone_fpn' in out:
                out = out['backbone_fpn']
            elif 'vision_features' in out:
                out = [out['vision_features']]
            else:
                raise RuntimeError(f'Unsupported SAM2 output keys: {list(out.keys())}')
        return self._order(out)


# -----------------------------
# Haar DWT / IWT
# -----------------------------

def _haar_filters(device:torch.device, dtype:torch.dtype) -> Tensor:
    f = torch.tensor([
        [[1.,  1.], [ 1.,  1.]],
        [[1., -1.], [ 1., -1.]],
        [[1.,  1.], [-1., -1.]],
        [[1., -1.], [-1.,  1.]],
    ], device=device, dtype=dtype) * 0.5
    return f[:,None,:,:]

def haar_dwt(x:Tensor) -> Tuple[Tensor,Tuple[int,int]]:
    """Returns band-major [B,4C,H/2,W/2] = [LL_allC,LH_allC,HL_allC,HH_allC]."""
    b,c,h,w = x.shape
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode='reflect')
    weight = _haar_filters(x.device, x.dtype).repeat(c,1,1,1)
    raw = F.conv2d(x, weight, stride=2, groups=c)  # interleaved per input channel
    _,_,h2,w2 = raw.shape
    bands = raw.view(b,c,4,h2,w2).permute(0,2,1,3,4).contiguous().view(b,4*c,h2,w2)
    return bands, (h,w)

def haar_iwt(bands:Tensor, original_hw:Tuple[int,int]) -> Tensor:
    b,fourc,h2,w2 = bands.shape
    if fourc % 4 != 0:
        raise ValueError('bands channels must be divisible by 4')
    c = fourc // 4
    raw = bands.view(b,4,c,h2,w2).permute(0,2,1,3,4).contiguous().view(b,4*c,h2,w2)
    weight = _haar_filters(bands.device, bands.dtype).repeat(c,1,1,1)
    x = F.conv_transpose2d(raw, weight, stride=2, groups=c)
    h,w = original_hw
    return x[..., :h, :w]

class DWTConv2d(nn.Module):
    def __init__(self, channels:int, levels:int=1):
        super().__init__()
        self.levels = levels
        self.mix = nn.ModuleList([nn.Conv2d(4*channels,4*channels,1,bias=False) for _ in range(levels)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(4*channels) for _ in range(levels)])
    def forward(self, x:Tensor) -> Tensor:
        lows: List[Tensor] = []
        highs: List[Tensor] = []
        shapes: List[Tuple[int,int]] = []
        cur = x
        for i in range(self.levels):
            if cur.shape[-2] < 2 or cur.shape[-1] < 2:
                break
            bands, hw = haar_dwt(cur)
            bands = F.gelu(self.bn[i](self.mix[i](bands)))
            ll, lh, hl, hh = torch.chunk(bands, 4, dim=1)
            lows.append(ll)
            highs.append(torch.cat([lh,hl,hh], dim=1))
            shapes.append(hw)
            cur = ll
        if not lows:
            return x
        z = torch.zeros_like(lows[-1])
        for i in reversed(range(len(lows))):
            z = haar_iwt(torch.cat([lows[i] + z, highs[i]], dim=1), shapes[i])
        return z


# -----------------------------
# WSPM / FE-RFB
# -----------------------------

class SpectralPoolingFilter(nn.Module):
    def __init__(self, n:int, lam:float, radius_mode:str='pow2'):
        super().__init__()
        if not 0 <= lam <= 1:
            raise ValueError('lam must be in [0,1]')
        self.n, self.lam, self.radius_mode = int(n), float(lam), radius_mode
    def _radius(self, h:int, w:int) -> int:
        limit = max(1, min(h,w)//2)
        if self.radius_mode == 'pow2':
            r = 2 ** self.n
        elif self.radius_mode == 'linear2n':
            r = 2 * self.n
        else:
            r = min(h,w)//4
        return max(1, min(int(r), limit))
    def forward(self, x:Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        _,_,h,w = xf.shape
        z = torch.fft.fftshift(torch.fft.fft2(xf, norm='ortho'), dim=(-2,-1))
        r = self._radius(h,w)
        yy = torch.arange(h, device=x.device) - h//2
        xx = torch.arange(w, device=x.device) - w//2
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        mask = ((gx.square() + gy.square()) <= r*r)[None,None,:,:]
        low = z * mask
        high = z - low
        mixed = self.lam * low + (1 - self.lam) * high
        y = torch.fft.ifft2(torch.fft.ifftshift(mixed, dim=(-2,-1)), norm='ortho').real
        return y.to(dtype=dtype)

class WSPM(nn.Module):
    def __init__(self, channels:int, n:int, wt_levels:int=1,
                 lambdas:Sequence[float]=(0.7,0.8), radius_mode:str='pow2'):
        super().__init__()
        self.dwt = DWTConv2d(channels, wt_levels)
        self.dw1 = ConvBNReLU(channels, channels, (1,n), padding=(0,n//2), groups=channels)
        self.dw2 = ConvBNReLU(channels, channels, (n,1), padding=(n//2,0), groups=channels)
        self.spf = nn.ModuleList([
            nn.Sequential(SpectralPoolingFilter(n, lam, radius_mode),
                          nn.Conv2d(channels, channels, 1, bias=False),
                          nn.BatchNorm2d(channels)) for lam in lambdas
        ])
        self.out = ConvBNReLU(channels, channels, 1)
    def forward(self, x:Tensor) -> Tensor:
        z = self.dw2(self.dw1(self.dwt(x)))
        m = torch.zeros_like(z)
        for branch in self.spf:
            m = m + branch(z)
        return self.out(F.relu(z + m, inplace=True))

class FERFB(nn.Module):
    def __init__(self, in_ch:int, out_ch:int=64, branch_ch:Optional[int]=None,
                 wt_levels:int=1, radius_mode:str='pow2'):
        super().__init__()
        bc = branch_ch or out_ch
        self.b0 = nn.Sequential(ConvBNReLU(in_ch,bc,1), ConvBNReLU(bc,bc,3,dilation=1,padding=1))
        self.b1 = nn.Sequential(ConvBNReLU(in_ch,bc,1), WSPM(bc,3,wt_levels,radius_mode=radius_mode), ConvBNReLU(bc,bc,3,dilation=3,padding=3))
        self.b2 = nn.Sequential(ConvBNReLU(in_ch,bc,1), WSPM(bc,5,wt_levels,radius_mode=radius_mode), ConvBNReLU(bc,bc,3,dilation=5,padding=5))
        self.b3 = nn.Sequential(ConvBNReLU(in_ch,bc,1), WSPM(bc,7,wt_levels,radius_mode=radius_mode), ConvBNReLU(bc,bc,3,dilation=7,padding=7))
        self.cat = ConvBNReLU(4*bc, out_ch, 3)
        self.res = ConvBNReLU(in_ch, out_ch, 1, relu=False)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x:Tensor) -> Tensor:
        y = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.act(self.cat(y) + self.res(x))


# -----------------------------
# FE-UNet
# -----------------------------

class FeatureReducer(nn.Module):
    def __init__(self, in_ch:int, out_ch:int=64):
        super().__init__()
        self.net = nn.Sequential(ConvBNReLU(in_ch,in_ch,3,groups=in_ch), ConvBNReLU(in_ch,out_ch,1))
    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int=64):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x:Tensor, skip:Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class FEUNet(nn.Module):
    def __init__(self, backbone:nn.Module, encoder_channels:Sequence[int]=(144,288,576,1152),
                 base_ch:int=64, num_classes:int=1, wt_levels:int=1, radius_mode:str='pow2',
                 freeze_backbone:bool=False, keep_adapters_trainable:bool=True):
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError('encoder_channels must contain 4 sizes')
        self.backbone = backbone
        if freeze_backbone:
            for name,p in self.backbone.named_parameters():
                p.requires_grad = bool(keep_adapters_trainable and 'adapter' in name.lower())
        self.reducers = nn.ModuleList([FeatureReducer(int(c), base_ch) for c in encoder_channels])
        self.ferfbs = nn.ModuleList([FERFB(base_ch, base_ch, wt_levels=wt_levels, radius_mode=radius_mode) for _ in encoder_channels])
        self.dec3 = DecoderBlock(base_ch*2, base_ch)
        self.dec2 = DecoderBlock(base_ch*2, base_ch)
        self.dec1 = DecoderBlock(base_ch*2, base_ch)
        self.head3 = nn.Conv2d(base_ch, num_classes, 1)
        self.head2 = nn.Conv2d(base_ch, num_classes, 1)
        self.head1 = nn.Conv2d(base_ch, num_classes, 1)
    def forward_features(self, x:Tensor) -> List[Tensor]:
        feats = self.backbone(x)
        if not isinstance(feats, (list,tuple)) or len(feats) != 4:
            raise RuntimeError('backbone must return 4 feature maps')
        return [ferfb(red(f)) for f, red, ferfb in zip(feats, self.reducers, self.ferfbs)]
    def forward(self, x:Tensor, upsample_logits:bool=True) -> List[Tensor]:
        hw = x.shape[-2:]
        f1,f2,f3,f4 = self.forward_features(x)
        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        outs = [self.head1(d1), self.head2(d2), self.head3(d3)]
        if upsample_logits:
            outs = [F.interpolate(o, size=hw, mode='bilinear', align_corners=False) for o in outs]
        return outs


# -----------------------------
# Loss / builders
# -----------------------------

def weighted_bce_iou_loss(logits:Tensor, mask:Tensor) -> Tensor:
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
    mask = mask.float()
    weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    wbce = (weight * bce).sum((2,3)) / weight.sum((2,3)).clamp_min(1e-6)
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weight).sum((2,3))
    union = ((pred + mask) * weight).sum((2,3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0).clamp_min(1e-6)
    return (wbce + wiou).mean()

def deep_supervision_loss(logits_list:Sequence[Tensor], mask:Tensor) -> Tensor:
    return sum(weighted_bce_iou_loss(y, mask) for y in logits_list)

def build_optimizer(model:nn.Module, lr:float=1e-3, weight_decay:float=1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

def build_cosine_scheduler(optimizer:torch.optim.Optimizer, epochs:int=20, min_lr:float=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

def build_feunet_sam2(model_cfg:str, ckpt_path:str, device:Union[str,torch.device]='cuda',
                      num_classes:int=1, base_ch:int=64, insert_adapters_flag:bool=True,
                      adapter_bottleneck_ratio:int=4, freeze_hiera:bool=True,
                      normalize:bool=True, input_range:str='0_1', wt_levels:int=1,
                      radius_mode:str='pow2') -> FEUNet:
    sam2 = load_official_sam2(model_cfg, ckpt_path, device=device, mode='eval', apply_postprocessing=False)
    backbone = SAM2HieraFeatureExtractor(
        sam2, insert_adapters_flag=insert_adapters_flag,
        adapter_bottleneck_ratio=adapter_bottleneck_ratio,
        freeze_trunk=freeze_hiera, normalize=normalize, input_range=input_range,
        expected_channels=(144,288,576,1152)
    )
    return FEUNet(backbone, (144,288,576,1152), base_ch, num_classes,
                  wt_levels=wt_levels, radius_mode=radius_mode, freeze_backbone=False).to(device)


# -----------------------------
# Model statistics
# -----------------------------

def count_parameters(model:nn.Module) -> Tuple[int,int,int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen, trainable + frozen

def measure_forward_flops(model:nn.Module, inputs:Tensor) -> Tuple[int,List[Tensor]]:
    """Count supported forward-pass FLOPs; some operators such as FFT may be omitted."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        raise RuntimeError("FLOP counting requires PyTorch 2.1 or newer.") from e

    was_training = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        outputs = model(inputs)
    model.train(was_training)
    return int(counter.get_total_flops()), outputs

def format_count(value:int) -> str:
    return f'{value:,}'

def format_flops(value:int) -> str:
    return f'{value / 1e9:.3f} GFLOPs'


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device(PROFILE_DEVICE)
    model = build_feunet_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    x = torch.randn(PROFILE_BATCH_SIZE, 3, PROFILE_IMAGE_SIZE, PROFILE_IMAGE_SIZE, device=device)
    trainable, frozen, total = count_parameters(model)
    flops, ys = measure_forward_flops(model, x)

    print('Model: FE-UNet + SAM2.1 Hiera-L')
    print(f'Input: {tuple(x.shape)} on {device}')
    print(f'Trainable parameters: {format_count(trainable)} ({trainable / total:.2%})')
    print(f'Frozen parameters:    {format_count(frozen)} ({frozen / total:.2%})')
    print(f'Total parameters:     {format_count(total)}')
    print(f'Forward FLOPs:        {format_count(flops)} ({format_flops(flops)})')
    print('Note: PyTorch FLOP counting can omit unsupported operators such as FFT.')
    print('Output shapes:', [tuple(y.shape) for y in ys])
