# FE-UNet for Polyp Segmentation

Triển khai PyTorch của **FE-UNet** dùng **SAM2 Hiera-L** làm bộ trích xuất đặc trưng để phân đoạn polyp trong ảnh nội soi.

## Kết quả

Kết quả của checkpoint huấn luyện ở kích thước ảnh `320 × 320`, threshold `0.5`:

| Bộ kiểm thử | Số ảnh | Mean Dice | Mean IoU |
|---|---:|---:|---:|
| Kvasir-SEG | 100 | 0.9295 | 0.8830 |
| CVC-ColonDB | 380 | 0.7814 | 0.7089 |
| CVC-300 | 60 | 0.8885 | 0.8234 |
| ETIS | 196 | 0.7787 | 0.7019 |

Dice và IoU được tính riêng trên từng ảnh rồi lấy trung bình toàn bộ tập kiểm thử.

## Cấu trúc mã nguồn

```text
FE-UNet/
├── fe_unet_sam2_full.py                 # Kiến trúc FE-UNet và hàm loss/optimizer
├── polyp_dataset.py                     # Dataset, augmentation và DataLoader
├── split_polyp_data.py                  # Chuẩn hóa và chia các bộ dữ liệu
├── train_polyp.py                       # Huấn luyện và validation
├── infer_feunet_polyp_overlay_panel.py  # Đánh giá, lưu mask và panel trực quan
└── README.md
```

Dữ liệu, checkpoint, thư mục `runs/` và mã nguồn SAM2 không nằm trong repo.

## Yêu cầu

- Python 3.10 trở lên
- PyTorch 2.5.1 trở lên


Tạo môi trường và cài các thư viện Python:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch torchvision
pip install albumentations numpy pillow matplotlib tqdm
```

Cài SAM2 chính thức vào đúng vị trí mà mã nguồn đang sử dụng:

```bash
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
```

Tải checkpoint `sam2.1_hiera_large.pt` vào `sam2/checkpoints/`. Trên Windows có thể tải checkpoint theo hướng dẫn của SAM2; trên Git Bash/Linux có thể chạy:

```bash
cd sam2/checkpoints
./download_ckpts.sh
cd ../..
```

Đường dẫn cuối cùng phải là:

```text
sam2/checkpoints/sam2.1_hiera_large.pt
```

## Chuẩn bị dữ liệu

Giải nén năm bộ dữ liệu tại thư mục gốc theo cấu trúc sau:

```text
FE-UNet/
├── Kvasir-SEG/images/ và masks/
├── CvC-ClinicDB/PNG/Original/ và PNG/Ground Truth/
├── CVC-ColonDB/images/ và masks/
├── CVC-300/images/ và masks/
└── ETIS/images/ và masks/
```

Sau đó chạy:

```bash
python split_polyp_data.py
```

Script dùng seed `42`, tạo `polyp_data/` với các split:

| Split | Thành phần | Số ảnh |
|---|---|---:|
| `train` | Kvasir-SEG (700) + CvC-ClinicDB (612) | 1312 |
| `val` | Kvasir-SEG | 200 |
| `kvasir_test` | Kvasir-SEG | 100 |
| `CVC-ColonDB_test` | CVC-ColonDB | 380 |
| `CVC-300_test` | CVC-300 | 60 |
| `ETIS_test` | ETIS | 196 |

Lưu ý: chạy lại script sẽ xóa và tạo lại các thư mục split do script quản lý bên trong `polyp_data/`.

## Huấn luyện

Các tham số được khai báo ở đầu `train_polyp.py`. Cấu hình mặc định quan trọng:

```python
DATA_ROOT = "polyp_data"
OUTPUT_DIR = "runs/polyp_feunet_320"
EPOCHS = 20
BATCH_SIZE = 8
LR = 0.001
IMAGE_SIZE = 320
USE_AMP = True
AMP_DTYPE = "bf16"
```

Chạy huấn luyện:

```bash
python train_polyp.py
```

Kết quả được lưu tại `OUTPUT_DIR`:

- `best.pt`: checkpoint có Dice validation tốt nhất.
- `last.pt`: checkpoint của epoch gần nhất.

Để tiếp tục huấn luyện, đặt `RESUME_TRAINING = True` và gán `RESUME_CHECKPOINT` tới `last.pt`. `IMAGE_SIZE` phải là bội số của 32, ví dụ `256`, `288`, `320`, `352` hoặc `384`.

## Đánh giá và trực quan hóa

Trước khi chạy, chỉnh các hằng số ở đầu `infer_feunet_polyp_overlay_panel.py` cho khớp phiên huấn luyện:

```python
CHECKPOINT = "runs/polyp_feunet_320/best.pt"
OUTPUT_DIR = "runs/polyp_feunet_320/inference_test"
IMAGE_SIZE = 320
SPLIT = "ETIS_test"
```

Các giá trị hợp lệ của `SPLIT` là `kvasir_test`, `CVC-ColonDB_test`, `CVC-300_test` và `ETIS_test`.

```bash
python infer_feunet_polyp_overlay_panel.py
```

Kết quả cho mỗi split được lưu tại `OUTPUT_DIR/<split>/`:

```text
<split>/
├── metrics.txt   # Mean Dice, Mean IoU và cấu hình đánh giá
├── panels/       # Image | GT overlay | Pred overlay
└── pred_masks/   # Mask nhị phân dự đoán
```

`IMAGE_SIZE`, kiến trúc SAM2 và đường dẫn checkpoint khi đánh giá phải khớp với cấu hình dùng để huấn luyện. Có thể điều chỉnh `THRESHOLD`, `MAX_PANELS` và `OVERLAY_ALPHA` trực tiếp trong script.

![alt text](runs/polyp_feunet_320/inference_test/kvasir_test/panels/00001_Kvasir-SEG_0002.png)
![alt text](runs/polyp_feunet_320/inference_test/kvasir_test/panels/00026_Kvasir-SEG_0027.png)
![alt text](runs/polyp_feunet_320/inference_test/CVC-ColonDB_test/panels/00003_CVC-ColonDB_0004.png)  
![alt text](runs/polyp_feunet_320/inference_test/CVC-300_test/panels/00006_CVC-300_0007.png)
![alt text](runs/polyp_feunet_320/inference_test/ETIS_test/panels/00004_ETIS_0005.png)
## Ghi chú

- Mô hình trả về ba đầu ra và dùng deep supervision trong quá trình huấn luyện.
- Augmentation cho tập train gồm flip, affine và thay đổi độ sáng/tương phản.
- Nếu GPU không hỗ trợ BF16, mã nguồn tự chuyển AMP sang FP16.

## Nguồn tham khảo

- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2)

