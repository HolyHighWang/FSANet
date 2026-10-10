# FSANet

FSANet is a small-object detection codebase built on
[MMYOLO](https://github.com/open-mmlab/mmyolo) **v0.6.0** with **YOLOv8s**.

Two modules are added on top of the upstream detector:

* a **wavelet-enhanced PAFPN** — Haar wavelet downsampling in the bottom-up fusion
  path, plus a learnable high-frequency residual gate on the shallow pyramid level;
* a **CLIP-guided detection head** — a frozen OpenCLIP ViT-L-14 text branch adds an
  auxiliary image-text matching loss while training.

The inference graph stays an ordinary YOLOv8s: the CLIP branch is used during
training only, and training results and checkpoints are not stored in this
repository.

## What this repository adds on top of MMYOLO v0.6.0

| Path | Change |
| --- | --- |
| `mmyolo/models/necks/yolov8_wavelet_pafpn.py` | new — `WaveletDownsample`, `WaveletFeatureEnhance`, `YOLOv8WaveletPAFPN` |
| `mmyolo/models/dense_heads/yolov8_wavelet_clip_head.py` | new — `YOLOv8WaveletCLIPHead` |
| `mmyolo/models/necks/__init__.py` | registers `YOLOv8WaveletPAFPN` |
| `mmyolo/models/dense_heads/__init__.py` | registers `YOLOv8WaveletCLIPHead` |
| `configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_dior.py` | new — full model, DIOR |
| `configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_ai_dot.py` | new — full model, AI-TODv1 |
| `configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_shiprs_lv0.py` … `_lv3.py` | new — full model, ShipRSImageNet levels 0–3 |
| `configs/yolov8/yolov8_s_wavelet_p3_limited_pafpn_dior.py` | new — ablation, wavelet PAFPN only |
| `configs/yolov8/yolov8_s_head_clip_p45_aux_plain_dior.py` | new — ablation, CLIP head only |
| `draw_detect_results.py` | new — draw selected prediction/annotation boxes from a `results.pkl` |

Everything else is unmodified MMYOLO v0.6.0 (upstream commit `8c4d9dc`): all
official configs, `tools/`, `projects/`, `docs/`, `demo/`, `tests/` and
`requirements/` are kept as upstream, so the change set can be reviewed with
`git diff 8c4d9dc`. Weight files and run outputs are not included. The upstream
README is preserved as [`README_MMYOLO.md`](README_MMYOLO.md).

## Repository layout

```
FSANet/
├── configs/yolov8/           # official YOLOv8 configs + the 8 FSANet configs
├── mmyolo/                   # MMYOLO v0.6.0 package (+ the two FSANet modules)
├── tools/                    # upstream train/test/analysis entry points
├── projects/, docs/, demo/, tests/, resources/, requirements/, docker/
├── draw_detect_results.py    # qualitative visualisation helper
├── pretrained/README.md      # how to obtain the weights (weights are not stored)
├── README.md                 # this file
└── README_MMYOLO.md          # upstream MMYOLO README
```

## Installation

```bash
conda create -n fsanet python=3.8 -y
conda activate fsanet

# PyTorch — pick the build matching your CUDA version
pip install torch==1.13.1 torchvision==0.14.1

# OpenMMLab dependencies (see requirements/mminstall.txt)
pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0" "mmengine>=0.7.1"

# extra dependencies of the CLIP-guided head
pip install open-clip-torch safetensors

# install this repository in editable mode
pip install -v -e .
```

## Weights

Weight files are not part of this repository; see
[`pretrained/README.md`](pretrained/README.md) for download commands. In short:

* **OpenCLIP ViT-L-14** (~1.71 GB, `safetensors`) for the semantic branch →
  `pretrained/open_clip/open_clip_model.safetensors`;
* **YOLOv8s COCO** checkpoint used as `load_from` →
  `pretrained/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth`.

## Datasets

All benchmarks are trained in **COCO detection format** with `img_scale` of
`(1000, 1000)` and `YOLOv5CocoDataset`.

| Dataset | `data_root` in the config | Annotations | Classes |
| --- | --- | --- | --- |
| DIOR | `/data2/wangzihao233ov/Datasets/DIOR/` | `train/annotations/DIOR_train.json`, `val/annotations/DIOR_val.json` | 20 |
| AI-TODv1 | `/data2/wangzihao233ov/Datasets/AI-DOT/` | `annotations/aitod_train_v1.json`, `annotations/aitod_val_v1.json` | 8 |
| ShipRSImageNet | `/data2/wangzihao233ov/Datasets/ShipRSImageNet_V1/` | `COCO_Format/ShipRSImageNet_bbox_{train,val}_level_{0..3}.json` | 2 / 4 / 25 / 50 |

Expected directory layout (DIOR shown; the others follow the same pattern):

```
DIOR/
├── train/annotations/DIOR_train.json
├── train/images/
├── val/annotations/DIOR_val.json
└── val/images/
```

ShipRSImageNet keeps the original images under
`YOLO_Format/ShipRS_original_{train,val}_bmp/`; the four task levels
(`shiprs_lv0` … `shiprs_lv3`) differ only in the annotation file and the number
of classes.

## Paths you have to change

The released configs keep the absolute paths of the training server they were
produced on:

* `data_root` — root of the dataset (see the table above);
* `load_from` —
  `/data2/wangzihao233ov/mmyolo-main/models/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth`
  → your local YOLOv8s COCO checkpoint;
* `clip_pretrained` — `pretrained/open_clip/open_clip_model.safetensors` (already
  relative, so it works once the weights are in place).

They can also be overridden on the command line instead of editing the files:

```bash
python tools/train.py configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_dior.py \
  --cfg-options data_root=/your/DIOR/ \
                load_from=/your/weights/yolov8_s_coco.pth
```

## Training

Single GPU:

```bash
python tools/train.py configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_dior.py
```

Multiple GPUs:

```bash
bash tools/dist_train.sh configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_shiprs_lv0.py 4
```

The eight released configurations:

| Config | Role | Dataset |
| --- | --- | --- |
| `..._head_clip_p45_aux_plain_dior.py` | full model | DIOR |
| `..._head_clip_p45_aux_plain_ai_dot.py` | full model | AI-TODv1 |
| `..._head_clip_p45_aux_plain_shiprs_lv0.py` … `_lv3.py` | full model | ShipRSImageNet levels 0–3 |
| `yolov8_s_wavelet_p3_limited_pafpn_dior.py` | ablation, wavelet PAFPN only | DIOR |
| `yolov8_s_head_clip_p45_aux_plain_dior.py` | ablation, CLIP head only | DIOR |

Full-model settings: `max_epochs = 100`; `batch_size = 8` (DIOR,
ShipRSImageNet) or `4` (AI-TOD); wavelet enhancement on P3 with
`wavelet_enhance_max_scale = 0.2`; CLIP supervision on P4/P5 with
`wavelet_clip_loss_weight = 0.015`, `wavelet_clip_center_ratio = 0.35`,
`wavelet_clip_feature_mode = 'plain'` and a cosine decay from epoch 45 to 75.

## Evaluation

```bash
python tools/test.py \
  configs/yolov8/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_dior.py \
  work_dirs/yolov8_s_wavelet_p3_limited_head_clip_p45_aux_plain_dior/best_coco_bbox_mAP_epoch_92.pth
```

COCO `bbox` metrics (`mAP`, `AP_s`, `AP_50`, `AP_75`, `AP_m`, `AP_l`) are reported
by `mmdet.CocoMetric`.

## Qualitative visualisation

`draw_detect_results.py` renders selected classes from a `results.pkl` produced by
`tools/test.py`:

```bash
python tools/test.py <config> <checkpoint> --out results.pkl

python draw_detect_results.py results.pkl \
  /path/to/DIOR/val/annotations/DIOR_val.json \
  /path/to/DIOR/val/images \
  --dataset dior --target-classes ship vehicle \
  --score-thr 0.3 --out-dir results/vis_dior_ship_vehicle
```

`--dataset` selects the class-name table (`dior` or `ai_dot`) and
`--target-classes` restricts the drawing to the classes of interest.

## Acknowledgement and licence

This project is a fork of [MMYOLO](https://github.com/open-mmlab/mmyolo) (v0.6.0,
Apache-2.0) and inherits its licence; see [`LICENSE`](LICENSE). Thanks to the
OpenMMLab team for the code base and to
[OpenCLIP](https://github.com/mlfoundations/open_clip) for the vision-language
model used in the semantic branch.

---

## 中文说明

FSANet 是一个面向小目标检测的代码工程，基于 MMYOLO v0.6.0 与 YOLOv8s，在官方检测
器上新增了两个模块：

1. **小波增强 PAFPN**（`mmyolo/models/necks/yolov8_wavelet_pafpn.py`）：自底向上融合
   路径中用固定的 Haar 小波变换替代步长为 2 的卷积下采样，四个子带拼接后用 1×1 卷积
   融合；并在指定层（P3）上只重建高频分量，通过一个可学习的、有上限的门控系数残差式
   加回原特征。
2. **CLIP 引导检测头**（`mmyolo/models/dense_heads/yolov8_wavelet_clip_head.py`）：
   用冻结的 OpenCLIP ViT-L-14 文本编码器编码类别名，把 P4/P5 特征投影到 CLIP 空间做
   逐位置余弦相似度匹配，只在真值框中心区域计算辅助损失，并在训练后段按余弦衰减。该
   分支只在训练时使用，推理结构与原版 YOLOv8s 一致。

本仓库相对官方 MMYOLO 只做了上表列出的改动，其余保持 v0.6.0 原样，可用
`git diff 8c4d9dc` 核对；权重文件与训练结果均未包含。使用前请按上文修改配置中的
`data_root` 与 `load_from`，并按 `pretrained/README.md` 自行下载权重。
