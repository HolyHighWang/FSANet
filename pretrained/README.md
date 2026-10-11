# Pretrained weights

Weight files are **not** stored in this repository — the OpenCLIP ViT-L-14
checkpoint alone is about 1.7 GB. Download them and place them as described
below. The whole `pretrained/` directory except this file is git-ignored.

## 1. OpenCLIP ViT-L-14 text encoder (required by the CLIP-guided head)

`YOLOv8WaveletCLIPHead` builds a frozen OpenCLIP model from
`clip_model_name='ViT-L-14'` and loads its weights from the local file given by
`clip_pretrained` (the released configs use
`pretrained/open_clip/open_clip_model.safetensors`).

```bash
mkdir -p pretrained/open_clip
# OpenCLIP ViT-L-14 (laion2B-s32B-b82K), mirrored on Hugging Face
wget -O pretrained/open_clip/open_clip_model.safetensors \
  https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_model.safetensors
```

* Expected size: 1,710,517,724 bytes (~1.71 GB).
* The file is a plain `safetensors` state dict (446 tensors, text width 768,
  vision width 1024). The head loads it with `strict=False` and keeps the model
  frozen in `eval()` mode, so the CLIP branch adds no trainable parameters and no
  inference cost.

Any equivalent OpenCLIP `ViT-L-14` checkpoint works: point `clip_pretrained` in
the config at that file, or at an `open_clip` `pretrained=` tag / Hugging Face
model name, in which case `open_clip` downloads the weights at build time.

Required packages: `open-clip-torch` and `safetensors`.

## 2. YOLOv8s COCO detection weights (used as `load_from`)

```bash
mkdir -p pretrained/yolov8
wget -O pretrained/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth \
  https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth
```

The released configs use the repository-relative path
`pretrained/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth`
for `load_from`, so this file only has to sit where the command above puts it. Only
`data_root` — shipped as a `/path/to/...` placeholder — has to be changed; see the
"Paths you have to change" section of the top-level README.
