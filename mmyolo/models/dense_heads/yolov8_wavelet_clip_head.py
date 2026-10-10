# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList
from mmengine.dist import get_dist_info
from mmengine import MessageHub
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..utils import gt_instances_preprocess
from .yolov8_head import YOLOv8Head


class HaarDWTMixin:
    """Fixed Haar wavelet helpers used by the head auxiliary branch."""

    @staticmethod
    def _pad_to_even(x: Tensor) -> Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    def haar_dwt(self, x: Tensor):
        x = self._pad_to_even(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def haar_high_reconstruct(lh: Tensor, hl: Tensor, hh: Tensor,
                              out_size: Sequence[int]) -> Tensor:
        y00 = (lh + hl + hh) * 0.5
        y01 = (-lh + hl - hh) * 0.5
        y10 = (lh - hl - hh) * 0.5
        y11 = (-lh - hl + hh) * 0.5

        out = y00.new_zeros(y00.shape[0], y00.shape[1], y00.shape[2] * 2,
                            y00.shape[3] * 2)
        out[:, :, 0::2, 0::2] = y00
        out[:, :, 0::2, 1::2] = y01
        out[:, :, 1::2, 0::2] = y10
        out[:, :, 1::2, 1::2] = y11
        return out[:, :, :out_size[0], :out_size[1]]


@MODELS.register_module()
class YOLOv8WaveletCLIPHead(YOLOv8Head, HaarDWTMixin):
    """YOLOv8 head with wavelet-assisted frozen CLIP text supervision.

    The detection branch is unchanged. During training, selected head input
    levels are decomposed with Haar DWT. The reconstructed high-frequency
    feature is concatenated with the original feature and projected to the CLIP
    text embedding space, where GT center regions receive a light semantic loss.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 clip_model_name: str = 'ViT-L-14',
                 clip_pretrained: str = 'openai',
                 clip_class_names: Sequence[str] = (),
                 prompt_templates: Optional[Sequence[str]] = None,
                 wavelet_clip_levels: Sequence[int] = (0,),
                 wavelet_clip_loss_weight: float = 0.03,
                 wavelet_clip_center_ratio: float = 0.5,
                 wavelet_clip_small_area_thr: float = 1024.0,
                 wavelet_clip_logit_scale: float = 20.0,
                 wavelet_clip_feature_mode: str = 'concat',
                 wavelet_clip_decay_start_epoch: int = 70,
                 wavelet_clip_decay_end_epoch: int = 90,
                 wavelet_clip_decay_type: str = 'linear',
                 wavelet_clip_min_scale: float = 0.0,
                 small_object_bbox_loss_weight: float = 0.0,
                 small_object_area_thr: float = 1024.0,
                 small_object_guard_strides: Sequence[int] = (),
                 small_object_loss_start_epoch: int = -1,
                 small_object_loss_end_epoch: int = -1,
                 init_cfg=None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.clip_class_names = tuple(clip_class_names)
        self.prompt_templates = tuple(prompt_templates or (
            'a remote sensing image of {}',
            'an aerial image of {}',
            'a small object of {} in aerial imagery',
        ))
        self.wavelet_clip_levels = tuple(wavelet_clip_levels)
        self.wavelet_clip_loss_weight = wavelet_clip_loss_weight
        self.wavelet_clip_center_ratio = wavelet_clip_center_ratio
        self.wavelet_clip_small_area_thr = wavelet_clip_small_area_thr
        self.wavelet_clip_logit_scale = wavelet_clip_logit_scale
        if wavelet_clip_feature_mode not in ('plain', 'high', 'concat'):
            raise ValueError(
                'wavelet_clip_feature_mode must be one of '
                '`plain`, `high`, or `concat`, but got '
                f'{wavelet_clip_feature_mode}.')
        self.wavelet_clip_feature_mode = wavelet_clip_feature_mode
        self.wavelet_clip_decay_start_epoch = wavelet_clip_decay_start_epoch
        self.wavelet_clip_decay_end_epoch = wavelet_clip_decay_end_epoch
        if wavelet_clip_decay_type not in ('linear', 'cosine'):
            raise ValueError(
                'wavelet_clip_decay_type must be `linear` or `cosine`, '
                f'but got {wavelet_clip_decay_type}.')
        self.wavelet_clip_decay_type = wavelet_clip_decay_type
        self.wavelet_clip_min_scale = wavelet_clip_min_scale
        self.small_object_bbox_loss_weight = small_object_bbox_loss_weight
        self.small_object_area_thr = small_object_area_thr
        self.small_object_guard_strides = tuple(small_object_guard_strides)
        self.small_object_loss_start_epoch = small_object_loss_start_epoch
        self.small_object_loss_end_epoch = small_object_loss_end_epoch

        self.clip_model, self.clip_tokenizer = self._build_clip_model()
        self.clip_embed_dim = self._infer_clip_embed_dim()
        self.text_features = None

        in_channels = self.head_module.in_channels
        norm_cfg = self.head_module.norm_cfg
        act_cfg = self.head_module.act_cfg
        self.wavelet_clip_projectors = nn.ModuleList()
        for channels in in_channels:
            projector_in_channels = channels * 2
            if self.wavelet_clip_feature_mode in ('plain', 'high'):
                projector_in_channels = channels
            self.wavelet_clip_projectors.append(
                nn.Sequential(
                    ConvModule(
                        projector_in_channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.Conv2d(channels, self.clip_embed_dim, kernel_size=1)))

    def _build_clip_model(self):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                'YOLOv8WaveletCLIPHead requires open-clip-torch. Install it '
                'with `pip install open-clip-torch` in the training '
                'environment.') from exc

        if self.clip_pretrained and os.path.isfile(self.clip_pretrained):
            try:
                model, _, _ = open_clip.create_model_and_transforms(
                    self.clip_model_name, pretrained=self.clip_pretrained)
            except Exception:
                model, _, _ = open_clip.create_model_and_transforms(
                    self.clip_model_name, pretrained=None)
                self._load_local_clip_checkpoint(model, self.clip_pretrained)
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.clip_pretrained)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model, open_clip.get_tokenizer(self.clip_model_name)

    def _load_local_clip_checkpoint(self, model: nn.Module,
                                    checkpoint_path: str) -> None:
        if checkpoint_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    'Loading .safetensors CLIP checkpoints requires '
                    '`safetensors`. Install it with `pip install safetensors`.'
                ) from exc
            checkpoint = load_file(checkpoint_path, device='cpu')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model', 'module'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        state_dict = {}
        for key, value in checkpoint.items():
            clean_key = key
            for prefix in ('module.', 'model.'):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            state_dict[clean_key] = value

        incompatible = model.load_state_dict(state_dict, strict=False)
        print_log(
            'Loaded local CLIP checkpoint '
            f'{checkpoint_path}. Missing keys: {len(incompatible.missing_keys)}, '
            f'unexpected keys: {len(incompatible.unexpected_keys)}.',
            logger='current')

    def _infer_clip_embed_dim(self) -> int:
        if hasattr(self.clip_model, 'text_projection'):
            return int(self.clip_model.text_projection.shape[1])
        if hasattr(self.clip_model.visual, 'output_dim'):
            return int(self.clip_model.visual.output_dim)
        raise AttributeError('Cannot infer CLIP embedding dimension.')

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self

    @torch.no_grad()
    def encode_clip_text(self, device: torch.device) -> Tensor:
        if len(self.clip_class_names) == 0:
            return torch.empty(0, device=device)

        prompts = []
        for class_name in self.clip_class_names:
            clean_name = class_name.replace('-', ' ')
            prompts.extend([template.format(clean_name)
                            for template in self.prompt_templates])

        tokens = self.clip_tokenizer(prompts).to(device)
        text_features = self.clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
        num_classes = len(self.clip_class_names)
        num_templates = len(self.prompt_templates)
        text_features = text_features.view(num_classes, num_templates,
                                           -1).mean(dim=1)
        return F.normalize(text_features, dim=-1)

    def get_text_features(self, device: torch.device) -> Tensor:
        if self.text_features is None or self.text_features.device != device:
            self.text_features = self.encode_clip_text(device)
        return self.text_features

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        losses = super().loss(x, batch_data_samples)
        loss_scale = self.get_wavelet_clip_loss_scale()
        if (self.wavelet_clip_loss_weight > 0 and loss_scale > 0
                and len(self.clip_class_names) > 0):
            if isinstance(batch_data_samples, list):
                batch_gt_instances = [
                    data_sample.gt_instances for data_sample in
                    batch_data_samples
                ]
                batch_img_metas = [
                    data_sample.metainfo for data_sample in batch_data_samples
                ]
            else:
                batch_gt_instances = batch_data_samples['bboxes_labels']
                batch_img_metas = batch_data_samples['img_metas']

            loss_clip = self.loss_wavelet_clip(x, batch_gt_instances,
                                               batch_img_metas)
            losses['loss_head_wavelet_clip'] = (
                loss_clip * self.wavelet_clip_loss_weight * loss_scale)
        return losses

    def get_wavelet_clip_loss_scale(self) -> float:
        if (self.wavelet_clip_decay_start_epoch < 0
                or self.wavelet_clip_decay_end_epoch <=
                self.wavelet_clip_decay_start_epoch):
            return 1.0

        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')
        if current_epoch is None:
            return 1.0
        current_epoch = int(current_epoch)

        if current_epoch < self.wavelet_clip_decay_start_epoch:
            return 1.0
        if current_epoch >= self.wavelet_clip_decay_end_epoch:
            return self.wavelet_clip_min_scale

        progress = ((current_epoch - self.wavelet_clip_decay_start_epoch) /
                    (self.wavelet_clip_decay_end_epoch -
                     self.wavelet_clip_decay_start_epoch))
        if self.wavelet_clip_decay_type == 'cosine':
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            scale = 1.0 - progress
        return max(self.wavelet_clip_min_scale, scale)

    def get_small_object_loss_scale(self) -> float:
        if self.small_object_bbox_loss_weight <= 0:
            return 0.0
        if (self.small_object_loss_start_epoch < 0
                or self.small_object_loss_end_epoch <=
                self.small_object_loss_start_epoch):
            return 1.0

        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')
        if current_epoch is None:
            return 1.0
        current_epoch = int(current_epoch)

        if current_epoch < self.small_object_loss_start_epoch:
            return 0.0
        if current_epoch >= self.small_object_loss_end_epoch:
            return 1.0

        return ((current_epoch - self.small_object_loss_start_epoch) /
                (self.small_object_loss_end_epoch -
                 self.small_object_loss_start_epoch))

    def loss_wavelet_clip(self, feats: Tuple[Tensor],
                          batch_gt_instances: Sequence[InstanceData],
                          batch_img_metas: Sequence[dict]) -> Tensor:
        text_features = self.get_text_features(feats[0].device)
        if text_features.numel() == 0:
            return feats[0].sum() * 0

        gt_info = gt_instances_preprocess(batch_gt_instances,
                                          len(batch_img_metas))
        total_loss = feats[0].sum() * 0
        total_pos = 0
        for level in self.wavelet_clip_levels:
            if level < 0 or level >= len(feats):
                continue

            feat = feats[level]
            if self.wavelet_clip_feature_mode == 'plain':
                semantic_feat = feat
            else:
                _, lh, hl, hh = self.haar_dwt(feat)
                high = self.haar_high_reconstruct(lh, hl, hh,
                                                  feat.shape[-2:])
                if self.wavelet_clip_feature_mode == 'high':
                    semantic_feat = high
                else:
                    semantic_feat = torch.cat([feat, high], dim=1)
            semantic_feat = self.wavelet_clip_projectors[level](
                semantic_feat.float())
            semantic_feat = F.normalize(semantic_feat, dim=1)
            logits = torch.einsum('bchw,nc->bnhw', semantic_feat,
                                  text_features)
            logits = logits * self.wavelet_clip_logit_scale

            target, pos_mask = self.build_wavelet_clip_targets(
                gt_info, logits.shape[-2:], self.featmap_strides[level],
                logits.device)
            if not pos_mask.any():
                continue

            level_logits = logits.permute(0, 2, 3, 1)[pos_mask]
            level_target = target[pos_mask]
            total_loss = total_loss + F.cross_entropy(
                level_logits, level_target, reduction='sum')
            total_pos += int(pos_mask.sum().item())

        if total_pos == 0:
            return feats[0].sum() * 0
        return total_loss / max(total_pos, 1)

    def build_wavelet_clip_targets(self, gt_info: Tensor,
                                   featmap_size: Sequence[int], stride: int,
                                   device: torch.device):
        num_imgs, max_gt, _ = gt_info.shape
        feat_h, feat_w = int(featmap_size[0]), int(featmap_size[1])
        target = torch.zeros(
            (num_imgs, feat_h, feat_w), dtype=torch.long, device=device)
        pos_mask = torch.zeros(
            (num_imgs, feat_h, feat_w), dtype=torch.bool, device=device)

        for img_idx in range(num_imgs):
            for gt_idx in range(max_gt):
                label = int(gt_info[img_idx, gt_idx, 0].item())
                bbox = gt_info[img_idx, gt_idx, 1:5]
                if bbox.sum() <= 0:
                    continue
                x1, y1, x2, y2 = [float(v.item()) for v in bbox]
                box_w = max(x2 - x1, 0.0)
                box_h = max(y2 - y1, 0.0)
                area = box_w * box_h
                if (self.wavelet_clip_small_area_thr > 0
                        and area > self.wavelet_clip_small_area_thr):
                    continue

                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                half_w = max(box_w * self.wavelet_clip_center_ratio * 0.5,
                             float(stride) * 0.5)
                half_h = max(box_h * self.wavelet_clip_center_ratio * 0.5,
                             float(stride) * 0.5)

                gx1 = max(int((center_x - half_w) / stride), 0)
                gy1 = max(int((center_y - half_h) / stride), 0)
                gx2 = min(int((center_x + half_w) / stride) + 1, feat_w)
                gy2 = min(int((center_y + half_h) / stride) + 1, feat_h)
                if gx2 <= gx1 or gy2 <= gy1:
                    continue
                target[img_idx, gy1:gy2, gx1:gx2] = label
                pos_mask[img_idx, gy1:gy2, gx1:gx2] = True
        return target, pos_mask

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        assigned_bboxes_pixel = assigned_bboxes
        assigned_bboxes = assigned_bboxes / self.stride_tensor
        flatten_pred_bboxes = flatten_pred_bboxes / self.stride_tensor

        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)

            small_scale = self.get_small_object_loss_scale()
            if small_scale > 0 and self.small_object_area_thr > 0:
                assigned_bboxes_pixel_pos = torch.masked_select(
                    assigned_bboxes_pixel, prior_bbox_mask).reshape([-1, 4])
                box_wh = (assigned_bboxes_pixel_pos[:, 2:] -
                          assigned_bboxes_pixel_pos[:, :2]).clamp(min=0)
                box_area = box_wh[:, 0] * box_wh[:, 1]
                small_mask = box_area <= self.small_object_area_thr
                if len(self.small_object_guard_strides) > 0:
                    stride_pos = torch.masked_select(
                        self.stride_tensor[..., 0].expand_as(
                            fg_mask_pre_prior), fg_mask_pre_prior)
                    stride_mask = torch.zeros_like(small_mask)
                    for stride in self.small_object_guard_strides:
                        stride_mask = stride_mask | (stride_pos == stride)
                    small_mask = small_mask & stride_mask
                small_weight = 1.0 + (
                    small_mask.to(bbox_weight.dtype).unsqueeze(-1) *
                    self.small_object_bbox_loss_weight * small_scale)
                bbox_weight = bbox_weight * small_weight

            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)
