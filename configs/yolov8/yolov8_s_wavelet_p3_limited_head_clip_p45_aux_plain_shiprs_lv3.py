_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

SHIPRS_LV3_CLASSES =  ('Other Ship','Other Warship','Submarine','Other Aircraft Carrier','Enterprise','Nimitz','Midway','Ticonderoga','Other Destroyer','Atago DD',
    'Arleigh Burke DD','Hatsuyuki DD','Hyuga DD','Asagiri DD','Other Frigate','Perry FF','Patrol','Other Landing','YuTing LL','YuDeng LL',
    'YuDao LL','YuZhao LL','Austin LL','Osumi LL''Wasp LL','LSD 41 LL','LHA LL','Commander','Other Auxiliary Ship','Medical Ship',
    'Test Ship','Training Ship','AOE','Masyuu AS','Sanantonio AS','EPF','Other Merchant','Container Ship','RoRo','Cargo',
    'Barge','Tugboat','Ferry','Yacht','Sailboat','Fishing Vessel','Oil Tanker','Hovercraft','Motorboat','Dock')

norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
max_epochs = 100

model = dict(
    neck=dict(
        _delete_=True,
        type='YOLOv8WaveletPAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        use_wavelet_downsample=True,
        wavelet_enhance_levels=(0,),
        wavelet_enhance_init_scale=0.0,
        wavelet_enhance_max_scale=0.2),
    bbox_head=dict(
        type='YOLOv8WaveletCLIPHead',
        head_module=dict(num_classes=50),
        clip_model_name='ViT-L-14',
        clip_pretrained='pretrained/open_clip/open_clip_model.safetensors',
        clip_class_names=SHIPRS_LV3_CLASSES,
        wavelet_clip_levels=(1, 2),
        wavelet_clip_loss_weight=0.015,
        wavelet_clip_center_ratio=0.35,
        wavelet_clip_small_area_thr=-1.0,
        wavelet_clip_logit_scale=20.0,
        wavelet_clip_feature_mode='plain',
        wavelet_clip_decay_start_epoch=45,
        wavelet_clip_decay_end_epoch=75,
        wavelet_clip_decay_type='cosine',
        wavelet_clip_min_scale=0.0),
        
    train_cfg=dict(
        assigner=dict(num_classes=50)
    )
)

data_root = '/data2/wangzihao233ov/Datasets/ShipRSImageNet_V1/'
metainfo = {
    'classes': ('Other Ship','Other Warship','Submarine','Other Aircraft Carrier','Enterprise','Nimitz','Midway','Ticonderoga','Other Destroyer','Atago DD',
    'Arleigh Burke DD','Hatsuyuki DD','Hyuga DD','Asagiri DD','Other Frigate','Perry FF','Patrol','Other Landing','YuTing LL','YuDeng LL',
    'YuDao LL','YuZhao LL','Austin LL','Osumi LL''Wasp LL','LSD 41 LL','LHA LL','Commander','Other Auxiliary Ship','Medical Ship',
    'Test Ship','Training Ship','AOE','Masyuu AS','Sanantonio AS','EPF','Other Merchant','Container Ship','RoRo','Cargo',
    'Barge','Tugboat','Ferry','Yacht','Sailboat','Fishing Vessel','Oil Tanker','Hovercraft','Motorboat','Dock'),
}

img_scale = (1000, 1000)  # width, height

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='COCO_Format/ShipRSImageNet_bbox_train_level_3.json',
        data_prefix=dict(img='YOLO_Format/ShipRS_original_train_bmp/')))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='COCO_Format/ShipRSImageNet_bbox_val_level_3.json',
        data_prefix=dict(img='YOLO_Format/ShipRS_original_val_bmp/')))
test_dataloader = val_dataloader


val_evaluator = dict(ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_val_level_3.json')
test_evaluator = val_evaluator


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5, dynamic_intervals=[((max_epochs - 30), 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = '/data2/wangzihao233ov/mmyolo-main/models/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'



train_pipeline = [
    
    dict(
        
        img_scale=img_scale),
    dict(
        
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2))
]

train_pipeline_stage2 = [
    
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        
        scale=img_scale)
    
]

test_pipeline = [
    
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        
        scale=img_scale
        
        )
]

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=5, save_best='auto', max_keep_ckpts=-1))

