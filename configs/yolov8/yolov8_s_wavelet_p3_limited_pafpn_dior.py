_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
    'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle', 'windmill')

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
        head_module=dict(num_classes=20)
    ),
        
    train_cfg=dict(
        assigner=dict(num_classes=20)
    )
)

data_root = '/path/to/DIOR/'
metainfo = {
    'classes': ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
                'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 
                'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
                'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
}

img_scale = (1000, 1000)  # width, height

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations/DIOR_train.json',
        data_prefix=dict(img='train/images/')))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotations/DIOR_val.json',
        data_prefix=dict(img='val/images')))
test_dataloader = val_dataloader


val_evaluator = dict(ann_file=data_root + 'val/annotations/DIOR_val.json')
test_evaluator = val_evaluator


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5, dynamic_intervals=[((max_epochs - 30), 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = 'pretrained/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'



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

