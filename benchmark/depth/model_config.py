_base_ = []

train_dataloader=dict(
    batch_size=26,
    num_workers=20,
)

teacher_model = None
student_model =  dict(
    #ckpt =  'work_dirs/edifat_prune_defront_1024x1216_BASELINE_200k_lite0thuge_yuv_coe1.4/iter_200000_0.55.pth', # modify this to your ckpt path
    data_preprocessor=dict(type='StereoDataPreProcessor'),
    depth_decoder=dict(
        act_cfg=dict(type='ReLU'),
        disp_loss=dict(
            loss_disp=dict(
                oa_param=1.0,
                suppress_swing=True,
                use_batch_weight=True,
                type='SequenceLoss'),
            loss_pseudo=dict(
                threshold=0.2, type='SequencePseudoLoss', weight=0.1)),
        feat_channels=[64, 64], #1/8, 1/16
        cxt_channels= [32, 32], #1/8, 1/16
        h_channels = [32, 32],  #1/8, 1/16,
        in_index = [1, 2],
        truncated_grad = False,
        net_type = "Basic",
        groups_d16=4,
        gru_type='SeqConv',
        iters=1,
        pseudo_loss_for_init=True,
        radius=4,
        radius_d16=24,
        type='IGEVStereoSlimDecoder',
        use_3dcnn=False,
        use_cvx_scale=True,
        use_disp16_pred=True,
        x5m_support='gn1'),
    seg_decoder = dict(
        type='FCNHead',
        truncated_grad = False,
        in_channels = [64, 64, 92], #1/8 1/16 1/32
        in_index=[0, 1, 2],
        channels = sum([64, 64, 92]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=3,  # 19
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=1.0)]),
    encoder=dict(
        #arch_type='lite0thuge',
        # arch_type='lite0tfat',
        arch_type='lite0m',
        backbone_pretrain_path=None,
        #prune_cfg = [48, 96, 144, 240, 480, 480, 672, 672, 1152, 1152], #lite0tfat, # set your prune_cfg here
        #prune_cfg =  [48, 128, 176, 184, 208, 288, 312, 304, 360, 840],
        #prune_cfg = [48, 88, 144, 136, 168, 80, 96, 168, 288, 216, 304, 272, 328, 400, 504, 848],
        prune_cfg = [40, 64, 112, 128, 104, 208, 168, 200, 200, 496], #0.67_reduceopt_reduceopt2_200k_200k 4.565ms
        in_channels=3,
        out_channels=[64, 64, 92], #1/8 1/16 1/32
        return_resolutions=3,
        type='EffliteFPN'),
    freeze_bn=False,
    fronzen_depth_features = False,
    fronzen_seg_features = False,
    fused_context=True,
    test_cfg=dict(),
    train_cfg=dict(),
    type='SepCxtNewRumStereo')

model=dict(
    type='StereoLabelDistiller',
    student = student_model,
    teacher = teacher_model,
    sr = 0.0,
    sr_coe = 1.3,
    freeze_bn = False,
    data_preprocessor=dict(type='StereoDataPreProcessor'),
    loss_cfg = dict(
        loss_warp = dict(
            type="warp",
            ssim_weight=0.85,
            scale=2,
            weight=2.0,
        ),
        loss_smooth = dict(
            type='dav2_smooth',
            weight=0.9,
        )
    )
)

# optimizer
LR = 0.00075
ITER_NUM = 500000
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=LR,
        eps=1e-08, amsgrad=False,
        weight_decay=0.0001, betas=(0.9, 0.999)),
    clip_grad = dict(max_norm=1.0),
)

param_scheduler = [dict(
    type='OneCycleLR',
    by_epoch=False,
    eta_max=LR,
    total_steps=ITER_NUM+100,
    pct_start=0.05,
    anneal_strategy='linear')
]

train_cfg = dict(by_epoch=False, max_iters=ITER_NUM, val_interval=20000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20000),
)

log_processor = dict(by_epoch=False)

#load_from = 'work_dirs/edifatprune_ft_defront_1024x1216_BASELINE_400k_yuv_lite0thuge/iter_300000.pth'