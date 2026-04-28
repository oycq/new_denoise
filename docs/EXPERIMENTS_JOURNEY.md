# N2N 降噪：实验过程与知识沉淀

> 这一份是**整个项目的总结性文档**，按时间顺序记录从最初的 L1 baseline
> 一路到最终配方的所有尝试、量化结果和经验教训。
>
> **当前线上配方 = N2N 论文原配方**（`L2 + γ·L_reg` + 非残差 UNet +
> BlurPool/PixelShuffle，`n2n/trainer.py` 默认配置，`python run_train.py 600`
> 即起 GUI 训 10 min）。**结论与所有补丁式失败的故事都在文档第 12 节**——
> 强烈建议先读那一节再回头看历史细节。
>
> 早期尝试过的 L1+TV / robust 训练 / VGG perceptual loss / 各种 mode
> penalty 都已废弃；其代码已经清理或归档到本地 `_archive_local/`（git 忽略），
> 但所有实验配置以文本形式完整保留在本文中，需要复现时按章节索引即可。

---

## 0. 任务定义与约束

- **输入**：1280 × 1088 的 16-bit Bayer (RGGB) raw，归一化到 `[0, 1]`。
- **目标**：最高 ISO（=16，即 `train_data/Data/<scene>/16/sensor*_raw.png`）
  做单图自监督降噪。
- **方法骨架**：Neighbor2Neighbor (Huang 2021) 子采样 + UNet 回归噪声残差。
- **硬件**：RTX 2070 (8 GB) · FP16 (`torch.amp`) · `cudnn.benchmark`。
  代码里 `_autodetect_amp_dtype` 在 Ampere+ (sm_80+) 上自动切到 bf16。
- **约束**：
  - 推理图必须保持单 UNet 前向，BPU 部署 0 改动 → 所有改进都只能放在**训练侧**。
  - 训练时长以分钟为单位（用户希望每候选 5–20 min 见效）。
  - 7 个场景 × 10 个 ISO × 2 个 sensor = 140 对，全部 GPU-resident。

---

## 1. 整体时间线

```
阶段 1   流程跑通 + 16-bit 显示链路修复            (commit 50db87c → 0e14105)
   |     N2N + GUI + 时间预算 + result/ + raw_to_display
   v
阶段 2   16 个策略横向扫描 (5 min × 16)             (commit c2e4176)
   |     loss 家族 / UNet 深度 / patch / 黑电平
   |     量化第一: ms_l1_charbonnier + p=256
   v
阶段 3   TV-λ 调优 + 时长扫描 + 7 场景 20 ROI       (commit 091298e)
   |     4 个 λ × 2 时长，视觉 16/20 ROI 给到 λ=0.01
   |     最终锁定 λ = 0.03（baseline 兼顾平/锐）
   v
阶段 4   7 小时自由探索 92 个候选                   (本仓库 _archive_local)
   |     loss / 网络 / seed / 组合，~±2% 内打平
   |     用户反馈：人眼感受跟 baseline 没差别 → 暗角团块没解决
   v
阶段 5   3 小时鲁棒性优化 (6 候选)                  (本仓库 _archive_local)
   |     诊断到 N2N 假设违反（FPN 在 cell 内完美相关）
   |     EMA 自蒸馏 / R2R / 像素掩码 / 亚像素抖动 / 暗区加权
   |     C4 组合显著降低暗角团块
   v
阶段 6   感知损失 6 候选 (1.5 h)                    (本仓库 _archive_local)
   |     用户反馈：清晰度 OK，但**残留噪声看着很恼人，不像自然噪声**
   |     chroma / CSF / MS-SSIM / VGG / 全套 5 种 + 对照
   |     VGG 是唯一让"残留长得像自然噪声"的项 ★
   v
当前     生产代码精简到 2 个推荐路径                (本 commit)
         · L1 + 0.03·TV（部署默认）
         · robust + VGG-16 feature L1（最佳感知）
```

---

## 2. 阶段 1：早期踩坑

### 显示管线的 staircase / posterization

**症状**：降噪后图像在大平面（天花板、墙壁）出现明显的色阶 / 梯形伪影。

**根因**：早期实现是

```
raw_uint16 / 256 - 9 -> astype(uint8) -> demosaic -> ...
```

中间那个 `astype(uint8)` 在去马赛克之前就把数据量化掉了，丢了 8 bit 精度。

**修复**：改成全程 16-bit / float32，只在最后一步出 8-bit 图：

```
raw_uint16
  -> 16-bit black-level subtract        (in raw_to_linear_bgr)
  -> 16-bit demosaic (cv2 BayerRG2BGR)
  -> float32 BGR 0-255
  -> gray-world WB (only for display)
  -> gamma 2.2
  -> uint8 (only here!)                 (in raw_to_display)
```

代码在 `n2n/raw_utils.py` 的 `raw_to_linear_bgr` / `raw_to_display`。

### 「团块状」低频残留

**症状**：5 min L1 训练后，平面区域出现尺寸 ~50–100 像素的块状色斑。

**根因**：默认 UNet `depth=4 + patch=128 (packed)`，最深一层的感受野
覆盖了整个 patch，模型实际上学到了「全图平均」的伪解。

**修复**：

- `depth → 3`（保留局部感受野）
- `patch_size → 256` packed（即 512 Bayer，远大于深层感受野）

后续所有候选都用 `depth=3 + patch=256 + base=48` 这个三件套。

---

## 3. 阶段 2：16 策略量化扫描

每个候选都用相同的 `Trainer` 和评估脚本（`blob/hf/edge/composite`）打分。

### 量化指标（每个候选只在固定一帧 sensor2 ISO=16 上测）

| 指标 | 含义 | 方向 |
|---|---|---|
| `blob_residual` | 平面 ROI 上的低频能量（5×5 box filter 的方差） | 越小越好 |
| `noise_floor`   | 平面 ROI 上的高频噪声底（残差的 std） | 越小越好 |
| `edge_sharpness`| 边缘 ROI 上的 Sobel 梯度均值 | 越大越好 |
| `composite`     | `0.5·blob + 0.25·hf + 0.25·edge` 加权 | 越大越好 |

### 16 个候选 + 关键观察

| 编号 | 配置 | 角色 |
|---|---|---|
| 00 | L1 (默认) | baseline |
| 01 | Charbonnier | 平滑 L1 |
| 02 | L1 + grad | 边缘保护 |
| 03 | multi-scale L1 | 抗团块 |
| 04 | Huber | L1+L2 混合 |
| 05 | L1 patch=256 | patch 增大 |
| 06 | multiscale + p=256 | 03 + 05 |
| 07 | UNet depth=4 + p=256 | 加深网络 |
| 08-09 | multiscale 4 尺度 × p=192/256 | 调多尺度 |
| **10** | **MS-Charbonnier + p=256** | **阶段 2 第一名 (composite=0.78)** |
| 11 | Charbonnier + p=256 | 验证多尺度增益 |
| 12 | L1 + black-level subtract | FP16 数值稳定性 |
| 13 | L1 + 0.05 · TV | 平面正则 |
| 14-15 | UNet d=4 + 多尺度变体 | 大网络对比 |

`10_ms_charbonnier_p256` 量化最高，但 `13_l1_plus_tv` 视觉**主观也很好**。
这成为阶段 3 的入口：「哪个 λ 才是真正的甜蜜点？」

---

## 4. 阶段 3：L1 + λ·TV 深度调优

### Total Variation 是什么

```
TV(I) = mean(|I(x+1,y) - I(x,y)|) + mean(|I(x,y+1) - I(x,y)|)
```

惩罚相邻像素差。在**平面**区域几乎无影响（相邻像素本来就接近），
但在**边缘**区域会把边缘也拉近 → λ 越大边缘越糊。

所以 **λ 是「平面干净 ↔ 细节锐利」的旋钮**。

### 第一轮：4 个 λ × 5 min

| λ | blob ↓ | hf ↓ | edge ↑ | composite ↑ | 视觉 |
|---|---|---|---|---|---|
| 0.005 | 1.033 | 0.902 | 18.27 | 0.493 | TV 太弱 ≈ baseline |
| 0.01  | 0.981 | 0.891 | 17.26 | 0.617 | 甜蜜点（5 min 时） |
| 0.02  | 0.983 | 0.809 | 16.54 | 0.612 | 角落更净 |
| 0.05  | 0.940 | 0.699 | 15.45 | 0.750 | 过度平滑（木纹消失） |
| baseline | 1.007 | 0.890 | 17.97 | 0.575 | 参考 |

### 第二轮：λ=0.01 的 5 vs 10 min — 时长可以救边缘

**关键发现**：5 min 时 `edge=17.26`（< baseline），看起来 TV 把边缘
拍扁了；但 10 min 时 `edge=18.07`（> baseline 17.97）。

**TV 带来的「边缘软化」是暂时的** ——给足训练时间，模型可以同时
学到「平 + 锐」两个目标，因为它们可以兼容，只是优化器需要时间。

### 第三轮：4 λ × 2 时长汇总

| λ | 5min edge | 10min edge | 恢复幅度 |
|---|---|---|---|
| 0.01 | 17.26 | 18.07 | **+4.7%** |
| 0.02 | 16.54 | 16.98 | +2.7% |
| 0.05 | 15.45 | 15.68 | +1.5% |

**TV 越强，时长能找回的细节越少**。λ=0.05 即使 10 min，高频细节也
救不回来——是真的被挤掉了。

### 第四轮：7 场景 20 ROI 视觉判定

20 个 ROI 覆盖 9 个内容类别（暗角 / 平面 / 网格 / 边缘 / 树叶 /
纹理 / 几何 / 玻璃 / 复杂场景）：

| 候选 | 胜的 ROI |
|---|---|
| **λ=0.01 10 min** | **16/20** |
| λ=0.02 10 min | 6/20 |
| λ=0.05 10 min | 4/20（仅纯平面） |
| baseline 5 min | 5/20 |

**重要的反直觉**：composite 排第一的 λ=0.05，在视觉上是最糟的「广义平均」
（树叶融成一团、栅栏糊、文字消失）。说明 composite 对**纹理损失**惩罚不足。
**视觉判定 > 单一标量指标**。

### 最终：L1 + 0.03·TV（生产 baseline）

经过四轮扫描后取 **λ = 0.03**：

- 比 λ=0.01 略强（角落更干净），比 λ=0.05 弱很多（不会过度平滑）
- 介于第二名（λ=0.02）和第三名（λ=0.05）之间，吸取了「角落需要更
  强 TV」的视觉判断
- 在 20 min 训练 + 中间快照 (2/5/10/20 min) 验证下表现稳定

`n2n/trainer.py` 的 `TrainConfig` 锁定如下，是 `python run_train.py`
默认会用的配置：

```python
loss_name    = "l1_plus_tv"
loss_kwargs  = {"lam": 0.03}
patch_size   = 256                  # packed-pixel
batch_size   = 6
unet_depth   = 3
base_channels= 48                   # ~3 M 参数, ~12 MB ckpt
use_fp16     = True
n2n_lambda   = 0.0                  # 不加 N2N 一致性正则，纯 L1+TV 主项
train_seconds= 1200                 # 20 min
intermediate_save_seconds = (120, 300, 600)
```

---

## 5. 阶段 4：7 小时自由探索 92 候选（已归档）

> 代码在 `_archive_local/experiments_7h/`，不再 git 跟踪。

### 设计

- 用 `experiments_7h/orchestrate.py` 跑 8 个 stage （A–I），每个 stage 一组
  loss / 网络 / 组合的扫描
- 总共 ~92 个候选，每个 2 min 训练 + 量化 + leaderboard.json
- 量化指标用阶段 2 同一套 (blob / hf / edge / composite) + seed-stability

### 涉及的损失家族（全部回归到 pixel-domain）

| 大类 | 代表 | 备注 |
|---|---|---|
| 多尺度 L1 / Charbonnier | `ms_l1`, `ms_charbonnier` | baseline 之上 ~+1% |
| TV / EATV | `l1_tv_eatv` (sigma 0.01–0.05) | 边缘自适应 TV，~baseline |
| FFT / 频域 | `l1_tv_fft`, `l1_tv_lapfft` | 高频损失，~baseline |
| Sobel / Laplacian / SDF | `l1_tv_sobel`, `l1_tv_lap` | ~baseline |
| Charbonnier + EATV + FFT (B61) | `charb_eatv_fft` | 量化第一名（marginal） |

### 涉及的网络变体

- `model_variants.py` 里写了 6 个 BPU-friendly 改架构（depthwise sep / ghost / SE attention 等）
- 都在 ±2% 范围内，没有显著突破

### 结论 + 用户反馈（导向阶段 5）

92 候选互相 `composite ±2%` 内打平。在用户的视觉测试里：
**「7 小时实验跑出来的效果和 L1+0.03·TV baseline 没啥区别 ——
暗角依然有团块、孤立暗点，人眼感受非常差。」**

**为什么 pixel-domain 改 loss 改不动**：所有这些损失只能控制
「多大幅度的误差被惩罚」，无法控制「哪种结构的误差被保留」。

---

## 6. 阶段 5：3 小时鲁棒性优化（C0–C5，已归档）

> 代码在 `_archive_local/experiments_3h_extra/`；唯一保留进 git 的是
> `n2n/robust_trainer.py`，它现在被感知损失阶段复用作为训练机制。

### 用户反馈推动的设计

> 「我不能做 FPN 校正（生产上没这个机会）。
> 网络要足够鲁棒，**各种奇怪的 FPN 都能在训练里被自动消化掉**。
> 试试 NN 方法做 loss。」

### 诊断

7h 实验全部都是 loss 改动（92 候选）。但 N2N 在 single-frame 下**假设违反**：

- 假设 = 2×2 cell 内两个邻居的噪声**独立**
- 现实 = sensor FPN（DSNU / PRNU / hot pixel）在 cell 内**完全相关**
- 后果 = 网络把 FPN 当 signal 学下来 → 暗角团块

任何 loss 函数微调都不能修复这个**系统性偏差**。需要**训练范式**的改动。

### 5 个训练时机制（推理 0 改动）

| 机制 | 作用 |
|---|---|
| **EMA 自蒸馏 (NN-as-loss)** | teacher = EMA(student, decay=0.999)；distill loss = `L1(student(x), stop_grad(teacher(x)))`。EMA 是 batch-bias 的低通滤波器，平均掉跨 batch 的 FPN bias。 |
| **R2R 噪声注入** | `input += N(0, 0.005)`，独立于 FPN，破坏 N2N 两侧 FPN 完美相关。来自 Pang 2021。 |
| **随机像素掩码 (J-invariance)** | 5% 像素替换为 4-邻域均值，强制网络从 context 推断，不能直接复制单像素 FPN。来自 Batson 2019 / Noise2Self。|
| **亚像素空间抖动** | 每 batch ±0.5 packed-pixel bilinear shift，FPN 在不同 batch 位置不一致，无法被 memorize。|
| **暗区加权损失** | weight ∝ `1/(local_luma + 0.02)`，把网络容量主动倾斜到暗角。|

### 6 个候选

| name | 配置 | 训练 |
|---|---|---|
| `C0_baseline`     | L1 + 0.03·TV（控制组） | 5 min |
| `C1_dark`         | 暗区加权 L1 + 0.03·TV | 5 min |
| `C2_ema`          | L1+TV + EMA 自蒸馏 | 6 min |
| `C3_robust_aug`   | L1+TV + R2R + mask + jitter | 5 min |
| `C4_combo`        | C1 + C2 + C3 全叠加 | 7 min |
| `C5_combo_charb`  | C4 + charb_eatv_fft_dark loss | 7 min |

### 结论

C2 (EMA) 和 C3 (R2R+mask+jitter) 各自在暗角团块和孤立暗点上肉眼可见地降低，
**C4 组合最强**：暗角的低频块状残留几乎消失，孤立暗点不再"漂"。
C5 加上 v3 dark-aware loss 的提升边际很小，没值得保留。

→ 进到阶段 6 时，**C4 的训练范式（EMA + R2R + mask + jitter）成为
所有感知损失候选的固定底座**，只换 loss 函数，公平比较。

---

## 7. 阶段 6：感知损失（6 候选，VGG ★ 当选）

> 代码精简后保留进 git：
>
> - `n2n/losses_perceptual.py` — VGG 提取器 + 亮度/色度 L1 + EATV + 组合
> - `n2n/robust_trainer.py` — C4 的 robust 训练循环
> - `train_vgg.py` — 顶层入口脚本，内嵌 P0_baseline + P4_vgg 两个候选定义
>   （其余 4 个候选已删除，配置在本节文档化）

### 用户反馈

> 「3h push 的 6 个候选把暗角团块解决掉了，但**残留噪声看着很恼人，
> 不像自然噪声，不符合人眼**。」

这一观感差距来自 pixel-domain L1 / charbonnier / TV 在频域、色度、
结构层面**毫无 HVS 偏好**——网络优化时不会主动避开人眼最敏感的失真模式。

### 4 类感知损失原理对照

| 类别 | 损失 | 关键洞察 |
|---|---|---|
| 亮度-色度 | `luma_chroma_l1`（YCoCg-R 4:1 加权） | HVS 对色度噪声远比亮度敏感 |
| CSF 频域 | `csf_fft_loss` (Mannos-Sakrison) | HVS 在 4-8 cpd 中频段最敏感 |
| 结构 | `ms_ssim_loss` | luminance / contrast / structure 三轴相似度 |
| **NN-as-loss** | **`vgg_perceptual_loss` (VGG-16 relu1_2 / 2_2 / 3_3)** | **特征空间 L1 = LPIPS 风格** |

### 6 个候选（loss 是唯一变量，其他全用 C4 同款 robust 训练）

```
P0_baseline      L1 + 0.03·TV                                300 s
P1_chroma        Y/Co/Cg 4:1 加权 L1 + TV                    300 s
P2_csf_fft       + CSF 加权频域 (lam_csf=0.05)                360 s
P3_msssim        + MS-SSIM (lam_msssim=0.30)                 360 s
P4_vgg           + VGG 特征 L1 (lam_vgg=0.05) + EATV         420 s   ★
P5_full_perceptual  Y/Co/Cg + CSF + MS-SSIM + VGG + EATV     540 s
```

每个候选都先按上面预算训一轮，再统一 +20 min 续训（resume 接上原 loss 曲线），
共做了完整 ~28k–34k step 的训练。

### Loss 曲线与相对收敛对比

> 中间过程的两张图（绝对损失 / 相对收敛）已经存到 `_archive_local/process_images/`
> （`loss_curves.png` 和 `loss_curves_normalised.png`），仅作过程留底。

定性观察：

- **绝对 loss**：P0 (~0.013) 远低于 P4_vgg (~0.096)、P5 (~0.096)，因为不同
  loss 的尺度差几个数量级，**绝对值不可比**。
- **相对 loss（loss / starting_loss）**：6 条曲线收敛到 87–96% 区间，
  P4 / P5 略高，P0 / P3 略低；同样**不能直接拿来排序**。
- 所有候选都已稳定收敛，没有发散 / 过拟合迹象。

→ 排序必须靠**视觉验证 + 量化指标**（noise_std / chroma_noise /
midfreq_blotch / fpn_resid_dark），不能看 loss 数字。

### 4 场景 4 ROI 量化（dark_tl / dark_bl / dark_br / mid，归一化到 noisy）

下表是 **chroma_noise_denoised / chroma_noise_noisy** 的均值（越小色噪压制越好）：

| 候选 | dark_tl | dark_bl | dark_br | mid | 均值 |
|---|---|---|---|---|---|
| P0_baseline | 0.30 | 0.11 | 0.18 | 0.59 | **0.295** |
| P1_chroma   | 0.31 | 0.13 | 0.21 | 0.59 | 0.310 |
| P2_csf_fft  | 0.32 | 0.14 | 0.22 | 0.60 | 0.320 |
| P3_msssim   | 0.34 | 0.13 | 0.22 | 0.61 | 0.325 |
| **P4_vgg**  | **0.31** | **0.18** | **0.31** | **0.61** | 0.353 |
| P5_full     | 0.31 | 0.18 | 0.31 | 0.59 | 0.348 |

→ 量化看，P0 反而"更狠地压色噪"。但**用户视觉判定相反**：

> P0 / P1 / P2 / P3 的残留噪声看着"很 plastic 不像自然噪声"，
> **P4_vgg 让残留噪声看着自然**——是真正符合人眼的。

这又一次复刻了阶段 3 那条经验：「composite 排第一 ≠ 视觉最优」。

### 结论

VGG 是 6 个候选里**唯一让"残留噪声看着像自然噪声"** 的项。
chroma / CSF / MS-SSIM 都让残留**看起来更"机器味"**——它们让网络
优化错了维度（pixel-amplitude 维度 vs HVS-perceived 维度）。

→ 收敛到生产推荐：**`P4_vgg = robust 训练 + VGG-16 feature L1`**

#### 当前生产 P4_vgg 配置（`train_vgg.py` 中内嵌）

```python
Candidate(
    name="P4_vgg",
    desc="★ VGG-16 特征 L1 + 亮度/色度 L1 + EATV",
    loss_name="vgg_l1_eatv",
    loss_kwargs={
        "w_luma": 4.0, "w_chroma": 1.0,
        "lam_eatv": 0.03,
        "lam_vgg": 0.05,
    },
    use_ema_distill=True,  ema_decay=0.999,  ema_lam=0.10,
    use_r2r=True,          r2r_sigma=0.005,
    use_mask=True,         mask_ratio=0.05,
    use_jitter=True,       jitter_amount=0.5,
    train_seconds=420.0,
)
```

`vgg_l1_eatv` 是被精简后的单一 loss：
`luma_chroma_l1 + 0.03·EATV + 0.05·VGG-16 feature L1`，
不再有 CSF / MS-SSIM 的死代码（之前混在 `perceptual_full` 里被
`lam=0` 屏蔽掉，但每步仍在算）。

#### 已删除候选的配置存档（备查）

如果以后想重跑这 4 个曾经被淘汰的候选，下面是它们的 loss kwargs，
配合当时已经在 `losses_perceptual.py` 里的 chroma / CSF / MS-SSIM 实现
（已删除，但可以从 git 历史恢复，或对照本节重写）即可：

```python
P1_chroma:   loss_name="luma_chroma_tv",
             loss_kwargs={"w_luma": 4.0, "w_chroma": 1.0, "lam_tv": 0.03}
             # luma_chroma_tv 仍保留在 losses_perceptual.py，可直接跑

P2_csf_fft:  loss_name="luma_chroma_csf",   # 已删
             loss_kwargs={"w_luma": 4.0, "w_chroma": 1.0,
                          "lam_csf": 0.05, "lam_tv": 0.03}

P3_msssim:   loss_name="perceptual_combo",   # 已删
             loss_kwargs={"w_luma": 4.0, "w_chroma": 1.0,
                          "lam_csf": 0.0, "lam_msssim": 0.30, "lam_eatv": 0.03}

P5_full:     loss_name="perceptual_full",    # 已删
             loss_kwargs={"w_luma": 4.0, "w_chroma": 1.0,
                          "lam_csf": 0.05, "lam_msssim": 0.10,
                          "lam_vgg": 0.05, "lam_eatv": 0.03}
```

CSF (Mannos-Sakrison) 实现要点：

```
A(f) = 2.6 · (0.0192 + 0.114·f) · exp(-(0.114·f)^1.1)
peak_cpd = 30  # 假设最大空间频率对应 30 cycles/degree
```

MS-SSIM 实现要点：3 个尺度（不是经典的 5 个，因为 patch=256），
window=7, sigma=1.5, weights=[0.0448, 0.2856, 0.6989] 截断/重归一。

---

## 8. 跨阶段 10 条经验

1. **量化精度链路是 #1 杀手**——别在显示前过早 `astype(uint8)`。
2. **UNet 感受野必须 < patch**，否则模型退化为全局平均，出团块。
3. **N2N 总 loss 上升 ≠ 训练失败**：reg 项随模型变好而变大，要分开看
   `main_loss` 和 `reg_loss`。极简方案直接 `n2n_lambda=0`。
4. **TV 对边缘的损害是暂时的**：5 min 看像糊了，10 min 模型能同时
   学到「平+锐」。但 λ 太大（≥0.05）救不回来。
5. **量化指标和视觉感受会冲突**——composite 把 blob 权重设得太高，
   会偏向过度平滑的策略。要做 ROI 视觉抽样验证。
6. **暗角是 TV 受益最大的区域**（高彩噪 + 局部平面），栅栏 / 木纹 /
   细文字是 TV 受损最严重的区域。如果你的关键内容是后者，少加 TV。
7. **Pixel-domain loss 改不出"看起来自然"的残留**——不管换多少种
   pixel-L1 变体，残留都会有"机器味"。要换到**特征域 / 感知域**
   才能改变残留的"长相"。
8. **N2N 假设违反在 FPN 严重的传感器上是系统性偏差**——不是 loss 调
   参能修的。需要训练范式改动（EMA 蒸馏 / J-invariance / 噪声注入 /
   亚像素抖动），且这些机制都是**推理 0 cost**。
9. **绝对 loss 不可跨候选比较**——loss 数值由 loss 函数本身的尺度决
   定，不同 loss 之间差几个数量级很常见，相对 loss（loss / 起始 loss）
   也只能看是否收敛，不能看哪个更好。最终判定要回到 ROI 视觉。
10. **VGG / NN-as-loss 的"扣分维度"和人眼对齐**——不是因为 VGG 更
    "聪明"，而是因为它惩罚的是 *what kind of error*（结构 / 纹理 /
    色度变化），而不是 *how big the error*。这是和 pixel-L1 的根
    本区别。

---

## 9. 当前代码结构（精简后）

```
new_denoise/
├── README.md                       # 项目入口
├── QUICKSTART.md                   # 新机器部署
├── requirements.txt
├── benchmark.py                    # 3 min 吞吐量基准
├── run_train.py                    # 一键启 GUI（默认 L1+0.03·TV）
├── train_gui.py                    # PyQt5 GUI
│
├── n2n/                            # 生产代码（生效）
│   ├── __init__.py
│   ├── raw_utils.py                # 16-bit 显示流水线 + I/O + Bayer pack
│   ├── n2n_sampler.py              # 矢量化 N2N 子采样
│   ├── dataset.py                  # CPU DataLoader fallback
│   ├── gpu_dataset.py              # GPU-resident 数据集（默认）
│   ├── model.py                    # UNet (residual head)
│   ├── losses.py                   # L1+0.03·TV 主损失 + 7 个备选 + perceptual lazy 注册
│   ├── losses_perceptual.py        # ★ VGG 特征损失 + luma/chroma + EATV
│   ├── trainer.py                  # 主 trainer（run_train.py 走这里）
│   ├── robust_trainer.py           # robust 训练 (EMA + R2R + mask + jitter; train_vgg.py 走这里)
│   └── infer.py                    # tile 推理 + ImageJ-stack 输出
│
├── run_train.py                    # 一键启 GUI（生产 baseline）
├── train_vgg.py                    # ★ 一键训 P0_baseline + P4_vgg
├── train_gui.py                    # PyQt5 GUI 实现
├── benchmark.py                    # 3 分钟吞吐量基准
│
├── docs/
│   └── EXPERIMENTS_JOURNEY.md      # （本文件）整个项目的来龙去脉
│
└── train_data/Data/...             # 数据
```

不再 git 跟踪、本地保留在 `_archive_local/` 里的内容（含中间过程 / 旧脚本 / 大件归档）：

```
_archive_local/
├── process_images/                 # 阶段 6 的两张 loss 曲线 PNG
├── docs_html/                      # 历史 HTML 报告（REPORT / TV_*.html / RESEARCH_7H_REPORT / REPORT_3H_ROBUST / REPORT_PERCEPTUAL …）
├── docs_assets/                    # demo_assets / research_7h_assets（HTML 报告引用的 PNG）
├── experiments_7h/                 # 92 候选 7h sweep 全套
├── experiments_3h_extra/           # 3h push 除 robust_trainer 外的所有产物 + build_report 等
├── experiments_perceptual_extra/   # 6 候选感知 sweep 的 ckpts/eval/assets/logs/build_report/run_extra/plot_curves
├── experiments_perceptual_dir_old/ # 上一版的 experiments_perceptual/ 目录骨架（candidates / run_all / README，已合并进 train_vgg.py）
├── extra_losses/                   # losses_v2.py / losses_v3.py / model_variants.py（v2/v3 fft/dark/charb 系列实现）
├── archives/                       # train_data.zip / report_*.tar.gz / .zip
├── reports_download/               # report_3h_download / report_perceptual_download 解压目录
└── old_tools/                      # tools/* 与 result_demo/* 早期 5/10 min 训练脚本
```

---

## 10. 怎么跑

### A. 生产 baseline（L1 + 0.03·TV，部署默认）

```bash
pip install -r requirements.txt
python run_train.py            # GUI
# 或 headless:
python -c "from n2n.trainer import Trainer, TrainConfig; Trainer(TrainConfig()).run()"
```

输出：`checkpoints/n2n_model.pt` + 2/5/10 min 中间快照。

### B. ★ VGG 推荐方案（最佳感知质量）

```bash
python train_vgg.py                       # 训 P0_baseline + P4_vgg 两个候选
python train_vgg.py --only P4_vgg         # 只训 VGG
python train_vgg.py --ckpt-dir my_ckpts/  # 自定义输出目录
```

输出（默认）：
- `ckpts/P0_baseline.pt`  — 同款 robust 训练 + L1+TV (对照)
- `ckpts/P4_vgg.pt`       — robust 训练 + VGG-16 feature L1 ★
- `ckpts/logs/<name>.log` — 每个候选的训练 stdout

两个 ckpt 推理路径完全一样（用 `n2n.infer.run_inference_set`），
**BPU 部署 0 改动**，仅训练时 VGG 多走一次 frozen forward
（~25% 训练慢一点）。

### C. 单测 / 烟雾测试

```bash
python -c "
import sys; sys.path.insert(0, '.')
from n2n.robust_trainer import RobustTrainer, RobustTrainConfig
RobustTrainer(RobustTrainConfig(
    train_seconds=20,
    use_ema_distill=True, ema_lam=0.10, ema_warmup_steps=10,
    use_r2r=True, use_mask=True, use_jitter=True,
    log_every=20,
    ckpt_path='/tmp/_smoke.pt',
)).run()
print('smoke OK')
"
```

---

## 11. 训练吞吐量备忘

```
RTX 2070 (Turing, 8 GB) + FP16 autocast (no GradScaler)
+ channels_last + fused Adam + GPU-resident dataset + cudnn benchmark
patch=256 (packed) + batch=6 + UNet depth=3 base=48

L1+0.03·TV       ~55–60 step/s (3-min sustained)
+ VGG forward    ~42–48 step/s (~25% slower; cost = frozen VGG forward)

samples_per_epoch = 1024  →  ~170 step/epoch  →  3.0–3.5 s/epoch
20 min 训练 ≈ 350 epoch (baseline) / 280 epoch (VGG)
```

新机器先 `python benchmark.py` 跑 3 min 看实测 sps，再调 `train_seconds`。

### Ampere+ 自动启用的优化

在 RTX 2070 / Windows 上无效，但代码已经写好，**上 4090 / A100 等 Ampere+
GPU + Linux 时自动启用**：

| 优化 | 触发条件 | 4090 上预期 |
|---|---|---|
| **bf16 autocast** 替代 fp16 | `cuda.major >= 8`（Ampere+） | 与 fp16 持平，省掉 GradScaler |
| **`torch.compile(reduce-overhead)`** | `import triton` 成功（Linux 默认装、Windows 无） | 4090 实测 +44% |

切换逻辑写在 `n2n/trainer.py::_autodetect_amp_dtype` /
`_try_torch_compile`，启动时打印当前 dtype。

---

## 12. 阶段 7（最终结论）：回到 N2N 论文原配方

文档头部介绍的"L1 + 0.03·TV"和"L1 + VGG"两条路径，**都已废弃**。
继续追低光 ISO=16 的 2×2 grid artifact 时发现：所有非平凡 loss
(VGG / TV / mode penalty / Gr-Gb 一致性 / channel std equalization)
都是给现象打补丁，**根因是 cleanup commit 把 Neighbor2Neighbor 论文
的一致性正则项删掉了**。

### 阶段 7 的一系列失败

按时间顺序记下被否决的方案，避免下次再走一遍：

| 方案 | 思路 | 失败原因 |
|---|---|---|
| `Gr-Gb \|·\|` 一致性 | 强制网络让 Gr 和 Gb 输出相等 | 视觉上 grid 没消，定量上只治住了 4 个 2×2 模式中的 1 个 |
| demosaic L1 | 在可微分 demosaic 输出上也做 N2N L1 | 子采样和目标都有同样的 grid，L1 互相抵消，没效果 |
| 对角 mode penalty (`X3`) | 用 `[[1,-1],[-1,1]]` 卷积惩罚特定 2×2 mode | 能量跑到 H/V mode → 输出**横条纹** |
| 全模式 mode penalty (`X5`) | 同时惩罚 H/V/D 三个 mode | 改善有限，每个 mode 实际权重只剩 1/3 |
| channel std equalize (`X6`) | 强制 R/Gr/Gb/B 残差 std 相等 | 网络放弃去噪以满足约束，输出几乎等于 noisy |
| Bilinear 上采样 + all-mode (`X9`) | 改架构去除 PixelShuffle 的 2×2 inductive bias | 视觉很干净但 X10 加额外 D-mode 后**过糊 + 绿色色偏** |

### 阶段 7 的根因

Neighbor2Neighbor 论文（Huang et al., CVPR 2021）的官方 loss 是：

```
L = ‖f(g₁) − g₂‖²  +  γ · ‖(f(g₁) − g₂) − (g₁(f(y)) − g₂(f(y)))‖²
                                                            ^^^^^^^^
                                              一致性正则——之前 commit 删掉了
```

第二项强制**子采样推理**和**全图推理**输出一致。没这一项时网络在
两种推理路径上 over-smooth / under-smooth 程度不同，**差异即可见的
2×2 grid**。我加的 mode penalty 是在治症状；论文这一项从源头不让
分裂发生。

### 阶段 7 的成功配方

加上一致性正则，并采纳 N2V2 (Hoeck et al. 2022) 的两个 anti-checkerboard
fix，方案降到极简：

- **网络**：非残差 UNet（`denoise(x) = forward(x)`，不是 `x − net(x)`）+
  BlurPool 下采样 + PixelShuffle 上采样（`n2n/model.py`）
- **loss**：L2 重建 + γ·L_reg（γ 从 0 渐进到 2），无 VGG / 无 TV /
  无任何 mode penalty
- **训练时长**：≥ 600 s（10 min）。非残差 UNet 没有零初始化恒等先验，
  5 min 以下有概率出现"灾难棋盘"——`tmp4/` 5 min 实测 sum_dark_roof
  达到 25（noisy 输入 7），10 min 立即降到 0.97。

10 / 15 / 20 min 实测对比（同一 ckpt 续训）：

| 训练时长 | sum_dark_roof | sum_pillar | sum_shadow |
|---|---|---|---|
| 10 min  | 0.979 | 2.090 | 2.595 |
| 15 min  | 0.926 | 2.175 | 2.521 |
| 20 min  | 0.869 | 2.015 | 2.344 |

10 → 20 min 给约 11% 量化改善，视觉上几乎不可见。**10 min 是甜蜜点**。

### 实测吞吐量（4090 + bf16 + torch.compile）

```
非残差 UNet + L2 + γ·L_reg (depth=3, patch=256, batch=6)  ~110 sps
```

比之前 L1+VGG 方案（45 sps）快 **2.5×**——因为没了 VGG forward，
只多了一次全图 forward (no_grad) 给一致性正则。

### 教训

1. **第一选择是论文原配方**。我跳过这一步直接组合"自家 trick"，
   走了几小时弯路。
2. **删功能要看清作用**。`n2n_lambda` 默认 0 看起来"未启用"，但论文
   推荐值是 1，而且对 Bayer raw 是必需的。删掉等于把 N2N 退化成
   普通自监督。
3. **mode-level 补丁是 whack-a-mole**。打掉一个模式，能量跑到另一个。
   只有从训练目标层面消除分裂源才根治。
4. **非残差网络对训练时长敏感**。残差网络的零初始化"恒等"先验是
   重要安全网；去掉以后必须保证训练时间够长。


## 阶段 8：主干换回 L1 + 0.01 · TV

### 背景

阶段 7 把主干切到了 paper N2N，结果好，但每个 step 多算一次整图
forward（一致性正则项），4090 上稳态从 ~290 sps 砍到 ~163 sps。
这次想做"训练时长扫描" + "λ 扫描" + "数据规模扫描"等多轮快速实验时，
sps 差一倍 = 实验吞吐慢一倍。

回头看视觉效果：在 7 个场景 × ISO 16 的高噪声测试集上，

| 配方 | 10 min wall-clock 后视觉 |
|---|---|
| paper N2N | 干净，但暗区还有微小 blob |
| L1 + 0.05·TV | 过度 piecewise-flat，墙面 / 树叶细节明显被 cartoonise |
| L1 + 0.03·TV | 接近 sweet-spot，但 lens-shading 角落仍偏平整化 |
| **L1 + 0.01·TV** | **细节保留最佳，平坦区 blob 残留对比 ISP 参考可接受** |
| L1 + 0.005·TV | 几乎等价纯 L1，dark roof 残留比 0.01 略多 |

L1+0.01·TV 在 10 min 内已经达到与 paper N2N 相当的视觉清洁度，且训练
快近一倍，做实验更利索。

5 / 10 / 20 min 训练时长扫描（同样 lam=0.01）：
- 5 min  loss=0.0162  暗区有可见 blob
- 10 min loss=0.0160  暗区显著改善，~5-10% 视觉提升
- 20 min loss=0.0156  比 10 min 微小提升，边际收益小

10 min 是 cost / quality 的甜点，定为 baseline。

### 决策

主干代码切回 L1 + 0.01·TV：

- `n2n/model.py` 用残差头版本（`denoise(x) = x − forward(x)`，零初始化输出
  端，对 600 s 预算特别友好）；不再用 N2V2 anti-checkerboard 修复版
  （只在 paper N2N 路径下需要）。
- `n2n/trainer.py` 简化成单一 loss 路径：`l1 + tv_lambda * tv`，loss 函数
  内联，`StepInfo` 只剩一个 `loss` 字段，去掉 loss 注册表 / `gamma_final` /
  `subtract_black_level` / `n2n_lambda` 等不再用的开关。
- `train_gui.py` 的 `LossPlot` 从三条线（rec / reg / total）简化成单条。
- 被替换掉的方法（paper N2N、L1+VGG、robust_trainer 等）全部沉淀到
  `docs/archived_recipes.txt`，包括完整 recipe / 复活 commit。

### 教训

5. **方法选择要平衡 sps 与视觉收益**。paper N2N 公式更优雅、抗 grid
   不需要 TV 强惩罚，但视觉收益不到 10%、sps 砍半，做日常迭代不划算。
6. **loss 平台 ≠ 视觉到顶**。L1+TV 在 5 min 时 loss 已经平台，但 10 min
   视觉上仍明显更干净。看视觉对比比看 loss 数字更靠谱。
7. **简化主干 + 文档归档**比保留多种代码路径更利于维护。复活某种老
   方法只需要 git checkout 历史 commit + 看 archived_recipes.txt。


## 阶段 9：评估指标驱动的优化（ysstereo EPE benchmark，2026-04-28）

### 背景

阶段 1–8 都在用"loss 数值 + 视觉对比"驱动迭代。这一阶段第一次把
ysstereo 流水线（去噪 → ISP → 校正 → depth → EPE）当成单一标量
评估指标接进训练循环。流程见 `docs/how_to_run_benchmark.txt`，
评估指标越小越好，并且要先过 ISO=100 降噪Δ < 0.6 的 QC。

### 关键 bug：N2N loss 把残差当去噪图喂进 L1（commit 738f5ac）

trainer 原代码：
```python
den_g1 = model_compiled(g1)        # ← forward(g1)，按 model.py 契约是噪声残差
loss   = L1(den_g1, g2) + λ·TV(den_g1)
```

但 `n2n/model.py` 的契约是 `denoise(x) = x − forward(x)`，即 `forward`
输出**噪声残差**，导出 wrapper 也按 `packed - residual` 减一次。trainer
这里把残差直接当去噪图喂给 L1 损失，会把 `forward(x)` 训练成 `≈ x`（因为
g₂ 在像素强度上 ≈ g₁），结果 `denoise(x) = x − forward(x) ≈ 0`，导出 ONNX
输出几乎全黑。

实测：buggy 训完 5 min 跑 ysstereo，QC 直接炸到 5.83（要求 < 0.6），完全
不能用。改成 `den_g1 = g1 - model_compiled(g1)`（等价 `model.denoise`）
后，5 min 训练 → QC = 0.31，评估指标 = 0.46。

教训：
- **三处契约要对齐**：model 端的 `forward / denoise`、preview 端、export
  wrapper 端。这次 bug 是只有训练循环偏离了三方共识。
- **loss 平台不能证明训练对**。bug 版 loss 也降到 0.0162，因为
  L1(forward(g₁), g₂) 在 forward≈identity 时本来就是噪声本身的强度。
- **盯单一标量评估指标比看 loss 靠谱很多**。loss 没异常、preview 也好像
  在动，但 ONNX 出去就是全黑——只有跑 ysstereo 的 EPE 才暴露出来。

### 9.1 阶段 9 优化扫描（2 h 预算，9 组实验）

目标：从 5 min baseline 的 ~0.46 压到 < 0.44。

| # | 配置 | 训练 | **score** | iso100Δ | 结论 |
|---|------|------|---------:|--------:|------|
| 1 | base=48, depth=3, L1+0.01TV   | 5 min  | 0.5646 | 0.32 | baseline；跑次方差 ~0.1 |
| 2 | tv_lambda=0                   | 3 min  | 0.7772 | 0.32 | TV 必须有 |
| 3 | base=64                       | 5 min  | 0.4628 | 0.32 | 加宽就涨 |
| 4 | Charbonnier+TV                | 5 min  | 0.5124 | 0.32 | 没拉差距，N2N 配对噪声 outlier 不多 |
| 5 | base=48, 10 min               | 10 min | 0.4435 | 0.32 | 文档基线 ≈ 复现（doc 0.4269） |
| 6 | base=64, 10 min               | 10 min | 0.4405 | 0.32 | 卡 0.44 |
| 7 | base=64, 15 min, lr=1e-4      | 15 min | 0.4407 | 0.32 | lr 减半 + 步数加倍无收益 |
| 8 | base=64, **depth=4**, 20 min  | 20 min | 0.4777 | 0.33 | 加深退化（receptive field 偏大） |
| 9 | **base=64, depth=3, 20 min**  | 20 min | **0.4188** | 0.32 | **采纳** |

最优 ckpt: `result/experiments/final_base64_d3_20min/{model.pt, model.onnx}`
（5.42 M 参数，base=48 那版是 3.05 M）。

### 教训（9.x）

8. **跑次方差很大（同 cfg 跨次跑 ~0.1 跳）**。loss 已在平台抖动，
   ysstereo ISP/depth 又是非确定性。判结论要看趋势，不要单点比。
9. **加宽 > 加深**。base 48 → 64 持续涨分；depth 3 → 4 反而退化，
   猜测 4 层下采样后感受野盖到 64×64 bayer，对拍摄噪声过参。
10. **loss 平台 ≠ score 平台（再次确认 9.6 的观察）**。loss 1 分钟内
    就到 0.0162，但 score 还会随训练时长持续小幅改善（5 → 10 → 20 min:
    0.46 → 0.44 → 0.42）。继续训练在压参数微调，不在压 loss。
11. **lr 减半 + 步数加倍无收益**（实验 6 vs 7 持平）。固定 wall-clock
    内涨分，加宽 / 延长比 schedule 玄学更直接。
12. **目标一旦从 loss 切到下游指标，"loss 改进 = 视觉 / 任务改进"
    的前提就不成立**。这次最显著的几个胜负点都跟 loss 数值无关
    （bugfix 把 0 分的输出救回来；加宽 base 让 loss 反而略升 0.0001
    但 score 涨 0.02）。
