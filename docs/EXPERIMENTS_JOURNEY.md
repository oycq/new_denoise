# N2N 降噪：实验过程与知识沉淀

> 这一份是**整个项目的总结性文档**，按时间顺序记录从最初的 L1 baseline
> 一路到当前最佳的所有尝试、量化结果和经验教训。
>
> **当前线上 trainer 配方 = L1 + 0.01·TV**（`n2n/trainer.py` 默认，
> `python run_train.py 600` 起 GUI 训 10 min；详见阶段 8）。
>
> **新对称-ISP benchmark 上的当前最佳 = method28**：3-model output-ensemble
> (m0 + m18 + m24) 各做 4-way correct-TTA，score = **0.3706**（用户目标 < 0.2
> 未达成；详见阶段 10 的攻坚总结）。
>
> 推荐阅读顺序：
> - 新读者 → **阶段 12** 先看主干结论 →（可选）回看 1-8 的细节
> - 想做新优化 → 直接跳 **阶段 10**（当前 benchmark + 30 个候选的复盘 +
>   还没试过的方向）
> - 阶段 9 的 score 数字基于**旧 benchmark**（vendor-ISP-asymmetric），
>   跟阶段 10 不可直接对比，已在该节顶部标注 ⚠
>
> 早期尝试过的 paper N2N consistency reg / robust 训练 / VGG perceptual loss
> / 各种 mode penalty 都已废弃；其代码归档到本地 `_archive_local/`（git 忽略），
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
├── speedtest.py                    # 1 min 训练吞吐量测试
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
├── speedtest.py                    # 1 分钟训练吞吐量测试
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

新机器先 `python speedtest.py` 跑 1 min 看实测 sps，再调 `train_seconds`。

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
  本文**附录 A**，包括完整 recipe / 复活 commit。

### 教训

5. **方法选择要平衡 sps 与视觉收益**。paper N2N 公式更优雅、抗 grid
   不需要 TV 强惩罚，但视觉收益不到 10%、sps 砍半，做日常迭代不划算。
6. **loss 平台 ≠ 视觉到顶**。L1+TV 在 5 min 时 loss 已经平台，但 10 min
   视觉上仍明显更干净。看视觉对比比看 loss 数字更靠谱。
7. **简化主干 + 文档归档**比保留多种代码路径更利于维护。复活某种老
   方法只需要 git checkout 历史 commit + 看本文附录 A。


## 阶段 9：旧 ysstereo benchmark 上的优化扫描（2026-04-28，**score 数字已废弃**）

> ⚠ **这一阶段所有 score 数字基于旧 ysstereo benchmark**：denoise=0 用 vendor
> 预渲染的 `sensor*_isp.png`，denoise=1 走我们的 `isp_process`。两条 ISP 路径
> 不一致，会给 EPE 加进一个 ~0.5 像素的"恒定偏置"，把降噪贡献淹没。
>
> 2026-04-29 重写的 `benchmark/run.py`（见 `benchmark/README.md` 的「ISP
> 对称性」节）让 denoise=0 / denoise=1 共用同一条 `raw → isp_process → rectify`
> 离线 ISP，identity denoiser 上 denoise=0 / denoise=1 byte-equal。这是**当前
> 评估管线**。
>
> **本阶段 9 的 score 数字（0.4188 / 0.4435 等）跟阶段 10 的新 benchmark 数字
> 不可直接比较**——保留下来只为复盘当时遇到的"残差当 denoise 喂进 L1"那
> 个 bug 和加宽 / 加深 / lr 扫描的 methodology。结论部分对新 benchmark **大
> 部分不复现**，详见阶段 10。

### 背景

阶段 1–8 都在用"loss 数值 + 视觉对比"驱动迭代。这一阶段第一次把
ysstereo 流水线（去噪 → ISP → 校正 → depth → EPE）当成单一标量
评估指标接进训练循环。流程见 `docs/how_to_run_benchmark.md`，
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


## 阶段 10：新对称-ISP benchmark 下的「< 0.2」攻坚（2026-04-29 / 30）

### 10.0 任务定义与约束

- 评估管线：**新版** `benchmark/run.py`（denoise=0 / denoise=1 共享同一条
  离线 ISP，identity-denoiser 在两条曲线上 byte-equal，参考 `benchmark/README.md`
  的「ISP 对称性」节）
- 起点：`L1 + 0.01·TV`、`base=48`、`depth=3`、3 min 训练 → score = **0.4131**
- 目标：score **< 0.2**（用户明确目标，见 `docs/how_to_improve.md`）
- 约束：单次训练 ≤ 10 min（保证公平，避免靠堆训练时间作弊）
- 硬件：双 RTX 4090，可并行训练
- 跑次方差 ~ **0.05**（depth 网络对 RGB 微扰敏感，参考 `benchmark/README.md`
  末尾"单次跑分有 ~0.05 量级的方差"）

评估指标关键性质（决定后面所有优化方向）：
- score = 把所有 (scene, ISO ∈ [100..1600], k=1) 的 denoised EPE pool 起来取平均
- baseline = 每个场景的 (ISO=100, denoise=0) 视差图
- 含义：denoiser 在 **gain=1（=ISO=100，最干净）上也要近似等价于不动**，
  否则 ISO=100 上 denoise=1 跟 denoise=0 不一致也会被算进 score
- 训练数据 `train_data/Data/<scene>/<gain>/sensor[23]_raw.png`，
  gain ∈ {1, 2, 3, 4, 6, 8, 10, 12, 14, 16}，trainer 默认 `find_raw_files`
  rglob `*_raw.png` 把所有 gain 都收进训练集

### 10.1 时间线（30+ 候选，分 9 轮）

```
Round 1   单纯堆配置（base=64 / +10min）       全部 ≤ baseline，跑次方差吃掉差异
   ↓
Round 2   推理时 4-way TTA                      score 看似掉到 0.346 ★
   ↓     （★ 但用户后来发现高 ISO 输出红偏！）
Round 3   多模型 ensemble + 不同 seed           ensemble 拉到方差均值，cap 0.37
   ↓
诊断：    method3 红偏 → 揭穿 channel-permute 是 metric gaming
   ↓
Round 4   SWA / tv=0 / 训练侧改                 没改善
   ↓
Round 5   训练时 channel-permute aug            红偏修了，分数也回到 0.42
   ↓     （证实 method3 的"收益"绝大部分是偏置 game metric，不是真降噪）
Round 6   监督训练 (gain_k → gain_1)            颜色完美但分数差
   ↓     （per-ISO 拆解发现 N2N / supervised 在不同 ISO 上严格互补）
Round 7   paper N2N + EMA distill              都失败（残差头 / bf16 NaN）
   ↓
Round 8   ★ 混合 N2N+supervised + ensemble      ★ method28 = 0.3706 ★
   ↓
Round 9   扩 ensemble (mixed-loss seed sweep)   mixed-loss 跨 seed 高方差，
                                                 method24 是侥幸；不稳
```

### 10.2 关键发现（按重要性排）

#### 1. **score 跟视觉清晰度反相关** —— depth 网络偏好的不是「干净」

对 `changqiao_qiaodi/16/sensor2_raw.png` 同帧做 sobel-gradient 检查（rectified
RGB 上的 Sobel 均值，越大代表越清晰 / 边缘越锐）：

| 输入 | Sobel 均值 | benchmark score |
|---|---:|---:|
| 真 gain=1 raw（参考干净） | **30.68** | — |
| method0 (N2N + L1+TV) | 19.68（**比真干净还低 36%**！） | 0.4131 |
| method18 (supervised gain_k→gain_1) | 23.35（接近真 gain=1） | 0.4338 |
| method23 (混合 loss) | 23.04 | **0.7532** |

→ 视觉最干净的 method23 在 benchmark 上反而最差。
→ **depth 网络 prefer 一种特定的 noise/blur balance，不是越接近真 clean 越好**。
   N2N 的"过度平滑但平滑得均匀"恰好处于这个 sweet spot；supervised 把信号往
   gain=1 的真实分布推反而出 depth net 的训练分布外了。
→ 这条经验也解释了阶段 8 (VGG perceptual / N2V2) 的"残留更自然但 EPE 没改善"
   的现象——视觉自然 ≠ depth-net-friendly。

#### 2. **TTA 在 RGGB lattice 上的 channel permute 数学正确但实践引入色偏**

method3 第一版 4-way TTA 的设计：在 packed 4-channel 空间做 h-flip / v-flip
/ 180-rot，每个 spatial flip 配对应的 channel permute（R↔Gr, R↔Gb, R↔B
等），保证 zero-residual 模型上 round-trip 严格 identity。

数学上这是 RGGB lattice 上的合法 D2 对称群操作——但 UNet 学的是 **per-channel
噪声模型**（R 通道有 R-style 噪声、Gr 通道有 Gr-style 等）。permute 让模型
看到 "Gr-content 在 R 位置" 这种通道内容跟标签错位的输入，模型用 R-style
残差去 denoise Gr-content，4 路平均后形成不对称残差：

```
method0 (无 TTA) effective residual [R, Gr, Gb, B] = [+0.00042, -0.00061, -0.00122, +0.00085]
method3 (permute-TTA)                              = [-0.00165, +0.00258, +0.00253, -0.00337]
```

→ TTA 让 R/B 被推上去、Gr/Gb 被拉下去，高 ISO 上累积成可见红偏（`changqiaoqiaodi_16_1.png`
   R/G 从 1.05 → 1.16）。
→ **修法 (`method11_tta_correct`)**：packed 空间 spatial flip 但**不 permute channel**，
   等价于"flip 原图 + 平移 1 像素恢复 RGGB 相位"，每个 packed channel 始终
   装载正确内容类型。round-trip identity max|err| = 0.0，颜色 R/G=1.030 干净。
→ method3 (0.3459) → method11 (0.4002) **3% 是真正的 TTA 收益，13% 是
   channel-bias metric gaming**。后续所有 ensemble 都用 correct-TTA，不再
   用 permute 版本。

#### 3. **gain=1 不是干净 ground-truth**——supervised 的天花板

训练数据每个 (scene, sensor) 在 gain ∈ {1..16} 拍的是同一帧（pixel corr coef
**0.99+**），gain=1 噪声最小。`method18_supervised_gain1_train.py` 利用这一点
做监督：

- input = packed(gain=k)，target = packed(gain=1)，同一 crop，L1 + TV loss
- 必须**也把 gain=1 当输入**（target=自己=identity）；否则推理时 gain=1 输入
  行为未定义，score 直接炸到 **0.8984**（method16）

修了 gain=1 输入后（method18，3min）：
- score = 0.4338（比 N2N baseline 0.4131 还差 +0.02）
- 但 R/G = 1.037（颜色完美，比 method0 的 1.031 都好）

为什么颜色完美 score 反而差？
- gain=1 跟 gain=k 不是 100% pixel-aligned（毫米级抖动 / 微动），
  supervised L1 在轻微错位的 (input, target) 上训出"轻度模糊"，丢了
  stereo matching 需要的高频特征
- 经 ISP 后丢失的细节正好是 depth net 用来匹配的关键信号

#### 4. **per-ISO 互补**：N2N 强在高 ISO，supervised 强在低 ISO

| ISO | method11 (N2N + correct TTA) | method18 (supervised) |
|---:|---:|---:|
| 100 | 0.160 | **0.068** ← supervised 完胜 |
| 400 | 0.227 | 0.222 |
| 1000 | 0.476 | 0.489 |
| 1600 | **0.854** ← N2N 完胜 | 1.079 |

→ 低 ISO 上 N2N 因为模型本身略动，比 "supervised 学了 identity" 多了点 EPE
→ 高 ISO 上 supervised 因为微糊，stereo 匹配特征比 N2N 弱
→ **混合 loss / 输出 ensemble 这两条路径能拿到双方优点**——这个洞察
   直接导出了 method24 / method28（见 10.4）

#### 5. **mixed-loss recipe 跨 seed 极不稳**

method23 = N2N L1 + supervised L1 + chperm aug + TV，10min 训练同 recipe
不同 seed：

| seed | score |
|---:|---:|
| 84 | **0.3777**（method24 = 唯一好的，进了 ★ ensemble） |
| 7  | 0.5396 |
| 21 | 0.7532 |
| 42 | 0.8291 |

→ N2N 单 recipe 跨 seed std ~0.03，mixed-loss recipe std ~0.30，**10× 倍差**。
→ method24 的 0.3777 大概率是侥幸，不应当作"可复现的最佳单模型"。
→ 后续 ensemble 扩充用 N2N 不同 seed 更稳（method30/31，0.39-0.42 区间）。

#### 6. **训练侧老 trick 在新 benchmark 不复现**

| 方案 | 阶段 9 旧 benchmark 报告 | 新 benchmark 实测 |
|---|---|---:|
| base=64 加宽 + 10min | 持续涨分 | **退化**到 0.4417 |
| 训得更长 (3→10min) | "10min 是甜点" | **打平** 0.4203 ≈ 跑次方差 |
| tv_lambda=0 | 不行 | 同样不行 0.4543 |
| paper N2N consistency reg | ~10% improvement | **退化**到 0.4730 |
| EMA self-distillation | docs claim 有效 | **NaN 训不动**（bf16 + compile + 双 forward） |
| SWA (snapshot weight avg) | n/a | 没改善 0.4162 |

→ 新 benchmark 下"训得更狠 / 训更长 / 用更深 trick"边际趋零。
→ paper N2N 配残差头不工作（docs 里 paper N2N 配的是非残差 UNet+N2V2 anti-checker，
   完整复活需要换整套架构，不是单点改 loss）。
→ EMA distill：torch.compile + autocast bf16 + (g1 + 全图) 两种 input shape 在
   step 1k 后 NaN；关 compile 后不 NaN 但 student 学不到东西。

#### 7. **ensemble 是当前唯一稳定的"涨分"路径**

correct-TTA 多模型 ensemble（`result/method25_m0_m18_ensemble/build_ensemble.py`
是通用模板）：

```
method0 alone (3min N2N)                                        0.4131
+ correct TTA (4-way D2 in packed space, no perm)               0.4002
+ method18 (supervised) ensemble                                0.3845
+ method24 (mixed N2N+sup, lucky run)                          ★ 0.3706
+ method14 (chperm-aug N2N) → 0.3730       (m14 跟 m0 太像，没增益)
+ method30 (N2N seed=42) → ?              (待试)
+ method31 (N2N seed=7, 单跑 0.3941) → ?  (待试)
```

每加一个 truly diverse 模型 ~ -0.005 到 -0.015，diminishing returns。

### 10.3 当前 Leaderboard（按 score 升序，颜色不干净的方案标 ⚠）

```
#    方法                                                          score   备注
---  ------------------------------------------------------------  ------  ----
★28  method28_3model_ensemble (m0+m18+m24) + correct TTA           0.3706  ★ 当前最佳
24c  method24 (mixed N2N+sup, 10min) + correct TTA                 0.3777  单模型最佳（跨 seed 不稳）
32   method32_4model (m0+m14+m18+m24) + correct TTA                0.3730  m14 跟 m0 太像
25   method25_2model (m0+m18) + correct TTA                        0.3845  最简洁有效的 ensemble
24n  method24 mixed + no TTA                                       0.3871
31   method31 N2N seed=7 10min                                     0.3941  单 N2N 最佳
3    method3 4-way permute-TTA                                     0.3459  ⚠ 红偏 R/G=1.16，已下架
4    method4 8-way D4 TTA                                          0.3440  ⚠ 同样红偏
8    method8_3model_ensemble + permute TTA                         0.3697  ⚠ 红偏
11   method11_tta_correct  4-way correct-TTA (no permute)          0.4002  曾经的 honest 最佳
0    method0_baseline_3min                                         0.4131  起点
12   method12_swa_correct_tta  SWA 4 snapshots                     0.4162
30   method30 N2N seed=42 10min                                    0.4199
2    method2_base48_d3_10min  仅加时长 3→10min                     0.4203  ≈ 跑次方差
18   method18_supervised  no TTA                                   0.4338
1    method1_base64_d3_10min  base=48→64 + 10min                   0.4417  反而变差
13   method13_tv0_correct_tta  tv=0 5min                           0.4543  TV 必须有
20n  method20 paper N2N (10min)                                    0.4730  残差头跟 paper 配方不匹配
22   method22_iterative_tta_n2  n=2 串行 correct TTA               0.4350  scene 7 过度去噪
27   method27 mixed-loss seed=7 10min                              0.5396  mixed-loss 不稳
23   method23 mixed-loss seed=21 3min                              0.7532  mixed-loss 不稳
21   method21 EMA distill                                          0.78-0.89  bf16 数值问题
26   method26 mixed-loss seed=42 10min                             0.8291  mixed-loss 不稳
16   method16_supervised (excluded gain=1 input)                   0.8984  灾难
```

### 10.4 还没试过 / 想试但没来得及（按预期价值排）

A. **score-aware 端到端反传**（最对齐 metric，难度最高）
   - 写 torch-纯实现的 ISP（demosaic 用 Malvar 5x5 conv，LSC mul，AWB 矩阵，γ pow）
   - rectify 用 `F.grid_sample`（已经天然可微）
   - depth net 是 PyTorch（mmengine 包了一层）→ 应该可微
   - 预计算 7 个场景的 disp(gain=1) 当 ground truth
   - L = L1(disp(denoised), disp_gt)，反传到 denoiser
   - 每 step ~1-2 s，10min 能跑 300-600 步。但 gradient 可能稀疏

B. **更大更稳的 ensemble**（已知能 work，diminishing returns）
   - 训 5+ 个 N2N 不同 seed (10min each)，全部 + correct TTA + ensemble
   - 预计 0.37 → 0.34-0.35（每个新模型 -0.005）

C. **修 method20 paper N2N**：换非残差头 + γ_final=1.0（更小 ramp）+
   N2V2 架构（BlurPool + PixelShuffle）整套换上，不是单改 loss

D. **修 method21 EMA distill**：去掉 bf16 autocast 跑 fp32，或 fp16+GradScaler

E. **EATV (Edge-Aware TV)** 替换均匀 TV：边缘处 TV 权重指数衰减

F. **大模型 + 蒸馏到小模型**：base=96 训 10min，蒸馏成 base=48 做 inference

### 10.5 < 0.2 是否可达？坦率讨论

不一定。变量分析：
- **跑次方差 ~0.05** ← 单次任何 score 都有 ±0.05 噪声
- **denoiser=identity baseline = 0.78** ← 理论可压 100%（→ 0）
- **但 depth net 自身在高噪输入上有显著 EPE**，不仅仅是去噪问题
- 7 场景里 scene 2/5（高噪 + 高细节）是天花板，noisy EPE 1.3-1.7，
  denoised 最低 0.45-0.50，再降 50% 可能受限于 depth net 自身性能
- 加上"score 跟视觉清晰度反相关"这个反直觉性质，意味着标准
  denoising loss（L1 / MSE / charb / VGG）跟 score 的相关性都不强

**最直接的路径是方向 A（端到端反传），但实现成本高**。中短期路线：扩大
N2N seed ensemble 到 5-6 个，应该能稳定到 0.34-0.36 区间。

### 10.6 工具与可复用脚本

代码沉淀（不再删掉，留作以后复用）：

- `result/method0_baseline_3min/` — 起点 ckpt（不动）
- `result/method11_tta_correct/build_tta_correct_onnx.py` — **正确 TTA**
  （4-way packed flip，无 channel permute，无红偏；推荐路径）
- `result/method3_tta_baseline/build_tta_onnx.py` — permute TTA（**已知有红偏**，
  历史保留作对比，**新工作不要用**）
- `result/method4_tta8/build_tta8_onnx.py` — 8-way D4 permute TTA（同样有红偏）
- `result/method25_m0_m18_ensemble/build_ensemble.py` — **多模型 correct-TTA
  ensemble**（推荐，可任意拼 ckpt）
- `result/method14_chperm_aug_train.py` — N2N + 训练时 channel permute aug
  （让模型变 channel-equivariant，permute-TTA 不再有红偏；但分数没真正涨）
- `result/method16_supervised_gain1_train.py` — 监督训练 (gain_k → gain_1)
- `result/method23_n2n_plus_sup_train.py` — 混合 N2N + supervised loss
  （★ recipe 不稳，跨 seed std 0.30，需要多 seed 试 pick best）

### 10.7 经验（13–20）

13. **score 跟视觉清晰度反相关**——depth 网络 prefer 一个特定的 noise/blur
    平衡，不是越接近真 clean 越好（method0 Sobel 19.68 比真 gain=1 的 30.68
    还低，但 score 反而是阶段 10 baseline 的最佳之一）。
14. **TTA 在 Bayer pattern 上做 channel permute 数学正确但引入不对称残差**：
    UNet 学的是 per-channel 噪声模型，permute 让它在 Gr-content 上用
    R-style denoising。修法是 packed-level flip 不 permute（等价 flip+平移
    1 像素，RGGB 相位保持）。
15. **gain=1 不是 ground-truth**：跟 gain=k 之间有微小 misregistration
    （毫米级抖动），supervised L1 在错位 pair 上训出"轻度模糊"，丢了
    stereo matching 高频特征。
16. **per-ISO 互补是真实的**：N2N 强在高 ISO，supervised 强在低 ISO；
    输出空间 ensemble / 混合 loss 能拿到双方优点。
17. **mixed-loss recipe 跨 seed std 极大**（0.30 vs N2N 单 recipe 的 0.03），
    method24 是侥幸；后续 ensemble 扩用稳定 recipe。
18. **bf16 + torch.compile + 一个 model 两种 input shape**（如 g1 = 128×128
    + full = 256×256）会在 step 1k 后 NaN。混合-shape 训练时禁用 compile
    或确保单一 shape。
19. **paper N2N 必须连同非残差头一起换**：单独搬 consistency reg 到残差头
    上不工作，要 N2V2 BlurPool + PixelShuffle 整套架构换上。
20. **新对称 ISP benchmark 让阶段 9 的"加宽涨分""10min 是甜点"等单变量
    结论失效**：新 benchmark 下这些都不复现，跑次方差 0.05 把所有"小改进"
    吃掉了。涨分必须是 ensemble / TTA / 训练范式级的改动，单点 hyperparam
    扫描 ROI 太低。


---

## 附录 A：主干配方复活索引（已废弃的 main，留作 git 复活用）

> 仓库主干历史上长成过几种不同的样子，截至当前（阶段 10）锁定为
> **L1 + 0.01·TV** + **method28 三模型 correct-TTA ensemble** 做最终交付。
>
> 下面记录被踢出主干的方法的 git 复活路径。**这些"为什么放弃"的理由
> 多半基于阶段 9 的旧 benchmark 或视觉 / loss 数值评估**，跟新对称-ISP
> benchmark **未必一致**——只有 paper N2N 在阶段 10 做过新 benchmark 复测
> （method20，score 0.4730，比 baseline 还差，详见 §10.2 第 6 条）。其它
> 方案在新 benchmark 下都没数据，复活前需要自己测。

### A.1 commit 演化简表（旧 → 当前）

```
47ba227   init
50bafdd~  early sweep, 16 candidates, registered losses
c171f66   adopt MS-Charbonnier + patch=256
31a8009   敲定 L1 + 0.03·TV，推理 / 评估流程
b098cd6   精简到两条路径：L1+0.03·TV + VGG perceptual
bab9f3b   robust_trainer 找回 5 项速度优化（l1+TV 65→120 sps、vgg 45→65 sps）
dc5c068   精简到 L1 + 0.05·VGG（删 robust_trainer / loss 注册表）
fd46c6d   回到 paper N2N（删 VGG，加一致性正则项，换非残差 UNet）
─────     主干切回 L1 + 0.01·TV（阶段 8）
─────     当前：trainer 仍为 L1 + 0.01·TV；交付端用 method28 ensemble（阶段 10）
```

### A.2 paper N2N（CVPR 2021 原配方）

公式：
```
L = ‖ f(g₁) − g₂ ‖²  +  γ(t) · ‖ (f(g₁) − g₂) − (g₁(f(y)) − g₂(f(y))) ‖²
```
- f：网络（**非残差** UNet，输出即去噪结果）
- γ(t)：从 0 线性升到 gamma_final（默认 2.0）

UNet 架构（N2V2 anti-checkerboard 修复版）：
- 非残差头：`denoise(x) = forward(x)`，不是 `x − forward(x)`
- BlurPool 下采样：3×3 [1,2,1] 可分离模糊 + avg_pool(2)，替代 max_pool
- PixelShuffle 上采样：1×1 conv into 4·c channels + nn.PixelShuffle(2)，
  替代 stride-2 ConvTranspose2d
- depth=3, base=48；输出端 1×1 conv 头用 Kaiming init

trainer.py 训练循环关键片段：
```python
# γ ramp
gamma_t = (elapsed / budget) * cfg.gamma_final

idx1, idx2 = generate_index_pair(batch)
g1 = subimage_by_idx(batch, idx1)
g2 = subimage_by_idx(batch, idx2)

with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
    den_g1 = model(g1)
    diff = den_g1 - g2

    with torch.no_grad():
        full = model(batch)
        full_g1 = subimage_by_idx(full, idx1)
        full_g2 = subimage_by_idx(full, idx2)
    exp_diff = full_g1 - full_g2

    rec_term = (diff ** 2).mean()
    reg_term = ((diff - exp_diff) ** 2).mean()
    loss = rec_term + gamma_t * reg_term
```

复活：
```bash
git checkout fd46c6d -- n2n/ benchmark.py run_train.py train_gui.py
```

新 benchmark 表现：阶段 10 method20 用残差头 + 这个 loss 跑 10min，score 0.4730，
比 L1+0.01·TV baseline (0.4131) 还差。**复活时必须连非残差头 + N2V2 一起换**。

### A.3 L1 + 0.05·VGG perceptual

公式：
```
L = L1(f(g₁), g₂) + 0.05 · ‖ VGG16_feat(f(y)) − VGG16_feat(y) ‖₁
```

实现要点：VGG-16 feature L1 + luma/chroma + EATV，组合在 `n2n/losses_perceptual.py`
里（已删，commit b098cd6）。配套 `robust_trainer.py`（EMA self-distillation +
R2R + 像素掩码 + 亚像素抖动，bab9f3b），有独立入口 `train_vgg.py`。

复活：
```bash
git checkout b098cd6 -- n2n/ train_vgg.py
python train_vgg.py    # 顶层入口，同时训 P0_baseline + P4_vgg
```

新 benchmark 下未测。注意阶段 10 的 method21（自己实现的 EMA distill）在
bf16 + torch.compile + 双 forward 下 NaN，复活 robust_trainer 时需要
确认数值稳定性（同 §10.2 第 6 条）。

### A.4 robust_trainer 路径（EMA + R2R + augmentation）

bab9f3b 的 `n2n/robust_trainer.py` 在 N2N main loss 之外加了：
- EMA teacher self-distillation（额外一次 teacher forward）
- R2R (Residual to Residual) 输入扰动
- 像素级掩码避免泄漏
- sub-pixel 亚像素抖动 augmentation

总开销 ~3× compute / step，4090 上 ~85 sps。

复活：
```bash
git show bab9f3b:n2n/robust_trainer.py > n2n/robust_trainer.py
```

新 benchmark 下未测；阶段 10 method21 自己重写的 EMA distill 在 bf16 下
不稳，可能整个 robust_trainer 路径需要回到 fp32 才能跑通。

### A.5 早期 16-strategy sweep（c171f66 时代）

16 个 candidate 比较 `blob_residual / hf_noise / edge_sharpness / composite`，
基于视觉 + composite 标量。完整 ranking 见 §3 / §4（阶段 2-3）。

代码里曾经有过一张 loss 注册表：
```python
LOSSES = {
    "l1": l1,
    "charbonnier": charbonnier,
    "l1_plus_grad": l1_plus_grad,
    "multiscale_l1": multiscale_l1,
    "huber": huber,
    "l1_plus_tv": l1_plus_tv,
    "ms_l1_charbonnier": ms_l1_charbonnier,
}
```
trainer 通过 `cfg.loss_name / loss_kwargs` 选 loss。dc5c068 commit 把整张
表 + losses_perceptual.py 一起删了，单一 loss 更省心，本次也不打算把表加回来。

复活某一个 loss：
```bash
git show bab9f3b:n2n/losses.py | grep -A 20 "def <name>"
# 把函数体复制到 n2n/trainer.py 里替换 F.l1_loss + TV 那几行
```

新 benchmark 下未测；composite 标量跟 score 不强相关（§10.2 第 1 条），
所以 16-sweep 的 ranking 在新 benchmark 下大概率不复现。

### A.6 L1 + λ·TV 的 λ 历史扫描

阶段 3（旧 benchmark / 视觉）的 λ 推荐：

| λ | 视觉评价 |
|---|---|
| 0.05 | 过强 piecewise-flat，墙面 / 树叶细节明显被 cartoonise |
| 0.03 | 接近 sweet-spot，但 lens-shading 角落仍偏平整化 |
| **0.01** | **细节保留最佳，平坦区 blob 残留对比 ISP 参考可接受 ✅** |
| 0.005 | 几乎等价纯 L1，dark roof 残留比 0.01 略多 |
| 0 | 纯 L1，参考 baseline，lens-shading 角落 blob 明显 |

新 benchmark 下：阶段 10 method13 测了 `tv=0` → score 0.4543（比 0.01 的
0.4131 明显差），证实 TV 是必要的；其它 λ 值（0.005 / 0.03 / 0.05）在新
benchmark 下未单独扫，但 method7（tv=0.005 + permute TTA = 0.3943）暗示
0.005 跟 0.01 接近。

要做新 λ 扫描：
```python
from dataclasses import replace
from n2n.trainer import Trainer, TrainConfig
cfg = replace(TrainConfig(), tv_lambda=0.005, ckpt_path="...")
Trainer(cfg).run()
```

### A.7 已归档但保留在本地的代码（git 不跟踪）

```
_archive_local/
├── process_images/                 # 阶段 6 的 loss 曲线 PNG
├── docs_html/                      # 历史 HTML 报告
├── docs_assets/                    # demo_assets / research_7h_assets
├── experiments_7h/                 # 92 候选 7h sweep 全套
├── experiments_3h_extra/           # 3h push 除 robust_trainer 外的所有产物
├── experiments_perceptual_extra/   # 6 候选感知 sweep
├── experiments_perceptual_dir_old/ # 上一版 experiments_perceptual/ 骨架
├── extra_losses/                   # losses_v2.py / losses_v3.py / model_variants.py
├── archives/                       # train_data.zip / report_*.tar.gz
├── reports_download/               # 解压目录
└── old_tools/                      # tools/* 与 result_demo/* 早期脚本
```

阶段 10 的代码（method14_chperm_aug_train.py / method16_supervised_gain1_train.py
/ method23_n2n_plus_sup_train.py / method3_tta_baseline / method11_tta_correct
/ method25_m0_m18_ensemble 等）目前仍 git 跟踪在 `result/`，作为
新-benchmark 路线的可复用工具沉淀（§10.6）。
