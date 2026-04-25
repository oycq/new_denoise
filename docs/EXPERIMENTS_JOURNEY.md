# N2N 降噪：实验过程与知识沉淀

> 本文档总结从「裸 L1 baseline」到「最终敲定 `L1 + 0.03·TV` 训练 ≥ 10 min」
> 之间的全部尝试。包含 23 个独立训练候选、5 个对比报告、20 ROI 跨场景验证。
> 实验脚本和中间产物已 gitignore（仍保留在本地 `experiments/` 和顶层
> `*.py` 测试脚本中），日后想复现某次扫描，照本文索引找回即可。

## 任务定义

- **输入**：1280 × 1088 的 16-bit Bayer (RGGB) raw，归一化到 `[0, 1]` float
- **目标**：最高 ISO（=16，即 train_data 下的 `<scene>/16/sensor*_raw.png`）
  做单图自监督降噪
- **方法**：Neighbor2Neighbor (Huang 2021) 子采样 + UNet 回归噪声残差
- **硬件**：RTX 2070 (8 GB) · FP16 (`torch.amp`) · `cudnn.benchmark`
- **约束**：训练时间在分钟级（用户希望每个候选 5–20 min 内见效）

## 整个过程的 5 个阶段

```
阶段 1     基础流程跑通                    commit 50db87c
   |       N2N + GUI + 时间预算 + result/ 输出
   v
阶段 2     精度 / 显示链路修复             commit 0e14105
   |       16-bit demosaic + float32 全程，消除 staircase
   v
阶段 3     16 个策略横向扫描 (5 min 每个)  commit c2e4176
   |       loss 家族 + UNet 深度 + patch 大小 + 黑电平
   |       composite 排名第一: ms_l1_charbonnier + patch=256
   v
阶段 4     TV-λ 调优 + 时长扫描 + 20 ROI   commit 091298e
   |       4 个 λ × 2 时长 + 7 场景 20 ROI 视觉判定
   |       composite 第一 ≠ 视觉最优；λ=0.01 在 16/20 ROI 胜出
   v
阶段 5     锁定 L1 + 0.03·TV，训练 20 min  当前
           移除测试代码，蒸馏知识到本文档
```

## 阶段 1–2：早期踩坑

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

**根因**：默认 UNet depth=4 + patch=128 (packed)，最深一层的感受野
覆盖了整个 patch，模型实际上学到了「全图平均」的伪解。

**修复**：

- depth → 3（保留局部感受野）
- patch_size → 256 packed（即 512 Bayer，远大于深层感受野）

后续所有候选都用 `depth=3 + patch=256 + base=48` 这个三件套。

## 阶段 3：16 策略扫描

每个候选都用相同的 `Trainer` 和 `evaluate_experiments.py` 评分（详见
[`REPORT.html`](REPORT.html)）。

### 量化指标（每个候选只在固定一帧 sensor2 ISO=16 上测）

| 指标 | 含义 | 方向 |
|---|---|---|
| `blob_residual` | 平面 ROI 上的低频能量（5×5 box filter 的方差） | 越小越好 |
| `noise_floor`   | 平面 ROI 上的高频噪声底（残差的 std） | 越小越好 |
| `edge_sharpness`| 边缘 ROI 上的 Sobel 梯度均值 | 越大越好 |
| `composite`     | `0.5·blob + 0.25·hf + 0.25·edge` 加权 | 越大越好 |

### 16 个候选的角色

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
| 10 | **MS-Charbonnier + p=256** | 阶段 3 第一名 |
| 11 | Charbonnier + p=256 | 验证多尺度增益 |
| 12 | L1 + black-level subtract | FP16 数值稳定性 |
| 13 | L1 + 0.05 · TV | 平面正则 |
| 14-15 | UNet d=4 + 多尺度变体 | 大网络对比 |

### 阶段 3 结论

`10_ms_charbonnier_p256` composite 最高 (0.78)：

- 大平面非常干净（多尺度损失抗 blob）
- 边缘保留得不错（Charbonnier 比 L2 软）

但当时已经看到 `13_l1_plus_tv` 的视觉效果**主观**也很好。这成为阶段 4 的起点。

## 阶段 4：L1 + λ·TV 深度调优

### Total Variation 是什么

```
TV(I) = mean(|I(x+1,y) - I(x,y)|) + mean(|I(x,y+1) - I(x,y)|)
```

惩罚相邻像素差。在 **平面** 区域几乎无影响（相邻像素本来就接近），
但在 **边缘** 区域会把边缘也拉近 → λ 越大边缘越糊。

所以 **λ 是「平面干净 ↔ 细节锐利」的旋钮**。

### 第一轮 4 个 λ × 5 min

详见 [`TV_LAMBDA_REPORT.html`](TV_LAMBDA_REPORT.html)。

| λ | blob ↓ | hf ↓ | edge ↑ | composite ↑ | 视觉 |
|---|---|---|---|---|---|
| 0.005 | 1.033 | 0.902 | 18.27 | 0.493 | TV 太弱 ≈ baseline |
| 0.01  | 0.981 | 0.891 | 17.26 | 0.617 | 甜蜜点 |
| 0.02  | 0.983 | 0.809 | 16.54 | 0.612 | 角落更净 |
| 0.05  | 0.940 | 0.699 | 15.45 | 0.750 | 过度平滑（木纹消失） |
| baseline | 1.007 | 0.890 | 17.97 | 0.575 | 参考 |

### 第二轮 λ=0.01 的 5 vs 10 min

详见 [`TV_LAM01_5VS10_REPORT.html`](TV_LAM01_5VS10_REPORT.html)。

**关键发现**：5 min 时 edge=17.26（< baseline），看起来 TV 把边缘
拍扁了；但 10 min 时 edge=18.07（> baseline 17.97）。

**TV 带来的「边缘软化」是暂时的**——给足训练时间，模型可以同时
学到「平 + 锐」两个目标，因为「平」的目标（TV）和「锐」的目标
（L1 让降噪后跟另一半子图一致）可以兼容，只是优化器需要时间。

### 第三轮 4 λ × 2 时长汇总

详见 [`TV_LAMBDA_DURATION_REPORT.html`](TV_LAMBDA_DURATION_REPORT.html)。

| λ | 5min edge | 10min edge | 恢复幅度 |
|---|---|---|---|
| 0.01 | 17.26 | 18.07 | **+4.7%** |
| 0.02 | 16.54 | 16.98 | +2.7% |
| 0.05 | 15.45 | 15.68 | +1.5% |

**TV 越强，时长能找回的细节越少**。λ=0.05 即使 10 min，高频细节
也救不回来——是真的被挤掉了。

### 第四轮 7 场景 20 ROI 视觉判定

详见 [`TV_20ROI_REPORT.html`](TV_20ROI_REPORT.html)。

20 个 ROI 覆盖 9 个内容类别（暗角 / 平面 / 网格 / 边缘 / 树叶 /
纹理 / 几何 / 玻璃 / 复杂场景）：

| 候选 | 胜的 ROI |
|---|---|
| **λ=0.01 10 min** | **16/20** |
| λ=0.02 10 min | 6/20 |
| λ=0.05 10 min | 4/20（仅纯平面） |
| baseline 5 min | 5/20 |

**重要的反直觉**：composite 排第一的 λ=0.05，在视觉上是最糟的「广义平均」
（树叶融成一团、栅栏糊、文字消失）。说明 composite 这个加权对
**纹理损失**惩罚不足。**视觉判定 > 单一标量指标**。

## 最终决定：L1 + 0.03·TV

经过四轮扫描后用户决定取 **λ = 0.03**：

- 比 λ=0.01 略强（角落更干净），比 λ=0.05 弱很多（不会过度平滑）
- 介于第二名（λ=0.02）和第三名（λ=0.05）之间，吸取了「角落需要更
  强 TV」的视觉判断
- 没有直接测过这个值，所以阶段 5 用 20 min 训练 + 中间快照 (2/5/10/20 min)
  来验证

### 当前默认配置

`n2n/trainer.py` 锁定为：

```python
loss_name    = "l1_plus_tv"
loss_kwargs  = {"lam": 0.03}
patch_size   = 256       # packed-pixel
batch_size   = 6
unet_depth   = 3
base_channels= 48
use_fp16     = True
n2n_lambda   = 0.0       # 纯 L1+TV 主项，不加 N2N 一致性正则
train_seconds= 1200      # 20 min
intermediate_save_seconds = (120, 300, 600)  # 2/5/10 min 各存一份
```

`n2n/losses.py` 还保留了所有 8 个损失函数以便日后实验，但**正式
训练只用 `l1_plus_tv`**。

## 跨阶段的 7 条经验

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
7. **同一个数据集，5 / 10 / 20 min 训练时长都能见到差异**。预算
   ≥10 min 才能享受 TV 的全部好处。

## 文件索引

### 当前生效的代码（git 跟踪）

```
n2n/                         # 生产代码
  raw_utils.py               # 16-bit display pipeline + I/O + Bayer pack
  n2n_sampler.py             # vectorised pair sampling
  model.py                   # configurable-depth UNet (residual head)
  dataset.py                 # patch + flip/transpose aug
  losses.py                  # 8 个 loss（默认只用 l1_plus_tv）
  trainer.py                 # FP16 训练循环 + 中间 checkpoint
  infer.py                   # tile inference + ImageJ-stack 输出
train_gui.py                 # PyQt5 GUI
run_train.py                 # 一键启动 GUI 的 wrapper
docs/                        # 知识总结（本文档 + 5 个 HTML 报告）
```

### 不再 git 跟踪、但保留在本地的产物

```
experiments/                 # 23 个候选的 checkpoint + isp 渲染
  00_baseline_l1/  ...  15_unet4_ms_charb_p256/
  13_l1_plus_tv/  13b_l1_plus_tv_lam02/  13c_l1_plus_tv_lam01/  ...
  _eval/                     # 评估期间生成的中间图（top5_*, roi_*_separate, ...）
  log.txt                    # 每次扫描的训练日志
  REPORT.md                  # 阶段 3 结论的 markdown 版

# 顶层测试脚本
run_experiments.py           # 16 策略扫描的入口
evaluate_experiments.py      # blob/hf/edge/composite 计算 + ROI 抽样
multi_scene_infer.py         # 7 场景 × N 模型推理（含 legacy state_dict 重映射）
generate_20rois.py           # 20 ROI 对比 strip 生成

experiments_*.log
multi_scene_infer.log
```

如果想再做一次扫描或追加新损失：

```powershell
# 把 .gitignore 里被排除的脚本恢复运行（git 不会跟踪结果）
python run_experiments.py --seconds 300 --only 00_baseline_l1
python evaluate_experiments.py
```

`run_experiments.py` 的 `STRATEGIES` 列表有完整的 16 + 7 个候选定义可参考。

## 5 个 HTML 报告速查

| 文件 | 内容 | 看点 |
|---|---|---|
| [`REPORT.html`](REPORT.html) | 16 策略初步排名（阶段 3） | top5 grid + composite 排序 |
| [`TV_LAMBDA_REPORT.html`](TV_LAMBDA_REPORT.html) | 4 个 λ 横向比较（5 min 各） | 量化指标 + 6 ROI 视觉 |
| [`TV_LAM01_5VS10_REPORT.html`](TV_LAM01_5VS10_REPORT.html) | λ=0.01 训练时长效应 | edge 反超 baseline 的证据 |
| [`TV_LAMBDA_DURATION_REPORT.html`](TV_LAMBDA_DURATION_REPORT.html) | 4 λ × 2 时长汇总 | 时长 × λ 的 trade-off |
| [`TV_20ROI_REPORT.html`](TV_20ROI_REPORT.html) | 7 场景 × 20 ROI 视觉判定 | 16/20 胜率给 λ=0.01 |

需要再补一个：阶段 5 的 **「λ=0.03 训练 2/5/10/20 min」对比报告** 由
`docs/TV_LAM03_DURATION_REPORT.html` 提供（一次 20-min 训练加 4 个
中间 checkpoint）。
