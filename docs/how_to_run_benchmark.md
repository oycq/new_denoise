# how_to_run_benchmark · 怎么把 ckpt 拿去跑评估指标

> ⚠ **这份文档大半已过时**：原本对接的 `/home/bobiou/ysstereo/run.sh` 已被
> 替换成仓库内的 `benchmark/run.py`。**当前实际跑 benchmark 的入口和合同
> 看 [`benchmark/README.md`](../benchmark/README.md)**（自包含、对称-ISP，identity-denoiser
> 上 denoise=0/1 byte-equal，是阶段 10 起的评估管线）。
>
> 本文保留下来是因为：
> - §2「ONNX 接口约定」里的「单通道 bayer_01 (1,1,1280,1088) → (1,1,1280,1088)」
>   合同**仍然适用**（`benchmark/run.py::_OnnxDenoiser` 走的就是这个）
> - §3「export_l1tv_onnx.py 用法」**仍然适用**
> - §4「mode 1/2 胶水」、§5「跑 run.sh」、§6「历史成绩 0.4188 / 0.4269」、§8
>   「reproduce 0.4269」等都是**旧 ysstereo 流程**，新读者直接跳过即可

## 1. ysstereo/run.sh 接口（旧流程，仅历史复盘用）

`/home/bobiou/ysstereo/run.sh` 用法：

```bash
./run.sh                                    # 用默认 ONNX + mode 1
./run.sh <onnx路径>                         # 只换模型
./run.sh <onnx路径> <mode>                  # 同时换模型和模式
```

两种模式（在 `denoise/nn_denoise/core.py` 里实现）：

| mode | 处理 |
|------|------|
| 1 = 旧方法 | `img - FPN(again, cam_id) -> onnx -> +9/255 偏置 -> clip` |
| 2 = 新方法 | `img -> onnx -> clip`（无 FPN、无偏置）|

不论 mode 几，**ONNX 的 I/O 都是 `(1,1,H,W)` 单通道 bayer_01 float32**。
mode 1/2 的差只在去噪函数外层（要不要减 FPN、要不要加偏置）。

run.sh 末尾会 print 一行结论：

```
评估指标 = 0.XXXX             ← 想拿的浮点
通过 QC 或 未通过 QC          ← QC = ISO=100 降噪Δ 平均 < 0.6
                                未过 QC 时不要报告分数
```

## 2. ONNX 接口约定（new_denoise ↔ benchmark 的合同，仍然适用）

benchmark 端期望 ONNX 是 **单通道 bayer**：

```
input  : (1, 1, 1280, 1088) float32, 值域 [0, 1]
output : (1, 1, 1280, 1088) float32, 值域 ≈ [0, 1]（外层会 clip）
```

注意 `cv2.imread('sensor*_raw.png', -1).shape == (1280, 1088)`，
即 H=1280 / W=1088，**不是** 1088×1280。导出时按 1280×1088 写就行。

但 new_denoise 主干 UNet 的输入是 4 通道 packed `(1, 4, H/2, W/2)`，
通道顺序 `R, Gr, Gb, B`（见 `n2n/raw_utils.py::pack_rggb`）。所以
**ONNX 内部要把 packing/UNet/residual/unpacking 全部封装进去**，
对外只暴露单通道接口，让 benchmark 端一行胶水都不用改：

```
bayer_01 (1,1,H,W)
    -> PixelUnshuffle(2)         # 等价 pack_rggb 的 R/Gr/Gb/B 顺序
    -> UNet (residual head)
    -> denoised_packed = packed - residual
    -> PixelShuffle(2)
    -> bayer_01 (1,1,H,W)
```

`PixelUnshuffle(2)` 默认通道顺序就是 `[0::2,0::2], [0::2,1::2], [1::2,0::2],
[1::2,1::2]` —— 跟 RGGB 的 `R, Gr, Gb, B` 完全一致，**不用 permute**。

为什么要把 residual 减法塞进 ONNX：
`n2n/model.py` 的 `UNet.forward` 输出"噪声残差"，`UNet.denoise(x) = x - forward(x)`
才是去噪图。`torch.onnx.export` 默认导的是 `forward`，导出来是残差不是去噪图。
塞进 wrapper 里能避免对面忘记减、或者两边对去噪含义理解不一致。

## 3. 一键导出脚本 `export_l1tv_onnx.py`

仓库里有 `export_l1tv_onnx.py`（持久脚本，跟 `speedtest.py` 平级），
默认从 `checkpoints/l1tv_lam01_10min.pt` 导成
`checkpoints/l1tv_lam01_10min.onnx`，分辨率 1280×1088，opset 11：

```bash
cd /home/bobiou/new_denoise
python export_l1tv_onnx.py
# 也可以指定其它 ckpt 或分辨率：
python export_l1tv_onnx.py \
    --ckpt checkpoints/<name>.pt \
    --out  checkpoints/<name>.onnx \
    --height 1280 --width 1088
```

脚本内部做了三件事：

1. 读 `ckpt['config']`，从里面拿 `base_channels` / `unet_depth` 去构造
   `UNet(in_ch=4, out_ch=4, base=base, depth=depth)`，避免硬编码
   （比如以后训了一份 base=64 的）。
2. strip 掉 `torch.compile` 留下的 `_orig_mod.` 前缀（虽然主干当前
   ckpt 没有，但保留这段更稳）。
3. 导完后用 onnxruntime CPU 跑一次随机输入，跟 PyTorch eager 输出对比，
   print `max|diff|`。健康值 ≤ 1e-5，2026-04-28 实测是 8.34e-7。

导出依赖 `onnx`（pkg）+ `protobuf>=3.20`，见末尾 §7「已知坑」。

> 阶段 10 还派生出几种 TTA / ensemble 包装版导出脚本，跟 `export_l1tv_onnx.py`
> 同样输出单通道 bayer ONNX，benchmark 端不用改。详见
> [`docs/EXPERIMENTS_JOURNEY.md`](EXPERIMENTS_JOURNEY.md) §10.6「工具与可复用脚本」。

## 4. 修 `ysstereo/denoise/nn_denoise/core.py` 的 mode 2 胶水（旧流程）

> 旧 ysstereo 用，新仓库内 `benchmark/run.py` 不需要这个补丁。

该文件历史上有过一个版本，mode 2 的 `run_raw` 直接把 `(H, W)` 的 numpy
丢给 ONNX session，跟我们 `(1,1,H,W)` 的合同对不上。修法是让 mode 2
跟 mode 1 共享 `pre_process`（加 batch + channel 维），并 squeeze 输出：

```python
def run_raw(self, inp: np.ndarray):
    inp_bayer_01 = self.pre_process(inp)
    pred = self.session.run(
        [self.output_name], {self.input_name: inp_bayer_01}
    )[0]
    pred = np.squeeze(pred, axis=(0, 1)).astype(np.float32)
    return pred
```

修过之后 mode 1/2 走完全相同的 ONNX I/O 形状，差异只在 `denoise()`
外面（要不要减 FPN、要不要加偏置）。这是和 ysstereo 维护方约定好的
合同，不要把"shape 适配"塞回 ONNX 侧——那样以后每个新模型都要重写
adapter，不划算。

## 5. 跑 benchmark（旧流程）

> 当前实际跑见 [`benchmark/README.md`](../benchmark/README.md)，命令是
> `python benchmark/run.py --onnx <path>`。

5.1 从 new_denoise 目录：

```bash
cd /home/bobiou/new_denoise
python export_l1tv_onnx.py
bash /home/bobiou/ysstereo/run.sh \
    /home/bobiou/new_denoise/checkpoints/l1tv_lam01_10min.onnx 2
```

run.sh 内部会：

- `cd ysstereo/denoise && python prepare_data_for_nn_depth.py`
  （对 ysstereo `Data/<scene>/<iso>/sensor[23]_raw.png` 全量去噪 + ISP + 校正）
- `cd ysstereo/depth && ./run.sh`（跑 depth 网络，140 帧，~10 s）
- `cd ysstereo/benchmark && python generate_plot.py`
  （算 EPE，print 评估指标，存 `Summary_Depth_EPE_Final.png`）

整条 pipeline 单次 ~2 分钟（CPU 推理 + GPU depth）。

5.2 解析输出：

```
最终结论
==================================================
通过 QC (ISO=100 降噪Δ 平均 = 0.3132 < 0.6)
评估指标 = 0.4269
```

- 第一行 "通过 QC" / "未通过 QC" 决定要不要看分：
  - 通过 → 报告分数
  - 未通过 → 这条 ckpt 在低噪声场景反而把指标搞坏了，只报告"未通过 QC"，
    不要报告评估指标
- 第二行的浮点就是要交付的"评估指标"，越小越好

## 6. 历史成绩参考（旧 benchmark，已废弃）

> ⚠ 下表 score 全部基于**旧 ysstereo benchmark**（vendor-ISP 跟我们 ISP 不一致），
> 跟阶段 10 的新 benchmark **不可直接对比**。详见
> [`EXPERIMENTS_JOURNEY.md`](EXPERIMENTS_JOURNEY.md) 阶段 9 顶部的 ⚠ 标注。

| 日期 | ckpt | mode | QC? | 评估指标 |
|------|------|------|-----|---------:|
| 2026-04-28 | `l1tv_lam01_10min` (base=48, depth=3) | 2 | 通过 | 0.4269 |
| 2026-04-28 | `final_base64_d3_20min` (base=64, depth=3) | 2 | 通过 | 0.4188 |

注：跑次方差较大（同 cfg 跨次能差 0.1）；判优劣看趋势别看单点。
完整对比扫描见 [`EXPERIMENTS_JOURNEY.md`](EXPERIMENTS_JOURNEY.md) 阶段 9
（已标 ⚠ 废弃，仅作 methodology 复盘）。

加一行新记录前先确认 ysstereo `Data/` 没动过（数据集变了分数没法横向比）。

## 7. 已知坑

- **protobuf 版本冲突**：该机器装了 `horizon-tc-ui`，它锁
  `protobuf==3.19.4`，但 `onnx>=1.15` 编出来的 `_pb2` 要 `protobuf>=3.20`，
  于是 `import onnx` 会炸：

  ```
  ImportError: cannot import name 'builder' from 'google.protobuf.internal'
  ```

  → `torch.onnx.export` 内部依赖 onnx pkg，连带也炸。修法：

  ```bash
  pip install --user 'protobuf>=3.20.2,<5'
  ```

  会让 horizon-tc-ui 报 dependency conflict 警告，但只是警告，
  暂不用 horizon 那条工具链就没影响。要恢复 horizon 时：

  ```bash
  pip install --user 'protobuf==3.19.4'
  ```

- **mode 1 vs mode 2 的偏置**（旧流程）：mode 1 在 ONNX 输出后加 `+9/255`，
  这是为旧 nikon ckpt 在训练时减了 8-bit 黑电平 9 设计的对冲；
  new_denoise 的 ckpt **训练时没减黑电平**（`subtract_black_level=False`），
  所以**必须用 mode 2** 跑，否则 ISP 会被偏到亮端，分数会糟。
  （这个是 mode 选 2 的根本理由，不只是"新方法"那么轻飘的描述。）

- **`cv2.imread` 返回 `(H, W) = (1280, 1088)`**：别被"1280×1088"的口语
  顺序绕进去，1280 是高、1088 是宽。`export_onnx` 也是 `(1, 1, 1280, 1088)`。

- **ckpt 里的 `_orig_mod.` 前缀**：主干 trainer 当前是 `model.state_dict()`
  存原始 model（不是 compiled wrapper），所以没前缀。但 `gen_compare.py`
  那边的兼容代码（见 [`how_to_compare.md`](how_to_compare.md) §3.4）也搬到
  `export_l1tv_onnx.py` 里了——以后哪天 trainer 改成存 compiled state_dict，
  导出脚本不用改。

- **ysstereo 数据集要存在**（旧流程）：没有 `/home/bobiou/ysstereo/denoise/Data`
  就没法跑，这个不在本仓库管辖范围。**新流程不需要**：`benchmark/run.py`
  直接读仓库内 `train_data/Data/`。

## 8. 怎么 reproduce 2026-04-28 的 0.4269（旧流程）

```bash
cd /home/bobiou/new_denoise
pip install --user 'protobuf>=3.20.2,<5'    # 如果首次跑
python export_l1tv_onnx.py                  # 默认 l1tv_lam01_10min
bash /home/bobiou/ysstereo/run.sh \
    $PWD/checkpoints/l1tv_lam01_10min.onnx 2
```

stdout 末尾应当出现：

```
通过 QC (ISO=100 降噪Δ 平均 = 0.3132 < 0.6)
评估指标 = 0.4269
```

> ⚠ 这个 0.4269 是旧 benchmark 的数字。新 benchmark 下 baseline 大约是 0.4131，
> 详见 [`EXPERIMENTS_JOURNEY.md`](EXPERIMENTS_JOURNEY.md) 阶段 10。
