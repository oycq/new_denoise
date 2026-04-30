# benchmark/

整套从 ONNX 去噪器 → ISP → 立体矫正 → 双目深度 → EPE 评估的流水线
搬到了项目内、用相对路径自包含、暴露成一个 Python 函数 + CLI。

来源：`/root/ysstereo/`（denoise + depth + benchmark）；本目录是干净版，
保留了"基础 ISP + 立体矫正 + 立体深度网络 + EPE 评估"的有效代码，
**剔除了 FPN 校正、加噪相关、模式 1 旧路径、AWB 调试可视化、
死掉的 TMO 调用、以及 main 入口里的 GUI 浏览代码**。
benchmark 用的图直接读仓库里的 `train_data/Data/`（跟训练集是同一份），
不再需要外部数据目录。

> **2026-04-29 更新**：`denoise=0` 和 `denoise=1` 两条 EPE 曲线**改成了
> 走同一条离线 ISP**（`raw → isp_process → rectify`）, 只差 `denoiser`
> 是否调用一次。之前 `denoise=0` 用的是 vendor 预渲染的 `sensor*_isp.png`,
> 跟 `denoise=1` 走 `isp_process` 不是一条路, ISP 差异会被算进 EPE 里把
> 降噪贡献淹没。改完之后 **`denoiser=identity` 时两条曲线点对点严格相
> 等**, 评估指标的 Δ 才真实反映"降噪本身"的贡献。详见下面 *ISP 对称性*
> 一节。

## 目录布局

```
benchmark/
├── run.py                  # 三阶段编排 + CLI: prepare → depth → eval
├── isp/                    # 1. 基础 ISP (单通道 bayer_01 → uint8 BGR)
│   ├── isp.py              #    BL → demosaic → LSC → AWB+CCM → sRGB γ
│   ├── lsc.py              #    启动时建增益图, 调用零开销
│   ├── awb.py              #    白点估计 + CCM 色温插值, 纯 numpy
│   └── assets/             #    awb_calib.json, cam{2,3}_lsc.png
├── rectify/                # 2. (顺手算的, 双目深度依赖) 立体矫正
│   ├── rectify.py          #    启动时建 remap maps, 调用零开销
│   └── assets/calib.json
├── depth/                  # 3. 双目深度推理
│   ├── runner.py           #    run_depth(img1_dir, img2_dir, out_dir) 函数接口
│   ├── ysstereo/           #    vendored mmengine-based 深度网络包
│   ├── model_config.py     #    网络配置 (原 edifatprune_ft_defront_…_joint.py)
│   └── weights.pth         #    iter_150000.pth 权重 (93 MB)
├── eval/                   # 4. EPE 聚合 + QC + 出图
│   └── epe.py              #    evaluate(depth_dir, plot_dir) -> EvalResult
└── output/                 # 中间产物 + 最终图 (run.py 写)
    ├── isp/{2,3}/          #   denoise + ISP + rectify 后的 BGR PNG
    ├── depth/              #   uint16 视差*100
    ├── visual/             #   可视化深度图
    └── Summary_Depth_EPE_Final.png
```

依赖（已经装在当前环境里）：
`onnxruntime`, `mmengine`, `mmcv`, `prettytable`, `pyexr`,
`numpy<2`, `torch`, `opencv-python`, `matplotlib`, `tqdm`。

模型权重的入库策略：

| 文件 | 大小 | 是否入库 | 说明 |
|---|---:|---|---|
| `benchmark/depth/weights.pth` | 89 MB | ✅ 入库 | 第三方立体深度网络，跨迭代稳定（vendored 自 `iter_150000.pth`） |
| `benchmark/checkpoint.onnx`   | ~10–25 MB | ❌ ignore | **是被测物本身**，每次训练都换；进库会让仓库越来越大 |

`benchmark/checkpoint.onnx` 是 `run.py` 默认会去找的去噪 ONNX：本机有就用、
没有就报 `--onnx` 参数缺失。clone 完仓库**不会**带这份；要想 `python benchmark/run.py`
不带任何参数就能跑，**自己往这个路径丢一份训好的 ONNX**：

```bash
python run_train.py 600                          # 训 10 min
python export_l1tv_onnx.py \
    --ckpt checkpoints/n2n_model.pt \
    --out  benchmark/checkpoint.onnx             # 直接落到默认位置
```

`benchmark/output/` 也被 ignore（每次跑都重生成，~400 MB）。

## 跑法

```bash
# 1. 默认 ONNX (benchmark/checkpoint.onnx, 本地存在时)
python benchmark/run.py

# 2. 显式指 ONNX 路径
python benchmark/run.py --onnx checkpoints/my_model.onnx

# 3. Python 调用
from benchmark.run import run_benchmark
res = run_benchmark(
    onnx_path="checkpoints/my_model.onnx",  # 不传就用 benchmark/checkpoint.onnx
    data_root="train_data/Data",            # 默认就是这个
    output_dir="benchmark/output",          # 默认就是这个
)
print(res.score)
```

`EvalResult` 结构：

```python
@dataclass
class EvalResult:
    score: float | None              # 唯一指标 (见下), 越小越好
    per_scene: list[dict]            # 各场景 EPE 曲线原始数据
    avg: dict                        # 全局平均
    plot_path: str                   # Summary_Depth_EPE_Final.png 绝对路径
```

## 指标定义 (单一基准, 单一数字)

每个场景内, **唯一基准** 就是该场景的 `(ISO=100, denoise=0)` 视差图 ——
也就是 "干净 ISO 不去噪" 那一张。该场景里其它每张视差图 (任意 ISO,
任意 denoise=0/1) 都跟它算 EPE：

```
EPE(scene, iso, k) = mean | disp(scene, iso, k) - disp(scene, 100, 0) | / 100
```

**评估指标** = 把所有场景 × 所有 ISO ∈ [100, 1600] 的 `EPE(scene, iso, denoise=1)`
全 pool 起来算一个均值, 一个浮点数。无 QC 阈值, 越小越好。

无降噪曲线 (`denoise=0`) 仍然画在每个子图上做视觉对照, 但 **不进指标**。
评估指标只看降噪那条曲线相对干净基准的距离。

之前那一版 (按 denoise 分别取 ISO=100 自己的 baseline、只在 ISO=1000
取一个点、加 0.6 的 QC 阈值) 已下线。理由是：
1. 给 denoise=1 单独配一个 ISO=100 denoise=1 的 baseline 等于让降噪
   方法 "自比", 没法揭露降噪在干净 ISO=100 上的扰动。
2. 单点 (ISO=1000) 抗噪声差, 跨次能差 0.1 量级。
3. QC 阈值是个二元 gate, 不利于做 sweep / 横向对比 (要不要把分数显
   示出来变成临界判定)。统一用 "全 ISO 平均, 单基准" 后这些都消失。

只想重跑 eval 不重新去噪 / 跑深度：

```bash
python benchmark/run.py --onnx ... --skip-prepare --skip-depth
```

## ISP 对称性 (denoise=0 / denoise=1 同管线)

`benchmark/run.py::_prepare_one` 现在两条路径**只差一个 `denoiser` 调用**:

```
denoise=0:  raw -> identity                 -> isp_process -> rectify -> output
denoise=1:  raw -> denoiser(bayer_01)       -> isp_process -> rectify -> output
```

数据树里的 vendor `sensor*_isp.png` **不再参与评估**, `_prepare_one` 直接
忽略掉。每帧 `sensor*_raw.png` 一次性产出 `_0.png` 和 `_1.png` 两张图。

为什么这么改：之前 `denoise=0` 用 vendor 预渲染的 `sensor*_isp.png`,
`denoise=1` 用我们的 `isp_process`。两条 ISP 在 demosaic 顺序、LSC、
AWB/CCM、gamma 上都不一样, 同一份 raw 出来的 RGB 都不一样, depth 网络
跑出来视差也就不同。结果 ISO=100 上即使 `denoiser=identity` 仍能看到
**~0.5 像素的"恒定 EPE 偏置"**, 这个偏置跟降噪好坏无关, 但会被算到
评估指标里, 把降噪贡献整个淹没——历史上看到 "降噪后 EPE 变差" 大半是
这个伪信号导致的。

**Sanity check**: 拿任一模型导出 `residual_scale=0` 的 ONNX 跑 benchmark
(等价于 `denoiser=identity`):

```
全局平均  无降噪 0.7805   降噪 0.7805   Δ +0.0000
```

每场景每 ISO 的 Δ 都是 `±0.0000`, 评估指标 = 0.7805 = 这套离线 ISP
本身在我们 depth 网络下的 "无降噪 baseline", 数字越小说明 raw 经过
ISP 后越接近 `(scene, 100, 0)` 那张参考。

实际跑法 (用 `bench_scale.py` 把 α 烘进 ONNX):

```bash
python bench_scale.py result/<run> 0.0 --name verify_alpha0
# stdout 末尾应该看到 全局平均 Δ +0.0000
```

> 注意原本搬过来的 0.4-0.6 历史成绩 (见 `docs/how_to_run_benchmark.md`
> 第 6 节) 用的是 ysstereo 的旧 ISP, 跟当前评估不同管线, **不能直接横
> 比**。重新跑一份新基线再开始评估。

## 与原 ysstereo 的对应关系 / 改动一览

| 原文件 | 本仓库位置 | 关键改动 |
|---|---|---|
| `denoise/isp/__init__.py` | `isp/__init__.py` | 删 `OMP_NUM_THREADS` 之类的全局副作用 |
| `denoise/isp/isp.py` | `isp/isp.py` | 删未被调用的 `tmo()`；矩阵运算改向量化 |
| `denoise/isp/awb.py` | `isp/awb.py` | **删 GUI 调试**（`cv2.namedWindow`/`imshow`/`putText` 等 ~150 行）；模块加载时只解析一次 calib，白点估计纯 numpy |
| `denoise/isp/lsc.py` | `isp/lsc.py` | `cam{2,3}` 增益图启动时建好；删硬编码 `'isp/...'` 相对 cwd 路径 |
| `denoise/isp/results.json` | `isp/assets/awb_calib.json` | 改名 |
| `denoise/isp/cam{2,3}_lsc.png` | `isp/assets/` | 搬过来 |
| `denoise/rectify/core.py` | `rectify/rectify.py` | **map 启动时算一次**（原代码每帧重算 stereoRectify + initUndistortRectifyMap，开销巨大）；删跑这文件就触发的副作用 |
| `denoise/rectify/CalibData.json` | `rectify/assets/calib.json` | 改名 |
| `denoise/nn_denoise/core.py` | (内联进 `run.py::_OnnxDenoiser`) | 只保留 mode 2 路径；FPN / mode 1 / +9/255 偏置全删 |
| `denoise/nn_denoise/fpn{2,3}/` | — | **删**（mode 1 旧路径资源） |
| `denoise/main.py`, `collect_data.py` | — | **删**（GUI 浏览 / 数据采集，跟 benchmark 无关） |
| `denoise/prepare_data_for_nn_depth.py` | `run.py::stage_prepare` | 入参化 + 多线程（原代码 cwd 强依赖） |
| `depth/main.py` | `depth/runner.py` | argparse 改函数签名；其余流水线 / DataLoader / IO 线程池保留 |
| `depth/run.sh` | — | **删**（之前硬编码 `/home/bobiou/...`，函数化后不需要） |
| `depth/ysstereo/` | `depth/ysstereo/` | 整体 vendor，没改内部代码 |
| `depth/ysstereo/iter_150000.pth` | `depth/weights.pth` | 改名 |
| `depth/ysstereo/edifat...joint.py` | `depth/model_config.py` | 改名 |
| `benchmark/generate_plot.py` | `eval/epe.py` | 入参化（原代码默认从 `../depth/output/depth` 读，cwd 强依赖）；返回结构化 `EvalResult` 而不是只 print |
| `benchmark/show_scenes.py` | — | 删（只是数据浏览器） |
| `run.sh` (顶层) | `run.py::run_benchmark` | 单一 Python 入口，可被其它模块直接 `import` |

## 验证

跑一次 3 分钟训练的 `l1tv_lam01_3min` ckpt (新对称-ISP 评估):

```
处理场景数: 7 | 最大ISO: 1600
基准: 每个场景的 (ISO=100, denoise=0)

=== EPE vs ISO=100 无降噪 基准 (ISO 100..1600 平均) ===
场景               无降噪        降噪         Δ(无降噪-降噪)
------------------------------------------------
场景 1         0.4418    0.2611           +0.1807
场景 2         1.3028    0.5967           +0.7062
场景 3         0.4885    0.4922           -0.0037
场景 4         0.4072    0.2904           +0.1169
场景 5         1.6828    0.5373           +1.1455
场景 6         0.7187    0.3763           +0.3424
场景 7         0.4223    0.4106           +0.0116
------------------------------------------------
全局平均          0.7806    0.4235           +0.3571

评估指标 = 0.4235
```

整套 wall-clock ~100s (prepare 75s, depth 12s, eval 1s; prepare 比之前
慢一档因为每帧多走一次 ISP——`denoise=0` 现在也是从 raw 算的)。

要点:
- "无降噪" 列那 0.7806 是这套 ISP 的"绝对下限基线"——`denoiser=identity`
  能拿的分。比之前用 vendor `*_isp.png` 时的 0.4162 高很多, **不是因
  为变差了, 是因为之前那个 0.4162 是个不同 ISP 的分数**。
- 想要分数有意义, 看 **Δ(无降噪-降噪)**: 正值就是降噪真有用; 负值就是
  降噪伤了 depth (常见原因: TV 过强 / 噪声放大 / 模型过激进, 见
  `docs/EXPERIMENTS_JOURNEY.md`)。
- 单次跑分有 ~0.05 量级的方差 (depth 网络对 RGB 微扰敏感), 比方法 A 比
  方法 B 时建议同样 ckpt 重复跑 2-3 次取均值, 别看单点。

输出落盘 (`benchmark/output/` 里):
- `isp/{2,3}/<scene>_<gain>_{0,1}.png` — 各 cam 校正后的 8-bit BGR PNG,
  `_0` 是无降噪经我们 ISP, `_1` 是降噪经我们 ISP, 文件名打平的 scene
  (下划线被去掉) 便于 EPE 模块按 `_<gain>_<denoise>` 解析。
- `depth/<scene>_<gain>_{0,1}.png` — uint16 视差 × 100, EPE 直接读这层。
- `visual/<scene>_<gain>_{0,1}.png` — 视差伪彩 + 输入并列的可视化图,
  人肉看 depth 质量很好用。
- `Summary_Depth_EPE_Final.png` — 7 场景子图 + 全局平均, 每张子图
  画无降噪曲线 (蓝) 和降噪曲线 (红); 标题写最终评估指标。

## 没搬 / 没动的

- `train_data/Data/` 没动（用作 benchmark 输入）。
- `n2n/`、`export_l1tv_onnx.py`、`run_train.py` 没动。
- `result/run_experiment.py` 还在用 `/root/ysstereo/run.sh`，**没改**。
  下次切到 `benchmark/` 入口可以让它脱离 `/root/ysstereo`，但本次任务
  没顺手做。
