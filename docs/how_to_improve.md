# how_to_improve · benchmark 优化任务说明（用户 spec）

1. 优化网络 / 损失函数 / 训练方法 使得双目降噪 benchmark 得分下降到 **0.2 以内**。
2. 单次优化网络训练时间最多为 **10 min**，这样是为了保证公平性，避免某个优化单纯是因为训练时间长而得分高。
3. 如果某个优化方案的 benchmark 指标不好，要多方面考虑原因，也有可能是 loss 震荡发散、学习率低等等，**不要因为结果不好就直接把一整个方案否定掉**，稍微复盘一下。
4. 如果有多块 NVIDIA 卡，**每张卡各跑一个优化训练**，提升训练速度。
5. 训练结果放到 `result/` 目录下。
6. `result/` 目录下还有一个 `best/` 文件夹，记录当前得分最高的方案，同时把 `benchmark/output` 放到这个 `best/` 文件夹里，方便随时看最佳方案的降噪图片效果以及曲线。
7. 其他的结果单也放到 `result/` 目录下，比如 `result/method1`（名字你来定，有区分度），里面**不需要存放 `benchmark/output`**，但是需要存放训练好的网络参数以及 `Summary_Depth_EPE_Final.png`，以及这次优化方案的介绍。
8. 如果没有达到 0.2，就**继续优化**。
9. `result/` 里面还要搞一个 doc 叫 `experience.txt`，记录经验总结简介，包含试过哪些方法、效果怎么样、benchmark 是多少、TODO 等等。  
   *(注：本轮 agent 已把 `experience.txt` 整合进 `docs/EXPERIMENTS_JOURNEY.md` 阶段 10。下一轮可以恢复 `experience.txt` 也可以继续往 EXPERIMENTS_JOURNEY 阶段 10 追写，二选一。)*
10. 如果我中间打断你的训练，**下次优化接着上次的进行**。
11. 如果重启了 agent，也需要你看一下经验总结，**接着经验总结进行训练**。
