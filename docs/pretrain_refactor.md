# 代码审阅与预训练项目迁移

## 审阅发现

原训练入口同时管理联合预训练和双层训练，还根据阶段选择 `root_pre` / `root_meta`。单一训练配置混用了 SDSD 预训练路径和 SDE 辅助训练路径，且 `is_indoor: true` 与室外 SDSD 路径不一致。独立推理配置又使用另一套数据路径，容易造成训练、推理数据和模型设置错配。

原 README 的 80 轮、学习率 `0.0002`、SSIM 权重 `0.1` 也与用户当前配置中的 50 轮、学习率 `0.00015`、SSIM 权重 `0` 不一致。此次以修改前工作区中的实际预训练配置为准。

## 调整后的结构

```text
train_event_dunet.py             # 完整单阶段联合训练
infer_event_dunet.py             # 按同一配置加载指定数据划分进行推理
configs/
  sde_in.yaml
  sde_out.yaml
  sdsd_in.yaml
  sdsd_out.yaml
adaptpro/
  training.py                   # 单阶段运行环境、训练、评估与保存
  losses.py                     # 原预训练重建目标
  data/                         # SDE / SDSD 路由和各自读取实现
  models/                       # 保留原网络构造、初始化和前向
tests/
docs/
```

四份配置自含全部设置，并分别指定 `data.dataset_id`、数据读取器、`data.root`、输出目录和推理权重。SDSD 的 `is_indoor` 明确区分事件来源：室内为 `low/`，室外为 `normal/`。其余网络和训练设置初始保持一致，不根据数据集名称猜测新的超参数。

移除了双层训练入口和辅助函数、额外数据加载、双层专属损失、参数分组及任务重置接口。独立任务微调脚本及旧的通用训练、微调、推理配置也已移除。

## 配置迁移

| 原设置 | 当前设置 |
|---|---|
| `data.root_pre` | 对应数据集配置中的 `data.root` |
| `data.root_meta`、额外数据划分 | 删除 |
| `train.stages.pretrain_all.epochs` | `train.epochs` |
| `train.stages.pretrain_all.lr` | `train.lr` |
| `train.stages.pretrain_all.lr_scheduler` | `train.lr_scheduler` |
| `train.stages.pretrain_all.min_lr` | `train.min_lr` |
| 双层阶段、专属损失配置 | 删除 |
| 单独的推理 YAML | 同一数据集 YAML 的 `inference` 节 |

原来已有用户修改的训练配置完整备份于 `outputs/pretrain_refactor_backup/event_mini_dunet_train.original.yaml`，便于核对实际设置。该备份不再是训练入口配置；`outputs/` 受项目忽略规则管理，迁移或打包项目时如需保留应单独带上。

## 保持不变的预训练行为

- EvLight 和参考分支的网络层、初始化、参数名称与形状、前向路径。
- 所有模型参数使用同一个 AdamW 优化器，每批一次反向传播。
- 原重建损失的数值定义、LPIPS 输入归一化及可选 FP32 SSIM。
- 原裁剪、参考帧选择、成对增强及 SDSD 的帧截取协议。
- 每轮先测试和保存模型，再更新学习率；最后再测试并保存最终权重。
- `pretrain_all_latest.pt`、`best.pt`、`global_last.pt` 的命名。
- `--resume` 仅初始化权重，不恢复优化器或调度进度。

后续按用户要求，`best.pt` 的选择指标由最高测试 PSNR 改为最高测试集平均 SSIM，最佳分数保存字段由 `best_psnr` 改为 `best_ssim`；分数相同保留较早的一轮。网络、损失、反向传播和学习率调度不受此调整影响，已有模型权重仍可加载。

这些约束用于保持重构前后的预训练行为一致，不代表已经完成四个真实数据集上的训练或获得新的性能结果。运行真实实验前，必须将四份配置的占位 `data.root` 分别替换为实际数据目录；运行方式见 [README](../README.md)。

## 验证结果

在 PyTorch 2.2.0 的 CPU 环境中，35 项测试全部通过。测试覆盖四配置的数据路由、成对增强、联合梯度、损失、学习率、权重加载，以及四类数据的合成样本从训练、保存权重到仅使用测试目录推理的完整流程。另验证了参考分支关闭或切换其他已有融合模式时，推理会跳过不适用的诊断，继续输出预测结果。

此外，以备份的原版源码为对照，在原配置的 `base_chs: 48`、32 事件通道下，用相同随机种子和 `32×32` 合成输入比较了 `reference_affine`、`latent_stats`、`spatial_align` 以及关闭参考分支四种情况。全部模型的初始参数名称、形状与数值相同，前向最大绝对误差为 `0`；两个批次的 AdamW 更新后，参数、梯度、训练与测试日志、学习率也逐值相同。保留的模型方法与损失计算方法还通过了语法树一致性检查。

本机缺少 LPIPS 包，数值对比和合成训练测试关闭了 LPIPS；真实的 LPIPS 特征网络和 CUDA 混合精度未作运行验证。原 LPIPS 构建与计算代码保持一致，`requirements.txt` 已补充该依赖，四份正式配置仍保留 `lpips_weight: 0.5`。本次未执行真实数据的 50 轮训练。

本地对照结果位于 `outputs/pretrain_refactor_backup/comparison.json`，完整测试日志为 `outputs/pretrain_refactor_tests.log`，均在忽略目录中。
