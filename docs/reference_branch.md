# 参考帧特征分支设计

## 设计目标

1. 保留 EvLight 的 SNR 引导图像–事件融合主干。
2. 参考帧只提供通道统计，不传递其空间结构，降低运动和视差造成的鬼影。
3. 通道门控控制参考亮度统计的使用强度，无效参考严格退化到 backbone。
4. 用单一配置开关完成干净消融；零初始化保证可从原 backbone 安全起步。

## 模块

`ReferenceFeatureInjector` 包含四步：

1. **双编码**：正常光参考 RGB 使用元编码器 `frame_encoder`；当前 light-up RGB 使用结构相同但参数独立的 `current_encoder`。构建模型和任务重置时，后者从元编码器复制初始化，随后仍作为 `theta` 独立更新。
2. **潜空间统计调制**：保留当前特征的归一化空间结构和通道标准差，只用参考特征的通道均值替换当前均值，即 `F_style = std_low * norm(F_low) + mean_ref`。这里没有 VAE 随机采样和 KL loss。
3. **通道门控**：由两侧潜特征的均值和标准差预测每通道置信度；`reference_valid=false` 时门控严格为零。
4. **线性残差注入**：拼接当前潜特征与门控调制后的潜特征，通过单个 `1x1 Conv`（逐位置 Linear）投影，并以固定 `residual_scale` 加到 EvLight holistic feature。投影零初始化，因此训练开始时与 baseline 完全一致。

旧的位移对齐融合保留为 `fusion_mode=spatial_align`，仅用于消融；默认
`fusion_mode=latent_stats` 不构建 flow predictor。

## 元表示定义

本仓库只把 `ReferenceFeatureInjector.frame_encoder` 定义为元表示 `omega`。`current_encoder`、统计门控、线性投影、`IllumiinationNet` 和 `ImageEnhanceNet` 全部属于任务网络 `theta`，不会进入 outer/meta optimizer。

在双层阶段，support set 用于固定参考编码器 `omega` 后适配 `theta`，query set 用于固定全部 `theta` 后只更新参考编码器。任务微调阶段保留并冻结已学习的参考特征提取器，只重新初始化和训练 `theta`。

## 与近期工作的关系

- EvLight（CVPR 2024）说明 SNR 引导的区域选择适合在低 SNR 区域使用事件结构，因此本实现保留其主干不动：<https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.html>
- RT-IRE（CVPRW 2024）表明高质量参考特征和残差式细节保留对多曝光恢复有效：<https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Xing_High_Quality_Reference_Feature_for_Two_Stage_Bracketing_Image_Restoration_CVPRW_2024_paper.html>
- ReBaIR（ICCVW 2025）采用“参考对齐—注意力融合—对 backbone 核心特征做残差细化”的通用参考式恢复范式，本实现采用相同的低侵入注入位置：<https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Bernasconi_ReBaIR_Reference-Based_Image_Restoration_ICCVW_2025_paper.pdf>
- LDIP（ICCV 2025）同样先对齐高质量参考特征，再进行像素级融合，支持把额外参考图像接入既有视频恢复模型：<https://openaccess.thecvf.com/content/ICCV2025/papers/Bernasconi_LDIP_Long_Distance_Information_Propagation_for_Video_Super-Resolution_ICCV_2025_paper.pdf>
- BiLIE（2025）强调双路特征直接融合会带入事件噪声与结构断裂，支持使用选择性融合而非无条件拼接：<https://arxiv.org/abs/2506.06120>

## 推荐实验

在相同数据划分、epoch、学习率和 seed 下比较：

1. `enabled=false`：EvLight baseline。
2. `enabled=true, fusion_mode=latent_stats`：参考均值潜空间调制。
3. `enabled=true, fusion_mode=spatial_align, use_alignment=true, residual_scale=1.0`：旧空间对齐融合。
4. 可选：`reference_dropout_probability` 取 `0/0.1/0.3`，验证对参考缺失的鲁棒性。
5. 检查 `meta_parameters()` 与 `recovery_parameters()` 参数 ID 无交集，确保 outer loop 没有更新 EvLight backbone。

除 PSNR/SSIM/LPIPS 外，建议检查运动边缘处的鬼影，并单独统计 `reference_valid=false` 样本，确认完整模型在缺参考时能退化到 baseline。
