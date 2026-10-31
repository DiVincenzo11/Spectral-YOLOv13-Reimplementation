# 论文对应与重建选择

来源：Yang et al., Sensors 2026, 26(10), 3265，DOI: https://doi.org/10.3390/s26103265。
以用户提供的36页正式PDF为依据。页码指PDF页码。本仓库不是找回的原始代码；没有原始权重或 CR-Mix++。

## 已实现的论文方法

1. **YOLOv13-L backbone，p.5 §3.1**：直接使用 iMoonLab/yolov13 官方模块，固定提交 `73289949533efac82bb5f72ec19b746618656bd2`。保留前9个 backbone 层的 DSC3k2、A2C2f 等真实结构，输出第4、6、8层的 P3/P4/P5。不是用普通CNN冒充YOLOv13。
2. **WIO-Neck，pp.6-7 §3.2，式1-5**：Db4 两级DWT，LL/LH增强、HL/HH抑制，独立1x1卷积和可学习强度，逆变换后执行上采样、拼接和卷积融合。默认按文字P3-P7设置5层；P6/P7由P5逐级stride-2卷积获得。
3. **CP-Head，pp.7-8 §3.3，式6-9**：每尺度线性-ReLU-线性投影；归一化嵌入与类别原型；按前景特征类别均值初始化；momentum更新；CIoU+CE+SupCon；最近原型分类。原型保存为buffer，不由梯度和EMA同时重复更新。
4. **Bio-Mamba，pp.9-11 §3.4**：1x1压缩，保留空间格点的时序流，Mamba-1因果深度卷积、输入依赖delta/B/C、selective scan和残差块，分别输出1/2/4周健康热图。提供真正的selective SSM，不是以GRU替代。纯PyTorch路径便于Windows/CPU验证；可配置Linux/CUDA selective_scan内核。
5. **训练，pp.11-15 §§3.5/3.7/4.1.2**：AdamW，LR=.001，余弦退火，5轮预热，300轮，weight decay=.0005，gradient clip=1；检测batch16、时序batch8可调；CUDA AMP；标准YOLO空间增强、HH系数[.5,2]增强、.1时序dropout；forecast loss权重.5。
6. **评测接口**：类别感知NMS、mAP@.5及mAP@.5:.95、每类AP、时序MAE/RMSE、PCA嵌入可视化。指标均由所提供数据计算，不使用论文数字充当程序输出。

## 论文不一致或未说明的地方

- **主干冲突**：§3.1写YOLOv13-L，§3.5.2写ResNet-50。采用题目及主要方法的YOLOv13-L；不声称恢复了两者同时成立的原实现。
- **尺度冲突**：§3.2写P3-P7，其他叙述接近三尺度检测。默认5层，配置 `pyramid_levels: 3` 可切换三层。官方YOLOv13原始HyperACE neck由论文描述的WIO+标准FPN替换；不是保留原neck后只改名称。
- **小波边界**：论文未给padding、系数排列、两级门控共享方式。采用 `pytorch_wavelets==1.3.0` 的Db4和symmetric边界，各深度独立门控；奇数尺寸先复制补齐并在重建后裁剪。测试验证关闭门控时含1x1、2x2、奇数图的重建近似恒等。periodization在短于Db4滤波器的小特征图上出现重建误差，未使用。
- **门控参数**：初值alpha=.5, gamma=.5, delta=.5, beta=.7。论文只给部分参数敏感性值，没有完整初始化。保持原式可学习标量，不强加未描述的截断。
- **原型维护**：§3.3同时称learnable prototypes又给EMA公式。采用式7的EMA，m=.1为重建假设；缺失类别保持未初始化状态，推理忽略未见类别；训练开始的临时中心为随机单位向量。验证/推理不更新原型。
- **背景分类**：四个健康原型并不能表示检测背景。增加一个训练用背景logit，与四个距离logit一起做CE；背景权重=.1，避免大量负anchor压倒前景。没有把背景当作第五个珊瑚健康类别。
- **置信度**：论文只说明相对边界距离，没有公式。采用 `(d_second-d_first)/max(d_second,eps)` 乘前景概率；分类仍严格取最近有效原型。这不是已经校准的概率，不承诺论文ECE。
- **检测匹配**：使用官方TaskAlignedAssigner，topk=10，alpha=.5,beta=6，属于实现选择。保留官方16-bin回归表示及期望解码，但论文损失未列DFL，默认DFL loss=0；可配置开启。
- **对比学习开销**：最多随机采样512个已匹配前景anchor参与SupCon；忽略没有正样本的anchor，排除自身。该上限论文未给。
- **时序长度冲突**：Fig.4显示四帧，正文写672步；数据又写2帧/天，四周应56帧。支持可变T但不自动伪造672个观测；manifest必须提供真实顺序与时间，未来目标必须在当前帧后7/14/28天。
- **时序模块细节缺失**：未给SSM层数、state size、空间压缩/解码结构。重建采用hidden256、state16、2块、8x8空间池化、各预测期独立latent head和共享卷积解码器。这些均非声称从原代码找回。
- **健康图定义缺失**：采用连续标量[0,1]，0=死亡、1=健康，中间值由数据提供者定义。没有从四类框标签自动伪造真实生态热图。
- **预测损失缺失**：采用MSE+0.01空间总变差，课程在总epoch的1/3、2/3处从1周增至2周、4周。论文只给方向，没有具体公式和分界。
- **梯度冲突**：§3.5.1称联合端到端，§3.5.2称预测分支冻结backbone。默认遵循后者，forecast特征stop-gradient且不更新backbone BatchNorm统计；joint任务仍以检测损失更新共享特征。可将 `detach_backbone: false` 改为真正联合梯度，此为替代实验。
- **Mamba连续性**：实现的是Mamba通常使用的离散化selective SSM，不宣称它无数值离散化误差或能直接证明连续生物机理。输入时间目前用于数据验证，不作为SSM的实际不规则时间步参数。
- **数据协议冲突**：§4.1.1同时出现1500帧、500 clips、其他部分500测试序列等口径；没有样本清单，不能恢复原划分。这里要求显式manifest并拒绝跨划分文件复用及同一观测序列的重叠时间窗。
- **计算资源与性能差异**：论文同时提及A100与双4090、多个速度值。重建版实际参数量/运行速度需重新测量，不人为调整到论文数字。

## 不包含的原始资产

没有CR-Mix++图像、健康框/热图标注、环境传感器元数据、原预训练checkpoint、训练日志及原始数据划分。也没有把论文图中个别样例扩充为8450张图。

环境温度、辐射、流速等可选外部变量的融合结构论文没有定义，本版不包含该输入。没有承诺复现物理浑浊模拟的NTU标定、TensorRT/Jetson部署、论文FPS、跨站点实验及所有可视化结果。

## GitHub身份说明建议

“This repository is a paper-based reimplementation of Spectral-YOLOv13. The original experimental assets are unavailable. Architectural ambiguities and reconstruction choices are documented. Software checks use explicitly synthetic data and do not establish reproduction of the published CR-Mix++ results.”

论文作者贡献按正式论文引用；本次重建代码不能冒充当年实验时的原始代码。上游YOLOv13/Ultralytics AGPL-3.0授权保留。
