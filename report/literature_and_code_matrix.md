# 文献与代码矩阵

> 检索日期：2026-07-15。优先列出原始论文、DOI/arXiv 与作者/论文代码。GitHub 的“未找到”仅表示本次公开检索未定位到，不能证明代码绝对不存在。

| 方法 | 核心机制 | 与本课题的关系 | 公开代码/可复现性 |
|---|---|---|---|
| [DPINN / DKAN, arXiv:2507.08338v2](https://arxiv.org/abs/2507.08338) | KAF + DyT/B-spline DKAN + 网格变换 + 可学习局部 AV | 直接起点；已完成“DKAN + PINN” | 未定位到该 shock-DPINN 的公开官方实现；PDF p.26 写明数据可按请求提供 |
| [PINNs-WE, arXiv:2206.03864](https://arxiv.org/abs/2206.03864) | 方程梯度加权，论文版本还包含 RH 与全局守恒 | 必须比较；与拟议 RH/守恒高度重叠 | [SeemSeam/PINN_WE](https://github.com/SeemSeam/PINN_WE)，含 Burgers、Sod/Lax、强激波和 2D 案例，但仓库较大且 notebook/结果文件混杂 |
| [gwPINNs, TechRxiv 20099957](https://doi.org/10.36227/techrxiv.20099957) | 梯度相关残差权重 + seq2seq | 用户指定基线；验证强形式加权能否达到相似效果 | 未定位到独立清洁仓库；可从 PINNs-WE 代码族复用思路 |
| [Adaptive localized AV-PINN, JCP 2023](https://doi.org/10.1016/j.jcp.2023.112265) | 学习全局或局部人工粘性图 | 用户指定 AV-PINN；也是原 DPINN 的直接前身 | 论文可复现公式清楚；本次未定位官方仓库 |
| [wPINNs, SIAM JNA 2024](https://doi.org/10.1137/22M1522504) | Kruzhkov 熵弱残差的 min-max 优化 | 理论强，但原方法偏标量守恒律、训练复杂 | 本次未定位官方实现 |
| [Coupled Integral PINN, arXiv:2411.11276](https://arxiv.org/abs/2411.11276) | 额外网络拟合积分解，绕开激波处不可导 | 用户所称 CI-PINN 的直接候选 | 本次未定位官方实现 |
| [WF-PINN, Scientific Reports 2025](https://doi.org/10.1038/s41598-025-24427-4) | Burgers 弱形式 + 熵条件；含正/逆问题 | 黏性/陡峭 Burgers 的近期强基线 | 需要按论文重建或联系作者 |
| [WE-PINNs, arXiv:2603.24819](https://arxiv.org/abs/2603.24819) | 动态时空控制体积 + 积分熵条件；报告标量 \(L_1\) 收敛率并测试 Euler | 2026 年最接近拟议弱守恒/熵模块的工作；新方法必须超越而非重复 | 本次未定位公开代码；论文称实现只需标准网络 |
| [WHC-PINN, Scientific Reports 2025](https://doi.org/10.1038/s41598-025-34263-1) | 梯度加权 + 硬约束，测试 Burgers/Euler | 近期强形式加权基线 | 代码状态待核验 |
| [Robust data-free PINN for shocks, Computers & Fluids 2026](https://doi.org/10.1016/j.compfluid.2026.106975) | 可消失 AV、无数据压缩流 PINN | 最新工程基线，尤其适合强激波/压缩流 | 代码状态待核验 |
| [Approximate Riemann solver PINN, Physics of Fluids 2025](https://doi.org/10.1063/5.0285282) | 在 PINN 中引入近似 Riemann 求解器思想 | 可作为“神经通量/Riemann”扩展基线 | 代码状态待核验 |
| WENO5-Z + SSP-RK3 | 高分辨率有限差分/有限体积参考 | 参考解、激波宽度与传统成本基准 | 可用 [Clawpack/PyClaw](https://github.com/clawpack/pyclaw) 交叉验证，自研实现需做网格收敛 |
| KAN 基础实现 | B-spline KAN | 构建 DKAN 层的底座 | [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) 为原始实现；[Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) 便于高效 PyTorch 实现 |

## GitHub 核查备注

- GitHub 用户 [leigq](https://github.com/leigq) 的公开资料与论文第一作者 Guoqiang Lei、北京理工大学信息相符。
- 其 [phase-field-DPINN](https://github.com/leigq/phase-field-DPINN) 是 2026 年更新的相场项目，不是输入 shock-DPINN 论文的实现。
- 该仓库若干文件名含 `DKAN`，但抽查代码主体为 Fourier 嵌入的 PirateNet/DynamicTanh，并未实现论文图 3 所示的 DyT + B-spline DKAN。因此只能作为相邻工作线索，不能用于复现输入论文。
- PINNs-WE 仓库的 canonical 名称当前重定向为 [SeemSeam/PINN_WE](https://github.com/SeemSeam/PINN_WE)；其目录包含 `PINNsrc/PINNs.py` 以及 Burgers、Sod、Lax、blast、2D Euler 等案例。

## 新颖性边界

以下单项已经有先例，不能单独声称为新：

- DKAN 代替 MLP；
- 局部/可学习人工粘性；
- 梯度加权；
- RH 条件；
- 全局守恒；
- 弱/积分形式；
- 熵约束。

较可信的新颖性应来自**结构化耦合**：用可学习激波流形把 DKAN 的间断基与分布意义的 RH/熵条件一一对应，再用混合强/控制体积损失在平滑区和间断区分别施加合适物理。这一主张仍需进一步做精确相似工作检索和实验验证。

