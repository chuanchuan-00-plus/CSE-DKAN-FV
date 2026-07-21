# 1D DKAN 激波研究 goal 完成审计

日期：2026-07-16

## 目标判定

当前 1D MVP goal 已完成，但结论发生了必要的模型边界调整：直接、数据无关的 CSE-DKAN-PINN 没有通过局部守恒/熵工程门；达到预注册 20% 目标的是监督式、物理门控、硬守恒的 DKAN–有限体积子单元混合模型。

## 完成项

- 无黏/黏性 1D Burgers：4096 WENO 参考，高黏度双分辨率收敛审计 20/20 通过。
- 1D Euler Sod/Riemann：精确理想气体 Riemann 解与 HLLC-MUSCL 有限体积骨架。
- PINN 基线：普通 MLP、梯度均衡 GW、人工粘性 AV、控制体积积分 CI，以及原始强形式 DKAN-PINN。
- 公平性：同方程 MLP 系列参数量完全一致；DKAN 与 MLP 参数量差异小于 3%；冻结步数、种子与 float64。
- 直接创新路径：显式间断流形、控制体积、Rankine–Hugoniot、熵与轨迹约束均已实现和测试；失败证据保留。
- 最终混合结构：零均值 DKAN 子单元修正、局部凸界、正性、TV 投影、跳跃传感器、解析黏性厚度门、Euler 信任域和分辨率回退。
- 最终确认：10 个训练种子，Burgers 600 对、Euler 300 对；跨方程中位增益 1.4259，bootstrap 95% CI [1.3952, 1.4523]。
- 关键消融：间断基、空间跳跃门、黏性厚度门、TV 投影、Euler 信任域和分辨率回退。
- 统一指标：状态 L1/L2/Linf、激波位置/宽度、质量/动量/能量、控制体积、RH、熵、正性、TV、极值、训练与在线成本。
- 软件验证：48 项单元/回归测试通过。

## 核心证据

- 最终协议：`protocols/final_1d_float64_v3.json`
- 最终汇总：`results/final_float64_v3/gate_summary.json`
- 最终状态：`report/final_float64_v3_status_2026-07-16.md`
- MLP/GW/AV/CI：`results/canonical_pinn_baselines_v1/summary.json`
- 原始 DKAN-PINN：`results/canonical_dkan_pinn_extension_v1/summary.json`
- 消融：`results/final_hybrid_ablations_v1/summary.json`
- 对照报告：`report/canonical_pinn_baselines_v1.md`
- 消融报告：`report/final_hybrid_ablations_v1.md`

## 保留的失败与限制

- `final_1d_float64_v1` 与 v2 均保留为失败协议；没有并入 v3 成功统计。
- 直接 CSE-DKAN-PINN 不允许声称达到 20%。
- Euler 单分支中位增益为 1.0893，不允许声称 Euler 单独提升 20%。
- 2D Riemann、双马赫反射和前向台阶未实现；这些是后续扩展，不属于当前 1D MVP 已验证范围。
- 论文级图片尚未在本 goal 中生成；所有原始 JSON 指标已具备制图条件。
- 当前目录不是 Git 仓库，成果没有提交或推送到 GitHub。
