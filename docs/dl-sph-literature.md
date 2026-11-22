# 用深度学习"学"这套 SPH：文献路线图

> 调研时间 2026-11-20。目的：给"怎么用深度学习学这个算法"找可落地的路线，
> 先看 UTK 李昊宸（Haochen Li）这条线在做什么，再看 surrogate / 可微 / 敏感性三族方法
> 各自能做到什么，最后落到本仓库（WCSPH + Riemann + 镜像 Riemann FSI，RTX 5080）。

## 0. 结论先行

1. **"学这个算法"有四种完全不同的意思**，别混：
   - 学**单步算子**（state → next state），替代 solver → 见 §2.1；
   - 把 solver 变**可微**，用梯度反演参数/做优化 → 见 §2.2；
   - 用**代理的梯度**做敏感性分析/UQ，回答"哪个参数重要" → 见 §2.3；
   - 用**伴随（adjoint）**从目标函数反推梯度，回答"目标对哪个参数/形状最敏感、该往哪调" → 见 §2.4。
     伴随算子在数学上作用在**对偶空间**（dual space），这大概就是"什么空间"的出处。
2. **李昊宸做的是第 1 + 第 3 种，不是第 2 种。** 他的"可微"指的是*神经网络代理*可微
   （automatic differentiation through the network），不是 SPH/CFD 求解器本身可微。
   这一点很容易记混。
3. **单步替代在本仓库不划算**：CUDA fp32 已经 1.64 ms/step（237k 粒子），
   同规模 GNN 一次前向大约 10–30 ms。代理的加速必须来自**时间粗化**（一次跳几十上百步）
   或**多查询场景**（优化、UQ、实时预报），不能来自"每步更快"。

## 1. UTK 李昊宸（Haochen Li）这条线

**身份**：University of Tennessee, Knoxville 土木与环境工程系助理教授，
Water Infrastructure Laboratory（Ψ Lab，<http://psilab.utk.edu/>）负责人。
UF 环境工程博士（2019）+ 机械工程硕士，Hohai 海岸工程本科。
研究标签：urban hydrology、CFD（LES/VOF/PIV）、physics-informed ML、operator learning。

**方法骨架**（三篇代表作串起来就是一条完整链路）：

| 年份 | 工作 | 做了什么 | 关键点 |
|---|---|---|---|
| 2024 | *Operator learning for urban water clarification hydrodynamics and particulate matter transport with physics-informed neural networks*, Water Research 251:121123 | 用 DeepONet 类算子学习 + PINN 学澄清池的水动力与颗粒物输运 | 学的是"输入条件 → 场"的**算子**，不是单步；32 次引用 |
| 2024 | *Unit Operation and Process Modeling with Physics-Informed Machine Learning*, J. Environ. Eng. | 把 PINN 用到单元工艺建模的方法论文章 | 与 Spelman、Sansalone 合作 |
| 2025/2026 | *Operator-Based Machine Learning Framework for Generalizable Prediction of Unsteady Treatment Dynamics in Stormwater Infrastructure*（arXiv:2507.04682；JEE 2026, doi:10.1061/joeedu.eeeng-8600） | CPNN 复合算子网络，预测 hydrodynamic separator 的水动力 + PM 浓度时程 | 水动力 R²>0.8 占 95.2% 测试例；**用代理的 AD 能力直接做敏感性分析**（暴雨负荷 → PM 输运） |
| 2025– | 洪水风险预报系统（AI Tennessee Initiative 资助，GPU/TPU） | 全流域 19,980 km²、2.02 亿自由度、5 m 分辨率的实时洪涝预报 | 目前主要是工程系统 + 新闻，学术论文尚未见到（截至 2026-11） |

**他这条线的三个特征，直接决定我们该怎么抄：**

- **数据来自高保真 CFD，不是实验也不是低阶模型**。代理学的是"贵的东西"，否则没有价值。
- **学算子而不是学一步**：输入是几何/边界/负荷条件，输出是整个时空场；这样一次训练可以泛化到
  没见过的工况，这正是工程设计优化需要的。
- **可微是手段，敏感性是目的**：网络可微 → 对输入求梯度 → 排序哪个工况/参数重要。
  他不需要 solver 可微，因为 solver 只在离线造数据时用一次。

> 对本仓库的启示：如果目标是"设计优化 / 参数排序 / 实时预报"，走"代理 + AD 敏感性"这条线，
> 和他是同一个故事，只是把 Eulerian CFD 换成 Lagrangian SPH——这反而是差异化。

## 2. 三条技术路线与代表作

### 2.1 代理模型 / 算子学习（surrogate）

| 文献 | 一句话 | 对本仓库的可用性 |
|---|---|---|
| **LagrangeBench**（NeurIPS 2023 D&B, arXiv:2309.16342） | 第一个 Lagrangian 粒子基准：7 个 SPH 数据集（含 dam break、Taylor-Green、反 Poiseuille），GNS/SEGNN/EGNN/PaiNN 的 JAX 实现，明确以**时间粗化**为目标 | ★★★ 先用它的 2D dam break 把 pipeline 跑通，再换自家数据，省掉造数据的坑 |
| **Neural SPH**（ICML 2024, arXiv:2402.06275） | 指出 GNN 代理的主要失效模式是**拉伸不稳定导致的粒子聚集**；把 SPH 的压力项/粘性项/外力项作为归纳偏置加进训练与 rollout，rollout 误差降一个量级以上 | ★★★ 这就是"用深度学习学这个算法"最正统的做法：不是黑箱替代，而是把算法结构喂给网络 |
| **GNS**（Sanchez-Gonzalez et al., ICML 2020） | 图网络学粒子动力学（沙、水、粘液）的开山之作，encode-process-decode + 一步预测 | ★★ 基线 |
| **Hybrid ISPH-GNN**（JCP 2025, doi:10.1016/j.jcp.2025.114277） | ISPH 与 GNN 混合，只用 wave-only 数据训练，做剧烈波浪-结构相互作用 | ★★★ 和本仓库的溃坝-结构碰撞最接近的公开工作，值到精读 |
| **PINO for 2D SWE**（2025/2026, ESSOAr/EGU） | 物理信息神经算子解二维浅水方程，水深相对误差 0.3%–10.9%，推理比数值求解器快 **4 个数量级** | ★★ 洪水预报场景的直接竞品/参照 |
| **CPNN**（Li 组，2025/2026） | 复合算子网络 + AD 敏感性 | ★★★ 方法论模板 |
| **Learning mesh-free discrete differential operators**（arXiv:2603.24641, 2026-03） | 用 GNN + 多项式矩约束**直接学 SPH 的离散微分算子**（局部 stencil → 算子权重），精度超过 SPH，且与分辨率无关 | ★★★ 最新、最"学算法本身"的一篇：学的是核函数/梯度算子，不是轨迹 |

### 2.2 可微 SPH（differentiable）

| 文献 | 一句话 | 备注 |
|---|---|---|
| **JAX-SPH**（arXiv:2403.04750, ICLR 2024 workshop） | JAX 写的可微 SPH，验证了梯度正确性，演示**反问题**（反演粘性）与 **solver-in-the-loop** | ★★★ 上手最快，代码 <https://github.com/tumaer/jax-sph> |
| **diffSPH**（arXiv:2507.21684；JCP 2026） | 纯 PyTorch + GPU，可压/弱可压/不可压三套格式，粒子平移、初值/参数反演、形状优化、solver-in-the-loop，梯度可穿过数百步 | ★★★ 与本仓库的 WCSPH 最接近，PyTorch 生态也最适合这台 Windows 机器 |
| **CraftSPH**（arXiv:2609.38208, 2026-09） | PyTorch 可组合框架，显式/隐式统一，AD 用于参数估计；强调"计算图里放哪些算子会改变梯度" | ★★ 新，工程化程度高 |
| **3DGS-SPH**（arXiv:2610.04336, 2026-10） | 3D Gaussian Splatting 与可微 SPH 耦合，从视频反演本构/边界参数 | ★ 思路有趣，离本仓库较远 |

**关键区别**：JAX-SPH/diffSPH 是"solver 可微"（梯度穿过时间步进）；
李昊宸的"可微"是"网络可微"（梯度穿过网络）。前者能做参数反演与最优控制，
后者只能做敏感性排序与代理优化。两条都要的话，工作量是两倍。

### 2.3 敏感性分析 / 不确定性量化（UQ）

| 文献 | 一句话 | 备注 |
|---|---|---|
| **Investigating the Effects of SPH Numerical Parameters for Dam-Break Flood Prediction**（Mathematics 2026, 14(15):2718, doi:10.3390/math14152718） | DualSPHysics + LiDAR 地形复现 Cleveland Dam，用 **Sobol 方差分解**量化时间积分格式、光滑长度、核函数、粒子间距、人工粘性对 WSE 的影响 | ★★★ 和本仓库的溃坝案例几乎同构，方法可直接照搬；结论：时间积分格式影响最大（0.87%/0.63%），α=0.2–0.3 几乎无影响，粒子间距在 298,188 之后收益饱和 |
| **S-SPH**（arXiv:2605.12540, 2026-05） | 多项式混沌 + KL 展开做随机 SPH，比 Monte Carlo 便宜 3 个数量级 | ★★ 如果要做"参数不确定 → 结果分布" |
| **CPNN 的 AD 敏感性**（Li 组） | 训练好的代理一次反传就得到 ∂输出/∂输入，不用重跑 | ★★★ 与 Sobol 互补：Sobol 定"谁重要"，AD 定"往哪调" |

### 2.4 伴随方法（adjoint）

这是"敏感性分析"的第三条路，也是 §2.2 里反向 AD 的数学本质。伴随算子把向量映到对偶空间，
老文献里因此有"伴随空间/对偶空间"的说法——记忆里那个"什么空间"多半来自这里。

| 方法 | 成本 | 给什么 | 什么时候用 |
|---|---|---|---|
| 有限差分 | 2D+2 次正向求解（D=参数个数） | 局部梯度 | D 很小 |
| **伴随** | **1 次正向 + 1 次反向 ≈ 2 次求解，与 D 无关** | 局部梯度 | D 大、只要一个工作点 |
| Sobol | N(D+2) 次采样 | 全局方差分解 | D≤10、要整个分布 |
| 反向 AD | ≈ 伴随，但自动实现 | 同上 | 有可微实现时首选 |

- **离散伴随**（discretize-then-optimize）：先有代码，再对代码反向求导。JAX/PyTorch 的 `grad`
  就是它——只要离散格式一致，梯度和手写伴随逐位一致。**本仓库推荐这条。**
- **连续伴随**（optimize-then-discretize）：先推伴随 PDE 再离散，网格无关；但 SPH 的 Riemann
  通量、核截断、自由面项推起来极痛苦，不建议。

**代表工作**
- diffSPH 的标题就是 *for Adjoint Optimization and Machine Learning*（arXiv:2507.21684）；
- JAX-SPH 用梯度做反问题与 solver-in-the-loop（arXiv:2403.04750）；
- [End-to-End Differentiable SPH for Fluid–Rigid Optimization](https://doi.org/10.1145/3860999)
  （ACM TOG 2025）——流体-刚体优化的现成模板，和本仓库的镜像 Riemann FSI 最接近；
- SPH 动弹性 + 拓扑优化的伴随敏度（JASCOME，日文）——固体侧的先行者；
- 经典综述：Giles & Pierce (2000) *An introduction to the adjoint approach to design*；
  Plessix (2006) *A review of the adjoint-state method*——后者就是地球物理里的 4D-Var，
  和文献同源。

**落到本仓库要过的四道坎**

1. **内存**：反向传播要存正向轨迹。237k 粒子 × ~10 个 float ≈ 9.5 MB/步，3000 步 ≈ 28 GB；
   1.2M 粒子 × 30,600 步 ≈ 1.5 TB。必须做 checkpoint（binomial / revolve），
   内存降到 O(√N)、计算变 2–3×。
2. **邻居表**：反向时冻结正向的邻居图（拓扑不可微，也不该微）。正向存 edge list，
   或用固定半径图——否则每次重算邻居会让梯度抖成噪声。
3. **不可微点**：Riemann 波速分支、核截断、自由面检测、接触摩擦、本构里的 max/min。
   要么平滑（softplus / 熵正则），要么固定分支（straight-through）。
4. **梯度检验**：先做 Taylor test——`(J(θ+εδ) − J(θ−εδ)) / 2ε` 与伴随梯度内积之比
   应随 ε 以 O(ε²) 收敛。不过这一关，后面的优化全是假的。

## 3. 落到本仓库：什么值得做，什么不值得

### 3.1 先算一笔账（为什么不能指望"单步更快"）

- 现状：237k 粒子 fp32 1.64 ms/step；1.2M 粒子 4.65 ms/step。
- 同规模 GNN（k≈30 邻居、4 层消息传递、64 维隐层）单次前向 ≈ 10–30 ms，
  比 solver 慢一个量级。**逐位精度也不可能达到**（fp32 solver 是确定性位级可复现的）。
- 所以代理的收益只能来自：
  1. **时间粗化**：一次预测 Δt = 50–500 步（LagrangeBench 的核心设定）；
  2. **多查询**：优化/UQ 要 10³–10⁶ 次评估，而 solver 一次就是分钟级；
  3. **梯度**：AD 一次反传 vs 有限差分 2D+2 次求解。

### 3.2 三个可做的实验（按风险从低到高）

**S1｜Sobol 敏感性分析（不训练任何网络，1–2 周，风险最低）**
- 参数：α（人工粘性）、c0（声速）、h/dp（光滑长度/粒子间距）、δ（δ-SPH 系数）、
  时间积分格式、固体 E/ν 与 Kelvin-Voigt 阻尼、耦合系数。
- 输出：WSE、峰值压力、固体位移/应力、漏水深度（NEXT_SESSION 里那个 z=-0.42 的问题）。
- 采样：Sobol N=256 → 6 参数约 3,584 次运行 × 5 s ≈ 5 GPU 小时（关掉结果输出）。
- 产出：一张"谁重要"的排序表 + 一套推荐参数。**这件事本身就能写一篇短文**，
  而且它告诉你后面代理该学什么、不该学什么。

**S2｜伴随小模型（2–4 周，核心）**
- 用 diffSPH 或 JAX-SPH 重写一个 2D 溃坝 + 弹性板的小算例（5k–20k 粒子），
  与本仓库 1.cpp 对齐（同一套 Riemann 通量、镜像 FSI）。
- 目标函数 J（挑一个标量）：板上峰值压力 / 某测点末时刻 WSE / 固体最大位移 / 漏水体积。
- 参数 θ：α、c0、δ、h/dp、固体 E/ν、Kelvin-Voigt 阻尼、闸门开启时刻。
- 流程：① Taylor test 验梯度（O(ε²)）→ ② 一次反传拿全部 ∂J/∂θ → ③ 排序并与 S1 的
  Sobol 指数对照（不一致就说明强非线性或代码有 bug）→ ④ 用它做标定（最小化
  J_sim − J_exp）。
- 量级：2 万粒子 × 3000 步在前向是分钟级，伴随 2–3× 仍是分钟级；1.2M × 30,600 步
  要 checkpoint，小时级。
- 风险：Windows 上 JAX GPU 需要 WSL；PyTorch（diffSPH 路线）原生可用。

**S3｜代理模型（1–3 个月）**
- 先在 LagrangeBench 的 2D dam break 上跑通 GNS/Neural SPH，确认 rollout 长度与稳定性；
- 再换成自家数据：需要改 `write_result` 输出 v/rho/itype（现在只有 x,y,z,p 四列），
  且存帧间隔要降到 1（`SPH_OUTCADENCE=1`）——数据量按 9.5 MB/帧算，1 万帧约 95 GB；
  建议先用 2D/小规模算例，别拿 1.2M 粒子开刀；
- 训练目标建议用**残差**：网络只学 solver 一步的修正量（Δt 内的增量），
  而不是绝对状态——这是 Neural SPH 那类工作能稳定 rollout 的关键。

### 3.3 一条更"论文友好"的组合拳

把四件事串成一条线，正好对得上李昊宸的叙事，但用 Lagrangian 方法做出差异：

> GPU SPH 造数据 → 可微/伴随标定物理参数 → GNN/算子代理（时间粗化）→ 代理 AD 求敏感性
> → 设计优化 / 实时预报

其中 S1（Sobol）负责"全局谁重要、给代理定输入维度"，S2（伴随）负责"局部梯度、标定物理参数"，
S3（代理）负责"多查询加速"。三者共享同一套算例与同一套评价指标（WSE/压力/位移/漏水），
不会做成三篇互不相干的文章。

## 4. 参考链接

**李昊宸 / Ψ Lab**
- 主页 <https://cee.utk.edu/people/haochen-li/>；实验室 <http://psilab.utk.edu/>
- Water Research 2024: <https://doi.org/10.1016/j.watres.2024.121123>
- CPNN: <https://arxiv.org/abs/2507.04682>；JEE 2026: <https://doi.org/10.1061/joeedu.eeeng-8600>
- 洪水预报系统报道: <https://engineer.utk.edu/li-develops-revolutionary-flood-hazard-forecasting-system/>

**代理 / 算子学习**
- LagrangeBench: <https://arxiv.org/abs/2309.16342>，代码 <https://github.com/tumaer/lagrangebench>，文档 <https://lagrangebench.readthedocs.io/>
- Neural SPH: <https://arxiv.org/abs/2402.06275>，代码 <https://github.com/tumaer/neuralsph>
- Hybrid ISPH-GNN: <https://doi.org/10.1016/j.jcp.2025.114277>
- 学习无网格微分算子: <https://arxiv.org/abs/2603.24641>
- PINO + 2D SWE: <https://doi.org/10.22541/essoar.176131914.46883575/v1>

**可微 SPH**
- JAX-SPH: <https://arxiv.org/abs/2403.04750>
- diffSPH: <https://arxiv.org/abs/2507.21684>
- CraftSPH: <https://arxiv.org/abs/2609.38208>
- 3DGS-SPH: <https://arxiv.org/abs/2610.04336>

**敏感性 / UQ / 伴随**
- Sobol + DualSPHysics 溃坝: <https://doi.org/10.3390/math14152718>
- S-SPH: <https://arxiv.org/abs/2605.12540>
- End-to-End Differentiable SPH for Fluid–Rigid Optimization（ACM TOG 2025）: <https://doi.org/10.1145/3860999>
- SPH 动弹性 + 拓扑优化伴随敏度（JASCOME）: <https://www.jstage.jst.go.jp/article/jascome/19/0/19_08-191201/_article/-char/en>
- Giles & Pierce, *An introduction to the adjoint approach to design* (2000)
- Plessix, *A review of the adjoint-state method* (2006) — 地球物理 4D-Var 的同一套数学
