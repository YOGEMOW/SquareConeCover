# 文献线索与检索词

> 本项目横跨三条线：面板化 / 表皮合理化、离散微分几何（可展与锥奇异点）、通用约束优化。以下为起步清单，欢迎在 PR 中追加。

## 1. 面板化 / 表皮合理化（最对口）

| 文献 | 一句话 | 与本项目的关系 |
| --- | --- | --- |
| Eigensatz, Kilian, Schiftner, Mitra, Pottmann, Pauly, *Paneling Architectural Freeform Surfaces*, SIGGRAPH 2010 | 平面面板覆盖自由曲面、优化布局与面板复用 | 问题结构几乎一一对应；"面板复用"对应我们的图元族 |
| Liu, Pottmann, Wallner, Yang, Wang, *Geometric Modeling with Conical Meshes and Developable Surfaces*, SIGGRAPH 2006 | 锥面网格与可展曲面建模 | 锥点/锥面的理论基石 |

检索词：paneling freeform surfaces, panelization, panel reuse, architectural geometry.

## 2. 离散微分几何（可展、T-网、锥奇异点）

| 文献 | 一句话 | 关系 |
| --- | --- | --- |
| Bobenko & Suris, *Discrete Differential Geometry: Integrable Structure*, 2008 | 系统介绍 T-网（每面为平行四边形） | 矩形 = 平行四边形 + 直角；对边平行线性约束的来源 |
| Zorich, *Flat Surfaces* (2006)；square-tiled surface / origami | 平曲面角亏与量化理论 | 解释"为什么一般不可能精确密铺"、角度量化的来源 |
| Ben-Chen, Gotsman, Bunin, *Conformal Flattening by Curvature Prescription and Metric Scaling*, CGF 2008 | 曲率预设 + 度量缩放 | 锥奇异点（曲率集中）策略的方法论 |
| Stein 等，可展性度量相关工作（2018 前后） | 度量曲面"可展程度" | 分区判定指标 |

检索词：discrete developable, conical mesh, cone singularities, flat surface, angle defect, curvature prescription.

## 3. 网格规整化与重网格化

| 文献 | 一句话 | 关系 |
| --- | --- | --- |
| Botsch & Kobbelt, *A Remeshing Approach to Multiresolution Modeling*, SGP 2004 | 各向同性重网格化经典 | 参考分割 |
| Surazhsky 等, *Isotropic Surface Remeshing*, SMI 2003 | 各向同性重网格化 | 同上 |
| Jakob, Tarini, Panozzo, Sorkine-Hornung, *Instant Field-Aligned Meshes*, SIGGRAPH Asia 2015 | Instant Meshes 论文 | 快速得到场对齐网格 |
| Bommes 等, *Quad-Mesh Generation and Processing: A Survey*, CGF 2013 | 四边网格综述 | quad-dominant / mixed meshing |
| Pottmann et al. 2007；Zadravec et al. 2010 | PQ 网格平面化；平面四边面化 | 矩形面网格规整化 |

检索词：isotropic remeshing, quad mesh generation, planarization, PQ mesh.

## 4. 区域分割与逐块拟合

| 文献 | 一句话 | 关系 |
| --- | --- | --- |
| Cohen-Steiner, Alliez, Desbrun, *Variational Shape Approximation*, SIGGRAPH 2004 | 区域生长 + 平面代理拟合 | 分区步的直接方法 |
| Julius, Kraevoy, Sheffer, *D-Charts: Quasi-Developable Mesh Segmentation*, Eurographics 2005 | 按可展程度分割 | "每片配一块可展图元" |
| Umeyama 1991 / Kabsch 1976 / Horn 1987 | Procrustes 配准 | 单块拟合的数学核心 |
| Hoppe 等, *Mesh Optimization*, SIGGRAPH 1993 | 距离+弹簧+形状能量 | 历史参照（顶点自由度版本） |

检索词：variational shape approximation, quasi-developable segmentation, Procrustes, shape matching.

## 5. 网格规整化求解器（Stage A 相关）

| 文献 | 一句话 | 关系 |
| --- | --- | --- |
| Bouaziz, Deuss, Schwartzburg, Weise, Pauly, *Shape-Up: Shaping Discrete Geometry with Projections*, SGP 2012 | 局部投影 + 全局求解的通用框架 | 路线 B 的直接理论依据 |
| Sorkine & Alexa, *As-Rigid-As-Possible Surface Modeling*, SGP 2007 | 局部-全局迭代的经典形式 | 同上 |
| Müller et al., *Position Based Dynamics*, 2007 | 约束投影 = 弹簧的视角 | 局部步的直觉来源 |
| Liu, Zhang, Xu, Gotsman, *A Local/Global Approach to Mesh Parameterization*, SGP 2008 | 局部-全局收敛性分析 | 收敛与稳定性参考 |
| 通用文献：预条件共轭梯度 + 稀疏最小二乘 | Gauss-Newton 路线 A 的实现基础 | Blender 无 scipy 时自写 CG |

检索词：Shape-Up, local-global, ARAP, position based dynamics, projection solver, Gauss-Newton conjugate gradient.

## 6. 精确极限（了解即可）

- 曲率量化与 flat surface：只有角亏能被"可用角度量子"整除时才可能无缝隙密铺；
- square-tiled surface / origami 数学：正方形拼接理论；
- 相关词：Veech surface, translation surface, cone angle quantization.

## 7. 待补充

- 建筑实践中"锥体 + 平面板"混合表皮的案例；
- 折纸/纸模型的可展近似；
- 3D 打印曲面分割（悬垂与支撑约束）。