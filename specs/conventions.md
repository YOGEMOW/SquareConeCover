# 约定（Conventions）

## 1. 坐标与单位

- 右手坐标系，+Y 朝上（C9）；
- 单位：米；角度内部计算用弧度，文档中出现"°"时表示度；
- 版本与日期：文档不写"最后更新日期"，变更写入 docs/01 §7 与 docs/07。

## 2. 图元默认朝向

| 图元 | 默认朝向 | 局部轴含义 | 名义尺寸 |
| --- | --- | --- | --- |
| square | 水平放置 | X/Z 在平面内，Y 为法向 | 边长 1 m |
| cone | 直立 | Y 为轴，底口圆在 XZ 平面 | 高 1 m，底半径 0.5 m |

## 3. 变换约定

- 顺序：x_world = T + R · diag(sx, sy, sz) · x_local（先缩放、再旋转、最后平移）；
- 旋转用单位四元数 [w, x, y, z]，JSON 中以长度为 4 的数组表示；
- 缩放分量取值于 [0.01, 50]，恒为正；
- 图元局部轴定义不可修改（C6）；方片的 sy 无几何意义（校验时仍要求落在区间内）。

## 4. 文件与命名

| 文件 | 用途 | 规范 |
| --- | --- | --- |
| scene.json | 实例列表 | 通过 specs/scene.schema.json 校验 |
| metrics.json | 指标报告 | 见 docs/05 §2 |
| render_*.png | 渲染图 | 命名含视图（front/side/iso） |
| energy.csv | 迭代能量曲线 | 两列：iteration,energy |

命名规则：编号 + 短横线 + 主题，例如 `E0001-cone-formula-numeric-check.md`；编号不复用、不跳号。

## 5. 术语中英对照

| 中文 | 英文 |
| --- | --- |
| 图元 | primitive |
| 实例 | instance |
| 曲率量子 / 顶角亏 | angular defect |
| 缝带 | gap band |
| 锥点 | cone point / cone singularity |
| 贴合残差 | fitting residual |
| 局部-全局迭代 | local-global iteration |

## 6. 数据校验清单

- scene.json 通过 JSON Schema 校验；
- 缩放三分量均在 [0.01, 50] 内且为正；
- 四元数归一化（‖q‖ = 1 ± 1e-6）；
- 实例 ID 唯一；
- 图元名只能是 square 或 cone。