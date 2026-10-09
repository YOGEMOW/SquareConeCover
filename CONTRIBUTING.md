# 协作协议（人类与 AI 通用）

## 0. 30 秒上手

1. 读 `docs/00-theory.md`（理论总纲）；
2. 读 `specs/conventions.md`（坐标、变换、数据格式）；
3. 找任务：`README.md` 路线图、`docs/07-open-questions.md`、`experiments/README.md`；
4. 动手前先开 Issue（引用 C/Q/E 编号），完成后提 PR。

## 1. 编号体系

| 前缀 | 含义 | 位置 |
| --- | --- | --- |
| C | 约束（当前全部为已定项） | docs/01-problem.md |
| Q | 开放问题 | docs/07-open-questions.md |
| E | 实验 | experiments/ |
| M | 里程碑 | README.md |

## 2. 提交规范

- 分支名：`feat/...`、`fix/...`、`docs/...`、`exp/E0001-...`；
- 提交信息：`类型(范围): 摘要`，例如 `docs(theory): add elliptical derivation`；
- 一个 PR 只做一件事；文档与代码尽量分开提交，便于审查。

## 3. 实验要求

- 必须可复现：环境、参数、随机种子、脚本版本、原始输出；
- 使用 `experiments/TEMPLATE.md` 的结构；
- 失败实验同样保留；结论必须回写 `docs/` 并引用编号。

## 4. AI 协作者要求

- 结论必须引用编号（C/Q/E/文档小节），不允许"凭感觉"；
- 不确定时标注"未验证"，禁止编造数值；
- 修改约束 C 必须同步更新 `docs/01-problem.md` 与 `docs/07-open-questions.md`；
- 提交前自查：数值可否复算、单位是否统一（米/毫米）、是否遵守 Y-up 与 T·R·S 约定。

## 5. 评审清单

- [ ] 是否引用相关编号？
- [ ] 数值是否可复现（含参数与随机种子）？
- [ ] 是否触及 C1–C12 中任何一条？
- [ ] 是否更新受影响的文档与实验索引？