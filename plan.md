# 项目执行计划

## 目标

修正原 Notebook 中的样本错配、标签泄漏、训练内重要性偏差、九体质贡献口径、三级风险规则失效与结果不可追溯问题，形成可从原始附件一键复现的三问完整求解链路。

## 项目结构

```text
code/
  solution.py          # 三问统一求解入口
  build_notebook.py    # 生成并执行根目录 Notebook
  outputs/             # 数值结果、中间过程与校验表
figures/               # 论文可引用 PDF 矢量图
reports/
  ANALYSIS_MODELING_REPORT.md
  RESULTS_REPORT.md
预处理结果/             # 清洗后的 CSV/XLSX 与 EDA 图
代码实现.ipynb          # 可复现 Notebook
```

## 执行顺序

1. 从附件 1 读取并校验 1000 行原始数据。
2. 问题一：分终点重复五折筛选；折外置换重要性与FDR；体质标签相对平和质的调整 OR、整体检验与连续积分敏感性。
3. 问题二：非血脂关联筛查模型；题面特征阈值生成唯一三级管理等级；患者级触发规则与痰湿质核心组合。
4. 问题三：对体质标签 5 的全部患者求解；输出 ID 1/2/3 逐月方案；归纳匹配规律。
5. 输出灵敏度、约束回代、CSV/JSON、PNG 与 PDF 图表。
6. 生成执行过的 Notebook 和结果报告，完成 Git 验收。
