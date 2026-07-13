# 项目执行计划

## 目标

修正原 Notebook 中的样本错配、三级风险失效、体质 OR 汇总错误与结果不可追溯问题，形成可从原始附件一键复现的三问完整求解链路。

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
2. 问题一：双终点稳定筛选；九体质多变量 Logistic OR、95%CI、P 值与 VIF。
3. 问题二：非血脂多维风险模型；训练集 OOF 阈值；三级风险验证；痰湿质核心组合。
4. 问题三：对体质标签 5 的全部患者求解；输出 ID 1/2/3 逐月方案；归纳匹配规律。
5. 输出灵敏度、约束回代、CSV/JSON、PNG 与 PDF 图表。
6. 生成执行过的 Notebook 和结果报告，完成 Git 验收。

