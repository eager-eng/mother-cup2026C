# 中老年人群高血脂症风险预警及干预方案优化

本项目从原始附件直接复现三个问题，已修正旧版本中的样本 1/2/3 参数错配、三级风险全员高危、体质 OR 汇总错位和中文图表乱码。

## 运行

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-py312.txt
.\.venv312\Scripts\python.exe code\build_notebook.py
```

运行后将覆盖根目录 `代码实现.ipynb` 和兼容 PNG，并生成：

- `reports/ANALYSIS_MODELING_REPORT.md`
- `reports/RESULTS_REPORT.md`
- `code/outputs/*.csv|json`
- `figures/*.pdf`

## 建模边界

附件的高血脂标签与四项血脂任一异常规则完全一致。项目将四项血脂用于临床诊断覆盖，不把近乎同义的诊断复现性能宣传为未病预测能力；未病预警模型仅使用体质、活动、代谢和基础信息。

