# 中老年人群高血脂症风险预警及干预方案优化

本项目从原始附件直接复现三个问题，已修正样本 1/2/3 参数错配、问题一标签泄漏与训练内重要性偏差、九体质贡献口径错位、问题二概率阈值替代特征规则、三级输出冲突和中文图表乱码。

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

附件的高血脂标签与四项血脂任一异常规则完全一致。四项血脂不会进入高血脂关联筛查模型；非血脂概率只解释为与当前异常的横断面关联分数。最终低、中、高三级管理等级由题面特征阈值规则生成，并为每名患者保留唯一等级与触发原因。附件没有随访结局，项目不宣称能够预测未来发病。
