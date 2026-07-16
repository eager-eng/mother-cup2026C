# 中老年人群高血脂症风险预警及干预方案优化

本提交包包含最终论文、原始题目材料、唯一建模代码、完整结果表、预处理成果和论文图表，可在保持当前目录结构的情况下直接复现。

## 运行环境

- Python 3.12
- 依赖见 `requirements-py312.txt`

## 运行方法

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-py312.txt
.\.venv312\Scripts\python.exe code\中老年人群高血脂症建模完整代码.py
```

完整运行约需 2—3 分钟。程序从 `题目材料/` 中的原始 Excel 读取数据，依次完成问题一、问题二和问题三，并在结束时输出“自动验收通过”。

## 目录说明

- `论文/`：76 页最终论文 PDF。
- `code/`：按公共预处理、问题一、问题二、问题三、结果汇总与验收、主入口排列的完整 Python 代码。
- `题目材料/`：原题 PDF 与附件 Excel。
- `results/`：问题一至三 CSV/JSON 结果及 `预处理结果/` 中的 1000 条完整数据。
- `figures/`：论文用矢量 PDF；`figures/png/` 为同名 300 dpi 预览图。

附件为横断面数据，非血脂模型结果解释为当前高血脂异常的关联筛查分数，不解释为未来发病概率。
