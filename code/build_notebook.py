from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def code_cell(source: str, output: str, count: int) -> dict:
    outputs = []
    if output:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output.splitlines(keepends=True)})
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def execute(source: str, namespace: dict) -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(source, "代码实现.ipynb", "exec"), namespace)
    except Exception:
        buffer.write(traceback.format_exc())
        raise
    return buffer.getvalue()


def main() -> None:
    namespace: dict = {}
    sources = [
        """from code.solution import (\n    PARAMS, OUTPUT_DIR, build_run_summary, configure_plotting, load_and_preprocess,\n    run_problem1, run_problem2, run_problem3, write_json, write_results_report\n)\nimport time\nstart_time = time.perf_counter()\nfont_name = configure_plotting()\nprint(f\"绘图字体：{font_name}\")\n""",
        """df, data_quality = load_and_preprocess()\nprint(data_quality)\n""",
        """res1 = run_problem1(df, PARAMS)\nprint(res1[\"endpoint_performance\"].round(4).to_string(index=False))\nprint(res1[\"screen\"].round(4).to_string(index=False))\nprint(res1[\"constitution\"].round(4).to_string(index=False))\nprint(res1[\"constitution_sensitivity\"].round(4).to_string(index=False))\n""",
        """res2 = run_problem2(df, PARAMS)\nprint(res2[\"candidate_models\"].round(4).to_string(index=False))\nprint(res2[\"test_tiers\"].round(4).to_string(index=False))\nprint(res2[\"profile\"].round(4).to_string(index=False))\nprint(res2[\"combinations\"].head(10).round(4).to_string(index=False))\n""",
        """res3 = run_problem3(df, PARAMS)\nprint(res3[\"summary\"][res3[\"summary\"][\"样本ID\"].isin([1,2,3])].round(4).to_string(index=False))\nprint(res3[\"plans_123\"].round(4).to_string(index=False))\nprint(res3[\"matching\"].round(4).to_string(index=False))\n""",
        """elapsed_seconds = time.perf_counter() - start_time\nwrite_results_report(data_quality, res1, res2, res3, elapsed_seconds)\nrun_summary = build_run_summary(font_name, data_quality, res1, res2, res3, elapsed_seconds)\nwrite_json(OUTPUT_DIR / \"run_summary.json\", run_summary)\nprint(f\"完整流程通过，耗时 {elapsed_seconds:.1f} 秒。\")\nprint(\"所有 CSV/JSON 已保存到 code/outputs，PDF 图已保存到 figures。\")\n""",
    ]
    cells = [
        markdown_cell(
            "# 中老年人群高血脂症风险预警及干预优化\n\n"
            "本 Notebook 从原始附件直接读取数据，依次完成问题一、问题二和问题三。"
            "所有患者参数、阈值、约束检查和论文引用数值均可追溯到 `code/outputs/`。\n"
        )
    ]
    for count, source in enumerate(sources, 1):
        output = execute(source, namespace)
        cells.append(code_cell(source, output, count))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "代码实现.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已覆盖并写入执行结果：代码实现.ipynb")


if __name__ == "__main__":
    main()
