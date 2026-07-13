from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "code" / "outputs"


def main() -> None:
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["data_quality"]["rows"] == 1000
    assert summary["data_quality"]["duplicate_sample_ids"] == 0
    assert summary["data_quality"]["diagnostic_rule_agreement"] == 1.0
    assert summary["question1_top_constitution"]["体质类型"] == "气虚质积分"

    tiers = pd.read_csv(OUTPUT / "question2_risk_tiers.csv")
    test_tiers = tiers[tiers["数据集"].eq("独立测试集")].sort_values("风险等级编码")
    assert test_tiers["人数"].gt(0).all()
    assert test_tiers["实际患病率"].is_monotonic_increasing
    predictions = pd.read_csv(OUTPUT / "question2_test_predictions.csv")
    assert predictions["预测概率"].between(0, 1).all()
    assert set(predictions["风险等级"]) == {"低危", "中危", "高危"}

    all_patients = pd.read_csv(OUTPUT / "question3_all_phlegm_patients.csv")
    assert len(all_patients) == 278
    assert all_patients["六个月总成本"].le(2000).all()
    expected = {
        1: (64.0, 2, 38.0),
        2: (58.0, 1, 40.0),
        3: (59.0, 1, 63.0),
    }
    for sample_id, values in expected.items():
        row = all_patients[all_patients["样本ID"].eq(sample_id)].iloc[0]
        actual = (float(row["初始痰湿积分"]), int(row["年龄组"]), float(row["活动量表总分"]))
        assert actual == values
    plans = pd.read_csv(OUTPUT / "question3_sample_1_2_3_monthly_plans.csv")
    assert plans.groupby("样本ID").size().eq(6).all()
    checks = json.loads((OUTPUT / "question3_constraint_checks.json").read_text(encoding="utf-8"))
    assert checks["all_plans_feasible"] is True

    notebook = json.loads((ROOT / "代码实现.ipynb").read_text(encoding="utf-8"))
    errors = [
        output for cell in notebook["cells"] for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors
    assert len(list((ROOT / "figures").glob("*.pdf"))) == 8
    assert all(path.stat().st_size > 10_000 for path in (ROOT / "figures").glob("*.pdf"))
    report = (ROOT / "reports" / "RESULTS_REPORT.md").read_text(encoding="utf-8")
    assert "TODO" not in report and "占位" not in report
    print("验收通过：数据、三级风险、真实样本、约束、Notebook、报告和 8 张 PDF 均有效。")


if __name__ == "__main__":
    main()

