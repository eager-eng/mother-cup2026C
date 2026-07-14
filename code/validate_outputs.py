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
    assert summary["question1_shared"] == []
    assert summary["question1_selected_phlegm"] == []
    assert summary["question1_selected_lipid"] == ["血尿酸"]
    assert 0 <= summary["question1_constitution_global_p"] <= 1

    screening = pd.read_csv(OUTPUT / "question1_dual_endpoint_screening.csv")
    lipid_rows = screening[screening["终点"].eq("高血脂关联筛查")]
    diagnostic = {"总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C", "高密度脂蛋白_HDL_C"}
    assert set(lipid_rows.loc[lipid_rows["是否诊断泄漏"], "指标名称"]) == diagnostic
    assert not lipid_rows.loc[lipid_rows["是否诊断泄漏"], "是否入选"].any()
    assert not screening.loc[screening["终点"].eq("痰湿严重度"), "是否入选"].any()
    constitution = pd.read_csv(OUTPUT / "question1_constitution_adjusted_or.csv")
    assert len(constitution) == 9
    assert constitution.loc[constitution["体质标签"].eq(1), "调整OR"].iloc[0] == 1.0
    assert constitution["整体LR检验P值"].between(0, 1).all()

    tiers = pd.read_csv(OUTPUT / "question2_risk_tiers.csv")
    test_tiers = tiers[tiers["数据集"].eq("独立测试集")].sort_values("风险等级编码")
    assert test_tiers["人数"].gt(0).all()
    assert test_tiers["实际患病率"].is_monotonic_increasing
    predictions = pd.read_csv(OUTPUT / "question2_test_predictions.csv")
    assert predictions["非血脂筛查概率"].between(0, 1).all()
    assert set(predictions["最终管理等级"]) == {"低危", "中危", "高危"}
    assert predictions["触发规则"].notna().all()
    assert not (
        predictions["诊断状态"].str.contains("异常") & predictions["最终管理等级"].eq("低危")
    ).any()
    leakage_features = {"总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C", "高密度脂蛋白_HDL_C", "任一血脂异常"}
    assert leakage_features.isdisjoint(summary["question2_metrics"]["model_features"])
    assert not (OUTPUT / "question2_threshold_bootstrap.csv").exists()
    assert (OUTPUT / "question2_rule_sensitivity.csv").exists()

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
    assert "双终点稳定筛选得到" not in report
    assert "气虚质积分，调整 OR" not in report
    print("验收通过：泄漏排除、折外筛选、分类体质OR、唯一三级管理规则、真实样本、约束、Notebook、报告和8张PDF均有效。")


if __name__ == "__main__":
    main()
