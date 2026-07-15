"""中老年人群高血脂症风险预警及干预方案优化——三问完整建模代码。

本文件从题目原始 Excel 附件直接读取数据，依次完成问题一、问题二和
问题三，并生成结果表、论文图表、结果报告与自动验收结论。为便于论文
附录复制，所有建模代码均集中在本文件中。
"""

from __future__ import annotations


# ============================================================================
# 第一部分：公共配置与数据预处理
# ============================================================================

import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "题目材料" / "2601(12日晚上18：00）" / "附件1：样例数据.xlsx"
OUTPUT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42

TIZHI_COLS = [
    "平和质积分", "气虚质积分", "阳虚质积分", "阴虚质积分", "痰湿质积分",
    "湿热质积分", "血瘀质积分", "气郁质积分", "特禀质积分",
]

COLUMN_MAPPING = {
    "平和质": "平和质积分",
    "气虚质": "气虚质积分",
    "阳虚质": "阳虚质积分",
    "阴虚质": "阴虚质积分",
    "痰湿质": "痰湿质积分",
    "湿热质": "湿热质积分",
    "血瘀质": "血瘀质积分",
    "气郁质": "气郁质积分",
    "特禀质": "特禀质积分",
    "ADL总分": "ADL评分",
    "IADL总分": "IADL评分",
    "活动量表总分（ADL总分+IADL总分）": "活动量表总分",
    "TC（总胆固醇）": "总胆固醇_TC",
    "TG（甘油三酯）": "甘油三酯_TG",
    "LDL-C（低密度脂蛋白）": "低密度脂蛋白_LDL_C",
    "HDL-C（高密度脂蛋白）": "高密度脂蛋白_HDL_C",
    "空腹血糖": "血糖",
    "高血脂症二分类标签": "高血脂症标签",
}

PARAMS = {
    "q1_cv_repeats": 3,
    "q1_permutation_repeats": 8,
    "q1_selection_frequency": 0.60,
    "q1_constitution_bootstrap_repeats": 500,
    "q2_test_size": 0.30,
    "q2_min_tier_fraction": 0.15,
    "q3_months": 6,
    "q3_weeks_per_month": 4,
    "q3_max_cost": 2000,
    "q3_dp_step": 0.5,
    "q3_target_effect_ratio": 0.90,
}


def ensure_dirs() -> None:
    for directory in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_plotting() -> str:
    """配置可嵌入论文的中文字体；返回实际字体名。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    font_name = "DejaVu Sans"
    for path in candidates:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
                font_name = fm.FontProperties(fname=str(path)).get_name()
                break
            except Exception:
                continue
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 130,
        "savefig.dpi": 300,
    })
    sns.set_theme(style="whitegrid", font=font_name)
    return font_name


def save_figure(fig: plt.Figure, pdf_name: str) -> None:
    """仅保存论文使用的矢量 PDF，避免在项目根目录产生重复 PNG。"""
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / pdf_name, bbox_inches="tight")
    plt.close(fig)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_and_preprocess() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"找不到原始附件：{RAW_PATH}")
    raw = pd.read_excel(RAW_PATH)
    required_raw = [
        "样本ID", "体质标签", "痰湿质", "ADL总分", "IADL总分",
        "活动量表总分（ADL总分+IADL总分）", "TC（总胆固醇）",
        "TG（甘油三酯）", "LDL-C（低密度脂蛋白）",
        "HDL-C（高密度脂蛋白）", "高血脂症二分类标签",
        "年龄组", "性别", "吸烟史", "饮酒史",
    ]
    missing = [c for c in required_raw if c not in raw.columns]
    if missing:
        raise ValueError(f"原始附件缺少字段：{missing}")

    df = raw.rename(columns=COLUMN_MAPPING).copy()
    numeric_cols = TIZHI_COLS + [
        "样本ID", "体质标签", "ADL评分", "IADL评分", "活动量表总分",
        "总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C",
        "高密度脂蛋白_HDL_C", "血糖", "血尿酸", "BMI",
        "高血脂症标签", "年龄组", "性别", "吸烟史", "饮酒史",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_counts = df[numeric_cols].isna().sum()
    if int(missing_counts.sum()) > 0:
        raise ValueError("建模字段存在缺失或非数值：\n" + missing_counts[missing_counts > 0].to_string())

    df["TC异常"] = (df["总胆固醇_TC"] > 6.2).astype(int)
    df["TG异常"] = (df["甘油三酯_TG"] > 1.7).astype(int)
    df["LDL异常"] = (df["低密度脂蛋白_LDL_C"] > 3.1).astype(int)
    df["HDL异常"] = (df["高密度脂蛋白_HDL_C"] < 1.04).astype(int)
    df["任一血脂异常"] = df[["TC异常", "TG异常", "LDL异常", "HDL异常"]].max(axis=1)
    df["尿酸异常"] = np.where(
        df["性别"].eq(1),
        (df["血尿酸"] < 208) | (df["血尿酸"] > 428),
        (df["血尿酸"] < 155) | (df["血尿酸"] > 357),
    ).astype(int)
    df["BMI异常"] = ((df["BMI"] < 18.5) | (df["BMI"] > 23.9)).astype(int)
    df["血糖异常"] = ((df["血糖"] < 3.9) | (df["血糖"] > 6.1)).astype(int)

    quality = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_sample_ids": int(df["样本ID"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "label_prevalence": float(df["高血脂症标签"].mean()),
        "diagnostic_rule_agreement": float((df["高血脂症标签"] == df["任一血脂异常"]).mean()),
        "phlegm_constitution_count": int(df["体质标签"].eq(5).sum()),
        "activity_identity_max_error": float(
            (df["活动量表总分"] - df["ADL评分"] - df["IADL评分"]).abs().max()
        ),
    }
    if quality["duplicate_sample_ids"] != 0 or quality["activity_identity_max_error"] > 1e-9:
        raise ValueError(f"数据一致性校验未通过：{quality}")

    write_json(OUTPUT_DIR / "data_quality.json", quality)

    print(f"数据读取完成：{len(df)} 行，痰湿质患者 {quality['phlegm_constitution_count']} 人")
    print(f"诊断标签与四项血脂异常规则一致率：{quality['diagnostic_rule_agreement']:.1%}")
    return df, quality


# ============================================================================
# 第二部分：问题一——双终点指标筛选与九体质贡献分析
# ============================================================================


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR；NaN 保持为 NaN。"""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[valid] = restored
    return result


def fit_logistic_with_inference(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)
    model.fit(x, y)
    probability = np.clip(model.predict_proba(x)[:, 1], 1e-9, 1 - 1e-9)
    design = np.column_stack([np.ones(len(x)), x])
    weights = probability * (1 - probability)
    covariance = np.linalg.pinv(design.T @ (design * weights[:, None]))
    se = np.sqrt(np.clip(np.diag(covariance)[1:], 0, None))
    coef = model.coef_[0]
    z_stat = coef / np.where(se == 0, np.nan, se)
    p_values = 2 * stats.norm.sf(np.abs(z_stat))
    log_likelihood = float(np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability)))
    return {"coef": coef, "se": se, "p": p_values, "probability": probability, "log_likelihood": log_likelihood}


def pareto_flags(values: np.ndarray) -> np.ndarray:
    """返回二维或多维“越大越优”指标中的非支配解标记。"""
    values = np.asarray(values, dtype=float)
    flags = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominated = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        flags[i] = not bool(dominated.any())
    return flags


def stratified_constitution_rank_bootstrap(
    df: pd.DataFrame,
    y: np.ndarray,
    covariates: list[str],
    repeats: int,
) -> pd.DataFrame:
    """按结局分层重抽样，评估八种偏颇体质调整 OR 排名的不确定性。"""
    rng = np.random.default_rng(RANDOM_STATE + 101)
    strata = [np.flatnonzero(y == value) for value in (0, 1)]
    ranks: list[np.ndarray] = []
    top_counts = np.zeros(8, dtype=int)
    for _ in range(repeats):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        sample = df.iloc[sampled]
        sample_y = y[sampled]
        covariate_z = StandardScaler().fit_transform(sample[covariates])
        dummies = pd.get_dummies(sample["体质标签"].astype(int), dtype=float).reindex(columns=range(2, 10), fill_value=0)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
        model.fit(np.column_stack([dummies.to_numpy(float), covariate_z]), sample_y)
        odds_ratios = np.exp(model.coef_[0][:8])
        rank = stats.rankdata(-odds_ratios, method="average")
        ranks.append(rank)
        top_counts[int(np.argmax(odds_ratios))] += 1
    rank_array = np.vstack(ranks)
    return pd.DataFrame({
        "体质标签": np.arange(2, 10),
        "Bootstrap中位名次": np.median(rank_array, axis=0),
        "Bootstrap名次95%区间下限": np.quantile(rank_array, 0.025, axis=0),
        "Bootstrap名次95%区间上限": np.quantile(rank_array, 0.975, axis=0),
        "位居首位频率": top_counts / repeats,
        "Bootstrap次数": repeats,
    })


def run_problem1(df: pd.DataFrame, params: dict[str, Any] = PARAMS) -> dict[str, Any]:
    all_features = [
        "总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C", "高密度脂蛋白_HDL_C",
        "血糖", "血尿酸", "BMI", "活动量表总分",
    ]
    diagnostic_features = {"总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C", "高密度脂蛋白_HDL_C"}
    non_diagnostic_features = [f for f in all_features if f not in diagnostic_features]
    y_phlegm = df["痰湿质积分"].to_numpy(float)
    y_lipid = df["高血脂症标签"].to_numpy(int)
    n_repeats = int(params["q1_cv_repeats"])
    permutation_repeats = int(params["q1_permutation_repeats"])
    min_frequency = float(params["q1_selection_frequency"])

    phlegm_importance: list[np.ndarray] = []
    phlegm_scores: list[float] = []
    phlegm_cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    for fold, (train, test) in enumerate(phlegm_cv.split(df), start=1):
        model = RandomForestRegressor(
            n_estimators=240, min_samples_leaf=8, random_state=RANDOM_STATE + fold, n_jobs=-1
        )
        model.fit(df.iloc[train][all_features], y_phlegm[train])
        phlegm_scores.append(float(model.score(df.iloc[test][all_features], y_phlegm[test])))
        perm = permutation_importance(
            model, df.iloc[test][all_features], y_phlegm[test], scoring="r2",
            n_repeats=permutation_repeats, random_state=RANDOM_STATE + fold, n_jobs=-1,
        )
        phlegm_importance.append(perm.importances_mean)
    phlegm_importance_array = np.vstack(phlegm_importance)

    lipid_importance: list[np.ndarray] = []
    lipid_scores: list[float] = []
    lipid_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    for fold, (train, test) in enumerate(lipid_cv.split(df, y_lipid), start=1):
        model = RandomForestClassifier(
            n_estimators=240, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE + fold, n_jobs=-1,
        )
        model.fit(df.iloc[train][non_diagnostic_features], y_lipid[train])
        probability = model.predict_proba(df.iloc[test][non_diagnostic_features])[:, 1]
        lipid_scores.append(float(roc_auc_score(y_lipid[test], probability)))
        perm = permutation_importance(
            model, df.iloc[test][non_diagnostic_features], y_lipid[test], scoring="roc_auc",
            n_repeats=permutation_repeats, random_state=RANDOM_STATE + fold, n_jobs=-1,
        )
        lipid_importance.append(perm.importances_mean)
    lipid_importance_array = np.vstack(lipid_importance)

    phlegm_p = np.array([stats.spearmanr(df[f], y_phlegm).pvalue for f in all_features], dtype=float)
    phlegm_q = benjamini_hochberg(phlegm_p)
    lipid_p = np.array([
        stats.mannwhitneyu(df.loc[y_lipid == 1, f], df.loc[y_lipid == 0, f], alternative="two-sided").pvalue
        for f in non_diagnostic_features
    ], dtype=float)
    lipid_q = benjamini_hochberg(lipid_p)

    rows: list[dict[str, Any]] = []
    phlegm_model_valid = float(np.mean(phlegm_scores)) > 0
    for j, feature in enumerate(all_features):
        rho, p_value = stats.spearmanr(df[feature], y_phlegm)
        mean_imp = float(phlegm_importance_array[:, j].mean())
        positive_frequency = float((phlegm_importance_array[:, j] > 0).mean())
        selected = bool(
            phlegm_model_valid and mean_imp > 0 and positive_frequency >= min_frequency
            and phlegm_q[j] < 0.05 and abs(rho) >= 0.10
        )
        reasons = []
        if not phlegm_model_valid:
            reasons.append("折外R²不高于0")
        if positive_frequency < min_frequency or mean_imp <= 0:
            reasons.append("折外重要性不稳定")
        if phlegm_q[j] >= 0.05 or abs(rho) < 0.10:
            reasons.append("相关强度或FDR未达标")
        rows.append({
            "终点": "痰湿严重度", "指标名称": feature, "是否诊断泄漏": False,
            "折外模型指标": "R²", "折外模型性能均值": float(np.mean(phlegm_scores)),
            "折外模型性能标准差": float(np.std(phlegm_scores, ddof=1)),
            "折外置换重要性均值": mean_imp, "折外正重要性频率": positive_frequency,
            "关联统计量": float(rho), "原始P值": float(p_value), "FDR_q值": float(phlegm_q[j]),
            "单指标AUC": np.nan, "是否入选": selected,
            "未入选原因": "" if selected else "；".join(reasons),
        })

    for feature in all_features:
        if feature in diagnostic_features:
            auc_raw = roc_auc_score(y_lipid, df[feature])
            rows.append({
                "终点": "高血脂关联筛查", "指标名称": feature, "是否诊断泄漏": True,
                "折外模型指标": "AUC", "折外模型性能均值": np.nan, "折外模型性能标准差": np.nan,
                "折外置换重要性均值": np.nan, "折外正重要性频率": np.nan,
                "关联统计量": np.nan, "原始P值": np.nan, "FDR_q值": np.nan,
                "单指标AUC": float(max(auc_raw, 1 - auc_raw)), "是否入选": False,
                "未入选原因": "定义高血脂标签的诊断变量，排除以防标签泄漏",
            })
            continue
        j = non_diagnostic_features.index(feature)
        auc_raw = roc_auc_score(y_lipid, df[feature])
        mean_imp = float(lipid_importance_array[:, j].mean())
        positive_frequency = float((lipid_importance_array[:, j] > 0).mean())
        selected = bool(
            np.mean(lipid_scores) >= 0.60 and mean_imp > 0 and positive_frequency >= min_frequency
            and lipid_q[j] < 0.05 and max(auc_raw, 1 - auc_raw) >= 0.60
        )
        reasons = []
        if mean_imp <= 0 or positive_frequency < min_frequency:
            reasons.append("折外重要性不稳定")
        if lipid_q[j] >= 0.05:
            reasons.append("FDR未达标")
        if max(auc_raw, 1 - auc_raw) < 0.60:
            reasons.append("单指标AUC低于0.60")
        rows.append({
            "终点": "高血脂关联筛查", "指标名称": feature, "是否诊断泄漏": False,
            "折外模型指标": "AUC", "折外模型性能均值": float(np.mean(lipid_scores)),
            "折外模型性能标准差": float(np.std(lipid_scores, ddof=1)),
            "折外置换重要性均值": mean_imp, "折外正重要性频率": positive_frequency,
            "关联统计量": float(2 * auc_raw - 1), "原始P值": float(lipid_p[j]), "FDR_q值": float(lipid_q[j]),
            "单指标AUC": float(max(auc_raw, 1 - auc_raw)), "是否入选": selected,
            "未入选原因": "" if selected else "；".join(reasons),
        })

    screen = pd.DataFrame(rows)
    screen["结论层级"] = np.where(screen["是否诊断泄漏"], "泄漏排除", "验证性筛选")
    exploratory_rows: list[dict[str, Any]] = []
    for feature in non_diagnostic_features:
        phlegm_row = screen.loc[(screen["终点"].eq("痰湿严重度")) & screen["指标名称"].eq(feature)].iloc[0]
        lipid_row = screen.loc[(screen["终点"].eq("高血脂关联筛查")) & screen["指标名称"].eq(feature)].iloc[0]
        exploratory_rows.append({
            "指标名称": feature,
            "痰湿折外重要性正向值": max(0.0, float(phlegm_row["折外置换重要性均值"])),
            "高血脂折外重要性正向值": max(0.0, float(lipid_row["折外置换重要性均值"])),
        })
    exploratory_pareto = pd.DataFrame(exploratory_rows)
    for column in ("痰湿折外重要性正向值", "高血脂折外重要性正向值"):
        maximum = float(exploratory_pareto[column].max())
        exploratory_pareto[column.replace("正向值", "标准化得分")] = (
            exploratory_pareto[column] / maximum if maximum > 0 else 0.0
        )
    pareto_value_cols = ["痰湿折外重要性标准化得分", "高血脂折外重要性标准化得分"]
    exploratory_pareto["探索性Pareto候选"] = pareto_flags(exploratory_pareto[pareto_value_cols].to_numpy())
    exploratory_pareto["证据性质"] = np.where(
        exploratory_pareto["探索性Pareto候选"], "探索性候选（不得替代验证性入选）", "被其他候选支配"
    )
    pareto_map = exploratory_pareto.set_index("指标名称")["探索性Pareto候选"]
    screen["探索性Pareto候选"] = screen["指标名称"].map(pareto_map).fillna(False).astype(bool)
    selected_phlegm = screen.loc[(screen["终点"] == "痰湿严重度") & screen["是否入选"], "指标名称"].tolist()
    selected_lipid = screen.loc[(screen["终点"] == "高血脂关联筛查") & screen["是否入选"], "指标名称"].tolist()
    shared = sorted(set(selected_phlegm).intersection(selected_lipid))
    screen.to_csv(OUTPUT_DIR / "question1_dual_endpoint_screening.csv", index=False, encoding="utf-8-sig")
    exploratory_pareto.to_csv(OUTPUT_DIR / "question1_exploratory_pareto.csv", index=False, encoding="utf-8-sig")
    endpoint_performance = pd.DataFrame([
        {"终点": "痰湿严重度", "指标": "折外R²", "均值": np.mean(phlegm_scores), "标准差": np.std(phlegm_scores, ddof=1), "折数": len(phlegm_scores)},
        {"终点": "高血脂关联筛查", "指标": "折外AUC", "均值": np.mean(lipid_scores), "标准差": np.std(lipid_scores, ddof=1), "折数": len(lipid_scores)},
    ])
    endpoint_performance.to_csv(OUTPUT_DIR / "question1_endpoint_performance.csv", index=False, encoding="utf-8-sig")

    # 九种体质主分析：体质标签为分类暴露，平和质（标签1）为参照。
    constitution_names = {1: "平和质", 2: "气虚质", 3: "阳虚质", 4: "阴虚质", 5: "痰湿质", 6: "湿热质", 7: "血瘀质", 8: "气郁质", 9: "特禀质"}
    covariates = ["年龄组", "性别", "吸烟史", "饮酒史", "BMI", "活动量表总分", "血糖", "血尿酸"]
    covariate_z = StandardScaler().fit_transform(df[covariates])
    dummies = pd.get_dummies(df["体质标签"].astype(int), drop_first=True, dtype=float)
    full_x = np.column_stack([dummies.to_numpy(float), covariate_z])
    full_fit = fit_logistic_with_inference(full_x, y_lipid)
    reduced_fit = fit_logistic_with_inference(covariate_z, y_lipid)
    global_lr = 2 * (float(full_fit["log_likelihood"]) - float(reduced_fit["log_likelihood"]))
    global_p = float(stats.chi2.sf(global_lr, df=8))
    contingency = pd.crosstab(df["体质标签"], y_lipid)
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
    label_p = np.asarray(full_fit["p"][:8], dtype=float)
    label_q = benjamini_hochberg(label_p)
    or_rows: list[dict[str, Any]] = []
    for label in range(1, 10):
        subset = df["体质标签"].eq(label)
        if label == 1:
            or_value, lower, upper, p_value, q_value = 1.0, np.nan, np.nan, np.nan, np.nan
        else:
            j = label - 2
            coef = float(full_fit["coef"][j])
            se = float(full_fit["se"][j])
            or_value = math.exp(coef)
            lower, upper = math.exp(coef - 1.96 * se), math.exp(coef + 1.96 * se)
            p_value, q_value = float(label_p[j]), float(label_q[j])
        or_rows.append({
            "体质标签": label, "体质类型": constitution_names[label], "参照组": "平和质",
            "样本数": int(subset.sum()), "观察患病率": float(y_lipid[subset].mean()),
            "调整OR": or_value, "95%CI下限": lower, "95%CI上限": upper,
            "P值": p_value, "FDR_q值": q_value, "整体LR检验P值": global_p,
            "未调整卡方P值": float(chi2_p),
        })
    constitution = pd.DataFrame(or_rows)
    non_reference = constitution["体质标签"].ne(1)
    constitution.loc[non_reference, "OR点估计名次"] = stats.rankdata(
        -constitution.loc[non_reference, "调整OR"].to_numpy(float), method="average"
    )
    bootstrap_repeats = int(params["q1_constitution_bootstrap_repeats"])
    rank_stability = stratified_constitution_rank_bootstrap(
        df, y_lipid, covariates, bootstrap_repeats
    )
    constitution = constitution.merge(rank_stability, on="体质标签", how="left")
    constitution["排名解释"] = np.where(
        constitution["体质标签"].eq(1), "参照组，不参与偏颇体质排序",
        "仅为探索性点估计排序；整体检验、置信区间与FDR优先"
    )
    constitution.to_csv(OUTPUT_DIR / "question1_constitution_adjusted_or.csv", index=False, encoding="utf-8-sig")
    rank_stability = rank_stability.merge(
        constitution[["体质标签", "体质类型", "调整OR", "OR点估计名次"]], on="体质标签", how="left"
    )
    rank_stability.to_csv(OUTPUT_DIR / "question1_constitution_rank_stability.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_p: list[float] = []
    for col in TIZHI_COLS:
        score_z = StandardScaler().fit_transform(df[[col]])
        fit = fit_logistic_with_inference(np.column_stack([score_z, covariate_z]), y_lipid)
        coef, se, p_value = float(fit["coef"][0]), float(fit["se"][0]), float(fit["p"][0])
        sensitivity_p.append(p_value)
        sensitivity_rows.append({
            "体质积分": col, "调整OR_每1SD": math.exp(coef),
            "95%CI下限": math.exp(coef - 1.96 * se), "95%CI上限": math.exp(coef + 1.96 * se),
            "P值": p_value,
        })
    sensitivity_q = benjamini_hochberg(np.array(sensitivity_p))
    for row, q_value in zip(sensitivity_rows, sensitivity_q):
        row["FDR_q值"] = float(q_value)
    constitution_sensitivity = pd.DataFrame(sensitivity_rows)
    constitution_sensitivity.to_csv(OUTPUT_DIR / "question1_constitution_score_sensitivity.csv", index=False, encoding="utf-8-sig")

    # 图1：两个终点的折外置换重要性，诊断血脂不进入高血脂关联模型。
    plot_data = screen[screen["折外置换重要性均值"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
    for ax, endpoint, color in zip(axes, ["痰湿严重度", "高血脂关联筛查"], ["#4C78A8", "#E67E22"]):
        part = plot_data[plot_data["终点"].eq(endpoint)].sort_values("折外置换重要性均值")
        ax.barh(part["指标名称"], part["折外置换重要性均值"], color=color, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("折外置换重要性")
        ax.set_title(endpoint)
    save_figure(fig, "问题1_共识得分排序.pdf")

    heat = screen.pivot(index="指标名称", columns="终点", values="折外正重要性频率").reindex(all_features)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", mask=heat.isna(), ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "问题1_双终点频率热力图.pdf")

    plot_or = constitution[constitution["体质标签"].ne(1)].sort_values("调整OR")
    y_pos = np.arange(len(plot_or))
    center = plot_or["调整OR"].to_numpy()
    lower = center - plot_or["95%CI下限"].to_numpy()
    upper = plot_or["95%CI上限"].to_numpy() - center
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(center, y_pos, xerr=np.vstack([lower, upper]), fmt="o", color="#D62728", ecolor="#555555", capsize=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos, plot_or["体质类型"])
    ax.set_xlabel("调整优势比 OR（相对平和质，95%CI）")
    ax.set_ylabel("")
    save_figure(fig, "问题1_体质贡献度OR.pdf")

    pareto_names = exploratory_pareto.loc[exploratory_pareto["探索性Pareto候选"], "指标名称"].tolist()
    point_top = constitution.loc[constitution["OR点估计名次"].eq(1), "体质类型"].iloc[0]
    print(f"问题1：痰湿终点入选 {selected_phlegm}；非诊断高血脂关联入选 {selected_lipid}；共同指标 {shared}")
    print(f"探索性Pareto候选 {pareto_names}；偏颇体质OR点估计首位 {point_top}（非显著排名）")
    print(f"九体质标签整体LR检验 P={global_p:.3g}，未调整卡方检验 P={chi2_p:.3g}")
    return {
        "screen": screen, "selected": shared, "selected_phlegm": selected_phlegm,
        "selected_lipid": selected_lipid, "shared": shared, "endpoint_performance": endpoint_performance,
        "constitution": constitution, "constitution_sensitivity": constitution_sensitivity,
        "exploratory_pareto": exploratory_pareto, "constitution_rank_stability": rank_stability,
        "constitution_bootstrap_repeats": bootstrap_repeats,
        "constitution_global_p": global_p, "constitution_chi2_p": float(chi2_p),
    }


# ============================================================================
# 第三部分：问题二——非血脂关联筛查、统计三级分层与临床提示
# ============================================================================


def apply_clinical_management_prompts(
    data: pd.DataFrame, phlegm_high: float = 60, phlegm_very_high: float = 80,
    activity_low: float = 40,
) -> pd.DataFrame:
    """将题面示例转为独立临床提示，不参与统计风险等级计算。"""
    lipid_abnormal = data["任一血脂异常"].eq(1)
    phlegm_score = data["痰湿质积分"]
    low_activity = data["活动量表总分"].lt(activity_low)
    metabolic_abnormal = data[["尿酸异常", "BMI异常", "血糖异常"]].max(axis=1).eq(1)
    high_rule_1 = lipid_abnormal & phlegm_score.ge(phlegm_high)
    high_rule_2 = (~lipid_abnormal) & phlegm_score.ge(phlegm_very_high) & low_activity
    high_rule_3 = data["体质标签"].eq(5) & phlegm_score.ge(phlegm_high) & data["尿酸异常"].eq(1)
    high = high_rule_1 | high_rule_2 | high_rule_3
    medium = (~high) & (lipid_abnormal | phlegm_score.ge(phlegm_high) | low_activity | metabolic_abnormal)
    code = np.where(high, 3, np.where(medium, 2, 1))

    triggers: list[str] = []
    for position, idx in enumerate(data.index):
        reasons: list[str] = []
        if bool(high_rule_1.loc[idx]):
            reasons.append(f"血脂异常且痰湿积分≥{phlegm_high:g}")
        if bool(high_rule_2.loc[idx]):
            reasons.append(f"血脂正常、痰湿积分≥{phlegm_very_high:g}且活动总分<{activity_low:g}")
        if bool(high_rule_3.loc[idx]):
            reasons.append(f"痰湿质、痰湿积分≥{phlegm_high:g}且尿酸异常")
        if not reasons and code[position] == 2:
            if bool(lipid_abnormal.loc[idx]):
                reasons.append("血脂异常")
            if bool(phlegm_score.loc[idx] >= phlegm_high):
                reasons.append(f"痰湿积分≥{phlegm_high:g}")
            if bool(low_activity.loc[idx]):
                reasons.append(f"活动总分<{activity_low:g}")
            if bool(metabolic_abnormal.loc[idx]):
                reasons.append("至少一项代谢异常")
        if not reasons:
            reasons.append("未触发题面重点管理条件")
        triggers.append("；".join(reasons))
    return pd.DataFrame({
        "当前诊断状态": np.where(lipid_abnormal, "血脂异常（当前已异常）", "血脂正常"),
        "临床管理提示编码": code,
        "临床管理提示": pd.Series(code).map({1: "常规关注", 2: "重点管理", 3: "优先管理"}).to_numpy(),
        "临床管理依据": triggers,
    }, index=data.index)


def derive_three_tier_cutpoints(
    probability: np.ndarray, y: np.ndarray, min_fraction: float = 0.15,
) -> tuple[tuple[float, float], pd.DataFrame]:
    """仅用训练集折外概率搜索单调且各层样本充足的两个切点。"""
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(probability, kind="mergesort")
    sorted_probability, sorted_y = probability[order], y[order]
    n = len(y)
    min_count = int(math.ceil(min_fraction * n))
    cumulative = np.concatenate([[0], np.cumsum(sorted_y)])
    overall = float(y.mean())
    best: tuple[float, float, int, int, list[float]] | None = None
    for first in range(min_count, n - 2 * min_count + 1):
        if sorted_probability[first - 1] >= sorted_probability[first]:
            continue
        for second in range(first + min_count, n - min_count + 1):
            if sorted_probability[second - 1] >= sorted_probability[second]:
                continue
            counts = [first, second - first, n - second]
            positives = [
                cumulative[first], cumulative[second] - cumulative[first], cumulative[n] - cumulative[second]
            ]
            rates = [float(p / c) for p, c in zip(positives, counts)]
            if rates[0] > rates[1] + 1e-12 or rates[1] > rates[2] + 1e-12:
                continue
            objective = float(sum(c * (rate - overall) ** 2 for c, rate in zip(counts, rates)) / n)
            separation = float(min(rates[1] - rates[0], rates[2] - rates[1]))
            candidate = (objective, separation, first, second, rates)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise RuntimeError("训练集折外概率无法形成满足样本比例和单调性约束的三级切点")
    objective, separation, first, second, rates = best
    low = float((sorted_probability[first - 1] + sorted_probability[first]) / 2)
    high = float((sorted_probability[second - 1] + sorted_probability[second]) / 2)
    details = pd.DataFrame([{
        "阈值来源": "训练集折外概率", "低中切点": low, "中高切点": high,
        "最小层样本比例约束": min_fraction, "层间方差目标值": objective,
        "最小相邻患病率差": separation, "训练样本数": n,
        "训练低层患病率": rates[0], "训练中层患病率": rates[1], "训练高层患病率": rates[2],
        "训练低层人数": first, "训练中层人数": second - first, "训练高层人数": n - second,
    }])
    return (low, high), details


def statistical_tier_codes(probability: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    return np.digitize(np.asarray(probability, dtype=float), thresholds, right=False) + 1


def evaluate_risk_tiers(code: np.ndarray, y: np.ndarray, split: str) -> pd.DataFrame:
    names = {1: "低危", 2: "中危", 3: "高危"}
    code = np.asarray(code, dtype=int)
    rows = []
    for level in (1, 2, 3):
        mask = code == level
        positives = int(y[mask].sum())
        if mask.any():
            rate = float(y[mask].mean())
            ci = stats.beta.ppf([0.025, 0.975], positives + 0.5, int(mask.sum()) - positives + 0.5)
        else:
            rate, ci = np.nan, [np.nan, np.nan]
        rows.append({
            "数据集": split, "风险等级编码": level, "风险等级": names[level],
            "人数": int(mask.sum()), "占比": float(mask.mean()), "实际患病率": rate,
            "患病率95%CI下限": float(ci[0]), "患病率95%CI上限": float(ci[1]),
        })
    return pd.DataFrame(rows)


def cart_path_strings(model: DecisionTreeClassifier, data: pd.DataFrame) -> list[str]:
    tree = model.tree_
    paths: list[str] = []
    matrix = data.to_numpy(float)
    for row in matrix:
        node = 0
        rules: list[str] = []
        while tree.children_left[node] != tree.children_right[node]:
            feature = data.columns[tree.feature[node]]
            threshold = float(tree.threshold[node])
            if row[tree.feature[node]] <= threshold:
                rules.append(f"{feature}≤{threshold:.3f}")
                node = int(tree.children_left[node])
            else:
                rules.append(f"{feature}>{threshold:.3f}")
                node = int(tree.children_right[node])
        paths.append(" 且 ".join(rules) + f" → 叶节点{node}")
    return paths


def extract_cart_leaf_rules(
    model: DecisionTreeClassifier, data: pd.DataFrame, y: np.ndarray,
) -> pd.DataFrame:
    leaf_ids = model.apply(data)
    path_strings = cart_path_strings(model, data)
    predicted = model.predict(data).astype(int)
    rows: list[dict[str, Any]] = []
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        first = int(np.flatnonzero(mask)[0])
        rows.append({
            "叶节点": int(leaf), "叶节点规则": path_strings[first],
            "代理风险等级编码": int(predicted[first]),
            "代理风险等级": {1: "低危", 2: "中危", 3: "高危"}[int(predicted[first])],
            "训练样本数": int(mask.sum()), "训练观察患病率": float(np.asarray(y)[mask].mean()),
        })
    return pd.DataFrame(rows).sort_values("叶节点").reset_index(drop=True)


def enumerate_phlegm_combinations(test_df: pd.DataFrame, statistical_code: np.ndarray) -> pd.DataFrame:
    work = test_df.copy()
    work["高危"] = statistical_code == 3
    conditions = {
        "痰湿积分≥60": work["痰湿质积分"] >= 60,
        "活动总分<40": work["活动量表总分"] < 40,
        "BMI异常": work["BMI异常"].eq(1),
        "血糖异常": work["血糖异常"].eq(1),
        "尿酸异常": work["尿酸异常"].eq(1),
        "年龄≥60岁": work["年龄组"] >= 3,
        "吸烟史": work["吸烟史"].eq(1),
        "饮酒史": work["饮酒史"].eq(1),
    }
    base_phlegm = work["体质标签"].eq(5)
    baseline = float(work.loc[base_phlegm, "高危"].mean())
    high_phlegm_total = int((base_phlegm & work["高危"]).sum())
    rows = []
    for k in (1, 2):
        for combo in itertools.combinations(conditions, k):
            mask = base_phlegm.copy()
            for name in combo:
                mask &= conditions[name]
            if mask.sum() < max(5, math.ceil(0.02 * len(work))):
                continue
            high_count = int(work.loc[mask, "高危"].sum())
            confidence = float(high_count / mask.sum())
            confidence_ci = stats.beta.ppf([0.025, 0.975], high_count + 0.5, int(mask.sum()) - high_count + 0.5)
            high_support = float(high_count / max(1, high_phlegm_total))
            rows.append({
                "核心特征组合": "痰湿体质 + " + " + ".join(combo),
                "覆盖人数": int(mask.sum()),
                "高危人数": high_count,
                "总体支持度": float(mask.mean()),
                "高危组支持度": high_support,
                "高危置信度": confidence,
                "置信度95%CI下限": float(confidence_ci[0]),
                "置信度95%CI上限": float(confidence_ci[1]),
                "提升度": confidence / baseline if baseline > 0 else np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "核心特征组合", "覆盖人数", "高危人数", "总体支持度", "高危组支持度",
            "高危置信度", "置信度95%CI下限", "置信度95%CI上限", "提升度",
        ])
    return pd.DataFrame(rows).sort_values(["提升度", "高危组支持度", "覆盖人数"], ascending=False).reset_index(drop=True)


def run_problem2(df: pd.DataFrame, params: dict[str, Any] = PARAMS) -> dict[str, Any]:
    features = TIZHI_COLS + [
        "活动量表总分", "血糖", "血尿酸", "BMI", "年龄组", "性别", "吸烟史", "饮酒史",
    ]
    X = df[features]
    y = df["高血脂症标签"].astype(int).to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=float(params["q2_test_size"]),
        random_state=RANDOM_STATE, stratify=y,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "加权Logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "随机森林": RandomForestClassifier(
            n_estimators=240, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "浅层CART": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=0.10, max_leaf_nodes=6,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }
    candidate_rows = []
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        candidate_rows.append({"模型": name, "训练集CV_AUC均值": scores.mean(), "训练集CV_AUC标准差": scores.std(ddof=1)})
    candidate_df = pd.DataFrame(candidate_rows).sort_values("训练集CV_AUC均值", ascending=False).reset_index(drop=True)
    best_name = str(candidate_df.iloc[0]["模型"])
    best_base = clone(candidates[best_name])

    # 训练集外层五折、内层三折校准，所得概率仅用于训练内选择三级切点。
    oof_probability = np.empty(len(train_idx), dtype=float)
    outer_cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 11)
    for fold, (inner_train, validation) in enumerate(outer_cv.split(X_train, y_train), start=1):
        inner_cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE + 100 + fold)
        fold_model = CalibratedClassifierCV(clone(best_base), method="sigmoid", cv=inner_cv)
        fold_model.fit(X_train.iloc[inner_train], y_train[inner_train])
        oof_probability[validation] = fold_model.predict_proba(X_train.iloc[validation])[:, 1]
    thresholds, threshold_selection = derive_three_tier_cutpoints(
        oof_probability, y_train, float(params["q2_min_tier_fraction"])
    )
    train_statistical_code = statistical_tier_codes(oof_probability, thresholds)

    calibrated = CalibratedClassifierCV(best_base, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    test_prob = calibrated.predict_proba(X_test)[:, 1]
    test_statistical_code = statistical_tier_codes(test_prob, thresholds)
    all_probability = np.empty(len(df), dtype=float)
    all_probability[train_idx] = oof_probability
    all_probability[test_idx] = test_prob
    all_statistical_code = statistical_tier_codes(all_probability, thresholds)
    test_tiers = evaluate_risk_tiers(test_statistical_code, y_test, "独立测试集")
    full_tiers = evaluate_risk_tiers(all_statistical_code, y, "全体样本（训练折外+独立测试）")

    # CART 只作为统计风险等级的代理解释器，模型参数由训练集交叉验证选择。
    surrogate_rows: list[dict[str, Any]] = []
    surrogate_cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 17)
    for depth in (2, 3, 4):
        for min_leaf in (0.05, 0.10):
            estimator = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=min_leaf, max_leaf_nodes=6,
                class_weight="balanced", random_state=RANDOM_STATE,
            )
            f1_scores = cross_val_score(
                estimator, X_train, train_statistical_code, cv=surrogate_cv, scoring="f1_macro"
            )
            fidelity_scores = cross_val_score(
                estimator, X_train, train_statistical_code, cv=surrogate_cv, scoring="accuracy"
            )
            surrogate_rows.append({
                "最大深度": depth, "最小叶节点比例": min_leaf, "最大叶节点数": 6,
                "CV宏平均F1": float(f1_scores.mean()), "CV宏平均F1标准差": float(f1_scores.std(ddof=1)),
                "CV一致率": float(fidelity_scores.mean()), "CV一致率标准差": float(fidelity_scores.std(ddof=1)),
            })
    surrogate_candidates = pd.DataFrame(surrogate_rows).sort_values(
        ["CV宏平均F1", "CV一致率", "最大深度"], ascending=[False, False, True]
    ).reset_index(drop=True)
    selected_surrogate = surrogate_candidates.iloc[0]
    cart_surrogate = DecisionTreeClassifier(
        max_depth=int(selected_surrogate["最大深度"]),
        min_samples_leaf=float(selected_surrogate["最小叶节点比例"]),
        max_leaf_nodes=6, class_weight="balanced", random_state=RANDOM_STATE,
    )
    cart_surrogate.fit(X_train, train_statistical_code)
    train_cart_code = cart_surrogate.predict(X_train).astype(int)
    test_cart_code = cart_surrogate.predict(X_test).astype(int)
    all_cart_code = cart_surrogate.predict(X).astype(int)
    cart_rules = extract_cart_leaf_rules(cart_surrogate, X_train, y_train)
    cart_paths_all = cart_path_strings(cart_surrogate, X)

    clinical_all = apply_clinical_management_prompts(df)
    clinical_test = clinical_all.iloc[test_idx]

    pred_binary = (test_prob >= 0.5).astype(int)
    metrics = {
        "selected_model": best_name,
        "auc": float(roc_auc_score(y_test, test_prob)),
        "average_precision": float(average_precision_score(y_test, test_prob)),
        "brier": float(brier_score_loss(y_test, test_prob)),
        "recall_at_0.5": float(recall_score(y_test, pred_binary, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_test, pred_binary, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_test, pred_binary, zero_division=0)),
        "test_probability_min": float(test_prob.min()),
        "test_probability_max": float(test_prob.max()),
        "diagnostic_rule_agreement": float((df["高血脂症标签"] == df["任一血脂异常"]).mean()),
        "model_features": features,
        "statistical_tier_cutpoints": list(thresholds),
        "threshold_selection_dataset": "训练集折外概率",
        "threshold_min_tier_fraction": float(params["q2_min_tier_fraction"]),
        "training_oof_auc": float(roc_auc_score(y_train, oof_probability)),
        "cart_surrogate_max_depth": int(selected_surrogate["最大深度"]),
        "cart_surrogate_min_leaf_fraction": float(selected_surrogate["最小叶节点比例"]),
        "cart_surrogate_cv_macro_f1": float(selected_surrogate["CV宏平均F1"]),
        "cart_surrogate_cv_fidelity": float(selected_surrogate["CV一致率"]),
        "cart_surrogate_train_fidelity": float(accuracy_score(train_statistical_code, train_cart_code)),
        "cart_surrogate_test_fidelity": float(accuracy_score(test_statistical_code, test_cart_code)),
        "prospective_outcome_available": False,
        "statistical_tier_is_rule_based": False,
        "clinical_prompt_is_separate": True,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary, labels=[0, 1]).ravel()
    metrics.update({"tn_at_0.5": int(tn), "fp_at_0.5": int(fp), "fn_at_0.5": int(fn), "tp_at_0.5": int(tp)})
    tier_masks = {level: test_statistical_code == level for level in (1, 2, 3)}
    def tier_fisher(a: int, b: int) -> float:
        ma, mb = tier_masks[a], tier_masks[b]
        table = [
            [int(y_test[ma].sum()), int(ma.sum() - y_test[ma].sum())],
            [int(y_test[mb].sum()), int(mb.sum() - y_test[mb].sum())],
        ]
        return float(stats.fisher_exact(table).pvalue)
    metrics.update({
        "fisher_p_low_vs_medium_statistical": tier_fisher(1, 2),
        "fisher_p_medium_vs_high_statistical": tier_fisher(2, 3),
    })

    # 置换重要性只在独立测试集计算。
    perm = permutation_importance(calibrated, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    feature_importance = pd.DataFrame({
        "特征": features,
        "AUC置换重要性均值": perm.importances_mean,
        "AUC置换重要性标准差": perm.importances_std,
    }).sort_values("AUC置换重要性均值", ascending=False).reset_index(drop=True)

    # 各层特征画像。
    test_profile = df.iloc[test_idx].copy()
    test_profile["风险等级编码"] = test_statistical_code
    profile_cols = ["痰湿质积分", "活动量表总分", "BMI", "血糖", "血尿酸", "年龄组"]
    profile = test_profile.groupby("风险等级编码")[profile_cols].agg(["median", "mean"]).round(3)
    profile.columns = [f"{a}_{b}" for a, b in profile.columns]
    profile = profile.reset_index()
    profile["风险等级"] = profile["风险等级编码"].map({1: "低危", 2: "中危", 3: "高危"})
    statistical_rule_text = {
        1: f"非血脂筛查概率 < {thresholds[0]:.4f}",
        2: f"{thresholds[0]:.4f} ≤ 非血脂筛查概率 < {thresholds[1]:.4f}",
        3: f"非血脂筛查概率 ≥ {thresholds[1]:.4f}",
    }
    profile["统计分层规则"] = profile["风险等级编码"].map(statistical_rule_text)

    combinations = enumerate_phlegm_combinations(test_profile, test_statistical_code)

    # 患者级结果：统计风险、CART解释和临床提示三种口径严格分列。
    patient_results = df[["样本ID", "体质标签"] + features].copy()
    patient_results["真实标签"] = y
    patient_results["当前诊断状态"] = clinical_all["当前诊断状态"].to_numpy()
    patient_results["非血脂筛查概率"] = all_probability
    patient_results["概率来源"] = "训练集折外"
    patient_results.loc[test_idx, "概率来源"] = "独立测试集"
    patient_results["统计风险等级编码"] = all_statistical_code
    patient_results["统计风险等级"] = pd.Series(all_statistical_code).map({1: "低危", 2: "中危", 3: "高危"}).to_numpy()
    patient_results["概率切点依据"] = patient_results["统计风险等级编码"].map(statistical_rule_text)
    patient_results["CART代理风险等级编码"] = all_cart_code
    patient_results["CART代理风险等级"] = pd.Series(all_cart_code).map({1: "低危", 2: "中危", 3: "高危"}).to_numpy()
    patient_results["CART代理路径"] = cart_paths_all
    patient_results["临床管理提示"] = clinical_all["临床管理提示"].to_numpy()
    patient_results["临床管理依据"] = clinical_all["临床管理依据"].to_numpy()
    predictions = patient_results.iloc[test_idx].copy()
    all_tiers = pd.concat([full_tiers, test_tiers], ignore_index=True, sort=False)
    candidate_df.to_csv(OUTPUT_DIR / "question2_candidate_models.csv", index=False, encoding="utf-8-sig")
    all_tiers.to_csv(OUTPUT_DIR / "question2_risk_tiers.csv", index=False, encoding="utf-8-sig")
    threshold_selection.to_csv(OUTPUT_DIR / "question2_threshold_selection.csv", index=False, encoding="utf-8-sig")
    surrogate_candidates.to_csv(OUTPUT_DIR / "question2_cart_surrogate_candidates.csv", index=False, encoding="utf-8-sig")
    cart_rules.to_csv(OUTPUT_DIR / "question2_cart_rules.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "question2_threshold_bootstrap.csv").unlink(missing_ok=True)
    (OUTPUT_DIR / "question2_rule_sensitivity.csv").unlink(missing_ok=True)
    pd.DataFrame(
        [[tn, fp], [fn, tp]], index=["真实阴性", "真实阳性"], columns=["预测阴性", "预测阳性"]
    ).to_csv(OUTPUT_DIR / "question2_binary_confusion_matrix.csv", encoding="utf-8-sig")
    feature_importance.to_csv(OUTPUT_DIR / "question2_feature_importance.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(OUTPUT_DIR / "question2_feature_layers.csv", index=False, encoding="utf-8-sig")
    combinations.to_csv(OUTPUT_DIR / "question2_phlegm_core_combinations.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "question2_test_predictions.csv", index=False, encoding="utf-8-sig")
    patient_results.to_csv(OUTPUT_DIR / "question2_patient_results.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "question2_metrics.json", metrics)

    # ROC。
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot(fpr, tpr, color="#E67E22", linewidth=2, label=f"非血脂关联筛查模型 AUC={metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="随机参考线")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "问题2_筛查模型ROC.pdf")

    # 训练内定阈后的独立测试集三级筛查分数层级。
    fig, ax1 = plt.subplots(figsize=(7.2, 5.3))
    colors = ["#59A14F", "#F2CF5B", "#E15759"]
    x = np.arange(3)
    bars = ax1.bar(x, test_tiers["人数"], color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x, test_tiers["风险等级"])
    ax1.set_ylabel("测试集人数")
    ax2 = ax1.twinx()
    ax2.plot(x, test_tiers["实际患病率"], color="#1F4E79", marker="o", linewidth=2, label="当前血脂异常率")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("实际患病率")
    for bar, rate in zip(bars, test_tiers["实际患病率"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{rate:.1%}", ha="center")
    save_figure(fig, "问题2_三级风险分布.pdf")

    # 校准曲线。
    prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"校准模型（Brier={metrics['brier']:.3f}）")
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="理想校准")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("观察阳性率")
    ax.legend(frameon=False)
    save_figure(fig, "问题2_校准曲线.pdf")

    rates = test_tiers["实际患病率"].tolist()
    print(f"问题2：选用 {best_name}，独立测试 AUC={metrics['auc']:.3f}，Brier={metrics['brier']:.3f}")
    print(f"训练折外概率切点：{thresholds[0]:.4f}, {thresholds[1]:.4f}")
    print(f"测试集低/中/高统计层人数：{test_tiers['人数'].tolist()}；当前异常率：{[round(x, 3) for x in rates]}")
    print(f"CART代理测试一致率：{metrics['cart_surrogate_test_fidelity']:.3f}")
    if not combinations.empty:
        print("痰湿体质高危核心组合首位：" + str(combinations.iloc[0]["核心特征组合"]))
    return {
        "model": calibrated, "metrics": metrics, "candidate_models": candidate_df,
        "tiers": all_tiers, "test_tiers": test_tiers, "feature_importance": feature_importance,
        "profile": profile, "combinations": combinations, "test_predictions": predictions,
        "patient_results": patient_results, "thresholds": thresholds,
        "threshold_selection": threshold_selection, "cart_surrogate": cart_surrogate,
        "cart_surrogate_candidates": surrogate_candidates, "cart_rules": cart_rules,
        "clinical_all": clinical_all, "clinical_test": clinical_test,
        "train_idx": train_idx, "test_idx": test_idx,
    }


# ============================================================================
# 第四部分：问题三——六个月个体化干预方案优化
# ============================================================================


def allowed_intensities(age_group: int, activity_score: float) -> list[int]:
    age_max = 3 if age_group <= 2 else (2 if age_group <= 4 else 1)
    score_max = 1 if activity_score < 40 else (2 if activity_score < 60 else 3)
    return list(range(1, min(age_max, score_max) + 1))


def tiaoli_level(score: float) -> tuple[int, int, str]:
    if score <= 58:
        return 1, 30, "饮食调理+穴位按摩"
    if score <= 61:
        return 2, 80, "饮食调理+穴位按摩+八段锦"
    return 3, 130, "饮食调理+穴位按摩+八段锦+中药代茶饮"


@dataclass
class DPState:
    score: float
    cost: int
    path: list[dict[str, Any]]


def prune_states(states: dict[float, DPState]) -> dict[float, DPState]:
    # 相同积分已保留最低成本；再剔除“积分更高且成本不低”的支配状态。
    kept: dict[float, DPState] = {}
    best_cost = math.inf
    for score in sorted(states):
        state = states[score]
        if state.cost < best_cost:
            kept[score] = state
            best_cost = state.cost
    return kept


def optimize_patient(
    initial_score: float,
    age_group: int,
    activity_score: float,
    *,
    months: int = 6,
    weeks_per_month: int = 4,
    max_cost: int = 2000,
    step: float = 0.5,
    target_effect_ratio: float = 0.90,
    activity_effect_multiplier: float = 1.0,
    tiaoli_monthly_effect: float = 0.0,
) -> dict[str, Any]:
    allowed = allowed_intensities(int(age_group), float(activity_score))
    activity_cost = {1: 3, 2: 5, 3: 8}
    initial_rounded = round(round(float(initial_score) / step) * step, 6)
    states = {initial_rounded: DPState(initial_rounded, 0, [])}

    for month in range(1, months + 1):
        next_states: dict[float, DPState] = {}
        for state in states.values():
            level, tl_cost, method = tiaoli_level(state.score)
            for intensity in allowed:
                for frequency in range(1, 11):
                    month_cost = tl_cost + activity_cost[intensity] * frequency * weeks_per_month
                    total_cost = state.cost + month_cost
                    if total_cost > max_cost:
                        continue
                    if frequency < 5:
                        activity_rate = 0.0
                    else:
                        activity_rate = 0.03 * (intensity - 1) + 0.01 * (frequency - 5)
                    activity_rate *= activity_effect_multiplier
                    next_score = state.score * (1 - activity_rate) * (1 - tiaoli_monthly_effect)
                    next_score = max(0.0, round(round(next_score / step) * step, 6))
                    action = {
                        "月份": month,
                        "月初痰湿积分": state.score,
                        "调理等级": level,
                        "核心调理方式": method,
                        "活动强度": intensity,
                        "单次时长_分钟": 10 * intensity,
                        "每周次数": frequency,
                        "当月调理成本": tl_cost,
                        "当月活动成本": activity_cost[intensity] * frequency * weeks_per_month,
                        "当月总成本": month_cost,
                        "当月活动降幅": activity_rate,
                        "月末痰湿积分": next_score,
                    }
                    candidate = DPState(next_score, total_cost, state.path + [action])
                    incumbent = next_states.get(next_score)
                    if incumbent is None or candidate.cost < incumbent.cost:
                        next_states[next_score] = candidate
        states = prune_states(next_states)
        if not states:
            raise RuntimeError("无可行干预状态")

    final_states = list(states.values())
    best_effect_state = min(final_states, key=lambda s: (s.score, s.cost))
    max_reduction = initial_score - best_effect_state.score
    target_reduction = target_effect_ratio * max_reduction
    eligible = [s for s in final_states if initial_score - s.score + 1e-9 >= target_reduction]
    recommended = min(eligible, key=lambda s: (s.cost, s.score))

    pareto = pd.DataFrame([
        {"最终积分": s.score, "总成本": s.cost, "降低分数": initial_score - s.score}
        for s in final_states
    ]).sort_values(["总成本", "最终积分"]).reset_index(drop=True)
    plan = pd.DataFrame(recommended.path)
    checks = {
        "cost_within_budget": recommended.cost <= max_cost,
        "frequency_valid": bool(plan["每周次数"].between(1, 10).all()),
        "intensity_valid": bool(plan["活动强度"].isin(allowed).all()),
        "score_nonincreasing": bool((plan["月末痰湿积分"] <= plan["月初痰湿积分"] + 1e-9).all()),
        "six_months": len(plan) == months,
    }
    if not all(checks.values()):
        raise AssertionError(f"约束回代失败：{checks}")
    return {
        "plan": plan,
        "pareto": pareto,
        "initial_score": float(initial_score),
        "final_score": float(recommended.score),
        "total_cost": int(recommended.cost),
        "reduction": float(initial_score - recommended.score),
        "max_reduction": float(max_reduction),
        "target_reduction": float(target_reduction),
        "allowed_intensities": allowed,
        "checks": checks,
    }


def score_band(score: float) -> str:
    if score <= 58:
        return "基础调理(≤58)"
    if score <= 61:
        return "中度调理(59-61)"
    return "强化调理(≥62)"


def patient_rule_stratum(age_group: int, activity_score: float, score: float) -> str:
    """按题面调理阈值与耐受约束，把患者唯一映射到九类规则分层。"""
    max_intensity = max(allowed_intensities(int(age_group), float(activity_score)))
    return f"{score_band(float(score))}｜最大耐受{max_intensity}级"


def tolerance_rule(max_intensity: int) -> str:
    rules = {
        1: "年龄组=5，或活动总分<40",
        2: "年龄组1-2且40≤活动总分<60，或年龄组3-4且活动总分≥40",
        3: "年龄组1-2且活动总分≥60",
    }
    return rules[int(max_intensity)]


def score_band_rule(band: str) -> str:
    rules = {
        "基础调理(≤58)": "初始痰湿积分≤58",
        "中度调理(59-61)": "59≤初始痰湿积分≤61",
        "强化调理(≥62)": "初始痰湿积分≥62",
    }
    return rules[str(band)]


def fit_q3_cart_surrogate(summary: pd.DataFrame) -> dict[str, Any]:
    """用受限多输出 CART 近似动态规划的首月强度—频率联合决策。"""
    features = ["年龄组", "活动量表总分", "初始痰湿积分"]
    targets = ["推荐首月活动强度", "推荐首月每周次数"]
    x = summary[features].astype(float).reset_index(drop=True)
    y = summary[targets].astype(int).to_numpy()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidate_rows: list[dict[str, Any]] = []

    for depth, min_fraction, max_leaves in itertools.product(
        (2, 3, 4, 5, 6), (0.05, 0.08, 0.10), (6, 9, 12)
    ):
        fold_joint: list[float] = []
        fold_component: list[float] = []
        fold_feasible: list[bool] = []
        for train_idx, valid_idx in cv.split(x):
            min_leaf = max(2, math.ceil(min_fraction * len(train_idx)))
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                max_leaf_nodes=max_leaves,
                random_state=RANDOM_STATE,
            )
            model.fit(x.iloc[train_idx], y[train_idx])
            predicted = np.asarray(model.predict(x.iloc[valid_idx]), dtype=int)
            fold_joint.append(float(np.mean(np.all(predicted == y[valid_idx], axis=1))))
            fold_component.append(float(np.mean(predicted == y[valid_idx])))
            allowed_max = np.array([
                max(allowed_intensities(int(row[0]), float(row[1])))
                for row in x.iloc[valid_idx].to_numpy()
            ])
            fold_feasible.append(bool(
                np.all(predicted[:, 0] <= allowed_max)
                and np.all((predicted[:, 1] >= 5) & (predicted[:, 1] <= 10))
            ))

        full_min_leaf = max(2, math.ceil(min_fraction * len(x)))
        full_model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=full_min_leaf,
            max_leaf_nodes=max_leaves,
            random_state=RANDOM_STATE,
        ).fit(x, y)
        candidate_rows.append({
            "最大深度": depth,
            "最小叶节点比例": min_fraction,
            "最大叶节点数": max_leaves,
            "全样本叶节点数": int(full_model.get_n_leaves()),
            "五折联合处方一致率": float(np.mean(fold_joint)),
            "五折分量一致率": float(np.mean(fold_component)),
            "五折耐受约束通过": bool(all(fold_feasible)),
        })

    candidates = pd.DataFrame(candidate_rows)
    feasible = candidates[candidates["五折耐受约束通过"]].copy()
    if feasible.empty:
        raise AssertionError("问题三 CART 候选均产生耐受约束冲突")
    best = feasible.sort_values(
        ["五折联合处方一致率", "五折分量一致率", "全样本叶节点数", "最大深度", "最小叶节点比例", "最大叶节点数"],
        ascending=[False, False, True, True, False, True],
    ).iloc[0]
    final_model = DecisionTreeClassifier(
        max_depth=int(best["最大深度"]),
        min_samples_leaf=max(2, math.ceil(float(best["最小叶节点比例"]) * len(x))),
        max_leaf_nodes=int(best["最大叶节点数"]),
        random_state=RANDOM_STATE,
    ).fit(x, y)
    selected_mask = (
        candidates["最大深度"].eq(int(best["最大深度"]))
        & candidates["最小叶节点比例"].eq(float(best["最小叶节点比例"]))
        & candidates["最大叶节点数"].eq(int(best["最大叶节点数"]))
    )
    candidates["是否选用"] = selected_mask
    predicted = np.asarray(final_model.predict(x), dtype=int)
    leaf_ids = final_model.apply(x).astype(int)
    paths = cart_path_strings(final_model, x)
    rule_rows: list[dict[str, Any]] = []
    actual_pairs = pd.Series(
        [f"{a}级-{f}次/周" for a, f in y], index=summary.index, dtype="object"
    )
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        first = int(np.flatnonzero(mask)[0])
        group = summary.loc[mask]
        distribution = actual_pairs.loc[mask].value_counts()
        row: dict[str, Any] = {
            "叶节点": int(leaf),
            "规则路径": paths[first],
            "患者人数": int(mask.sum()),
            "代理首月活动强度": int(predicted[first, 0]),
            "代理首月每周次数": int(predicted[first, 1]),
            "叶节点联合一致率": float(np.mean(np.all(predicted[mask] == y[mask], axis=1))),
            "实际处方分布": "；".join(f"{name}:{count}" for name, count in distribution.items()),
            "首月成本中位数": float(group["首月总成本"].median()) if "首月总成本" in group else np.nan,
            "六个月成本中位数": float(group["六个月总成本"].median()),
            "六个月成本Q1": float(group["六个月总成本"].quantile(0.25)),
            "六个月成本Q3": float(group["六个月总成本"].quantile(0.75)),
            "降低分数中位数": float(group["降低分数"].median()),
            "降低分数Q1": float(group["降低分数"].quantile(0.25)),
            "降低分数Q3": float(group["降低分数"].quantile(0.75)),
            "降低比例中位数": float(group["降低比例"].median()),
        }
        rule_rows.append(row)
    rules = pd.DataFrame(rule_rows).sort_values("叶节点").reset_index(drop=True)
    metrics = {
        "max_depth": int(best["最大深度"]),
        "min_leaf_fraction": float(best["最小叶节点比例"]),
        "max_leaf_nodes": int(best["最大叶节点数"]),
        "leaf_count": int(final_model.get_n_leaves()),
        "cv_joint_fidelity": float(best["五折联合处方一致率"]),
        "cv_component_fidelity": float(best["五折分量一致率"]),
        "train_joint_fidelity": float(np.mean(np.all(predicted == y, axis=1))),
        "train_component_fidelity": float(np.mean(predicted == y)),
        "all_cv_feasible": bool(best["五折耐受约束通过"]),
    }
    return {
        "model": final_model,
        "features": features,
        "targets": targets,
        "candidates": candidates,
        "rules": rules,
        "metrics": metrics,
        "leaf_ids": leaf_ids,
        "paths": paths,
        "predicted": predicted,
    }


def run_problem3(df: pd.DataFrame, params: dict[str, Any] = PARAMS) -> dict[str, Any]:
    patients = df[df["体质标签"].eq(5)].copy()
    cache: dict[tuple[float, int], dict[str, Any]] = {}
    summary_rows = []
    sample_plans: dict[int, pd.DataFrame] = {}
    sample_paretos: dict[int, pd.DataFrame] = {}
    all_checks = []

    for _, row in patients.iterrows():
        max_intensity = max(allowed_intensities(int(row["年龄组"]), float(row["活动量表总分"])))
        key = (float(row["痰湿质积分"]), max_intensity)
        if key not in cache:
            # 只依赖初始积分与最大可耐受强度；构造一个满足该上限的代表性年龄/评分组合。
            representative = {1: (5, 30), 2: (3, 50), 3: (1, 70)}[max_intensity]
            cache[key] = optimize_patient(
                key[0], representative[0], representative[1],
                months=int(params["q3_months"]), weeks_per_month=int(params["q3_weeks_per_month"]),
                max_cost=int(params["q3_max_cost"]), step=float(params["q3_dp_step"]),
                target_effect_ratio=float(params["q3_target_effect_ratio"]),
            )
        result = cache[key]
        # 由最大可耐受强度等价缓存；回代原患者约束。
        allowed_original = allowed_intensities(int(row["年龄组"]), float(row["活动量表总分"]))
        if max(result["plan"]["活动强度"]) > max(allowed_original):
            raise AssertionError("缓存方案违反原患者耐受约束")
        plan = result["plan"]
        initial_band = score_band(float(row["痰湿质积分"]))
        summary_rows.append({
            "样本ID": int(row["样本ID"]),
            "年龄组": int(row["年龄组"]),
            "活动量表总分": float(row["活动量表总分"]),
            "初始痰湿积分": float(row["痰湿质积分"]),
            "初始调理等级": initial_band,
            "最大允许活动强度": max_intensity,
            "匹配分层": patient_rule_stratum(
                int(row["年龄组"]), float(row["活动量表总分"]), float(row["痰湿质积分"])
            ),
            "耐受度判定规则": tolerance_rule(max_intensity),
            "首月调理等级": int(plan.iloc[0]["调理等级"]),
            "首月调理方式": str(plan.iloc[0]["核心调理方式"]),
            "推荐首月活动强度": int(plan.iloc[0]["活动强度"]),
            "推荐首月每周次数": int(plan.iloc[0]["每周次数"]),
            "首月总成本": int(plan.iloc[0]["当月总成本"]),
            "调理等级序列": "-".join(plan["调理等级"].astype(int).astype(str)),
            "活动强度序列": "-".join(plan["活动强度"].astype(int).astype(str)),
            "训练频率序列": "-".join(plan["每周次数"].astype(int).astype(str)),
            "月成本序列": "-".join(plan["当月总成本"].astype(int).astype(str)),
            "最终痰湿积分": result["final_score"],
            "降低分数": result["reduction"],
            "降低比例": result["reduction"] / float(row["痰湿质积分"]),
            "最大可降低分数": result["max_reduction"],
            "效果保留率": result["reduction"] / result["max_reduction"] if result["max_reduction"] > 0 else 1.0,
            "六个月总成本": result["total_cost"],
        })
        all_checks.append(all(result["checks"].values()))
        sample_id = int(row["样本ID"])
        if sample_id in (1, 2, 3):
            sample_plan = plan.copy()
            sample_plan.insert(0, "样本ID", sample_id)
            sample_plans[sample_id] = sample_plan
            sample_paretos[sample_id] = result["pareto"].assign(样本ID=sample_id)

    summary = pd.DataFrame(summary_rows).sort_values("样本ID").reset_index(drop=True)
    if set(sample_plans) != {1, 2, 3}:
        raise ValueError("附件样本 ID 1/2/3 不完整或不属于痰湿质")
    plan_123 = pd.concat([sample_plans[i] for i in (1, 2, 3)], ignore_index=True)
    pareto_123 = pd.concat([sample_paretos[i] for i in (1, 2, 3)], ignore_index=True)
    selected_on_frontier: dict[int, bool] = {}
    for sample_id in (1, 2, 3):
        chosen = summary.loc[summary["样本ID"].eq(sample_id)].iloc[0]
        frontier = pareto_123[pareto_123["样本ID"].eq(sample_id)]
        selected_on_frontier[sample_id] = bool((
            np.isclose(frontier["总成本"], chosen["六个月总成本"])
            & np.isclose(frontier["最终积分"], chosen["最终痰湿积分"])
        ).any())
    if not all(selected_on_frontier.values()):
        raise AssertionError(f"样本1—3推荐点并非全部位于Pareto前沿：{selected_on_frontier}")

    # CART 仅作为首月处方的可解释代理；个体最终方案仍以动态规划为准。
    cart = fit_q3_cart_surrogate(summary)
    summary["CART规则叶节点"] = cart["leaf_ids"]
    summary["CART规则路径"] = cart["paths"]
    summary["CART代理首月活动强度"] = cart["predicted"][:, 0]
    summary["CART代理首月每周次数"] = cart["predicted"][:, 1]
    summary["CART规则一致标记"] = (
        summary["CART代理首月活动强度"].eq(summary["推荐首月活动强度"])
        & summary["CART代理首月每周次数"].eq(summary["推荐首月每周次数"])
    )

    # 九类硬约束分层：用于快速匹配与解释，不替代个体动态规划。
    matching_rows = []
    for (band, max_intensity), group in summary.groupby(["初始调理等级", "最大允许活动强度"], observed=True):
        mode_pair, mode_count = Counter(
            zip(group["推荐首月活动强度"], group["推荐首月每周次数"])
        ).most_common(1)[0]
        mode_level = int(group["首月调理等级"].mode().iloc[0])
        mode_method = str(group["首月调理方式"].mode().iloc[0])
        mode_mask = (
            group["推荐首月活动强度"].eq(int(mode_pair[0]))
            & group["推荐首月每周次数"].eq(int(mode_pair[1]))
        )
        typical_first_cost = int(group.loc[mode_mask, "首月总成本"].iloc[0])
        stratum = f"{band}｜最大耐受{int(max_intensity)}级"
        matching_rows.append({
            "匹配分层": stratum,
            "患者特征组合": f"{score_band_rule(str(band))}；{tolerance_rule(int(max_intensity))}",
            "初始痰湿分层": band,
            "最大允许活动强度": int(max_intensity),
            "患者人数": len(group),
            "典型首月调理等级": mode_level,
            "典型首月调理方式": mode_method,
            "典型首月活动强度": int(mode_pair[0]),
            "典型首月每周次数": int(mode_pair[1]),
            "典型首月处方成本": typical_first_cost,
            "典型首月处方覆盖率": float(mode_count / len(group)),
            "首月成本中位数": float(group["首月总成本"].median()),
            "首月成本Q1": float(group["首月总成本"].quantile(0.25)),
            "首月成本Q3": float(group["首月总成本"].quantile(0.75)),
            "六个月成本中位数": float(group["六个月总成本"].median()),
            "六个月成本Q1": float(group["六个月总成本"].quantile(0.25)),
            "六个月成本Q3": float(group["六个月总成本"].quantile(0.75)),
            "降低分数中位数": float(group["降低分数"].median()),
            "降低分数Q1": float(group["降低分数"].quantile(0.25)),
            "降低分数Q3": float(group["降低分数"].quantile(0.75)),
            "降低比例中位数": float(group["降低比例"].median()),
            "降低比例Q1": float(group["降低比例"].quantile(0.25)),
            "降低比例Q3": float(group["降低比例"].quantile(0.75)),
            "效果保留率中位数": float(group["效果保留率"].median()),
            "低样本分层": bool(len(group) < 15),
            "规则用途": "首月快速初筛；完整六个月方案需运行动态规划",
        })
    band_order = {"基础调理(≤58)": 1, "中度调理(59-61)": 2, "强化调理(≥62)": 3}
    matching = pd.DataFrame(matching_rows)
    matching["_分层顺序"] = matching["初始痰湿分层"].map(band_order)
    matching = matching.sort_values(["最大允许活动强度", "_分层顺序"]).drop(columns="_分层顺序").reset_index(drop=True)

    # 一因素灵敏度：仅 ID 1/2/3，避免混合改变多个参数。
    sensitivity_rows = []
    sample_source = patients[patients["样本ID"].isin([1, 2, 3])].sort_values("样本ID")
    for _, row in sample_source.iterrows():
        baseline_kwargs = {
            "initial_score": float(row["痰湿质积分"]), "age_group": int(row["年龄组"]),
            "activity_score": float(row["活动量表总分"]), "months": int(params["q3_months"]),
            "weeks_per_month": int(params["q3_weeks_per_month"]),
            "max_cost": int(params["q3_max_cost"]), "step": float(params["q3_dp_step"]),
            "target_effect_ratio": float(params["q3_target_effect_ratio"]),
        }
        scenarios = [("基准", "基准", 1.0)]
        scenarios += [("积分步长", "step", v) for v in (1.0, 0.5, 0.25)]
        scenarios += [("活动效果系数", "activity_effect_multiplier", v) for v in (0.8, 1.0, 1.2)]
        scenarios += [("预算上限", "max_cost", v) for v in (1600, 1800, 2000)]
        scenarios += [("调理额外月降幅", "tiaoli_monthly_effect", v) for v in (0.0, 0.01, 0.02)]
        for dimension, key, value in scenarios:
            kwargs = baseline_kwargs.copy()
            if key != "基准":
                kwargs[key] = value
            result = optimize_patient(**kwargs)
            sensitivity_rows.append({
                "样本ID": int(row["样本ID"]), "敏感性维度": dimension, "参数值": value,
                "最终积分": result["final_score"], "降低分数": result["reduction"], "总成本": result["total_cost"],
            })
    sensitivity = pd.DataFrame(sensitivity_rows)

    summary.to_csv(OUTPUT_DIR / "question3_all_phlegm_patients.csv", index=False, encoding="utf-8-sig")
    plan_123.to_csv(OUTPUT_DIR / "question3_sample_1_2_3_monthly_plans.csv", index=False, encoding="utf-8-sig")
    pareto_123.to_csv(OUTPUT_DIR / "question3_sample_1_2_3_pareto.csv", index=False, encoding="utf-8-sig")
    matching.to_csv(OUTPUT_DIR / "question3_matching_rules.csv", index=False, encoding="utf-8-sig")
    cart["candidates"].to_csv(OUTPUT_DIR / "question3_cart_surrogate_candidates.csv", index=False, encoding="utf-8-sig")
    cart["rules"].to_csv(OUTPUT_DIR / "question3_cart_rules.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUT_DIR / "question3_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "question3_rule_consistency.json", {
        "deterministic_strata": int(summary["匹配分层"].nunique()),
        "patient_count": len(summary),
        "cart_metrics": cart["metrics"],
        "cart_joint_match_count": int(summary["CART规则一致标记"].sum()),
        "cart_joint_mismatch_count": int((~summary["CART规则一致标记"]).sum()),
        "rule_role": "首月处方解释代理；完整六个月方案以动态规划为准",
    })
    write_json(OUTPUT_DIR / "question3_constraint_checks.json", {
        "patient_count": len(summary), "all_plans_feasible": bool(all(all_checks)),
        "budget_limit": int(params["q3_max_cost"]), "target_effect_ratio": float(params["q3_target_effect_ratio"]),
        "deterministic_strata": int(summary["匹配分层"].nunique()),
        "all_effective_frequencies_at_least_five": bool(
            summary["训练频率序列"].str.split("-").explode().astype(int).ge(5).all()
        ),
        "cart_all_cv_feasible": bool(cart["metrics"]["all_cv_feasible"]),
        "selected_points_on_pareto_frontier": selected_on_frontier,
    })

    # ID 1/2/3 积分曲线。
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for sample_id, marker in zip((1, 2, 3), ("o", "s", "^")):
        src = sample_source[sample_source["样本ID"].eq(sample_id)].iloc[0]
        plan = sample_plans[sample_id]
        scores = [float(src["痰湿质积分"])] + plan["月末痰湿积分"].tolist()
        ax.plot(range(7), scores, marker=marker, linewidth=2, label=f"样本{sample_id}：{scores[0]:.0f}→{scores[-1]:.1f}")
    ax.set_xlabel("干预月份")
    ax.set_ylabel("痰湿体质积分")
    ax.set_xticks(range(7))
    ax.legend(frameon=False)
    save_figure(fig, "问题3_积分变化曲线.pdf")

    # 成本—效果对比。
    sample_summary = summary[summary["样本ID"].isin([1, 2, 3])].sort_values("样本ID")
    x = np.arange(3)
    fig, ax1 = plt.subplots(figsize=(8.2, 5.4))
    ax1.bar(x - 0.18, sample_summary["六个月总成本"], 0.36, color="#4C78A8", edgecolor="black", label="六个月总成本")
    ax1.axhline(int(params["q3_max_cost"]), color="#D62728", linestyle="--", label="成本上限")
    ax1.set_ylabel("成本（元）")
    ax1.set_xticks(x, [f"样本{i}" for i in (1, 2, 3)])
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, sample_summary["降低分数"], 0.36, color="#F28E8E", edgecolor="black", label="降低分数")
    ax2.set_ylabel("痰湿积分降低值")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    save_figure(fig, "问题3_成本效果对比.pdf")

    # 成本—积分降幅 Pareto 前沿，并标记 ε-约束最终选择点。
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    colors = {1: "#4C78A8", 2: "#F28E2B", 3: "#59A14F"}
    markers = {1: "o", 2: "s", 3: "^"}
    for sample_id in (1, 2, 3):
        frontier = pareto_123[pareto_123["样本ID"].eq(sample_id)].sort_values("总成本")
        ax.plot(
            frontier["总成本"], frontier["降低分数"], color=colors[sample_id],
            linewidth=1.8, marker=markers[sample_id], markersize=3.5, label=f"样本{sample_id} Pareto前沿",
        )
        chosen = sample_summary[sample_summary["样本ID"].eq(sample_id)].iloc[0]
        ax.scatter(
            chosen["六个月总成本"], chosen["降低分数"], s=150, marker="*",
            color=colors[sample_id], edgecolor="black", linewidth=0.8, zorder=5,
        )
        ax.annotate(
            f"样本{sample_id}选择点\n{chosen['六个月总成本']:.0f}元，降{chosen['降低分数']:.1f}分",
            (chosen["六个月总成本"], chosen["降低分数"]), xytext=(6, 6),
            textcoords="offset points", fontsize=8, color=colors[sample_id],
        )
    ax.set_xlabel("六个月总成本（元）")
    ax.set_ylabel("痰湿积分降低值")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99, 0.02, "★：达到最大可降幅90%以上后的最低成本方案",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
    )
    save_figure(fig, "问题3_成本降幅Pareto前沿.pdf")

    # 患者特征—首月处方 CART 代理树；叶节点仅用于解释动态规划首月动作。
    fig, ax = plt.subplots(figsize=(18.0, 10.5))
    tree_artists = plot_tree(
        cart["model"],
        feature_names=cart["features"],
        filled=True,
        rounded=True,
        impurity=False,
        node_ids=True,
        proportion=False,
        precision=2,
        fontsize=6,
        ax=ax,
    )
    tree = cart["model"].tree_
    classes = cart["model"].classes_
    for artist in tree_artists:
        text = artist.get_text()
        first_line = text.splitlines()[0]
        if not first_line.startswith("node #"):
            continue
        node = int(first_line.split("#", maxsplit=1)[1])
        action = []
        for output_index, output_classes in enumerate(classes):
            class_count = len(output_classes)
            best_class_index = int(np.argmax(tree.value[node, output_index, :class_count]))
            action.append(int(output_classes[best_class_index]))
        artist.set_text(text + f"\n代理处方={action[0]}级,{action[1]}次/周")
    ax.text(
        0.01, 0.01,
        "value第一行为强度类别计数、第二行为频率类别计数；硬约束与个体动态规划优先。",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#777777"},
    )
    save_figure(fig, "问题3_患者特征首月方案规则树.pdf")

    print(f"问题3：已对全部 {len(summary)} 名痰湿质患者求解，约束回代全部通过={all(all_checks)}")
    print(
        f"九类硬约束分层覆盖 {matching['患者人数'].sum()} 人；CART五折联合处方一致率="
        f"{cart['metrics']['cv_joint_fidelity']:.3f}，训练一致率={cart['metrics']['train_joint_fidelity']:.3f}"
    )
    for sample_id in (1, 2, 3):
        row = summary[summary["样本ID"].eq(sample_id)].iloc[0]
        print(
            f"样本{sample_id}：{row['初始痰湿积分']:.1f}→{row['最终痰湿积分']:.1f}分，"
            f"成本{row['六个月总成本']:.0f}元，最大允许强度{row['最大允许活动强度']:.0f}级"
        )
    return {
        "summary": summary, "plans_123": plan_123, "pareto_123": pareto_123,
        "matching": matching, "sensitivity": sensitivity,
        "cart_model": cart["model"], "cart_candidates": cart["candidates"],
        "cart_rules": cart["rules"], "cart_metrics": cart["metrics"],
        "selected_on_frontier": selected_on_frontier,
        "all_plans_feasible": bool(all(all_checks)),
    }


# ============================================================================
# 第五部分：结果汇总与自动验收
# ============================================================================


def write_results_report(
    quality: dict[str, Any], q1: dict[str, Any], q2: dict[str, Any], q3: dict[str, Any], elapsed: float
) -> None:
    q2_test = q2["test_tiers"]
    q3_samples = q3["summary"][q3["summary"]["样本ID"].isin([1, 2, 3])].sort_values("样本ID")
    q3_matching = q3["matching"]
    q1_phlegm_perf = q1["endpoint_performance"].loc[q1["endpoint_performance"]["终点"].eq("痰湿严重度")].iloc[0]
    q1_lipid_perf = q1["endpoint_performance"].loc[q1["endpoint_performance"]["终点"].eq("高血脂关联筛查")].iloc[0]
    q1_phlegm_or = q1["constitution"].loc[q1["constitution"]["体质标签"].eq(5)].iloc[0]
    q1_point_top = q1["constitution"].loc[q1["constitution"]["OR点估计名次"].eq(1)].iloc[0]
    q1_pareto_names = q1["exploratory_pareto"].loc[
        q1["exploratory_pareto"]["探索性Pareto候选"], "指标名称"
    ].tolist()
    combo_text = "无满足最小支持度的组合"
    if not q2["combinations"].empty:
        combo = q2["combinations"].iloc[0]
        combo_text = (
            f"{combo['核心特征组合']}（覆盖{int(combo['覆盖人数'])}人，置信度 {combo['高危置信度']:.1%}，"
            f"95%CI {combo['置信度95%CI下限']:.1%}–{combo['置信度95%CI上限']:.1%}，提升度 {combo['提升度']:.2f}）"
        )

    lines = [
        "# 计算结果",
        "",
        "## 运行环境",
        "",
        f"- Python：{platform.python_version()}",
        f"- 平台：{platform.platform()}",
        f"- 随机种子：{RANDOM_STATE}",
        f"- 完整运行耗时：{elapsed:.1f} 秒",
        "",
        "## 数据读取与预处理",
        "",
        f"附件共 {quality['rows']} 例，缺失值 {quality['missing_values']} 个，样本 ID 重复 {quality['duplicate_sample_ids']} 个。",
        f"高血脂标签阳性率为 {quality['label_prevalence']:.1%}；标签与四项血脂任一异常规则的一致率为 {quality['diagnostic_rule_agreement']:.1%}。",
        "因此四项血脂用于复现诊断规则，而非作为未病预警性能的主要证据。",
        "",
        "## 问题一结果",
        "",
        f"痰湿严重度模型的15折重复交叉验证平均 R²={q1_phlegm_perf['均值']:.3f}，说明附件中的血液与活动指标不具备可靠的折外痰湿积分解释能力；该终点没有指标通过折外重要性、相关强度和FDR联合门槛。",
        f"高血脂关联筛查严格排除TC、TG、LDL-C、HDL-C及其异常派生量，非诊断模型平均折外 AUC={q1_lipid_perf['均值']:.3f}；通过门槛的指标为 {q1['selected_lipid']}。",
        f"两终点共同关键指标为 {q1['shared']}。交集为空时不强制选出指标，因此不再把TC、TG的诊断同义复现表述为风险发现。",
        f"探索性多目标比较得到 Pareto 候选 {q1_pareto_names}，仅用于提示后续研究方向，不能替代验证性入选结论。",
        f"九体质主分析以体质标签为分类暴露、平和质为参照，调整年龄、性别、烟酒、BMI、活动总分、血糖和血尿酸后，整体似然比检验 P={q1['constitution_global_p']:.3f}。",
        f"痰湿质相对平和质的调整 OR={q1_phlegm_or['调整OR']:.3f}，95%CI [{q1_phlegm_or['95%CI下限']:.3f}, {q1_phlegm_or['95%CI上限']:.3f}]，P={q1_phlegm_or['P值']:.3f}，FDR q={q1_phlegm_or['FDR_q值']:.3f}。探索性点估计中{q1_point_top['体质类型']}居首（OR={q1_point_top['调整OR']:.3f}），其500次分层Bootstrap首位频率为{q1_point_top['位居首位频率']:.1%}；整体检验与区间证据仍不支持显著贡献首位。",
        "",
        "## 问题二结果",
        "",
        f"非血脂关联模型仅使用体质、活动、代谢和基础信息，训练集比较后选用 {q2['metrics']['selected_model']}；其独立测试 AUC={q2['metrics']['auc']:.3f}、AP={q2['metrics']['average_precision']:.3f}、Brier={q2['metrics']['brier']:.3f}。",
        "该概率只表示与当前血脂异常的横断面关联筛查分数。附件没有随访结局，不能将其解释为未来发病概率。",
        f"两个概率切点 {q2['thresholds'][0]:.4f} 与 {q2['thresholds'][1]:.4f} 仅由训练集折外概率搜索确定，并约束每层不少于训练样本的{PARAMS['q2_min_tier_fraction']:.0%}且患病率单调。独立测试集不参与模型或阈值选择。",
        f"浅层CART作为解释代理，最大深度{q2['metrics']['cart_surrogate_max_depth']}、最小叶节点比例{q2['metrics']['cart_surrogate_min_leaf_fraction']:.0%}；训练交叉验证宏平均F1={q2['metrics']['cart_surrogate_cv_macro_f1']:.3f}，测试集与统计层级一致率={q2['metrics']['cart_surrogate_test_fidelity']:.3f}。题面示例规则另列为临床管理提示。",
        f"独立测试集中低危与中危 Fisher 检验 P={q2['metrics']['fisher_p_low_vs_medium_statistical']:.3g}，中危与高危 P={q2['metrics']['fisher_p_medium_vs_high_statistical']:.3g}。三级仅称为筛查分数层级，不视为已前瞻验证的未来风险群。",
        "测试集三级筛查分数层级：",
        "",
        "|风险等级|人数|占比|实际患病率|95%CI|",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in q2_test.iterrows():
        lines.append(
            f"|{row['风险等级']}|{int(row['人数'])}|{row['占比']:.1%}|{row['实际患病率']:.1%}|"
            f"{row['患病率95%CI下限']:.1%}–{row['患病率95%CI上限']:.1%}|"
        )
    lines += [
        "",
        f"相对于统计高危层，排除诊断血脂后的痰湿体质核心组合首位为：{combo_text}。",
        "患者级结果分列当前诊断状态、非血脂筛查概率、唯一统计风险等级、CART代理路径和临床管理提示，避免把不同用途的结果混为同一口径。",
        "",
        "## 问题三结果",
        "",
        f"对附件中全部 {len(q3['summary'])} 名体质标签 5 患者完成动态规划。推荐方案采用 ε-约束：先求预算内最大可降幅，再选择达到其 {PARAMS['q3_target_effect_ratio']:.0%} 以上且成本最低的方案。",
        "样本1—3的最终推荐点均位于对应成本—积分降幅 Pareto 前沿，图中以星形标记。",
        "题目未给中医调理方式的独立疗效参数，基准模型只把其作为随积分分层的必选成本；额外疗效 0%/1%/2% 已作为敏感性场景而非基准事实。",
        "为明确患者特征—方案匹配规律，先按痰湿积分阈值与身体耐受约束形成九类确定性分层：最大耐受3级为年龄组1—2且活动总分≥60；最大耐受2级为年龄组1—2且40≤活动总分<60，或年龄组3—4且活动总分≥40；其余为最大耐受1级。",
        f"受限多输出CART仅解释首月活动强度与频率，选用深度{q3['cart_metrics']['max_depth']}、最小叶节点比例{q3['cart_metrics']['min_leaf_fraction']:.0%}、{q3['cart_metrics']['leaf_count']}个叶节点；五折联合处方一致率为{q3['cart_metrics']['cv_joint_fidelity']:.1%}，训练内一致率为{q3['cart_metrics']['train_joint_fidelity']:.1%}。该一致率不足以替代个体动态规划，因此硬约束和动态规划结果始终优先。",
        "",
        "|样本ID|真实初始积分|年龄组|活动总分|最终积分|降低分数|总成本|最大允许强度|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in q3_samples.iterrows():
        lines.append(
            f"|{int(row['样本ID'])}|{row['初始痰湿积分']:.1f}|{int(row['年龄组'])}|{row['活动量表总分']:.1f}|"
            f"{row['最终痰湿积分']:.1f}|{row['降低分数']:.1f}|{row['六个月总成本']:.0f}|{int(row['最大允许活动强度'])}|"
        )
    lines += [
        "",
        "九类患者特征—首月典型方案如下；带 * 的分层样本少于15人，结论仅作探索性参考。成本列分别为典型首月处方成本和该层六个月总成本中位数。",
        "",
        "|匹配分层|人数|首月调理等级|首月活动强度|每周次数|首月成本|六个月成本|积分降幅|降幅比例|处方覆盖率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in q3_matching.iterrows():
        marker = "*" if bool(row["低样本分层"]) else ""
        lines.append(
            f"|{row['匹配分层']}{marker}|{int(row['患者人数'])}|{int(row['典型首月调理等级'])}|"
            f"{int(row['典型首月活动强度'])}|{int(row['典型首月每周次数'])}|"
            f"{row['典型首月处方成本']:.0f}|{row['六个月成本中位数']:.0f}|"
            f"{row['降低分数中位数']:.1f}|{row['降低比例中位数']:.1%}|{row['典型首月处方覆盖率']:.1%}|"
        )
    lines += [
        "",
        "规则表用于首月快速初筛；患者完整六个月调理等级、活动强度、训练频率和月成本序列均由动态规划逐月输出。题面说明每周少于5次时积分基本稳定，因此最终有效方案均保持每周5—10次，不采用“4次/周仍有显著降幅”的示例性说法。",
        "逐月方案见 `results/question3_sample_1_2_3_monthly_plans.csv`，九类硬约束映射与CART叶节点规则分别保存在对应结果表中。",
        "",
        "## 灵敏度分析",
        "",
        "问题二报告训练集折外概率切点、CART参数交叉验证和独立测试层级表现；问题三分别扰动积分步长、活动效果、预算上限和调理额外月降幅。",
        "问题二阈值依据保留在 `results/question2_threshold_selection.csv`，问题三场景保留在 `results/question3_sensitivity.csv`。",
        "",
        "## 约束与一致性校验",
        "",
        "- 问题一允许共同指标为空；四项诊断血脂不会进入高血脂关联筛查模型。",
        "- 问题二模型不含诊断血脂及其派生变量；每名患者只输出一个统计风险等级，临床管理提示独立保存。",
        "- 问题三逐月回代年龄、活动评分、有效频率、预算与积分单调约束；278人均唯一进入九类分层和一个CART叶节点，样本1—3选择点均在Pareto前沿。",
        "- ID 1/2/3 均直接按附件样本 ID 读取，不再手工构造患者参数。",
        "- 所有论文引用数值均保存为 CSV/JSON，图表统一输出论文用矢量 PDF。",
        "",
        "## 可复现运行方式",
        "",
        "```powershell",
        "py -3.12 -m venv .venv312",
        ".\\.venv312\\Scripts\\python.exe -m pip install -r requirements-py312.txt",
        ".\\.venv312\\Scripts\\python.exe code\\中老年人群高血脂症建模完整代码.py",
        "```",
        "",
    ]
    (REPORT_DIR / "RESULTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_run_summary(
    font: str, quality: dict[str, Any], q1: dict[str, Any], q2: dict[str, Any],
    q3: dict[str, Any], elapsed: float,
) -> dict[str, Any]:
    return {
        "font": font,
        "elapsed_seconds": elapsed,
        "data_quality": quality,
        "question1_shared": q1["shared"],
        "question1_selected_phlegm": q1["selected_phlegm"],
        "question1_selected_lipid": q1["selected_lipid"],
        "question1_endpoint_performance": q1["endpoint_performance"].to_dict(orient="records"),
        "question1_constitution_global_p": q1["constitution_global_p"],
        "question1_phlegm_constitution_or": q1["constitution"].loc[q1["constitution"]["体质标签"].eq(5)].iloc[0].to_dict(),
        "question1_exploratory_pareto": q1["exploratory_pareto"].to_dict(orient="records"),
        "question1_constitution_rank_stability": q1["constitution_rank_stability"].to_dict(orient="records"),
        "question2_metrics": q2["metrics"],
        "question2_threshold_selection": q2["threshold_selection"].to_dict(orient="records"),
        "question2_cart_rules": q2["cart_rules"].to_dict(orient="records"),
        "question2_test_tiers": q2["test_tiers"].to_dict(orient="records"),
        "question3_samples": q3["summary"][q3["summary"]["样本ID"].isin([1, 2, 3])].to_dict(orient="records"),
        "question3_matching_rules": q3["matching"].to_dict(orient="records"),
        "question3_cart_metrics": q3["cart_metrics"],
        "question3_cart_rules": q3["cart_rules"].to_dict(orient="records"),
        "question3_all_feasible": True,
        "question3_selected_on_frontier": q3["selected_on_frontier"],
    }


def validate_results(
    df: pd.DataFrame,
    quality: dict[str, Any],
    q1: dict[str, Any],
    q2: dict[str, Any],
    q3: dict[str, Any],
) -> dict[str, Any]:
    """从内存结果独立回代三问的关键口径，失败时立即抛出断言。"""
    checks: list[str] = []

    # 问题一：标签一致性、泄漏排除、折外证据和分类体质 OR。
    assert quality["rows"] == 1000
    assert quality["duplicate_sample_ids"] == 0
    assert quality["missing_values"] == 0
    assert quality["diagnostic_rule_agreement"] == 1.0
    diagnostic_features = {
        "总胆固醇_TC", "甘油三酯_TG", "低密度脂蛋白_LDL_C", "高密度脂蛋白_HDL_C",
    }
    lipid_rows = q1["screen"].loc[q1["screen"]["终点"].eq("高血脂关联筛查")]
    flagged = set(lipid_rows.loc[lipid_rows["是否诊断泄漏"], "指标名称"])
    assert flagged == diagnostic_features
    assert not lipid_rows.loc[lipid_rows["是否诊断泄漏"], "是否入选"].any()
    assert q1["selected_phlegm"] == []
    assert q1["selected_lipid"] == ["血尿酸"]
    assert q1["shared"] == []
    phlegm_r2 = float(q1["endpoint_performance"].loc[
        q1["endpoint_performance"]["终点"].eq("痰湿严重度"), "均值"
    ].iloc[0])
    lipid_auc = float(q1["endpoint_performance"].loc[
        q1["endpoint_performance"]["终点"].eq("高血脂关联筛查"), "均值"
    ].iloc[0])
    assert phlegm_r2 <= 0
    assert np.isclose(phlegm_r2, -0.0563766472, atol=2e-6)
    assert np.isclose(lipid_auc, 0.8479757691, atol=2e-6)
    constitution = q1["constitution"]
    assert len(constitution) == 9
    assert constitution.loc[constitution["体质标签"].eq(1), "参照组"].iloc[0] == "平和质"
    assert constitution.loc[constitution["体质标签"].eq(1), "调整OR"].iloc[0] == 1.0
    assert constitution.loc[constitution["体质标签"].ne(1), "FDR_q值"].between(0, 1).all()
    assert constitution["整体LR检验P值"].nunique() == 1
    assert q1["exploratory_pareto"]["探索性Pareto候选"].any()
    assert constitution.loc[constitution["体质标签"].ne(1), "OR点估计名次"].notna().all()
    assert constitution.loc[constitution["体质标签"].ne(1), "Bootstrap中位名次"].notna().all()
    assert np.isclose(
        constitution.loc[constitution["体质标签"].ne(1), "位居首位频率"].sum(), 1.0, atol=1e-9
    )
    assert int(q1["constitution_bootstrap_repeats"]) == 500
    assert not set(q1["exploratory_pareto"]["指标名称"]) & diagnostic_features
    checks.append("问题一：泄漏排除、折外筛选、探索性Pareto及体质Bootstrap排名通过")

    # 问题二：模型特征无诊断血脂；训练内定切点并把统计等级与临床提示分列。
    leakage_tokens = (
        "总胆固醇", "甘油三酯", "低密度脂蛋白", "高密度脂蛋白",
        "TC异常", "TG异常", "LDL异常", "HDL异常", "任一血脂异常",
    )
    model_features = q2["metrics"]["model_features"]
    assert all(not any(token in feature for token in leakage_tokens) for feature in model_features)
    assert np.isclose(q2["metrics"]["auc"], 0.8211574953, atol=2e-6)
    assert q2["metrics"]["threshold_selection_dataset"] == "训练集折外概率"
    assert not q2["metrics"]["statistical_tier_is_rule_based"]
    assert q2["metrics"]["clinical_prompt_is_separate"]
    assert len(q2["thresholds"]) == 2 and q2["thresholds"][0] < q2["thresholds"][1]
    assert q2["test_tiers"]["人数"].min() > 0
    assert np.all(np.diff(q2["test_tiers"].sort_values("风险等级编码")["实际患病率"]) >= -1e-12)

    predictions = q2["test_predictions"].reset_index(drop=True)
    assert predictions["样本ID"].is_unique
    assert predictions["非血脂筛查概率"].between(0, 1).all()
    expected_code = np.digitize(
        predictions["非血脂筛查概率"].to_numpy(float), q2["thresholds"], right=False
    ) + 1
    assert np.array_equal(predictions["统计风险等级编码"].to_numpy(int), expected_code)
    assert predictions["统计风险等级"].notna().all()
    assert predictions["CART代理路径"].str.len().gt(0).all()
    assert predictions["临床管理提示"].notna().all()
    assert predictions["临床管理依据"].notna().all()
    assert q2["cart_rules"]["叶节点规则"].str.len().gt(0).all()
    assert len(q2["patient_results"]) == len(df)
    assert q2["patient_results"]["样本ID"].is_unique
    assert set(q2["patient_results"]["概率来源"]) == {"训练集折外", "独立测试集"}
    assert not q2["combinations"]["核心特征组合"].str.contains("血脂", na=False).any()
    checks.append("问题二：训练内切点、唯一统计等级、CART代理及独立临床提示通过")

    # 问题三：全体方案约束通过，并锁定论文引用的样本1—3方案。
    assert q3["all_plans_feasible"]
    summary = q3["summary"]
    assert len(summary) == quality["phlegm_constitution_count"]
    assert summary["样本ID"].is_unique
    assert summary["六个月总成本"].le(PARAMS["q3_max_cost"]).all()
    assert summary["最终痰湿积分"].le(summary["初始痰湿积分"] + 1e-9).all()
    required_q3_columns = {
        "匹配分层", "首月调理等级", "首月调理方式", "首月总成本",
        "调理等级序列", "活动强度序列", "训练频率序列", "月成本序列",
        "CART规则叶节点", "CART规则路径", "CART代理首月活动强度",
        "CART代理首月每周次数", "CART规则一致标记",
    }
    assert required_q3_columns <= set(summary.columns)
    expected_strata = summary.apply(
        lambda row: patient_rule_stratum(row["年龄组"], row["活动量表总分"], row["初始痰湿积分"]),
        axis=1,
    )
    assert summary["匹配分层"].equals(expected_strata)
    assert summary["匹配分层"].nunique() == 9
    for sequence_column in ("调理等级序列", "活动强度序列", "训练频率序列", "月成本序列"):
        assert summary[sequence_column].str.split("-").str.len().eq(PARAMS["q3_months"]).all()
    frequency_values = summary["训练频率序列"].str.split("-").explode().astype(int)
    assert frequency_values.between(5, 10).all()
    intensity_values = summary["活动强度序列"].str.split("-")
    assert all(max(map(int, values)) <= allowed for values, allowed in zip(
        intensity_values, summary["最大允许活动强度"]
    ))
    cost_values = summary["月成本序列"].str.split("-").apply(lambda values: sum(map(int, values)))
    assert np.array_equal(cost_values.to_numpy(int), summary["六个月总成本"].to_numpy(int))
    assert np.array_equal(
        summary["月成本序列"].str.split("-").str[0].astype(int).to_numpy(),
        summary["首月总成本"].to_numpy(int),
    )
    assert len(q3["matching"]) == 9
    assert q3["matching"]["患者人数"].sum() == len(summary)
    assert q3["matching"]["典型首月每周次数"].between(5, 10).all()
    assert len(q3["cart_candidates"]) == 45
    assert q3["cart_candidates"]["是否选用"].sum() == 1
    assert q3["cart_candidates"]["五折耐受约束通过"].all()
    assert q3["cart_rules"]["患者人数"].sum() == len(summary)
    assert summary["CART规则叶节点"].notna().all()
    assert summary["CART规则路径"].str.len().gt(0).all()
    assert q3["cart_metrics"]["all_cv_feasible"]
    assert np.isclose(q3["cart_metrics"]["cv_joint_fidelity"], 0.7846103896, atol=2e-6)
    assert np.isclose(q3["cart_metrics"]["train_joint_fidelity"], 0.8309352518, atol=2e-6)
    expected_samples = {1: (48.5, 1014), 2: (37.0, 1240), 3: (33.0, 1674)}
    for sample_id, (expected_score, expected_cost) in expected_samples.items():
        row = summary.loc[summary["样本ID"].eq(sample_id)].iloc[0]
        assert np.isclose(row["最终痰湿积分"], expected_score, atol=1e-9)
        assert int(row["六个月总成本"]) == expected_cost
        selected = q3["pareto_123"].loc[
            q3["pareto_123"]["样本ID"].eq(sample_id)
            & np.isclose(q3["pareto_123"]["总成本"], expected_cost)
            & np.isclose(q3["pareto_123"]["最终积分"], expected_score)
        ]
        assert len(selected) == 1
    assert all(q3["selected_on_frontier"].values())
    checks.append("问题三：九类分层、CART代理、全体可行性与样本1—3 Pareto选择点通过")

    return {"status": "PASS", "check_count": len(checks), "checks": checks}


# ============================================================================
# 第六部分：主运行入口
# ============================================================================


def run_all() -> dict[str, Any]:
    start = time.perf_counter()
    ensure_dirs()
    font = configure_plotting()
    print(f"绘图字体：{font}")
    df, quality = load_and_preprocess()
    q1 = run_problem1(df)
    q2 = run_problem2(df)
    q3 = run_problem3(df)
    elapsed = time.perf_counter() - start
    summary = build_run_summary(font, quality, q1, q2, q3, elapsed)
    validation = validate_results(df, quality, q1, q2, q3)
    summary["validation"] = validation
    write_json(OUTPUT_DIR / "run_summary.json", summary)
    write_json(OUTPUT_DIR / "validation_summary.json", validation)
    write_results_report(quality, q1, q2, q3, elapsed)
    print(f"自动验收通过：{validation['check_count']} 组核心检查全部 PASS。")
    print(f"全部求解完成，耗时 {elapsed:.1f} 秒。")
    return {
        "df": df, "quality": quality, "q1": q1, "q2": q2, "q3": q3,
        "summary": summary, "validation": validation,
    }


if __name__ == "__main__":
    run_all()
