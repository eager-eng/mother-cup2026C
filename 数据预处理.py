import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ===================== 1. 配置与路径设置 =====================
# 基于脚本所在目录定位文件，兼容 Windows 且不依赖启动目录
base_dir = Path(__file__).resolve().parent
input_file_path = base_dir / '题目材料' / '2601(12日晚上18：00）' / '附件1：样例数据.xlsx'
output_dir = base_dir / '预处理结果'
output_dir.mkdir(parents=True, exist_ok=True)

# 处理后数据保存路径
output_csv_path = output_dir / '高血脂症数据_预处理完成.csv'
output_excel_path = output_dir / '高血脂症数据_预处理完成.xlsx'

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 数据读取 =====================
# 读取原始数据
df_raw = pd.read_excel(input_file_path, sheet_name='Sheet1')
df_processed = df_raw.copy()

print("===================== 原始数据基础信息 =====================")
print(f"原始数据总行数：{df_processed.shape[0]}，总列数：{df_processed.shape[1]}")
print(f"原始数据缺失值总数：{df_processed.isnull().sum().sum()}")

# ===================== 3. 异常值处理 =====================
print("\n===================== 异常值处理 =====================")
# 3.1 定义各指标的临床合理范围（来源：权威临床指南）
clinical_range = {
    'TC（总胆固醇）': (0, 20),  # mmol/L，临床极端上限20
    'TG（甘油三酯）': (0, 20),  # mmol/L，临床极端上限20
    'HDL-C（高密度脂蛋白）': (0, 5),  # mmol/L
    'LDL-C（低密度脂蛋白）': (0, 20),  # mmol/L
    '空腹血糖': (0, 11.1),  # mmol/L，糖尿病诊断上限11.1
    '血尿酸': (0, 800),  # μmol/L，临床极端上限800
    'BMI': (10, 40),  # kg/m²，临床合理范围10-40
    '痰湿质': (0, 100),
    '平和质': (0, 100),
    '气虚质': (0, 100),
    '阳虚质': (0, 100),
    '阴虚质': (0, 100),
    '湿热质': (0, 100),
    '血瘀质': (0, 100),
    '气郁质': (0, 100),
    '特禀质': (0, 100),
    'ADL总分': (0, 50),
    'IADL总分': (0, 50),
    '活动量表总分（ADL总分+IADL总分）': (0, 100)
}

# 3.2 IQR法异常值检测与处理
numeric_cols = [col for col in df_processed.columns if col in clinical_range.keys()]
outlier_info = []

for col in numeric_cols:
    # 1. 临床范围校验
    min_val, max_val = clinical_range[col]
    clinical_outlier = df_processed[(df_processed[col] < min_val) | (df_processed[col] > max_val)].shape[0]
    
    # 2. IQR法检测
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outlier = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].shape[0]
    
    # 3. 合并异常值标记
    df_processed[f'{col}_是否异常'] = np.where(
        (df_processed[col] < min_val) | (df_processed[col] > max_val) | 
        (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound),
        1, 0
    )
    
    # 4. 异常值填充：用该列中位数替换
    outlier_mask = df_processed[f'{col}_是否异常'] == 1
    if outlier_mask.sum() > 0:
        fill_median = df_processed[col].median()
        df_processed.loc[outlier_mask, col] = fill_median
        outlier_info.append({
            '特征列': col,
            '临床范围异常数': clinical_outlier,
            'IQR法异常数': iqr_outlier,
            '总异常数': outlier_mask.sum(),
            '填充值（中位数）': round(fill_median, 4)
        })

# 输出异常值处理结果
outlier_df = pd.DataFrame(outlier_info)
print("异常值处理详情：")
print(outlier_df)
print(f"\n总异常值处理数量：{outlier_df['总异常数'].sum()}")

# ===================== 4. 特征工程与转换 =====================
print("\n===================== 特征工程与转换 =====================")
# 4.1 体质相关特征
# 偏颇体质积分总和（平和质为正常，其余8种为偏颇体质）
biased_constitution_cols = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
df_processed['偏颇体质总分'] = df_processed[biased_constitution_cols].sum(axis=1)
# 痰湿质标记（体质标签=5为痰湿质）
df_processed['是否痰湿质'] = np.where(df_processed['体质标签'] == 5, 1, 0)

# 4.2 血脂相关特征
# 致动脉粥样硬化指数（AIP）= log(TG/HDL-C)，心血管风险核心指标
df_processed['致动脉粥样硬化指数(AIP)'] = np.log(df_processed['TG（甘油三酯）'] / df_processed['HDL-C（高密度脂蛋白）'])
# 血脂异常分级（已存在，保留）
df_processed['血脂异常分级'] = df_processed['血脂异常分型标签（确诊病例）']

# 4.3 活动能力相关特征
# 活动能力分级：0=正常(61-100)、1=轻度受损(41-60)、2=中度受损(21-40)、3=重度受损(0-20)
df_processed['活动能力分级'] = pd.cut(
    df_processed['活动量表总分（ADL总分+IADL总分）'],
    bins=[0, 20, 40, 60, 100],
    labels=[3, 2, 1, 0],
    right=False
).astype(int)

# 4.4 标准化处理（Z-score标准化）
# 需标准化的数值特征列
standardize_cols = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
    'TC（总胆固醇）', 'TG（甘油三酯）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'BMI', '偏颇体质总分', '致动脉粥样硬化指数(AIP)'
]

# 标准化处理
for col in standardize_cols:
    df_processed[f'{col}_标准化'] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()

print(f"已完成{len(standardize_cols)}个数值特征的Z-score标准化")
print(f"已新增特征列：偏颇体质总分、是否痰湿质、致动脉粥样硬化指数(AIP)、活动能力分级，以及所有标准化列")

# ===================== 5. 预处理后数据可视化 =====================
print("\n===================== 预处理后数据可视化 =====================")
# 5.1 预处理后核心指标分布箱线图
plt.figure(figsize=(16, 10))
# 选取核心指标
core_numeric_cols = [
    '痰湿质', 'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）',
    '活动量表总分（ADL总分+IADL总分）', 'BMI', '空腹血糖'
]
sns.boxplot(data=df_processed[core_numeric_cols], orient='h')
plt.title('预处理后核心数值指标分布箱线图', fontsize=16)
plt.xlabel('数值范围', fontsize=12)
plt.ylabel('特征列', fontsize=12)
plt.tight_layout()
processed_boxplot_path = output_dir / '预处理后核心指标箱线图.png'
plt.savefig(processed_boxplot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"预处理后核心指标箱线图已保存至：{processed_boxplot_path}")

# 5.2 预处理后核心指标相关性热力图
plt.figure(figsize=(12, 10))
core_cols = [
    '痰湿质', 'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）',
    'HDL-C（高密度脂蛋白）', '活动量表总分（ADL总分+IADL总分）',
    'BMI', '空腹血糖', '血尿酸', '高血脂症二分类标签', '致动脉粥样硬化指数(AIP)'
]
corr_matrix = df_processed[core_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('预处理后核心指标相关性热力图', fontsize=16)
plt.tight_layout()
processed_corr_path = output_dir / '预处理后核心指标相关性热力图.png'
plt.savefig(processed_corr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"预处理后核心指标相关性热力图已保存至：{processed_corr_path}")

# ===================== 6. 数据保存 =====================
print("\n===================== 数据保存 =====================")
# 保存为CSV格式（通用格式）
df_processed.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
# 保存为Excel格式
df_processed.to_excel(output_excel_path, index=False)

print(f"预处理完成数据已保存至：")
print(f"CSV格式：{output_csv_path}")
print(f"Excel格式：{output_excel_path}")

# ===================== 7. 数据完整性验证 =====================
print("\n===================== 数据完整性验证 =====================")
print(f"处理后数据总行数：{df_processed.shape[0]}，总列数：{df_processed.shape[1]}")
print(f"处理后数据缺失值总数：{df_processed.isnull().sum().sum()}")
print(f"处理后异常值标记列总数：{len([col for col in df_processed.columns if '_是否异常' in col])}")
print(f"处理后标准化列总数：{len([col for col in df_processed.columns if '_标准化' in col])}")
