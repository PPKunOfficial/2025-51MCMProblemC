import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)  # 导入评估指标
from scipy.stats import randint, uniform  # 用于定义随机搜索的参数分布
import warnings
import os  # 引入os模块检查文件是否存在

# 忽略可能的警告
warnings.filterwarnings("ignore")

# --- 配置 ---
file_path = "program/a1.csv"  # 确保文件路径正确
prediction_date = datetime(2024, 7, 21)
historical_start_date = datetime(2024, 7, 11)
historical_end_date = datetime(2024, 7, 20)
lookback_days = 3  # 使用前 k 天的数据作为特征，这里 k=3

# --- 检查文件是否存在 ---
if not os.path.exists(file_path):
    print(f"错误：文件未找到 - {file_path}")
    print("请确保 '附件1.csv' 文件与脚本在同一目录下。")
    exit()

# --- 1. 数据加载与预处理 ---
print(f"正在加载数据: {file_path}...")
try:
    # 尝试更高效的读取方式，指定dtype
    dtype_spec = {
        "UserID": "category",
        "UserBehaviour": "int8",
        "BloggerID": "category",
        "Time": "object",  # 先读object，再转datetime
    }
    df = pd.read_csv(file_path, dtype=dtype_spec, low_memory=False)
    print("数据加载完成.")

    # 转换时间列
    print("正在转换时间格式...")
    df["Time"] = pd.to_datetime(df["Time"])
    df["Date"] = df["Time"].dt.date
    print("时间格式转换完成.")

    # 过滤历史数据范围
    print(
        f"正在过滤数据范围至 {historical_start_date.date()} - {historical_end_date.date()}..."
    )
    df_history = df[
        (df["Date"] >= historical_start_date.date())
        & (df["Date"] <= historical_end_date.date())
    ].copy()
    print(f"过滤完成，剩余 {len(df_history)} 条数据.")

    # 释放原始大DataFrame内存
    del df

except Exception as e:
    print(f"数据加载或初步处理失败: {e}")
    exit()  # 如果数据加载失败，直接退出

# 获取所有独特的博主ID，后续预测需要为所有这些博主进行
all_blogger_ids = df_history["BloggerID"].unique()
print(f"共有 {len(all_blogger_ids)} 位博主在历史数据中出现.")

# --- 2. 计算每日互动计数 ---
print("正在计算每日博主互动计数...")
# 使用pivot_table更方便地获取各种行为的每日计数
daily_interactions = (
    pd.pivot_table(
        df_history,
        values="UserID",  # 任意一个值列即可，我们只关心计数
        index=["BloggerID", "Date"],
        columns="UserBehaviour",
        aggfunc="count",
        fill_value=0,  # 关键：用0填充没有发生某种行为的日期/博主组合
    )
    .rename(
        columns={
            1: "watch_count",
            2: "like_count",
            3: "comment_count",
            4: "follow_count",
        }
    )
    .reset_index()
)

# 确保所有博主在所有历史日期都有记录，即使计数为0
# 创建所有博主和历史日期的组合
all_dates_in_history = pd.date_range(
    start=historical_start_date.date(), end=historical_end_date.date(), freq="D"
).date
all_blogger_date_combinations = pd.MultiIndex.from_product(
    [all_blogger_ids, all_dates_in_history], names=["BloggerID", "Date"]
)
daily_interactions = (
    daily_interactions.set_index(["BloggerID", "Date"])
    .reindex(all_blogger_date_combinations, fill_value=0)
    .reset_index()
)

print("每日互动计数计算完成.")
# print(daily_interactions.head())

# --- 3. 构建训练数据集 ---
print(f"正在构建训练数据集 (使用前 {lookback_days} 天数据预测当天)...")

X_train = []
y_train = []
train_samples_meta = []  # 存储样本对应的博主和日期

# 训练的目标日期范围：从 historical_start_date + lookback_days 到 historical_end_date
train_start_date = historical_start_date + timedelta(days=lookback_days)
train_end_date = historical_end_date

# 确保训练数据日期范围有效
if train_start_date > historical_end_date:
    print(
        f"错误：历史数据范围 ({historical_start_date.date()} - {historical_end_date.date()}) 太短，无法构建训练集 (需要至少 {lookback_days + 1} 天)."
    )
    exit()

current_train_date = train_start_date
while current_train_date <= train_end_date:
    lookback_end = current_train_date - timedelta(days=1)
    lookback_start = current_train_date - timedelta(days=lookback_days)

    # 提取当前训练日期的目标数据
    target_day_data = daily_interactions[
        daily_interactions["Date"] == current_train_date.date()
    ]

    # 提取当前训练日期的特征数据（前 lookback_days 的汇总）
    feature_window_data = daily_interactions[
        (daily_interactions["Date"] >= lookback_start.date())
        & (daily_interactions["Date"] <= lookback_end.date())
    ]

    # 按博主ID汇总特征窗口内的数据
    features_aggregated = (
        feature_window_data.groupby("BloggerID")[
            ["watch_count", "like_count", "comment_count", "follow_count"]
        ]
        .sum()
        .reset_index()
    )
    features_aggregated.rename(
        columns={
            "watch_count": "sum_watch_prev",
            "like_count": "sum_like_prev",
            "comment_count": "sum_comment_prev",
            "follow_count": "sum_follow_prev",
        },
        inplace=True,
    )

    # 迭代所有博主，为每个博主在当前训练日期构建样本
    for blogger_id in all_blogger_ids:
        # 获取特征 (处理可能没有数据的博主)
        blogger_features = features_aggregated[
            features_aggregated["BloggerID"] == blogger_id
        ]
        if blogger_features.empty:
            features_row = [0, 0, 0, 0]  # 如果前lookback_days没有数据，特征为0
        else:
            features_row = (
                blogger_features[
                    [
                        "sum_watch_prev",
                        "sum_like_prev",
                        "sum_comment_prev",
                        "sum_follow_prev",
                    ]
                ]
                .iloc[0]
                .tolist()
            )

        # 获取目标 (处理当天没有关注的博主)
        blogger_target = target_day_data[target_day_data["BloggerID"] == blogger_id]
        target_value = (
            blogger_target["follow_count"].iloc[0] if not blogger_target.empty else 0
        )  # 如果当天没数据，目标为0

        X_train.append(features_row)
        y_train.append(target_value)
        train_samples_meta.append(
            (blogger_id, current_train_date.date())
        )  # 记录样本元数据

    current_train_date += timedelta(days=1)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"训练数据集构建完成，共 {len(X_train)} 个样本.")
# print("X_train sample:", X_train[:5])
# print("y_train sample:", y_train[:5])

# --- 4. 参数调优 ---
print("\n正在进行参数调优 (RandomizedSearchCV)...")

# 定义参数搜索空间
param_distributions = {
    "n_estimators": randint(50, 301),  # 树的数量在 50 到 300 之间随机整数
    "max_depth": [None]
    + list(
        randint(10, 51).rvs(size=20)
    ),  # 最大深度可以是 None，或者 10 到 50 之间的一些随机整数 (减少样本数量)
    "min_samples_split": randint(2, 21),  # 分裂所需的最小样本数在 2 到 20 之间随机整数
    "min_samples_leaf": randint(1, 11),  # 叶节点所需的最小样本数在 1 到 10 之间随机整数
    "max_features": ["sqrt", "log2", 1.0],  # 寻找最佳分裂时考虑的特征数量
}

# 交叉验证策略
# 使用 KFold with shuffle。cv=5 表示 5折交叉验证。
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化 RandomizedSearchCV
# n_iter: 随机采样的参数组合数量。这里设置为 50 次尝试，可以根据计算资源调整。
# scoring: 使用负平均绝对误差作为评估指标，越高越好 (因为是负的误差)。
# random_state: 保证结果可复现。
# n_jobs=-1: 利用所有核心进行并行计算。
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),  # 传入一个基础模型实例
    param_distributions=param_distributions,
    n_iter=50,  # 尝试 50 个不同的参数组合，这个值可以根据计算资源调整
    cv=cv_strategy,
    scoring="neg_mean_absolute_error",
    verbose=1,  # 打印进度信息
    random_state=42,
    n_jobs=-1,
)

# 在训练数据上运行随机搜索
random_search.fit(X_train, y_train)

print("\n参数调优完成.")
print("最佳参数组合:", random_search.best_params_)
print("最佳交叉验证得分 (负MAE):", random_search.best_score_)  # 这是负MAE，值越大越好

# --- 5. 使用最优参数训练最终模型 ---
print("\n正在使用最优参数训练最终模型...")
final_model = random_search.best_estimator_  # 获取带有最佳参数的训练好的模型实例
# 注意：best_estimator_ 已经在训练数据上拟合过了，可以直接用于预测

print("最终模型训练完成 (使用最佳参数).")

# --- 6. 模型在训练集上的评价 ---
print("\n正在评价最终模型在训练集上的性能...")

# 使用训练好的模型对训练集进行预测
y_train_pred = final_model.predict(X_train)

# 确保预测结果非负并四舍五入取整，以便与真实值对比
y_train_pred_rounded = np.round(y_train_pred).astype(int)
y_train_pred_rounded[y_train_pred_rounded < 0] = 0

# 计算评估指标
mae_train = mean_absolute_error(y_train, y_train_pred_rounded)
mse_train = mean_squared_error(y_train, y_train_pred_rounded)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred_rounded)

print(f"训练集评估结果:")
print(f"  平均绝对误差 (MAE): {mae_train:.4f}")
print(f"  均方误差 (MSE): {mse_train:.4f}")
print(f"  均方根误差 (RMSE): {rmse_train:.4f}")
print(f"  决定系数 (R²): {r2_train:.4f}")

print("模型评价完成.")

# --- 7. 构建预测数据集 ---
print(
    f"\n正在构建预测数据集 (使用 {prediction_date.date()-timedelta(days=lookback_days)} - {prediction_date.date()-timedelta(days=1)} 数据预测 {prediction_date.date()})..."
)

# 预测窗口： 7.18 到 7.20 (如果 lookback_days=3)
prediction_feature_start_date = prediction_date - timedelta(days=lookback_days)
prediction_feature_end_date = prediction_date - timedelta(days=1)

# 确保预测特征窗口在历史数据范围内
if (
    prediction_feature_start_date.date() < historical_start_date.date()
    or prediction_feature_end_date.date() > historical_end_date.date()
):
    print(
        f"错误：预测特征窗口 ({prediction_feature_start_date.date()} - {prediction_feature_end_date.date()}) 超出历史数据范围 ({historical_start_date.date()} - {historical_end_date.date()})."
    )
    exit()


# 提取预测日期的特征数据（前 lookback_days 的汇总）
prediction_feature_window_data = daily_interactions[
    (daily_interactions["Date"] >= prediction_feature_start_date.date())
    & (daily_interactions["Date"] <= prediction_feature_end_date.date())
]

X_pred = []
predict_blogger_ids = []

# 确保为所有在历史数据中出现过的博主生成预测特征
for blogger_id in all_blogger_ids:
    blogger_features = prediction_feature_window_data[
        prediction_feature_window_data["BloggerID"] == blogger_id
    ]

    if blogger_features.empty:
        features_row = [0, 0, 0, 0]  # 如果前lookback_days没有数据，特征为0
    else:
        # 汇总前 lookback_days 的数据
        sum_features = (
            blogger_features[
                ["watch_count", "like_count", "comment_count", "follow_count"]
            ]
            .sum()
            .tolist()
        )
        features_row = sum_features

    X_pred.append(features_row)
    predict_blogger_ids.append(blogger_id)

X_pred = np.array(X_pred)

print(f"预测数据集构建完成，共 {len(X_pred)} 个样本.")
# print("X_pred sample:", X_pred[:5])

# --- 8. 进行预测 ---
print("正在使用最终模型进行预测...")
predictions = final_model.predict(X_pred)
print("预测完成.")

# --- 9. 后处理与结果输出 ---
print("正在处理预测结果并生成排名...")

# 确保预测结果非负并四舍五入取整
predicted_follows = np.round(predictions).astype(int)
predicted_follows[predicted_follows < 0] = 0

# 构建结果DataFrame
results_df = pd.DataFrame(
    {"BloggerID": predict_blogger_ids, "PredictedFollows_20240721": predicted_follows}
)

# 按预测关注数降序排序
results_df = results_df.sort_values(by="PredictedFollows_20240721", ascending=False)

# 选取前5位博主
top_5_bloggers = results_df.head(5)

print("\n预测完成，2024.7.21 当日新增关注数最多的5位博主：")
# 按照表1格式输出
print("表1: 2024.7.21 当日新增关注数最多的5位博主")
print("-" * 40)
# 使用 to_markdown 或 to_string 打印为表格格式
try:
    # 如果安装了 tabulate 库
    from tabulate import tabulate

    print(tabulate(top_5_bloggers, headers="keys", tablefmt="github", showindex=False))
except ImportError:
    # 否则使用 to_string
    print(
        top_5_bloggers.rename(
            columns={"PredictedFollows_20240721": "新增关注数"}
        ).to_string(index=False)
    )

print("-" * 40)

# 可选：保存完整预测结果
# results_df.to_csv('predicted_follows_20240721_all_bloggers.csv', index=False)
# print("\n所有博主的预测结果已保存至 'predicted_follows_20240721_all_bloggers.csv'")
