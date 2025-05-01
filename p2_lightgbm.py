import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm.notebook import tqdm  # 使用tqdm显示进度条
import gc # 垃圾回收

# --- 1. 数据加载与预处理 ---
print("Loading data...")
# 为了节省内存，可以指定部分列的dtype，特别是ID类和行为类别
dtype_spec = {
    'UserID': 'category',
    'UserBehaviour': 'int8',
    'BloggerID': 'category',
    'Time': 'str' # 先读作字符串，再转换
}
# 注意：文件路径需要根据实际情况修改
df_hist = pd.read_csv('program/a1.csv', dtype=dtype_spec)
df_future_interactions = pd.read_csv('program/a2.csv', dtype=dtype_spec)

print("Preprocessing data...")
# 转换时间列
df_hist['Time'] = pd.to_datetime(df_hist['Time'])
df_future_interactions['Time'] = pd.to_datetime(df_future_interactions['Time'])

# 提取日期方便筛选
df_hist['Date'] = df_hist['Time'].dt.date
df_future_interactions['Date'] = df_future_interactions['Time'].dt.date

# 将日期转换为可比较的格式
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
df_future_interactions['Date'] = pd.to_datetime(df_future_interactions['Date'])

# 设定关键日期
train_end_date = pd.to_datetime('2024-07-19')
label_date = pd.to_datetime('2024-07-20')
predict_date = pd.to_datetime('2024-07-22')
hist_end_date_for_predict = pd.to_datetime('2024-07-20') # 用于预测时，历史数据截止到20号

# 为了处理类别特征，可以先获取所有用户和博主的列表
all_users = pd.concat([df_hist['UserID'], df_future_interactions['UserID']]).astype(str).unique()
all_bloggers = pd.concat([df_hist['BloggerID'], df_future_interactions['BloggerID']]).astype(str).unique()

# 将 UserID 和 BloggerID 转换为数值编码可能更方便处理，但LightGBM也能直接处理category类型
# 这里我们保留category类型，LightGBM可以处理
df_hist['UserID'] = df_hist['UserID'].astype('category')
df_hist['BloggerID'] = df_hist['BloggerID'].astype('category')
df_future_interactions['UserID'] = df_future_interactions['UserID'].astype('category')
df_future_interactions['BloggerID'] = df_future_interactions['BloggerID'].astype('category')

print("Data loaded and preprocessed.")
gc.collect() # 回收内存

# --- 2. 特征工程函数 ---
# 定义一个函数计算特征，避免代码重复
# reference_date: 计算历史特征的截止日期 (不包含)
# interaction_data: 用于计算近期特征的数据 (通常是 reference_date 的后一天 或 预测当天)
def create_features(target_pairs_df, historical_data, interaction_data, reference_date):
    """
    为给定的 (UserID, BloggerID) 对创建特征.

    Args:
        target_pairs_df (pd.DataFrame): 包含 'UserID' 和 'BloggerID' 列的 DataFrame.
        historical_data (pd.DataFrame): 用于计算历史特征的数据 (时间 <= reference_date).
        interaction_data (pd.DataFrame): 用于计算当天互动特征的数据 (通常是 reference_date + 1天 或 预测当天).
        reference_date (pd.Timestamp): 历史数据的截止日期 (不包含).

    Returns:
        pd.DataFrame: 包含特征的 DataFrame.
    """
    print(f"Creating features based on data up to {reference_date}...")
    features_list = []
    
    # 筛选历史数据
    hist_ref = historical_data[historical_data['Date'] <= reference_date].copy()
    
    # --- 计算全局/用户/博主级别的聚合特征 (只需要计算一次) ---
    print("Calculating global/user/blogger features...")
    # 用户历史特征
    user_agg = hist_ref.groupby('UserID').agg(
        user_hist_interactions=('BloggerID', 'count'),
        user_hist_unique_bloggers=('BloggerID', 'nunique'),
        user_hist_follows=('UserBehaviour', lambda x: (x == 4).sum()),
        user_hist_views=('UserBehaviour', lambda x: (x == 1).sum()),
        user_hist_likes=('UserBehaviour', lambda x: (x == 2).sum()),
        user_hist_comments=('UserBehaviour', lambda x: (x == 3).sum()),
        user_active_days=('Date', 'nunique'),
        user_last_active_date=('Date', 'max')
    ).reset_index()
    user_agg['user_follow_rate'] = user_agg['user_hist_follows'] / user_agg['user_hist_interactions'].replace(0, 1)
    user_agg['user_days_since_last_active'] = (reference_date - user_agg['user_last_active_date']).dt.days

    # 博主历史特征
    blogger_agg = hist_ref.groupby('BloggerID').agg(
        blogger_hist_interactions=('UserID', 'count'),
        blogger_hist_unique_users=('UserID', 'nunique'),
        blogger_hist_follows=('UserBehaviour', lambda x: (x == 4).sum()),
        blogger_hist_views=('UserBehaviour', lambda x: (x == 1).sum()),
        blogger_hist_likes=('UserBehaviour', lambda x: (x == 2).sum()),
        blogger_hist_comments=('UserBehaviour', lambda x: (x == 3).sum()),
        blogger_active_days=('Date', 'nunique'),
        blogger_last_active_date=('Date', 'max')
    ).reset_index()
    blogger_agg['blogger_follow_rate'] = blogger_agg['blogger_hist_follows'] / blogger_agg['blogger_hist_interactions'].replace(0, 1)
    blogger_agg['blogger_days_since_last_active'] = (reference_date - blogger_agg['blogger_last_active_date']).dt.days
    
    gc.collect()

    # --- 计算用户-博主交互特征 & 当天互动特征 ---
    print("Calculating user-blogger interaction features...")
    
    # 历史交互特征 (User-Blogger)
    user_blogger_hist_agg = hist_ref.groupby(['UserID', 'BloggerID']).agg(
        ub_hist_interactions=('Time', 'count'),
        ub_hist_views=('UserBehaviour', lambda x: (x == 1).sum()),
        ub_hist_likes=('UserBehaviour', lambda x: (x == 2).sum()),
        ub_hist_comments=('UserBehaviour', lambda x: (x == 3).sum()),
        ub_first_interaction_date=('Date', 'min'),
        ub_last_interaction_date=('Date', 'max'),
        ub_interaction_days=('Date', 'nunique')
    ).reset_index()
    user_blogger_hist_agg['ub_days_since_first_interaction'] = (reference_date - user_blogger_hist_agg['ub_first_interaction_date']).dt.days
    user_blogger_hist_agg['ub_days_since_last_interaction'] = (reference_date - user_blogger_hist_agg['ub_last_interaction_date']).dt.days
    user_blogger_hist_agg['ub_interaction_frequency'] = user_blogger_hist_agg['ub_hist_interactions'] / user_blogger_hist_agg['ub_interaction_days'].replace(0, 1)

    # 当天互动特征 (从 interaction_data 计算)
    # 当天互动特征 (从 interaction_data 计算)
    print("Calculating 'current day' interaction features...")
    interaction_agg = interaction_data.groupby(['UserID', 'BloggerID']).agg(
        current_interactions=('Time', 'count'),
        # --- 修改下面三行 ---
        current_has_view=('UserBehaviour', lambda x: int(1 in x.values)),
        current_has_like=('UserBehaviour', lambda x: int(2 in x.values)),
        current_has_comment=('UserBehaviour', lambda x: int(3 in x.values)),
        # --- 修改结束 ---
    ).reset_index()


    # --- 合并特征 ---
    print("Merging features...")
    # 将特征合并到目标对上
    data = target_pairs_df.copy()
    data = pd.merge(data, user_agg, on='UserID', how='left')
    data = pd.merge(data, blogger_agg, on='BloggerID', how='left')
    data = pd.merge(data, user_blogger_hist_agg, on=['UserID', 'BloggerID'], how='left')
    data = pd.merge(data, interaction_agg, on=['UserID', 'BloggerID'], how='left')

    # 填充缺失值 (例如，某个用户/博主/交互对在历史数据中没有记录)
    # 对于计数特征，填充0；对于比率，可能填充0或平均值；对于天数，可能填充一个较大的值
    count_cols = [col for col in data.columns if 'interactions' in col or 'follows' in col or 'views' in col or 'likes' in col or 'comments' in col or 'days' in col or 'has_' in col or 'unique' in col or 'frequency' in col]
    data[count_cols] = data[count_cols].fillna(0)
    # 对于天数相关的，可以考虑填充一个较大的值或者-1表示从未发生
    day_cols = [col for col in data.columns if 'days_since' in col]
    data[day_cols] = data[day_cols].fillna(data[day_cols].max() + 30) # 填充一个比最大值更大的数
    # 比率填充0
    rate_cols = [col for col in data.columns if 'rate' in col]
    data[rate_cols] = data[rate_cols].fillna(0)
    
    # 删除不再需要的日期列
    drop_cols = [col for col in data.columns if 'date' in col.lower() and col not in ['UserID', 'BloggerID', 'label']] # 小心别误删目标列
    data = data.drop(columns=drop_cols, errors='ignore')


    print("Feature creation finished.")
    gc.collect()
    return data

# --- 3. 构建训练数据 (预测 2024-07-20 的关注) ---
print("Building training data...")
# 标签日期的互动数据 (用于提取当天互动特征)
label_day_interactions = df_hist[df_hist['Date'] == label_date].copy()

# 标签日期的关注行为 (正样本)
positive_samples = label_day_interactions[label_day_interactions['UserBehaviour'] == 4][['UserID', 'BloggerID']].drop_duplicates()
positive_samples['label'] = 1

# 找出在标签日期之前已经关注的对，用于过滤
hist_follows = df_hist[(df_hist['Date'] < label_date) & (df_hist['UserBehaviour'] == 4)][['UserID', 'BloggerID']].drop_duplicates()
hist_follows['already_followed'] = 1

# 从正样本中移除那些在标签日期之前已经关注的 (理论上不应该有，根据假设2)
positive_samples = pd.merge(positive_samples, hist_follows, on=['UserID', 'BloggerID'], how='left')
positive_samples = positive_samples[positive_samples['already_followed'].isna()]
positive_samples = positive_samples[['UserID', 'BloggerID', 'label']]

# 负样本：标签日期有互动(1,2,3)但没有关注(4)的用户-博主对
negative_candidates = label_day_interactions[label_day_interactions['UserBehaviour'].isin([1, 2, 3])][['UserID', 'BloggerID']].drop_duplicates()
# 移除当天关注了的 (正样本)
negative_samples = pd.merge(negative_candidates, positive_samples[['UserID', 'BloggerID']].assign(is_positive=1), on=['UserID', 'BloggerID'], how='left')
negative_samples = negative_samples[negative_samples['is_positive'].isna()]
# 移除之前已经关注了的
negative_samples = pd.merge(negative_samples, hist_follows, on=['UserID', 'BloggerID'], how='left')
negative_samples = negative_samples[negative_samples['already_followed'].isna()]
negative_samples = negative_samples[['UserID', 'BloggerID']]
negative_samples['label'] = 0

# 合并正负样本 (可以考虑采样负样本以平衡数据，这里简单合并)
# 注意：如果负样本过多，需要进行采样，否则模型可能倾向于预测负类
# sample_ratio = len(positive_samples) * 5 / len(negative_samples) # 例如，负样本数是正样本的5倍
# if sample_ratio < 1:
#    negative_samples = negative_samples.sample(frac=sample_ratio, random_state=42)

train_df_targets = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
print(f"Training data: {len(positive_samples)} positive, {len(negative_samples)} negative samples.")

# 为训练集创建特征
# 历史数据截止到 train_end_date (7.19)
# 当天互动数据使用 label_day_interactions (7.20)
train_features = create_features(train_df_targets[['UserID', 'BloggerID']], 
                                 df_hist, 
                                 label_day_interactions, 
                                 train_end_date)
train_labels = train_df_targets['label']

# 合并特征和标签
train_data = pd.concat([train_features, train_labels], axis=1)

# 清理内存
del positive_samples, negative_samples, negative_candidates, label_day_interactions, train_df_targets
gc.collect()

# --- 4. 模型训练 ---
print("Training LightGBM model...")

# 定义特征列和类别特征
feature_cols = [col for col in train_data.columns if col not in ['UserID', 'BloggerID', 'label']]
categorical_features = ['UserID', 'BloggerID'] # 指定类别特征
# 检查类别特征是否真的在特征列中（如果之前转换成了数值就不用指定了）
categorical_features_in_model = [f for f in categorical_features if f in feature_cols]


# LightGBM 数据集
# 注意：LightGBM可以直接处理Pandas的Category类型，需要指定 categorical_feature
# 如果 UserID/BloggerID 之前未转换为 category 类型，需要先转换
# train_data['UserID'] = train_data['UserID'].astype('category')
# train_data['BloggerID'] = train_data['BloggerID'].astype('category')

X_train = train_data[feature_cols]
y_train = train_data['label']

# 定义模型参数 (可以根据需要调优)
params = {
    'objective': 'binary',
    'metric': 'auc', # 使用AUC作为评估指标
    'boosting_type': 'gbdt',
    'num_leaves': 31, # 根据数据量和特征复杂度调整
    'learning_rate': 0.05,
    'feature_fraction': 0.9, # 防止过拟合
    'bagging_fraction': 0.8, # 防止过拟合
    'bagging_freq': 5,
    'verbose': -1, # 控制输出信息级别
    'n_estimators': 1000, # 迭代次数，可以使用early_stopping来确定最佳次数
    'n_jobs': -1, # 使用所有可用CPU核心
    'seed': 42,
    #'is_unbalance': True # 如果正负样本不平衡可以尝试开启
}

# 训练模型 (可以加入验证集和早停)
# 这里为了简化，直接在全量训练数据上训练，实际应用中建议划分验证集
model = lgb.LGBMClassifier(**params)

# 确保类别特征被正确处理
# 在 fit 方法中传入 categorical_feature 参数
# 注意：列名必须完全匹配
categorical_cols_for_lgb = [col for col in categorical_features if col in X_train.columns]

model.fit(X_train, y_train, 
          categorical_feature=categorical_cols_for_lgb,
          # eval_set=[(X_val, y_val)], # 如果有验证集
          # callbacks=[lgb.early_stopping(100, verbose=True)] # 早停回调
         )

print("Model training finished.")
# 可以打印特征重要性
# lgb.plot_importance(model, max_num_features=30)

# 清理内存
del X_train, y_train, train_data
gc.collect()

# --- 5. 准备预测数据 (针对指定用户和 2024-07-22) ---
print("Preparing prediction data...")
target_users = ['U7', 'U6749', 'U5769', 'U14990', 'U52010']

# 筛选附件2中目标用户的互动记录
predict_day_interactions = df_future_interactions[
    (df_future_interactions['Date'] == predict_date) &
    (df_future_interactions['UserID'].isin(target_users))
].copy()

# 获取这些用户在当天互动的 (UserID, BloggerID) 候选对
prediction_candidates = predict_day_interactions[['UserID', 'BloggerID']].drop_duplicates()

# 找出目标用户在预测日期之前已经关注的博主 (使用附件1全部历史数据)
all_hist_follows = df_hist[df_hist['UserBehaviour'] == 4][['UserID', 'BloggerID']].drop_duplicates()
target_user_hist_follows = all_hist_follows[all_hist_follows['UserID'].isin(target_users)].copy()
target_user_hist_follows['already_followed'] = 1

# 从候选对中移除已经关注的
predict_targets = pd.merge(prediction_candidates, target_user_hist_follows, on=['UserID', 'BloggerID'], how='left')
predict_targets = predict_targets[predict_targets['already_followed'].isna()]
predict_targets = predict_targets[['UserID', 'BloggerID']]

print(f"Found {len(predict_targets)} potential new follow pairs for target users.")

# 为预测集创建特征
# 历史数据使用附件1的全部数据 (截止到 7.20)
# 当天互动数据使用 predict_day_interactions (7.22 的数据)
predict_features = create_features(predict_targets[['UserID', 'BloggerID']],
                                   df_hist, # 全部历史数据
                                   predict_day_interactions, # 7.22 的互动数据
                                   hist_end_date_for_predict) # 历史数据截止日期为 7.20

# 确保预测数据的列与训练数据一致且顺序相同
predict_features = predict_features[feature_cols] # 保证列顺序和使用的特征一致

# 处理可能出现的新的UserID或BloggerID (如果模型训练时没见过)
# LightGBM 在 predict 时如果遇到未知的类别特征值会报错或处理为NaN
# 确保UserID和BloggerID在预测时也转换为category类型，并且categories与训练时一致
# predict_features['UserID'] = pd.Categorical(predict_features['UserID'], categories=train_data['UserID'].cat.categories)
# predict_features['BloggerID'] = pd.Categorical(predict_features['BloggerID'], categories=train_data['BloggerID'].cat.categories)
# 注意：如果create_features函数中已经正确处理了category类型，这里可能不需要再次转换，但要确保categories信息被保留

# --- 6. 进行预测 ---
print("Making predictions...")
predictions_proba = model.predict_proba(predict_features)[:, 1] # 获取正类（关注）的概率

# 将概率添加到预测目标DataFrame中
predict_targets['follow_probability'] = predictions_proba

# 选择概率大于阈值的作为预测结果 (阈值可以调整，例如0.5)
prediction_threshold = 0.2
predicted_follows = predict_targets[predict_targets['follow_probability'] > prediction_threshold]

print(f"Predicted {len(predicted_follows)} new follows with threshold > {prediction_threshold}.")

# --- 7. 输出结果 ---
print("Generating final results for Table 2...")
# 按用户分组，收集预测关注的博主ID
final_results = predicted_follows.groupby('UserID')['BloggerID'].apply(list).reset_index()

# 创建结果表
table2_data = {'UserID': target_users}
table2_df = pd.DataFrame(table2_data)

# 合并预测结果
table2_df = pd.merge(table2_df, final_results, on='UserID', how='left')

# 格式化输出，将列表转换为逗号分隔的字符串 (如果需要) 或保持列表形式
# 题目要求填入博主ID，如果关注多个，均填入。这里用列表表示比较清晰。
# 如果需要逗号分隔字符串：
# table2_df['NewFollowBloggerID'] = table2_df['BloggerID'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else '')

# 直接使用列表，如果未关注则为NaN或空列表
table2_df.rename(columns={'BloggerID': '新关注博主ID'}, inplace=True)
# 将NaN替换为空列表或提示信息
table2_df['新关注博主ID'] = table2_df['新关注博主ID'].apply(lambda x: x if isinstance(x, list) else []) 

print("\n--- 问题2 预测结果 ---")
print(table2_df.to_string(index=False))

# 示例输出填充到表2:
# U7: [Bxxx, Byyy]
# U6749: []
# ... etc.

# 可以将结果保存到文件
# table2_df.to_csv('problem2_predictions.csv', index=False, encoding='utf-8-sig')

print("\nDone.")
