# 加载示例数据
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lifelines import CoxPHFitter
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# import xgboost as xgb
# import lightgbm as lgb
# from pycox.models import LogisticHazard
# import torchtuples as tt
# import torch
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('/mnt/e/liangdong/Machine_Learning/github/DataAnalysis/predict_app/filter_data1.csv')

# 删除ID列
df = data.drop(columns=['ID'])  # 确保删除正确的ID列

# 定义时间和事件列
time_col = 'OS.time'
event_col = 'OS'

# 检查缺失值并删除包含缺失值的行
df = df.dropna()

# 分离特征和标签
X = df.drop(columns=[time_col, event_col])
y = df[[time_col, event_col]]

# 将分类变量转换为独热编码
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)

# 标准化数值特征
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 合并X和y训练数据
train_data = X_train.copy()
train_data[time_col] = y_train[time_col]
train_data[event_col] = y_train[event_col]

# 构建Cox回归模型
cph = CoxPHFitter()
cph.fit(train_data, duration_col=time_col, event_col=event_col)

# 定义时间点
time_points = [365, 3*365, 5*365, 10*365]

# 计算预测生存概率
def predict_survival_probabilities(model, X, time_points):
    surv_func = model.predict_survival_function(X)
    survival_probabilities = pd.DataFrame()
    for t in time_points:
        if t in surv_func.index:
            survival_probabilities[t] = surv_func.loc[t]
        else:
            closest_time = min(surv_func.index, key=lambda x: abs(x-t))
            survival_probabilities[t] = surv_func.loc[closest_time]
    return survival_probabilities

# 预测生存概率
train_surv_probs = predict_survival_probabilities(cph, X_train, time_points)
test_surv_probs = predict_survival_probabilities(cph, X_test, time_points)

# 计算和绘制ROC曲线
def plot_roc_curve(y_true, y_pred, time_point, dataset_type):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{dataset_type} {time_point/365:.1f}-year (AUC = {roc_auc:.2f})')

# 绘制并保存每个时间点的ROC曲线
for t in time_points:
    plt.figure()
    plot_roc_curve(y_train[event_col], 1 - train_surv_probs[t], t, 'Train')
    plot_roc_curve(y_test[event_col], 1 - test_surv_probs[t], t, 'Test')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'most_model/ROC Curve at {t/365:.1f} years')
    plt.legend(loc='lower right')
    # plt.savefig(f"most_model/roc_curve_{int(t/365)}_years.pdf")
    plt.show()

# 保存模型为 pkl 文件
with open('prediction_model.pkl', 'wb') as file:
    pickle.dump(cph, file)

print("模型已保存为 prediction_model.pkl")

