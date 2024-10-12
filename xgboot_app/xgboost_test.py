import pandas as pd
import xgboost as xgb
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载示例数据
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 保存模型为 pkl 文件
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("模型已保存为 xgboost_model.pkl")

