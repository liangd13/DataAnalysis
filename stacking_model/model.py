import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_excel('2024-11-27公众号Python机器学习AI.xlsx')

from sklearn.model_selection import train_test_split, KFold

X = df.drop(['Y'],axis=1)
y = df['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

# 定义一级学习器
base_learners = [
    ("RF", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("XGB", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
    ("LGBM", LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
    ("GBM", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ("AdaBoost", AdaBoostRegressor(n_estimators=100, random_state=42)),
    ("CatBoost", CatBoostRegressor(n_estimators=100, random_state=42, verbose=0))
]

# 定义二级学习器
meta_model = LinearRegression()

# 创建Stacking回归器
stacking_regressor = StackingRegressor(estimators=base_learners, final_estimator=meta_model, cv=5)

# 训练模型
stacking_regressor.fit(X_train, y_train)


# import joblib
# joblib.dump(stacking_regressor, "stacking_regressor_model.pkl")
import pickle
# 保存模型为 pkl 文件
with open('stacking_regressor_model.pkl', 'wb') as file:
    pickle.dump(stacking_regressor, file)

print("模型已保存为 xgboost_model.pkl")
