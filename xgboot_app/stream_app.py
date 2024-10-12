import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# 加载模型
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit 应用界面
st.title("XGBoost 模型预测")

# 输入特征
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# 创建输入数据框
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# 预测按钮
if st.button("预测"):
    prediction = model.predict(input_data)
    st.write(f"预测结果: {prediction[0]}")