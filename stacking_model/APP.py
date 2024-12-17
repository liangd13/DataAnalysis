import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import pickle
import os

# 加载模型
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'stacking_regressor_model.pkl')
# model_path = "stacking_regressor_model.pkl"
# stacking_regressor = joblib.load(model_path)
with open(model_path, 'rb') as file:
    stacking_regressor = pickle.load(file)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
X_1 = st.sidebar.number_input("特征 X_1 (范围: 5.5-20.0)", min_value=5.5, max_value=20.0, value=10.0)
X_2 = st.sidebar.number_input("特征 X_2 (范围: 7-57)", min_value=7, max_value=57, value=30)
X_3 = st.sidebar.number_input("特征 X_3 (范围: 6.71-146.91)", min_value=6.71, max_value=146.91, value=20.0)
X_4 = st.sidebar.number_input("特征 X_4 (范围: 5.53-39.07)", min_value=5.53, max_value=39.07, value=20.0)
X_5 = st.sidebar.number_input("特征 X_5 (范围: 23-12208)", min_value=23, max_value=12208, value=1000)
X_6 = st.sidebar.number_input("特征 X_6 (范围: 6.28-22.18)", min_value=6.28, max_value=22.18, value=10.0)
X_7 = st.sidebar.number_input("特征 X_7 (范围: 41.72-46.95)", min_value=41.72, max_value=46.95, value=45.0)
X_8 = st.sidebar.number_input("特征 X_8 (范围: -119.3到-114.77)", min_value=-119.3, max_value=-114.77, value=-117.0)

# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("预测结果")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8]).reshape(1, -1)

        # 模型预测
        prediction = stacking_regressor.predict(input_array)[0]

        # 显示预测结果
        st.success(f"预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# 可视化展示
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. 第一层基学习器")
st.write("基学习器（RandomForest、XGB、LGBM 等）的特征贡献分析。")
first_layer_img = os.path.join(current_dir, "summary_plot.png")
# first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. 第二层元学习器")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_imgs = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
meta_layer_img = os.path.join(current_dir, meta_layer_imgs)
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_imga = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
overall_img = os.path.join(current_dir, overall_imga)
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")
