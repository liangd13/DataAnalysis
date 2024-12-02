import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径
# import shap  # 导入 SHAP 库，用于解释模型

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'prediction_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("Prediction 模型预测")

# 在侧边栏中输入特征
st.sidebar.header("输入特征")  # 侧边栏的标题
# 使用滑动条接收花萼长度，设置范围为 0.0 到 10.0，默认值为 5.0
Age = st.sidebar.slider("花萼长度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
# 使用滑动条接收花萼宽度，设置范围为 0.0 到 10.0，默认值为 5.0
HR = st.sidebar.slider("花萼宽度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
# 使用滑动条接收花瓣长度，设置范围为 0.0 到 10.0，默认值为 5.0
Hb = st.sidebar.slider("花瓣长度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
# 使用滑动条接收花瓣宽度，设置范围为 0.0 到 10.0，默认值为 5.0
HDL = st.sidebar.slider("花瓣宽度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

albumin = st.sidebar.slider("花瓣宽度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
fibrinogen = st.sidebar.slider("花瓣宽度 (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'Age': [Age],
    'HR': [HR],
    'Hb': [Hb],
    'HDL': [HDL],
    'albumin': [albumin],
    'fibrinogen': [fibrinogen]
})


# 添加预测按钮，用户点击后进行模型预测
if st.button("预测"):
    prediction = model.predict(input_data)  # 使用加载的模型进行预测
    st.write(f"预测结果: {prediction[0]}")

    # # 计算 SHAP 值
    # explainer = shap.Explainer(model)  # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值
    # shap_values = explainer(input_data)
    #
    # # 提取单个样本的 SHAP 值和期望值
    # sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值
    # expected_value = explainer.expected_value[0]  # 获取对应输出的期望值
    #
    # # 创建 Explanation 对象
    # explanation = shap.Explanation(
    #     values=sample_shap_values[:, 0],  # 选择特定输出的 SHAP 值
    #     base_values=expected_value,
    #     data=input_data.iloc[0].values,
    #     feature_names=input_data.columns.tolist()
    # )
    #
    # # 保存为 HTML 文件
    # shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))
    #
    # # 在 Streamlit 中显示 HTML
    # st.subheader("模型预测的力图")
    # with open("shap_force_plot.html") as f:
    #     st.components.v1.html(f.read(), height=600)
