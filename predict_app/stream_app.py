import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'prediction_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("Comprehensive mortality prediction in older patients with coronary heart disease through advanced machine learning approaches: a prospective study with a 10-year follow-up")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
st.sidebar.subheader("Picking up parameters")
Age = st.sidebar.slider("Age (year)", min_value=59, max_value=104, value=85, step=1)
HR = st.sidebar.slider("HR (bpm)", min_value=39, max_value=106, value=56, step=1)
Hb = st.sidebar.slider("Hb (g/L)", min_value=57, max_value=185, value=75, step=1)
HDL = st.sidebar.slider("HDL (mmol/L)", min_value=0.19, max_value=2.71, value=0.52, step=0.01)
albumin = st.sidebar.slider("Albumin (g/L)", min_value=23.7, max_value=50.0, value=25.0, step=0.1)
fibrinogen = st.sidebar.slider("Fibrinogen (g/L)", min_value=0.78, max_value=9.02, value=2.18, step=0.01)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'Age': [Age],
    'HR': [HR],
    'Hb': [Hb],
    'HDL': [HDL],
    'albumin': [albumin],
    'fibrinogen': [fibrinogen]
})

time_points = [365, 3*365, 5*365, 10*365]

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


if st.button("Calculate"):
    test_surv_probs = predict_survival_probabilities(model, input_data, time_points)

    print(type(test_surv_probs[365]))
    one_year = test_surv_probs[365]
    three_year = test_surv_probs[1095]
    five_year = test_surv_probs[1825]
    ten_year = test_surv_probs[3650]
    ten_year_p = 0
    for ten in ten_year:
        ten_year_p = ten
    ten_rounded_num = round(ten_year_p * 100, 3)
    st.write(f"10-year predicted survival probability: {ten_rounded_num}%")

