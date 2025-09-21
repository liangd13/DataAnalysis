import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用OneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 关闭TensorFlow日志
os.environ['MPLBACKEND'] = 'Agg'  # 强制使用无GUI后端

import util
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier

data_path = "/mnt/e/liangdong/Machine_Learning/github/DataAnalysis/binary_classification_models/cnn/analysis_data_select_lasso.csv"
result_var_name = "Prognosis"
test_set_size = 0.4
cat_var_num = 2
new_directory_name = "result"


output_dir = util.makeDirectory(new_directory_name)

# 1. 数据加载
print("\n******** 数据集加载 *********")
data = pd.read_csv(data_path)
print("数据集shape:", data.shape)
print(data.head())
print("数据集类型:\n", data.dtypes)

print("\n******** 区分分类变量和连续变量 *********")
## 分类变量
catnames = [data.columns.to_list()[i] for i in range(1, cat_var_num)]
print("分类变量:", catnames)
## 连续变量
connames = [i for i in data.columns.to_list() if i not in catnames + [result_var_name]]
print("连续变量:", connames)
## 所有变量
allnames = catnames + connames
print("所有变量:", allnames)

# 分类变量转换为category
for i in catnames:
    data[i] = data[i].astype("category")
## 设定因变量
data[result_var_name] = pd.Categorical(data[result_var_name])
data[result_var_name].cat.categories
## 数据列类型
print("转换后数据类型:\n", data.dtypes)

# 描述统计表
util.exportDataDescriptionTable(data, output_dir=output_dir)

# 数字变量相关系数热图
util.exportConVarHeatMap(data, connames=connames, output_dir=output_dir)


# 2. 拆分数据集（训练集和测试集）
train_x, test_x, train_y, test_y = util.trainTestSplit(data, allnames=allnames,
                                                       result_var_name=result_var_name,
                                                       test_set_size=test_set_size,
                                                       output_dir=output_dir)


# 3. 数据的预处理
prep_mlp = util.dataPreprocessing(catnames, connames)

# 构建MLP的搜索管道
mlp_steps = [("prep", prep_mlp), ("mlp", MLPClassifier(random_state=42))]
# 超参数调优空间
mlp_hpspace = dict(mlp__hidden_layer_sizes=[(12,), (12, 12), (12, 12, 12)],
                   mlp__activation=["tanh", "relu"],
                   mlp__batch_size=[64, 128])

mlp = util.searchMostOptiModel(datax=train_x, datay=train_y,
                               set_steps=mlp_steps,
                               set_searchtype="GridSearchCV",
                               set_hpspace=mlp_hpspace,
                               set_indicator="roc_auc",
                               set_cv=util.kFoldTest(split=5))

util.tuneplot(mlp, output_dir)


# 4. 模型的评估
util.evalfunc(model=mlp,
              actualx=train_x, actualy=train_y,
              modelname="MLP",
              datasetname="train_data",
              positive=1,
              negative=0,
              output_dir=output_dir)

util.evalfunc(model=mlp,
              actualx=test_x, actualy=test_y,
              modelname="MLP",
              datasetname="test_data",
              positive=1,
              negative=0,
              output_dir=output_dir)

util.plotFeatureImportance(mlp, train_x,  modelname="mlp", output_dir=output_dir)

# Partial dependence display
# util.plotSinglePartialDependence(mlp, train_x, connames=connames, output_dir=output_dir)
#
# util.plotTwoPartialDependence(mlp, train_x, "Latency", "NSE", output_dir)
# util.plotTwoPartialDependence(mlp, train_x, "Latency", "IL-8", output_dir)
# util.plotTwoPartialDependence(mlp, train_x, "NSE", "IL-8", output_dir)

# shap
shapvalues_mlp, final_train_x = util.getModelShapeValues(mlp, "mlp", train_x)

util.plotShapBeeswarmViolinBar(shapvalues_mlp, final_train_x, output_dir)
var_names = {"Latency", "IL-8", "MVG_Grading_1", "NSE", "Albumin", "TP"}
util.plotShapSingleVar(shapvalues_mlp, var_names, output_dir)

util.plotShapWaterForceDesicion(shapvalues_mlp, final_train_x, output_dir)


raise ValueError("--------")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)

# 重塑数据为CNN格式
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


# 4. 构建CNN模型
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(64, 2, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# 5. 模型训练
cnn_model = build_cnn_model((X_train_cnn.shape[1], 1))
print("模型架构概要:")
cnn_model.summary()

callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, verbose=1, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("\n开始训练模型...")
history = cnn_model.fit(
    X_train_cnn, train_y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=callbacks
)

# 6. 模型评估
print("\n评估结果:")
test_loss, test_acc, test_auc = cnn_model.evaluate(X_test_cnn, test_y, verbose=0)
print(f"测试准确率: {test_acc:.4f}")
print(f"测试AUC: {test_auc:.4f}")

y_pred_probs = cnn_model.predict(X_test_cnn).flatten()
fpr, tpr, thresholds = roc_curve(test_y, y_pred_probs)
roc_auc = roc_auc_score(test_y, y_pred_probs)

train_y_pred_probs = cnn_model.predict(X_train_cnn).flatten()
train_fpr, train_tpr, train_thresholds = roc_curve(train_y, train_y_pred_probs)
train_roc_auc = roc_auc_score(train_y, train_y_pred_probs)

# 7. 安全渲染ROC曲线（解决字体问题）
def save_roc_plot(fpr, tpr, roc_auc, train_fpr, train_tpr, train_roc_auc):
    """安全保存ROC曲线，规避字体问题"""
    plt.figure(dpi=120)

    # 仅使用ASCII字符安全渲染
    plt.plot(fpr, tpr, color='darkorange', label=f'Test AUC = {roc_auc:.4f}')
    plt.plot(train_fpr, train_tpr, color='blue', label=f'Train AUC = {train_roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # 避免使用中文标签
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve - CNN Model')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('safe_roc_curve.png', bbox_inches='tight')
    print("ROC曲线已安全保存为 safe_roc_curve.png")


# 生成安全的ROC图像
save_roc_plot(fpr, tpr, roc_auc, train_fpr, train_tpr, train_roc_auc)


def plot_calibration(train_y, train_proba, test_y, test_proba):
    plt.figure()
    for prob, y_, name in [(train_proba, train_y, 'Train'), (test_proba, test_y, 'Test')]:
        prob_true, prob_pred = calibration_curve(y_, prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Ensemble Model")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./cnn_calibration.png", dpi=300)


plot_calibration(train_y, train_y_pred_probs, test_y, y_pred_probs)

print("\n模型训练和评估完成！")




