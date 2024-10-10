import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置页面标题
st.title("多子图数据可视化示例")

# 创建随机数生成器
rng = np.random.default_rng(1234)

# 生成训练集和测试集数据
def generate_data():
    x_train = rng.uniform(1, 38, size=50)
    y_train = x_train + rng.normal(size=50)
    x_test = rng.uniform(1, 38, size=50)
    y_test = x_test + rng.normal(size=50)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = generate_data()

# 拟合整体数据的回归线
x_combined = np.concatenate([x_train, x_test])
y_combined = np.concatenate([y_train, y_test])
b_combined, a_combined = np.polyfit(x_combined, y_combined, deg=1)

# 计算训练集和测试集的预测值
y_train_pred = a_combined + b_combined * x_train
y_test_pred = a_combined + b_combined * x_test

# 计算残差
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# 创建画布
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 5], width_ratios=[2, 6, 1])

# 绘制散点图
ax1 = fig.add_subplot(gs[1, 1])
ax1.scatter(x_train, y_train, s=40, alpha=0.7, c='b', label='Train set')
ax1.scatter(x_test, y_test, s=40, alpha=0.7, c='r', label='Test set')
ax1.plot(np.linspace(0, 40, num=100), a_combined + b_combined * np.linspace(0, 40, num=100), color="k", lw=1.5)
ax1.set_xlabel('Measured, d[mm]', fontsize=13)
ax1.set_ylabel('Predicted, d[mm]', fontsize=13)
ax1.legend(fontsize=12)
ax1.grid(True)
ax1.set_xlim(left=0, right=40)
ax1.set_ylim(bottom=0, top=40)

# 绘制堆积图（x 轴的分布图）
ax3 = fig.add_subplot(gs[0, 1])
edges = np.histogram(np.concatenate([x_train, x_test]), bins=10)[1]
counts_train, _ = np.histogram(x_train, bins=edges)
counts_test, _ = np.histogram(x_test, bins=edges)
ax3.bar(edges[:-1], counts_train, width=np.diff(edges), color='b', alpha=0.5)
ax3.bar(edges[:-1], counts_test, width=np.diff(edges), bottom=counts_train, color='r', alpha=0.5)
ax3.set_ylabel('Count', fontsize=13)
ax3.set_xticklabels([])

# 绘制堆积图（y 轴的分布图）
ax4 = fig.add_subplot(gs[1, 2])
edges = np.histogram(np.concatenate([y_train, y_test]), bins=10)[1]
counts_train, _ = np.histogram(y_train, bins=edges)
counts_test, _ = np.histogram(y_test, bins=edges)
ax4.barh(edges[:-1], counts_train, height=np.diff(edges), color='b', alpha=0.5)
ax4.barh(edges[:-1], counts_test, height=np.diff(edges), left=counts_train, color='r', alpha=0.5)
ax4.set_xlabel('Count', fontsize=13)

# 绘制残差散点图
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(residuals_train, x_train, s=40, alpha=0.7, c='b')
ax2.scatter(residuals_test, x_test, s=40, alpha=0.7, c='r')
ax2.axvline(0, color='k', lw=2, linestyle='--')
ax2.set_xlabel('Residuals', fontsize=13)

# 绘制残差的横向堆积图（在左下子图）
ax5 = fig.add_subplot(gs[0, 0])
edges = np.histogram(np.concatenate([residuals_train, residuals_test]), bins=10)[1]
counts_train, _ = np.histogram(residuals_train, bins=edges)
counts_test, _ = np.histogram(residuals_test, bins=edges)
ax5.bar(edges[:-1], counts_train, width=np.diff(edges), color='b', alpha=0.5)
ax5.bar(edges[:-1], counts_test, width=np.diff(edges), bottom=counts_train, color='r', alpha=0.5)
ax5.set_ylabel('Count', fontsize=13)
ax5.set_xticklabels([])

# 调整布局并显示图形
plt.subplots_adjust(hspace=0.3, wspace=0.1)
st.pyplot(fig)