import os
from tableone import TableOne
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score

import shap

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d  %(message)s",
    datefmt="%H:%M:%S"  # 设置时间格式为时:分:秒
)
logger = logging.getLogger(__name__)

def makeDirectory(dir_name):
    logger.info(f"尝试创建新目录 {dir_name}")
    current_path = os.getcwd()
    output_dir = os.path.join(current_path, dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logger.info(f"已在当前目录 {current_path} 下创建了目录 {dir_name}")
    else:
        logger.info(f"目录 {dir_name} 已存在，无需创建。")

    return output_dir


def loadProcessData(data_path, result_var_name, cat_var_num, output_dir):
    logger.info("******** 数据集加载 *********")
    data = pd.read_csv(data_path)
    logger.info(f"数据集shape: {data.shape}")
    logger.info(f"数据集类型:\n{data.dtypes}")

    logger.info("******** 区分分类变量和连续变量 *********")
    # 分类变量
    catnames = [data.columns.to_list()[i] for i in range(1, cat_var_num)]
    logger.info(f"分类变量:{catnames}")
    # 连续变量
    connames = [i for i in data.columns.to_list() if i not in catnames + [result_var_name]]
    logger.info(f"连续变量:{connames}")
    # 所有变量
    allnames = catnames + connames
    logger.info(f"所有变量:{allnames}")

    # 分类变量转换为category
    for i in catnames:
        data[i] = data[i].astype("category")
    # 设定因变量
    data[result_var_name] = pd.Categorical(data[result_var_name])
    data[result_var_name].cat.categories
    # 数据列类型
    logger.info(f"转换后数据类型:\n{data.dtypes}")

    # 描述统计表
    exportDataDescriptionTable(data, output_dir=output_dir)
    # 数字变量相关系数热图
    exportConVarHeatMap(data, connames=connames, output_dir=output_dir)

    return data, catnames, connames


def exportDataDescriptionTable(data, output_dir):
    # 描述统计表
    logger.info("******** Export 数据集描述统计表 *********")
    table1 = TableOne(data, dip_test=True, normal_test=True, tukey_test=True)
    table1.to_csv(output_dir + "/table1.csv")
    logger.info(f"数据集统计表\n{table1}")
    logger.info(f">> 数据集描述统计表已保存在：{output_dir}/table1.pdf")


def exportConVarHeatMap(data, connames, output_dir):
    logger.info("******** Export 连续变量的相关系数热图 *********")
    sns.heatmap(data.loc[:, connames].corr(), vmin=-1, vmax=1, center=0,
                linecolor="white",
                linewidths=0.1,
                cmap="RdBu_r")
    plt.savefig(output_dir + "/heatmap.pdf")
    plt.close()
    logger.info(f">> 连续变量相关系数热图已保存在：{output_dir}/heatmap.pdf")


def trainTestSplit(data, allnames, result_var_name, test_set_size, output_dir):
    logger.info("******** 划分训练集和测试集数据 *********")
    ## 自变量部分
    data_x = data.loc[:, allnames]
    ## 因变量部分
    data_y = data.loc[:, result_var_name]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_set_size,
                                                        random_state=432)
    logger.info(f"训练集自变量：{train_x.shape}")
    logger.info(f"测试集自变量：{test_x.shape}")
    logger.info(f"训练集因变量：{train_y.shape}")
    logger.info(f"测试集因变量：{test_y.shape}")

    logger.info("******** Export 训练集和测试集统计表 *********")
    warnings.filterwarnings("ignore")
    data_tmp = data.copy()
    data_tmp["dataset"] = "test"
    data_tmp["dataset"].iloc[train_x.index] = "train"
    table2 = TableOne(data_tmp, groupby=result_var_name, pval=True, htest_name=True)
    table2.to_csv(output_dir + "/table_train_test.csv")
    logger.info(f">> 训练集和测试集统计表已保存在：{output_dir}/table_train_test.csv")

    return train_x, test_x, train_y, test_y


def dataPreprocessing(catnames, connames):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    cattfppl = Pipeline(steps=[("impute", SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
                               ("ohe", OneHotEncoder(sparse_output=False, drop="first"))])
    contfppl = Pipeline(steps=[("impute", SimpleImputer(missing_values=pd.NA, strategy='median')),
                               ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("categorical", cattfppl, catnames),
        ("numerical", contfppl, connames)
    ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    return prep


def kFoldTest(split=5):
    # 设置交叉验证策略
    # 使用KFold进行5折交叉验证，数据在每次分割前都会被打乱，以确保随机性
    from sklearn.model_selection import KFold
    mlp_cv_hpo = KFold(n_splits=split, shuffle=True, random_state=42)


def searchMostOptiModel(
    datax,
    datay,
    set_steps,
    set_searchtype,
    set_hpspace,
    set_indicator,
    set_cv,
    n_jobs=-1,  # 新增默认参数：并行计算（-1表示用所有CPU核心）
    refit=True,  # 新增默认参数：用最佳超参数重新拟合全量数据
    verbose=1    # 新增默认参数：打印搜索过程信息
):
    """
    超参数优化（Hyperparameter Optimization）封装函数，支持网格搜索等策略

    Parameters:
    -----------
    datax : pd.DataFrame or np.ndarray
        训练特征数据（X）
    datay : pd.Series or np.ndarray
        训练标签数据（y）
    set_steps : list
        机器学习管道的步骤列表
    set_searchtype : str
        超参数搜索类型，当前仅支持 "GridSearchCV"
    set_hpspace : dict
        超参数搜索空间，键为"管道中模型名__超参数名"
    set_indicator : str
        模型评价指标
    set_cv : cross-validation generator or int
        交叉验证策略
    n_jobs : int, default=-1
        并行运行的CPU核心数
    refit : bool, default=True
        是否用最佳超参数重新拟合模型
    verbose : int, default=1
        搜索过程的日志详细程度

    Returns:
    --------
    sklearn.model_selection.GridSearchCV
        超参数搜索对象
    """
    # --------------------------
    # 1. 参数合法性校验
    # --------------------------
    supported_search_types = ["GridSearchCV"]
    if set_searchtype not in supported_search_types:
        raise ValueError(
            f"不支持的搜索类型 '{set_searchtype}'！当前仅支持：{', '.join(supported_search_types)}"
        )

    if not isinstance(set_steps, list) or len(set_steps) == 0:
        raise ValueError("set_steps 必须是非空列表，格式为 [('步骤名', 步骤对象), ...]")
    for step in set_steps:
        if not isinstance(step, tuple) or len(step) != 2 or not isinstance(step[0], str):
            raise ValueError(f"管道步骤 {step} 格式错误！需为 ('步骤名', 步骤对象) 元组")

    if not isinstance(set_hpspace, dict) or len(set_hpspace) == 0:
        raise ValueError("set_hpspace 必须是非空字典，键为 '模型名__超参数名'")
    model_names = [step[0] for step in set_steps if hasattr(step[1], "get_params")]
    for param_key in set_hpspace.keys():
        if "__" not in param_key:
            raise ValueError(f"超参数键 '{param_key}' 格式错误！需为 '模型名__超参数名'")

        model_name = param_key.split("__")[0]
        if model_name not in model_names:
            logger.warning(f"超参数键 '{param_key}' 中的模型名 '{model_name}' 未在管道 set_steps 中找到！")

    # --------------------------
    # 2. 构建机器学习管道
    # --------------------------
    try:
        pipeline = Pipeline(steps=set_steps)
        logger.info(f"成功构建管道，包含 {len(set_steps)} 个步骤：{[step[0] for step in set_steps]}")
    except Exception as e:
        raise RuntimeError(f"管道构建失败：{str(e)}")

    # --------------------------
    # 3. 初始化超参数搜索器
    # --------------------------
    try:
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=set_hpspace,
            scoring=set_indicator,
            cv=set_cv,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            return_train_score=True
        )
        logger.info(f"初始化 GridSearchCV 搜索器，评价指标：{set_indicator}，交叉验证策略：{type(set_cv).__name__}")
    except Exception as e:
        raise RuntimeError(f"GridSearchCV 初始化失败：{str(e)}")

    # --------------------------
    # 4. 执行超参数搜索（拟合数据）
    # --------------------------
    logger.info(f"开始超参数搜索，共 {len([v for v in set_hpspace.values() if isinstance(v, list)])} 个参数维度，{_count_param_combinations(set_hpspace)} 种组合")
    try:
        searcher.fit(datax, datay)
    except Exception as e:
        raise RuntimeError(f"超参数搜索失败：{str(e)}")

    # --------------------------
    # 5. 输出搜索结果
    # --------------------------
    logger.info("=" * 50)
    logger.info(f"超参数搜索完成！")
    logger.info(f"最佳超参数组合：{searcher.best_params_}")
    logger.info(f"最佳交叉验证得分（{set_indicator}）：{searcher.best_score_:.4f}")
    if refit:
        logger.info("已用最佳超参数在全量训练数据上重新拟合模型")
    logger.info("=" * 50)

    # --------------------------
    # 6. 返回搜索结果对象
    # --------------------------
    return searcher


# 辅助函数：计算超参数组合总数
def _count_param_combinations(hp_space):
    """计算超参数空间的总组合数"""
    from functools import reduce
    from operator import mul
    try:
        counts = [len(v) for v in hp_space.values() if isinstance(v, (list, tuple))]
        return reduce(mul, counts, 1)
    except Exception as e:
        logger.warning(f"计算超参数组合数时出现异常：{str(e)}")
        return "未知"


def tuneplot(grid_search_results, output_dir, scoreindim=True):
    results = pd.DataFrame(grid_search_results.cv_results_)
    param_names = [key for key in results.columns if key.startswith('param_')]

    plt.figure(figsize=(12, 6))
    # 绘制得分与每个超参数的关系
    for param_name in param_names:
        # 检查参数值是否为可绘制的数值
        if results[param_name].dtype == 'object':
            # 如果参数是对象，尝试提取数值
            values = results[param_name].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        else:
            values = results[param_name]

        plt.subplot(1, len(param_names), param_names.index(param_name) + 1)
        plt.scatter(values, results['mean_test_score'], alpha=0.5)
        plt.title(param_name)
        plt.xlabel(param_name)
        plt.ylabel('Mean Test Score')

        if scoreindim:
            plt.grid()

    plt.tight_layout()

    plt.savefig(output_dir + "/超参数调优结果.pdf", format='pdf')
    plt.close()  # 关闭图形以释放内存
    logger.info(f"参数调优的结果图已保存到：{output_dir}/超参数调优结果.pdf")


def evalfunc(model, actualx, actualy, modelname, datasetname, positive, negative, output_dir):
    # 预测概率
    predicted_probs = None
    if modelname == "MLP":
        predicted_probs = model.predict_proba(actualx)[:, 1]
    else:
        raise ValueError("No supported model name ", modelname)

    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(actualy, predicted_probs, pos_label=positive)
    roc_auc = auc(fpr, tpr)

    # 计算 PR 曲线
    precision, recall, _ = precision_recall_curve(actualy, predicted_probs, pos_label=positive)
    ap_value = average_precision_score(actualy, predicted_probs)

    # 计算混淆矩阵
    predicted_classes = (predicted_probs >= 0.5).astype(int)  # 使用 0.5 作为默认阈值
    cm = confusion_matrix(actualy, predicted_classes, labels=[negative, positive])

    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(actualy, predicted_probs, n_bins=6)

    _plotRocCurve(fpr, tpr, modelname, datasetname, output_dir)
    _plotPrCurve(recall, precision, ap_value, modelname, datasetname, output_dir)
    _plotCalibrationCurve(prob_pred, prob_true, modelname, datasetname, output_dir)
    _plotConfusionMatrix(cm, modelname, datasetname, output_dir)
    _plotDCACurve(predicted_probs, actualy, modelname, datasetname, output_dir)


def _plotDCACurve(pred_probs, actual_classes, modelname, datasetname, output_dir):
    def calculate_net_benefit(pred_probs, actual_classes, threshold):
        pred_classes = np.where(pred_probs >= threshold, 1, 0)
        tp = np.sum((pred_classes == 1) & (actual_classes == 1))
        fp = np.sum((pred_classes == 1) & (actual_classes == 0))
        n = len(actual_classes)
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))

    def calculate_all_benefit(actual_classes, threshold):
        tp = np.sum(actual_classes == 1)
        fp = np.sum(actual_classes == 0)
        n = len(actual_classes)
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))

    thresholds = np.arange(0, 1, 0.01)
    # real net benefit DCA
    net_benefits = []
    for threshold in thresholds:
        net_benefit = calculate_net_benefit(pred_probs, actual_classes, threshold)
        net_benefits.append(net_benefit)

    # all net benefit DCA
    net_benefits_all = []
    for threshold in thresholds:
        net_benefit = calculate_all_benefit(actual_classes, threshold)
        net_benefits_all.append(net_benefit)

    # 计算'none'的净收益（水平直线）
    net_benefits_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, color='red',
             label='{} {} DCA'.format(modelname, datasetname))
    plt.plot(thresholds, net_benefits_all, color='black', label='All')
    plt.plot(thresholds, net_benefits_none, color='gray', label='None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.25, 0.75])
    plt.legend()
    file_path = output_dir + "/" + modelname + "_" + datasetname + "_dca.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Model {modelname} based on {datasetname}'s DCA save to：{file_path}")


def _plotConfusionMatrix(cm, modelname, datasetname, output_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    file_path = output_dir + "/" + modelname + "_" + datasetname + "_confusionmatrix.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Model {modelname} based on {datasetname}'s confusionmatrix save to：{file_path}")


def _plotCalibrationCurve(prob_pred, prob_true, modelname, datasetname, output_dir):
    # 绘制校准曲线
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o',
             label='{} {} calibration'.format(modelname, datasetname))
    plt.plot([0, 1], [0, 1], linestyle='--', color='red',
             label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability (Positive class: 1)')
    plt.ylabel('Fraction of Positives (Positive class: 1)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    file_path = output_dir + "/" + modelname + "_" + datasetname + "_calibration.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Model {modelname} based on {datasetname}'s calibration save to：{file_path}")


def _plotRocCurve(fpr, tpr, modelname, datasetname, output_dir):
    # 绘制 ROC 曲线
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label='{} {} ROC (AUC = {:.2f})'.format(modelname, datasetname, roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('False Positive Rate (Positive label: 1)')
    plt.ylabel('True Positive Rate (Positive label: 1)')
    plt.legend(loc='lower right')
    file_path = output_dir + "/" + modelname + "_" + datasetname + "_roc.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Model {modelname} based on {datasetname}'s ROC save to：{file_path}")


def _plotPrCurve(recall, precision, ap_value, modelname, datasetname, output_dir):
    # 绘制 PR 曲线
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='blue',
             label='{} {} ROC (AP = {:.2f})'.format(modelname, datasetname, ap_value))
    plt.xlim([0.0, 1.03])
    plt.ylim([0.0, 1.03])
    plt.xlabel('Recall (Positive label: 1)')
    plt.ylabel('Precision (Positive label: 1)')
    # plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    file_path = output_dir + "/" + modelname + "_" + datasetname + "_pr.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Model {modelname} based on {datasetname}'s PR save to：{file_path}")


def plotFeatureImportance(orig_model, train_x, modelname, output_dir):
    model = orig_model.best_estimator_.named_steps[modelname]
    logger.info(f"最佳模型类型 {type(model)}")
    if not hasattr(model, "coefs_"):
        raise ValueError("The provided model does not have coefs_ attribute.")

    # 取出第一层的权重（假设输入层到隐藏层）
    weights = model.coefs_[0]

    # 计算特征的重要性（每个特征的权重之和的绝对值）
    feature_importance = np.sum(np.abs(weights), axis=1)

    # 创建数据框
    feature_importance_df = pd.DataFrame({
        'Importance': feature_importance,
        'Feature': train_x.columns.tolist()
    })

    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # 绘制特征重要性图
    plt.figure(figsize=(8, 6))
    # plt.subplot(1, 2, 1)
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'],
             color='#1f77b4', height=0.4)
    plt.xlabel('Importance')
    plt.xlim(0, max(feature_importance_df['Importance']) * 1.1)  # 适当扩展 x 轴
    plt.gca().invert_yaxis()  # 反转 y 轴，使得最重要的特征在顶部

    file_path = output_dir + "/" + modelname + "_feature_importance.pdf"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # 关闭图形以释放内存

    logger.info(f"Model {modelname}'s feature_importance save to：{file_path}")


def plotSinglePartialDependence(model, train_x, connames, output_dir):
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(model,
                                            train_x.dropna(),
                                            connames,
                                            kind="both",
                                            random_state=42,
                                            ax=ax)
    file_path = output_dir + "/partial_dependence_single_var.pdf"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Single var's partial dependence save to：{file_path}")


def plotTwoPartialDependence(model, train_x, var1, var2, output_dir):
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(model,
                                            train_x.dropna(),
                                            [var1, var2, (var1, var2)],
                                            ax=ax)
    file_path = output_dir + "/partial_dependence_" + var1 + "_" + var2 + ".pdf"
    plt.savefig(file_path)
    logger.info(f"{var1} and {var2} partial dependence save to：{file_path}")


def plotShapBeeswarmViolinBar(shapvalues, final_train_x, output_dir):
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shapvalues, max_display=len(final_train_x.columns), show=False)
    plt.tight_layout()
    file_path = output_dir + "/shap_value_impact_on_model_output_beeswarm.pdf"
    plt.savefig(file_path, format='pdf')
    plt.close()
    logger.info(f"Shap value beeswarm save to：{file_path}")

    plt.figure(figsize=(10, 6))
    shap.plots.violin(shapvalues, plot_type="layered_violin", show=False)
    plt.tight_layout()
    file_path = output_dir + "/shap_value_impact_on_model_output_violin.pdf"
    plt.savefig(file_path, format="pdf")
    plt.close()
    logger.info(f"Shap value violin save to：{file_path}")

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shapvalues, max_display=len(final_train_x.columns), show=False)
    plt.tight_layout()
    file_path = output_dir + "/shap_mean_bar.pdf"
    plt.savefig(file_path, format="pdf")
    plt.close()
    logger.info(f"Shap value mean bar save to：{file_path}")


def plotShapSingleVar(shapvalues, var_names, output_dir):
    for var_name_cell in var_names:
        shap.plots.scatter(shapvalues[:, var_name_cell], show=False)
        plt.tight_layout()
        plt.savefig(output_dir + "/shap_single_" + var_name_cell + "_0.pdf", format="pdf")
        plt.close()

        shap.plots.scatter(shapvalues[:, var_name_cell], color=shapvalues, show=False)
        plt.tight_layout()
        plt.savefig(output_dir + "/shap_single_" + var_name_cell + "_1.pdf", format="pdf")
        plt.close()


def plotShapWaterForceDesicion(shapvalues, final_train_x, output_dir):
    shap.plots.waterfall(shapvalues[0], show=False)
    plt.tight_layout()
    plt.savefig(output_dir + "/shap_waterfall.pdf", format="pdf")
    plt.close()

    shap.initjs()
    shap_force_html = shap.plots.force(shapvalues[0])
    shap.save_html(output_dir + "/shap_force.html", shap_force_html)

    shap_force2_html = shap.plots.force(shapvalues)
    shap.save_html(output_dir + "/shap_force2.html", shap_force2_html)

    shap.plots.heatmap(shapvalues, instance_order=shapvalues.sum(1), show=False)
    plt.tight_layout()
    plt.savefig(output_dir + "/shap_heat_map.pdf", format="pdf")
    plt.close()

    shap.decision_plot(shapvalues.base_values[1], shapvalues.values, final_train_x, ignore_warnings=True, show=False)
    plt.tight_layout()
    plt.savefig(output_dir + "/shap_decision_plot.pdf", format="pdf")
    plt.close()


def getModelShapeValues(model, model_name, train_x):
    if model_name == "mlp":
        def shapmodel_mlp(x):
            return model.best_estimator_[model_name].predict_proba(x)[:, 1]

        final_train_x = model.best_estimator_["prep"].transform(train_x)

        explainer_mlp = shap.Explainer(shapmodel_mlp, final_train_x)
        shapvalues_mlp = explainer_mlp(final_train_x)
        return shapvalues_mlp, final_train_x
    elif model_name == "svm":
        def shapmodel_svm(x):
            return model.best_estimator_[model_name].predict_proba(x)[:, 1]

        final_train_x = model.best_estimator_["prep"].transform(train_x)

        explainer_svm = shap.Explainer(shapmodel_svm, final_train_x)
        shapvalues_svm = explainer_svm(final_train_x)
        return shapvalues_svm, final_train_x
    else:
        logger.error(f"Model {model_name} not supported.")
        return None
