import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve

# 自动保存图像函数
def save_zcp(savefile, figname, width=8, height=6):
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    plt.gcf().set_size_inches(width, height)
    for ext in ['pdf', 'png', 'svg']:
        path = os.path.join(savefile, f"{figname}.{ext}")
        plt.savefig(path, format=ext, dpi=300, bbox_inches='tight')

# 学习曲线

def plot_learning_curve(estimator, X, y, title='Learning Curve', cv=10,
                             savefile='Figure', figname='learning_curve',
                             width=6, height=5, ylim=(0.8, 1.0),
                             train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator, X=X, y=y, train_sizes=train_sizes, cv=cv,
        scoring=scoring, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(width, height))
    plt.plot(train_sizes, train_mean, color='#CC5490', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#CC5490')
    plt.plot(train_sizes, test_mean, color='#9589DF', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#9589DF')

    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim(*ylim)
    plt.title(title)
    plt.tight_layout()

    save_zcp(savefile=savefile, figname=figname, width=width, height=height)
    plt.show()

# 验证曲线

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=10, scoring='accuracy',
                               xlabel=None, title='Validation Curve', logx=True, ylim=(0.8, 1.0),
                               savefile=None, figname=None, width=6, height=5):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                 param_range=param_range, cv=cv,
                                                 scoring=scoring, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(width, height))
    plt.plot(param_range, train_mean, color='#CC5490', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#CC5490')
    plt.plot(param_range, test_mean, color='#9589DF', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#9589DF')

    if logx:
        plt.xscale('log')
    plt.xlabel(xlabel or param_name)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(*ylim)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)

    plt.show()

# GridSearchCV - 单参数曲线

def plot_grid_search_curve(grid_search, param_name, logx=True, xlabel=None, title='Grid Search Result',
                            ylim=(0.8, 1.0), savefile=None, figname=None, width=6, height=5):
    results = grid_search.cv_results_
    param_range = results['param_' + param_name].data
    mean_test_score = results['mean_test_score']
    std_test_score = results['std_test_score']

    plt.figure(figsize=(width, height))
    plt.plot(param_range, mean_test_score, color='#9589DF', marker='s', linestyle='-', label='Validation score')
    plt.fill_between(param_range, mean_test_score - std_test_score, mean_test_score + std_test_score, alpha=0.15, color='#9589DF')

    if logx:
        plt.xscale('log')
    plt.xlabel(xlabel or param_name)
    plt.ylabel('Score')
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)

    plt.show()

# GridSearchCV - 多曲线对比

def plot_grid_search_multicurve(grid_search, x_param, curve_param, logx=True,
                                 title='Multi-Param Validation Curve', xlabel=None, ylabel='Validation Score',
                                 ylim=(0.8, 1.0), savefile=None, figname=None, width=6, height=5):
    results = grid_search.cv_results_
    x_vals = sorted(set(results['param_' + x_param].data))
    curve_vals = sorted(set(results['param_' + curve_param].data))

    plt.figure(figsize=(width, height))
    for val in curve_vals:
        mask = results['param_' + curve_param] == val
        x = results['param_' + x_param][mask].data
        y = results['mean_test_score'][mask]
        std = results['std_test_score'][mask]
        plt.plot(x, y, marker='o', label=f'{curve_param}={val}')
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    if logx:
        plt.xscale('log')
    plt.xlabel(xlabel or x_param)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(*ylim)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)

    plt.show()

# GridSearchCV - 2D 热力图

def plot_grid_search_heatmap(grid_search, param_x, param_y, scoring_name='Validation Score',
                              fmt='.3f', cmap='viridis', title='Grid Search 2D Heatmap',
                              savefile=None, figname=None, width=7, height=5):
    results = grid_search.cv_results_
    scores = results['mean_test_score']
    x_vals = sorted(list(set(results['param_' + param_x].data)))
    y_vals = sorted(list(set(results['param_' + param_y].data)))

    score_matrix = np.empty((len(y_vals), len(x_vals)))
    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            mask = (results['param_' + param_x] == x_val) & (results['param_' + param_y] == y_val)
            score_matrix[i, j] = scores[mask][0]

    df_scores = pd.DataFrame(score_matrix, index=y_vals, columns=x_vals)
    plt.figure(figsize=(width, height))
    sns.heatmap(df_scores, annot=True, fmt=fmt, cmap=cmap, xticklabels=x_vals, yticklabels=y_vals)
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(title)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)

    plt.show()


# 混淆矩阵

def plot_confusion_matrix(y_true, y_pred, labels=None, cmap='Blues', title='Confusion Matrix',
                               savefile=None, figname=None, width=6, height=5):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(width, height))
    disp.plot(cmap=cmap, ax=ax, values_format='d')
    plt.title(title)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)
    plt.show()

# ROC 曲线
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             roc_auc_score)

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    bootstrapped_fpr = []
    bootstrapped_tpr = []
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_score[indices]))
        fpr_i, tpr_i, _ = roc_curve(y_true[indices], y_score[indices])
        bootstrapped_fpr.append(fpr_i)
        bootstrapped_tpr.append(tpr_i)
    sorted_scores = np.sort(aucs)
    lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    return lower, upper, aucs, bootstrapped_fpr, bootstrapped_tpr

# 带置信区间阴影的 ROC 曲线（基于 bootstrap ROC 曲线集成）
def plot_roc_curve(y_true, y_score, title='ROC Curve with CI',
                            savefile=None, figname=None, width=6, height=5,
                            n_bootstraps=1000, ci=0.95, seed=42,
                            curve_color='darkorange', ci_color='orange', ref_color='navy'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ci_low, ci_high, aucs, boot_fprs, boot_tprs = bootstrap_auc_ci(
        y_true, y_score, n_bootstraps=n_bootstraps, ci=ci, seed=seed)

    # 主图
    fig, ax = plt.subplots(figsize=(width, height))

    # 绘制 bootstrap ROC 曲线（半透明）
    for fpr_b, tpr_b in zip(boot_fprs, boot_tprs):
        ax.plot(fpr_b, tpr_b, color=ci_color, alpha=0.03)

    # 绘制主 ROC 曲线
    ax.plot(fpr, tpr, color=curve_color, lw=2,
             label=f'ROC (AUC = {roc_auc:.3f}, CI: {ci_low:.3f}–{ci_high:.3f})')
    ax.plot([0, 1], [0, 1], color=ref_color, lw=1, linestyle='--')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(False)
    #ax.grid(False)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)
    plt.show()



# Precision-Recall 曲线
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y_true, y_score, title='Precision-Recall Curve', savefile=None, figname=None, width=6, height=5):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(width, height))
    plt.plot(recall, precision, lw=2, color='teal')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)
    plt.show()


from sklearn.base import clone

def update_model_with_best_params(grid_search):
    """
    从 GridSearchCV 中提取最优参数，返回更新后的 estimator 模型（兼容 pipeline 或单一模型）

    参数：
    - grid_search: 已拟合好的 GridSearchCV 对象

    返回：
    - 一个重新构造、参数已更新的 estimator（即 best_estimator_ 的深拷贝）
    """
    best_params = grid_search.best_params_
    base_estimator = clone(grid_search.estimator)  # 深拷贝原始模型
    base_estimator.set_params(**best_params)
    return base_estimator

# 交叉验证 ROC 曲线
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

def plot_cv_mean_roc(X_train, y_train, pipeline=None, n_splits=10, width=7, height=5,savefile=None, figname=None):
    if pipeline is None:
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(
                penalty='l2',
                max_iter=10000,
                random_state=1,
                solver='lbfgs',
                C=100
            )
        )

    X_train_np = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
    y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train

    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1).split(X_train_np, y_train_np))

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    plt.figure(figsize=(width, height))

    for i, (train_idx, test_idx) in enumerate(cv):
        probas = pipeline.fit(X_train_np[train_idx], y_train_np[train_idx])\
                         .predict_proba(X_train_np[test_idx])

        fpr, tpr, _ = roc_curve(y_train_np[test_idx], probas[:, 1])
        roc_auc = auc(fpr, tpr)

        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        plt.plot(fpr, tpr, label=f'ROC fold {i} (area = {roc_auc:.2f})', alpha=0.3)

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='#D63A79', label=f'Mean ROC (area = {mean_auc:.2f})', lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance', alpha=0.8)
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='gray', label='Perfect', alpha=0.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(False)
    if savefile and figname:
        save_zcp(savefile, figname, width=width, height=height)
    plt.show()

