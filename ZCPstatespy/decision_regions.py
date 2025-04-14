from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def plot_decision_regions(X, y, classifier, pca=None, resolution=0.02,
                          xlabel='Feature 1', ylabel='Feature 2', title='Decision Regions',
                          highlight_errors=False, show_accuracy=True, accuracy_position='title'):
    """
    参数：
    - X, y : 输入数据和标签（通常是 2D PCA 降维后的）
    - classifier : 分类器对象（必须有 .predict() 方法）
    - pca : 如果使用了 PCA，传入已拟合的对象以显示解释度
    - highlight_errors : 是否高亮预测错误的点
    - show_accuracy : 是否显示准确率
    - accuracy_position : 准确率显示位置，'title' 或 'corner'
    """
    markers = ('o', 's', '^', 'v', '<')
    colors = ('#E03870', '#7550C8', '#4874CB', '#00B050', '#44546A')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 创建决策区域网格
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 预测 & 分类
    y_pred = classifier.predict(X)

    for idx, cl in enumerate(np.unique(y)):
        is_class = (y == cl)
        plt.scatter(X[is_class, 0], X[is_class, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    if highlight_errors:
        correct_idx = np.where(y_pred == y)[0]
        error_idx = np.where(y_pred != y)[0]

        plt.scatter(X[correct_idx, 0], X[correct_idx, 1],
                    facecolors='none',
                    edgecolors='green',
                    marker='o',
                    s=100,
                    linewidths=1.5,
                    label='Correct')

        plt.scatter(X[error_idx, 0], X[error_idx, 1],
                    facecolors='none',
                    edgecolors='red',
                    marker='o',
                    s=100,
                    linewidths=1.5,
                    label='Wrong')

    # 坐标轴与标题
    accuracy = accuracy_score(y, y_pred)

    if show_accuracy and accuracy_position == 'title':
        full_title = f"{title} (Accuracy: {accuracy:.2%})"
    else:
        full_title = title

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(full_title, fontsize=14, fontweight='bold', loc='center')
    plt.legend()

    # 精度显示在角落
    if show_accuracy and accuracy_position == 'corner' and accuracy_position != 'title':
        plt.text(0.01, 0.01,
                 f"Accuracy: {accuracy:.2%}",
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 ha='left', va='bottom')

    # 显示 PCA 主成分解释度
    if pca is not None and hasattr(pca, 'explained_variance_ratio_'):
        var_ratio = pca.explained_variance_ratio_
        pc1 = f"PC1: {var_ratio[0]*100:.2f}%"
        pc2 = f"PC2: {var_ratio[1]*100:.2f}%"
        plt.text(0.99, 0.01,
                 f"{pc1}\n{pc2}",
                 transform=plt.gca().transAxes,
                 ha='right', va='bottom',
                 fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #plt.tight_layout()
    #plt.show()
