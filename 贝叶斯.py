import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb  # 关键库

# ================= 配置区域 =================
DATA_DIR = "./data"
LABEL_FILES = [
    "Annotation_Base11_update20161024.xls", "Annotation_Base12.xls",
    "Annotation_Base13_update20161024.xls", "Annotation_Base14.xls",
    "Annotation_Base21.xls", "Annotation_Base22.xls",
    "Annotation_Base23.xls", "Annotation_Base24.xls",
    "Annotation_Base31.xls", "Annotation_Base32.xls",
    "Annotation_Base33.xls", "Annotation_Base34.xls"
]
IMG_SIZE = (256, 256)


def load_data_all():
    print(">>> 正在读取所有图片数据 (HOG + 纯像素)...")
    raw_pixels = []
    hog_features = []
    labels = []
    image_paths = {}

    # 1. 扫描图片
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.tif', '.jpg', '.png')):
                image_paths[file] = os.path.join(root, file)

    # 2. 读取表格并提取特征
    count = 0
    for excel_file in LABEL_FILES:
        if not os.path.exists(excel_file): continue
        try:
            df = pd.read_excel(excel_file)
            for index, row in df.iterrows():
                img_name = str(row.iloc[0])
                if not img_name.lower().endswith('.tif'): img_name += '.tif'
                grade = int(row.iloc[2])

                if img_name in image_paths:
                    img = cv2.imread(image_paths[img_name], cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)

                        # 1. 存纯像素 (用于 PCA 方案)
                        raw_pixels.append(img.flatten())

                        # 2. 存 HOG (用于 HOG 方案)
                        fd = hog(img, orientations=18, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
                        hog_features.append(fd)

                        labels.append(grade)
                        count += 1
                        if count % 100 == 0: print(f"    已处理 {count} 张...")
        except:
            pass

    print(f">>> 数据加载完毕，共 {len(labels)} 张。")
    return np.array(raw_pixels), np.array(hog_features), np.array(labels)


def get_accuracy_cv(X, y, use_smote=False):
    """核心函数：计算不同维度下的十折交叉验证准确率"""
    dims = range(10, 201, 10)
    acc_scores = []

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 十折交叉验证器
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print(f"    正在进行 10折交叉验证 (SMOTE={use_smote})...")

    for d in dims:
        # 1. PCA 降维
        pca = PCA(n_components=d)
        X_pca = pca.fit_transform(X_scaled)

        # 2. 构建分类器流水线
        if use_smote:
            # 如果用 SMOTE，必须用 pipeline，确保只在训练集上做 SMOTE，验证集不动
            model = make_pipeline_imb(SMOTE(random_state=42), GaussianNB())
        else:
            model = GaussianNB()

        # 3. 计算 10 次的平均分
        scores = cross_val_score(model, X_pca, y, cv=cv, scoring='accuracy')
        avg_score = scores.mean()
        acc_scores.append(avg_score)

        # 打印进度 (可选)
        # print(f"       Dim={d}: {avg_score:.2%}")

    return dims, acc_scores


def main():
    # 1. 一次性加载数据
    X_raw, X_hog, y = load_data_all()
    if len(y) == 0: return

    # 2. 运行 4 组实验

    print("\n=== 实验 1: PCA + HOG (蓝线) ===")
    dims, acc_hog = get_accuracy_cv(X_hog, y, use_smote=False)

    print("\n=== 实验 2: Pure PCA (橙线) ===")
    _, acc_pca = get_accuracy_cv(X_raw, y, use_smote=False)

    print("\n=== 实验 3: PCA + HOG + SMOTE (灰线) ===")
    _, acc_hog_smote = get_accuracy_cv(X_hog, y, use_smote=True)

    print("\n=== 实验 4: PCA + SMOTE (黄线) ===")
    _, acc_pca_smote = get_accuracy_cv(X_raw, y, use_smote=True)

    # 3. 统一绘图
    plt.figure(figsize=(10, 6))

    plt.plot(dims, acc_hog, marker='o', label='PCA + HOG')  # 蓝
    plt.plot(dims, acc_pca, marker='o', label='PCA')  # 橙
    plt.plot(dims, acc_hog_smote, marker='o', label='PCA+HOG+SMOTE')  # 灰
    plt.plot(dims, acc_pca_smote, marker='o', label='PCA+SMOTE')  # 黄

    plt.title('Naive Bayes Accuracy (10-Fold CV)', fontsize=14)
    plt.xlabel('PCA Dimensions (10-200)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0.2, 0.6)  # 根据报告图 3.6-1 锁定纵坐标范围

    plt.show()



if __name__ == "__main__":
    main()