import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 引入绘图库
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================= 配置 =================
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


def load_data_hog():
    print(">>> [方案2] 正在读取数据并提取 HOG (18方向)...")
    print("    (这一步比较慢，请耐心等待，提取完后会开始快速循环测试)")
    features, labels = [], []
    image_paths = {}

    # 1. 扫描图片
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.tif', '.jpg', '.png')):
                image_paths[file] = os.path.join(root, file)

    # 2. 读取表格并提取 HOG
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
                        # 报告参数: orientations=18, Gamma校正
                        fd = hog(img, orientations=18, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
                        features.append(fd)
                        labels.append(grade)
                        count += 1
                        if count % 100 == 0:
                            print(f"    已提取 {count} 张图片的特征...")
        except:
            pass

    print(f">>> HOG 特征提取完毕，共 {len(features)} 张。")
    return np.array(features), np.array(labels)


def main():
    # 1. 加载 HOG 数据 (只做一次)
    X, y = load_data_hog()
    if len(X) == 0:
        print("错误：未读取到数据！")
        return

    # 2. 标准化 (只做一次)
    print(">>> 正在进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 定义测试维度：10 到 200，步长 10
    dimensions = range(10, 201, 10)
    accuracies = []

    print("\n" + "=" * 45)
    print(f"{'维度 (Dims)':<15} | {'准确率 (Accuracy)':<15}")
    print("-" * 45)

    # 3. 循环测试不同维度
    for dim in dimensions:
        # A. PCA 降维
        pca = PCA(n_components=dim)
        X_pca = pca.fit_transform(X_scaled)

        # B. 划分数据
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

        # C. 训练 KNN (无 SMOTE)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        # D. 记录结果
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"{dim:<15} | {acc:.2%}")

    print("=" * 45)

    # 4. 绘制折线图
    try:
        plt.figure(figsize=(10, 6))
        # 橙色线条，对应报告中的 PCA+HOG 颜色
        plt.plot(dimensions, accuracies, marker='o', linestyle='-', color='orange', label='KNN + PCA + HOG')

        plt.title('KNN Accuracy vs Dimensions (PCA + HOG)', fontsize=14)
        plt.xlabel('PCA Dimensions (10-200)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 标记最高点
        max_acc = max(accuracies)
        max_dim = dimensions[accuracies.index(max_acc)]
        plt.annotate(f'Max: {max_acc:.2%}', xy=(max_dim, max_acc), xytext=(max_dim, max_acc + 0.01),
                     arrowprops=dict(facecolor='red', shrink=0.05))

        plt.show()
        print("\n>>> 折线图已生成！")
    except Exception as e:
        print(f"\n[提示] 绘图失败，但数据已生成。错误: {e}")


if __name__ == "__main__":
    main()