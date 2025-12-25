import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 引入绘图库
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


def load_data_pixels():
    """读取数据 (纯像素模式)"""
    print(">>> 正在读取数据 (纯像素模式)...")
    images, labels = [], []
    image_paths = {}

    # 1. 扫描图片
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.tif', '.jpg', '.png')):
                image_paths[file] = os.path.join(root, file)

    # 2. 读取表格
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
                        images.append(img.flatten())  # 展平像素
                        labels.append(grade)
        except:
            pass
    return np.array(images), np.array(labels)


def main():
    # 1. 加载数据
    X, y = load_data_pixels()
    if len(X) == 0:
        print("错误：未读取到数据！")
        return

    # 2. 标准化 (只做一次，节省时间)
    print(">>> 正在进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 定义要测试的维度列表：从 10 到 200，步长为 10
    dimensions = range(10, 201, 10)
    accuracies = []  # 用来存每次的准确率

    print("\n" + "=" * 40)
    print(f"{'维度 (Dims)':<15} | {'准确率 (Accuracy)':<15}")
    print("-" * 40)

    # 3. 循环测试不同维度
    for dim in dimensions:
        # A. PCA 降维
        pca = PCA(n_components=dim)
        X_pca = pca.fit_transform(X_scaled)

        # B. 划分数据 (保持 random_state 一致，确保对比公平)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

        # C. 训练 KNN (K=5)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        # D. 记录结果
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"{dim:<15} | {acc:.2%}")

    print("=" * 40)

    # 4. 绘制折线图 (模仿报告中的图)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, accuracies, marker='o', linestyle='-', color='b', label='KNN + Pure PCA')

        plt.title('KNN Accuracy vs PCA Dimensions', fontsize=14)
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
        print(f"\n[提示] 绘图失败 (可能是缺少 matplotlib 库)，但上方数据已列出。\n错误信息: {e}")


if __name__ == "__main__":
    main()