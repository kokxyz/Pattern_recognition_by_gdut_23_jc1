import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def setup_logging():
    """设置日志系统"""
    import logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(f'logs/project_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def memory_usage():
    """检查内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB"

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"✅ GPU可用: {[gpu.name for gpu in gpus]}"
        else:
            return "未检测到GPU，使用CPU模式"
    except ImportError:
        return "TensorFlow未安装，无法检测GPU"

def progress_bar(iterable, desc=None):
    """增强的进度条"""
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

def visualize_sample_images(df, images, num_samples=5):
    """可视化样本图像"""
    # 确保目录存在并跳过空图像
    out_dir = os.path.join('results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        idx = np.random.randint(0, len(df))
        img = images[idx]
        if img is None:
            # 显示空白占位
            axes[0, i].text(0.5, 0.5, 'Missing', ha='center', va='center')
            axes[1, i].text(0.5, 0.5, 'Missing', ha='center', va='center')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            continue

        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Edema Risk: {df.iloc[idx]['Risk of macular edema']}")
        axes[0, i].axis('off')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes[1, i].imshow(gray, cmap='gray')
        axes[1, i].set_title(f"Edema Risk: {df.iloc[idx]['Risk of macular edema']}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()

def check_class_balance(df, task_columns):
    """检查类别平衡性"""
    for column in task_columns:
        value_counts = df[column].value_counts().sort_index()
        print(f"\n{column} 类别分布:")
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  类别 {value}: {count} 张图像 ({percentage:.1f}%)")
    
    # 绘制类别分布图（支持任意数量的任务）
    try:
        n = max(1, len(task_columns))
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

        # 当只有一个子图时 axes 不是数组，统一处理为可迭代
        if n == 1:
            axes = [axes]

        for i, column in enumerate(task_columns):
            value_counts = df[column].value_counts().sort_index()
            axes[i].bar(value_counts.index, value_counts.values)
            axes[i].set_title(f'{column} Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')

        plt.tight_layout()

        # 确保保存目录存在
        out_dir = os.path.join('results', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'class_distribution.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"类别分布图已保存: {out_path}")
        return out_path

    except Exception as e:
        print(f"绘制类别分布图时出错: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close()
        except:
            pass
        return None

def generate_requirements_file():
    """生成requirements.txt文件"""
    requirements = """
tensorflow>=2.8.0
opencv-python>=4.5.5
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.60.0
scikit-image>=0.19.0
joblib>=1.1.0
xgboost>=1.5.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("requirements.txt 文件已生成")