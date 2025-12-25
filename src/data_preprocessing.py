import random
import os
import re
import traceback
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

def augment_image(img, config):
    """对单张图像进行数据增强，增强参数由config['augmentation']指定"""
    import cv2
    import numpy as np
    aug_cfg = config.get('augmentation', {})
    if not aug_cfg.get('enable', False):
        return img

    # 随机旋转
    angle = random.uniform(-aug_cfg.get('rotation', 0), aug_cfg.get('rotation', 0))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 随机平移
    ws = aug_cfg.get('width_shift', 0)
    hs = aug_cfg.get('height_shift', 0)
    tx = int(random.uniform(-ws, ws) * w)
    ty = int(random.uniform(-hs, hs) * h)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 随机缩放
    zoom = 1 + random.uniform(-aug_cfg.get('zoom', 0), aug_cfg.get('zoom', 0))
    if zoom != 1:
        nh, nw = int(h * zoom), int(w * zoom)
        img = cv2.resize(img, (nw, nh))
        # 裁剪或填充回原尺寸
        if zoom > 1:
            # 裁剪中心
            startx = (nw - w) // 2
            starty = (nh - h) // 2
            img = img[starty:starty+h, startx:startx+w]
        else:
            # 填充
            pad_x = (w - nw) // 2
            pad_y = (h - nh) // 2
            img = cv2.copyMakeBorder(img, pad_y, h-nh-pad_y, pad_x, w-nw-pad_x, cv2.BORDER_REFLECT)

    # 随机亮度/对比度
    brightness = aug_cfg.get('brightness', 0)
    contrast = aug_cfg.get('contrast', 0)
    if brightness > 0 or contrast > 0:
        alpha = 1 + random.uniform(-contrast, contrast)
        beta = int(255 * random.uniform(-brightness, brightness))
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 随机水平/垂直翻转
    if aug_cfg.get('horizontal_flip', False) and random.random() < 0.5:
        img = cv2.flip(img, 1)
    if aug_cfg.get('vertical_flip', False) and random.random() < 0.5:
        img = cv2.flip(img, 0)

    return img
class DataPreprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r',encoding='UTF-8') as file:
            self.config = yaml.safe_load(file)
        
        self.annotations_dir = self.config['data_paths']['annotations_dir']
        self.annotations_csv_dir = self.config['data_paths']['annotations_csv_dir']
        self.images_base_dir = self.config['data_paths']['images_base_dir']
        
        # 创建必要的目录
        os.makedirs(self.annotations_csv_dir, exist_ok=True)
        os.makedirs(self.config['data_paths']['processed_dir'], exist_ok=True)
    
    def convert_xls_to_csv(self):
        """将XLS文件转换为CSV格式"""
        print("开始转换XLS文件到CSV格式...")
        # 自动发现 annotations_dir 下的所有 xls/xlsx 文件并转换
        files = os.listdir(self.annotations_dir) if os.path.exists(self.annotations_dir) else []
        xls_files = [f for f in files if f.lower().endswith(('.xls', '.xlsx'))]

        if not xls_files:
            print(f"在目录 {self.annotations_dir} 中未发现任何 .xls/.xlsx 文件")
            return

        def try_read_excel(path):
            """尝试使用多种方法读取 Excel 文件，返回 DataFrame 或 None。"""
            ext = os.path.splitext(path)[1].lower()
            last_err = None

            # 优先按扩展名选择 engine
            if ext == '.xlsx':
                for engine in ('openpyxl', None):
                    try:
                        if engine:
                            return pd.read_excel(path, engine=engine)
                        else:
                            return pd.read_excel(path)
                    except Exception as e:
                        last_err = e

            elif ext == '.xls':
                # 旧式 xls 文件通常需要 xlrd (<=1.2) 支持
                for engine in ('xlrd', None):
                    try:
                        if engine:
                            return pd.read_excel(path, engine=engine)
                        else:
                            return pd.read_excel(path)
                    except Exception as e:
                        last_err = e

            # 最后尝试以 HTML 表格解析（有些 .xls 实际上是 HTML）
            try:
                tables = pd.read_html(path)
                if tables:
                    return tables[0]
            except Exception as e:
                last_err = e

            # 作为最后尝试，尝试按文本解析为 CSV（如果文件实际上是以制表符/逗号分隔）
            try:
                return pd.read_csv(path, encoding='utf-8', engine='python')
            except Exception as e:
                last_err = e

            # 无法读取
            raise RuntimeError(f"无法读取 Excel 文件 {path}. 最后错误: {last_err}")

        for fname in xls_files:
            xls_file = os.path.join(self.annotations_dir, fname)
            base_name = os.path.splitext(fname)[0]
            csv_file = os.path.join(self.annotations_csv_dir, f"{base_name}.csv")

            try:
                df = try_read_excel(xls_file)

                 #如果读取成功，打印简单诊断信息
                if hasattr(df, 'shape'):
                    print("   ")
                if hasattr(df, 'columns'):
                    print("   ")
                    df.columns = df.columns.str.strip()

                if getattr(df, 'empty', False):
                    print(f"警告: 从 {xls_file} 读取到空表，跳过保存 CSV。请检查文件内容或尝试用 Excel 打开并另存为 .xlsx")
                    continue

                # 保存为CSV（使用 utf-8-sig 保持 Excel 兼容 BOM）
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')

            except Exception as e:
                print(f"转换文件 {xls_file} 时出错: {e}")
                traceback.print_exc()
 
    def load_and_merge_annotations(self):
        """加载并合并所有标注文件"""
        print("加载并合并标注文件...")
        
        all_annotations = []
        # 遍历 annotations_csv_dir 下所有 CSV 文件，智能推断所属 Base 子目录
        if not os.path.exists(self.annotations_csv_dir):
            print(f"标注 CSV 目录不存在: {self.annotations_csv_dir}")

        csv_files = [os.path.join(self.annotations_csv_dir, f) for f in os.listdir(self.annotations_csv_dir)
                     if f.lower().endswith('.csv')]

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # 试图从文件名中推断 base folder，例如 Annotation_Base11 -> Base11
                basename = os.path.basename(csv_file)
                m = re.search(r'(Base\d+)', basename, re.IGNORECASE)
                if m:
                    base_folder = m.group(1)
                else:
                    # 如果文件名中找不到，通过查看 images 目录下哪个子目录包含首个样本图像名来推断
                    base_folder = None
                    sample_names = df['Image name'].dropna().astype(str).head(5).tolist()
                    for sample in sample_names:
                        for candidate in os.listdir(self.images_base_dir):
                            candidate_path = os.path.join(self.images_base_dir, candidate, sample.strip())
                            if os.path.exists(candidate_path):
                                base_folder = candidate
                                break
                        if base_folder:
                            break

                if base_folder is None:
                    # 无法推断，则记录警告并继续（此文件可能需要人工检查）
                    print(f"无法从文件或图像目录推断 Base 文件夹: {basename}，将尝试默认路径")
                    base_folder = ''

                # 添加图像路径信息
                if base_folder:
                    df['base_folder'] = base_folder
                    df['image_path'] = df['Image name'].apply(
                        lambda x: os.path.join(self.images_base_dir, base_folder, str(x).strip())
                    )
                else:
                    # 当没有 base_folder 时，尝试在 images_base_dir 的所有子目录中寻找图像
                    def find_image_path(img_name):
                        img_name = str(img_name).strip()
                        for candidate in os.listdir(self.images_base_dir):
                            candidate_path = os.path.join(self.images_base_dir, candidate, img_name)
                            if os.path.exists(candidate_path):
                                return candidate_path
                        return os.path.join(self.images_base_dir, img_name)

                    df['base_folder'] = df['Image name'].apply(lambda x: '')
                    df['image_path'] = df['Image name'].apply(find_image_path)

                # 验证图像路径是否存在
                df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))

                # 只保留存在的图像
                df = df[df['image_exists'] == True]

                if not df.empty:
                    all_annotations.append(df)
                    #print(f"{os.path.basename(csv_file)}: 加载 {len(df)} 条有效记录 (base: {df['base_folder'].iloc[0]})")

            except Exception as e:
                print(f"加载文件 {csv_file} 时出错: {e}")
        
        if not all_annotations:
            raise ValueError("没有找到任何有效的标注数据")
        
        # 合并所有数据
        combined_df = pd.concat(all_annotations, ignore_index=True)
        
        # 数据清洗
        combined_df = self.clean_annotation_data(combined_df)
        
        #print(f"总共加载 {len(combined_df)} 条有效记录")
        return combined_df
    
    def clean_annotation_data(self, df):
        """清理标注数据（仅保留黄斑水肿风险标签）"""
        # 只保留并校验黄斑水肿风险标签
        df = df.dropna(subset=['Risk of macular edema'])

        # 确保标签为整数
        df['Risk of macular edema'] = df['Risk of macular edema'].astype(int)

        return df
    
    def load_images(self, df, target_size=(224, 224)):
        """加载并预处理图像（支持数据增强）"""
        print("开始加载图像...")

        images = []
        failed_images = []

        # 图像质量统计
        quality_stats = {
            'too_small': 0,
            'corrupted': 0,
            'success': 0
        }

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                image_path = row['image_path']

                if os.path.getsize(image_path) < 1024:
                    quality_stats['too_small'] += 1
                    raise ValueError("图像文件过小，可能已损坏")

                img = cv2.imread(image_path)

                if img is None:
                    quality_stats['corrupted'] += 1
                    try:
                        img = np.array(Image.open(image_path))
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        raise ValueError(f"PIL读取失败: {e}")

                if img.shape[0] < 50 or img.shape[1] < 50:
                    quality_stats['too_small'] += 1
                    raise ValueError("图像尺寸过小")

                # 调整图像大小
                img = cv2.resize(img, target_size)

                # 图像增强：对比度自适应直方图均衡
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
                img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

                # 高斯模糊去噪
                img = cv2.GaussianBlur(img, (3, 3), 0)

                # 数据增强（仅对小类样本增强，主类不增强）
                label = row.get('Risk of macular edema', None)
                aug_cfg = self.config.get('augmentation', {})
                n_aug_1 = aug_cfg.get('n_aug_1', 8)
                n_aug_2 = aug_cfg.get('n_aug_2', 12)
                if aug_cfg.get('enable', False) and label in [1, 2]:
                    n_aug = n_aug_1 if label == 1 else n_aug_2
                    for _ in range(n_aug):
                        aug_img = augment_image(img, self.config)
                        images.append(aug_img)
                        quality_stats['success'] += 1

                images.append(img)
                quality_stats['success'] += 1

            except Exception as e:
                print(f"加载图像失败: {image_path}, 错误: {e}")
                failed_images.append(image_path)
                images.append(None)

        print(f"\n图像加载质量统计:")
        print(f"成功加载: {quality_stats['success']}")
        print(f"文件过小: {quality_stats['too_small']}")
        print(f"损坏文件: {quality_stats['corrupted']}")

        if failed_images:
            print(f"总共 {len(failed_images)} 张图像加载失败")

        return images, failed_images
    
    #新增：配置验证方法
    def validate_config(self):
        """验证配置文件完整性"""
        required_paths = [
            self.annotations_dir,
            self.images_base_dir
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                print(f"警告: 路径不存在: {path}")
                return False
        
        print("配置文件验证通过")
        return True
    
    #新增：数据完整性检查
    def check_data_integrity(self, df):
        """检查数据完整性（仅针对黄斑水肿风险标签）"""
        print("\n数据完整性检查:")
        print(f"总记录数: {len(df)}")
        print(f"有效图像路径: {df['image_exists'].sum()}")
        missing_labels = df['Risk of macular edema'].isnull().sum()
        print(f"缺失标签: {missing_labels}")

        # 检查类别平衡
        self._check_class_balance(df)
        return True
    
    def _check_class_balance(self, df):
        """检查类别平衡性（仅黄斑水肿风险）"""
        print("\n类别分布:")
        value_counts = df['Risk of macular edema'].value_counts().sort_index()
        print("\nRisk of macular edema:")
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  类别 {value}: {count} 条 ({percentage:.1f}%)")

    
    def prepare_data_splits(self, df, images, task_column):
        """准备训练和测试数据分割"""
        # 过滤掉加载失败的图像
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        valid_images = [images[i] for i in valid_indices]
        
        print(f"有效图像数量: {len(valid_images)}")
        
        # 提取标签
        labels = valid_df[task_column].values
        
        # 分层分割训练集和测试集（8/2）
        X_train, X_test, y_train, y_test = train_test_split(
            range(len(valid_images)),  # 使用索引而不是实际图像数据
            labels,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=labels
        )
        
        # 根据索引获取实际的图像数据
        train_images = [valid_images[i] for i in X_train]
        test_images = [valid_images[i] for i in X_test]
        train_df = valid_df.iloc[X_train]
        test_df = valid_df.iloc[X_test]
        
        return train_images, test_images, train_df, test_df, y_train, y_test
    
    def save_processed_data(self, data_dict, filename):
        """保存处理后的数据"""
        filepath = os.path.join(self.config['data_paths']['processed_dir'], filename)
        np.save(filepath, data_dict)
        print(f"数据已保存到: {filepath}")
    
    def load_processed_data(self, filename):
        """加载处理后的数据"""
        filepath = os.path.join(self.config['data_paths']['processed_dir'], filename)
        if os.path.exists(filepath):
            return np.load(filepath, allow_pickle=True).item()
        return None