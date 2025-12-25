class MultiDeepFeatureExtractor:
    """多模型深度特征融合：EfficientNetB3+ResNet50"""
    def __init__(self, config):
        self.config = config
        # EfficientNetB3
        from tensorflow.keras.applications import EfficientNetB3
        eff_input_size = (300, 300)
        eff_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=eff_input_size + (3,))
        from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
        self.eff_model = tf.keras.models.Model(inputs=eff_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(eff_model.output))
        self.eff_preprocess = eff_preprocess
        self.eff_input_size = eff_input_size
        eff_model.trainable = False

        # ResNet50
        from tensorflow.keras.applications import ResNet50
        res_input_size = (224, 224)
        res_model = ResNet50(weights='imagenet', include_top=False, input_shape=res_input_size + (3,))
        from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
        self.res_model = tf.keras.models.Model(inputs=res_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(res_model.output))
        self.res_preprocess = res_preprocess
        self.res_input_size = res_input_size
        res_model.trainable = False

        print("已加载EfficientNetB3与ResNet50用于特征融合")

    def extract_features(self, images):
        eff_imgs = []
        res_imgs = []
        for img in images:
            # EfficientNetB3预处理
            eff_img = cv2.resize(img, self.eff_input_size)
            eff_img = self.eff_preprocess(eff_img.astype(np.float32))
            eff_imgs.append(eff_img)
            # ResNet50预处理
            res_img = cv2.resize(img, self.res_input_size)
            res_img = self.res_preprocess(res_img.astype(np.float32))
            res_imgs.append(res_img)
        eff_batch = np.array(eff_imgs)
        res_batch = np.array(res_imgs)
        eff_feats = self.eff_model.predict(eff_batch, verbose=0)
        res_feats = self.res_model.predict(res_batch, verbose=0)
        # 拼接特征
        return np.hstack([eff_feats, res_feats])
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import joblib
from tqdm import tqdm

class DeepFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_pre_trained_model()
    
    def load_pre_trained_model(self):
        """加载预训练模型"""
        model_name = self.config['feature_extraction']['deep_learning']['model_name']
        input_size = tuple(self.config['feature_extraction']['deep_learning']['input_size'])
        name = model_name.lower()
        # 根据模型选择相应的预处理函数和基础模型
        if name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_size + (3,)
            )
            self.preprocess = tf.keras.applications.resnet50.preprocess_input

        elif name == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_size + (3,)
            )
            self.preprocess = tf.keras.applications.vgg16.preprocess_input

        elif name == 'efficientnetb3':
            from tensorflow.keras.applications import EfficientNetB3
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=input_size + (3,)
            )
            # EfficientNet 的预处理函数
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
            self.preprocess = eff_preprocess

        elif name == 'densenet201':
            from tensorflow.keras.applications import DenseNet201
            base_model = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=input_size + (3,)
            )
            from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess
            self.preprocess = dn_preprocess

        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 获取指定层的输出
        layer_name = self.config['feature_extraction']['deep_learning'].get('feature_layer', 'avg_pool')
        if layer_name == 'avg_pool':
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            self.model = Model(inputs=base_model.input, outputs=x)
        else:
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

        # 冻结基础模型权重
        base_model.trainable = False

        print(f"已加载预训练模型: {model_name}")
    
    def extract_features(self, images):
        """提取深度特征"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 预处理图像
        processed_images = []
        for img in images:
            # 调整图像大小
            target_size = self.config['feature_extraction']['deep_learning']['input_size']
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size)
            
            # 预处理输入（使用模型对应的 preprocess）
            preprocess_fn = getattr(self, 'preprocess', None)
            if preprocess_fn is None:
                # 回退到 ResNet 的 preprocess（向后兼容）
                from tensorflow.keras.applications.resnet50 import preprocess_input as _pre
                preprocess_fn = _pre
            img = preprocess_fn(img.astype(np.float32))
            processed_images.append(img)
        
        # 转换为批量格式
        batch = np.array(processed_images)
        
        # 提取特征
        features = self.model.predict(batch, verbose=0)
        return features
    # 已移除重复的 extract_deep_features，keep 一致的 extract_features 接口

class TraditionalFeatureExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_surf_features(self, image):
        """提取SURF特征"""
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            kp, descriptors = surf.detectAndCompute(image, None)
        except Exception:
            orb = cv2.ORB_create(nfeatures=500)
            kp, descriptors = orb.detectAndCompute(image, None)

        if descriptors is None or len(descriptors) == 0:
            # 返回固定长度的零向量（便于后续拼接）
            return np.zeros(128)

        # 限制描述符数量并计算统计量
        max_features = int(self.config.get('feature_extraction', {}).get('traditional', {}).get('surf_features', 64))
        descriptors = descriptors[:max_features]
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        return np.hstack([mean_desc, std_desc])
    
    def extract_color_histogram(self, image, bins=32):
        """提取颜色直方图特征"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 计算各通道直方图
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # 归一化
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        
        return np.hstack([hist_h, hist_s, hist_v]).flatten()
    
    def extract_lbp_features(self, image):
        """提取LBP纹理特征"""
        # 为简化，使用灰度直方图代替完整的 LBP 实现，作为纹理近似
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist.astype('float32')
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist
    
    def extract_hybrid_features(self, image):
        """提取混合特征"""
        surf_features = self.extract_surf_features(image)
        color_features = self.extract_color_histogram(image)
        texture_features = self.extract_lbp_features(image)
        
        # 合并特征
        hybrid_features = np.hstack([surf_features, color_features, texture_features])
        return hybrid_features

class FeatureProcessor:
    def select_features(self, X, y, model=None, method='rfe', n_features_to_select=10, step=1, cv=5, scoring='f1_macro'):
        """
        特征选择：支持RFE和RFECV
        X: 特征矩阵
        y: 标签
        model: 基学习器（如RandomForestClassifier）
        method: 'rfe' 或 'rfecv'
        n_features_to_select: RFE保留特征数
        step: 每次移除特征数
        cv: RFECV交叉验证折数
        scoring: 评估指标
        返回: 选择后的特征、选择器对象
        """
        from sklearn.feature_selection import RFE, RFECV
        if model is None:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if method == 'rfe':
            selector = RFE(model, n_features_to_select=n_features_to_select, step=step)
            X_selected = selector.fit_transform(X, y)
            print(f"RFE选择特征数: {X_selected.shape[1]}")
        elif method == 'rfecv':
            selector = RFECV(model, step=step, cv=cv, scoring=scoring, n_jobs=-1)
            X_selected = selector.fit_transform(X, y)
            print(f"RFECV选择特征数: {X_selected.shape[1]}")
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        return X_selected, selector
    def __init__(self, config):
        self.config = config
        self.deep_extractor = None
        self.multi_deep_extractor = None
        if config['feature_extraction']['method'] == 'deep':
            self.deep_extractor = DeepFeatureExtractor(config)
        elif config['feature_extraction']['method'] == 'multi_deep':
            self.multi_deep_extractor = MultiDeepFeatureExtractor(config)
        self.traditional_extractor = TraditionalFeatureExtractor(config)
        self.pca = None
        self.scaler = None
    
    def extract_features_batch(self, images, method=None):
        """批量提取特征，支持直接PCA特征"""
        if method is None:
            method = self.config['feature_extraction']['method']

        if method == 'pca':
            # 先用hybrid提取，再直接PCA
            base_features = []
            failed_extractions = 0
            for i, image in enumerate(tqdm(images)):
                try:
                    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                        raise ValueError("无效的图像数据")
                    hybrid = self.traditional_extractor.extract_hybrid_features(image)
                    base_features.append(hybrid)
                except Exception as e:
                    print(f"特征提取失败第 {i} 张图像: {e}")
                    base_features.append(np.zeros(100))
                    failed_extractions += 1
            base_features = np.array(base_features)
            print(f"原始特征 shape: {base_features.shape}")
            # 归一化
            self.scaler = StandardScaler()
            base_features = self.scaler.fit_transform(base_features)
            # PCA
            n_components = self.config['feature_extraction'].get('pca', {}).get('n_components', 30)
            self.pca = PCA(n_components=n_components)
            features_array = self.pca.fit_transform(base_features)
            print(f"PCA特征 shape: {features_array.shape}")
            return features_array

        features = []
        failed_extractions = 0
        print(f"使用 {method} 方法提取特征...")
        for i, image in enumerate(tqdm(images)):
            try:
                if image is None:
                    raise ValueError("图像为空")
                if not isinstance(image, np.ndarray) or image.size == 0:
                    raise ValueError("无效的图像数据")
                if method == 'deep':
                    feature = self.deep_extractor.extract_features([image])[0]
                elif method == 'multi_deep':
                    feature = self.multi_deep_extractor.extract_features([image])[0]
                elif method == 'traditional':
                    feature = self.traditional_extractor.extract_hybrid_features(image)
                elif method == 'hybrid':
                    deep_feature = self.deep_extractor.extract_features([image])[0]
                    traditional_feature = self.traditional_extractor.extract_hybrid_features(image)
                    feature = np.hstack([deep_feature, traditional_feature])
                else:
                    raise ValueError(f"不支持的特征提取方法: {method}")
                if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                    raise ValueError("特征包含NaN或无限值")
                features.append(feature)
            except Exception as e:
                print(f"特征提取失败第 {i} 张图像: {e}")
                feature_shape = 2048 if method == 'deep' else 500
                features.append(np.zeros(feature_shape))
                failed_extractions += 1
        if failed_extractions > 0:
            print(f"特征提取失败: {failed_extractions} 张图像")
        features_array = np.array(features)
        print(f"特征提取完成，形状: {features_array.shape}")
        return features_array

    
    def reduce_dimensionality(self, features, n_components=100):
        """兼容旧接口，PCA已在extract_features_batch中完成时直接返回"""
        if self.config['feature_extraction']['method'] == 'pca':
            return features, self.pca
        if features.shape[1] <= n_components:
            return features, None
        self.pca = PCA(n_components=n_components)
        features_reduced = self.pca.fit_transform(features)
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA降维: {features.shape[1]} -> {n_components} 维，解释方差: {explained_variance:.3f}")
        return features_reduced, self.pca
    
    def normalize_features(self, features):
        """特征标准化"""
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(features)
        return features_normalized, self.scaler
    
    def save_preprocessing_objects(self, save_dir):
        """保存预处理对象"""
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(save_dir, 'pca.pkl'))
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
    
    def load_preprocessing_objects(self, save_dir):
        """加载预处理对象"""
        pca_path = os.path.join(save_dir, 'pca.pkl')
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)