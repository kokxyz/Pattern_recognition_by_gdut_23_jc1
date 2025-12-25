import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib
import os
import yaml

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

class ModelTrainer:
    def __init__(self, config_path="configs/config.yaml", config=None):
        # 简化配置加载：优先使用传入 config，否则用 utf-8-sig 读取 yaml
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                self.config = yaml.safe_load(f)
        
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        # 如果 config 中包含 training.epochs，则存储方便使用
        # 不对 epochs 做特殊覆盖处理，保持配置文件为单一事实源
        
        # 创建模型保存目录
        os.makedirs('models', exist_ok=True)
    
    def initialize_models(self):
        """初始化分类器模型"""
        model_config = self.config['models']
        classifiers = model_config.get('classifiers', [])

        for name in classifiers:
            params = self.config['models'].get(name, {})
            n_est = params.get('n_estimators', None)
            if isinstance(n_est, list) and len(n_est) > 0:
                try:
                    n_est = int(n_est[0])
                except Exception:
                    n_est = None


            if name == 'random_forest':
                if n_est:
                    self.models['random_forest'] = RandomForestClassifier(n_estimators=n_est, random_state=self.config['training']['random_state'], class_weight='balanced')
                else:
                    self.models['random_forest'] = RandomForestClassifier(random_state=self.config['training']['random_state'], class_weight='balanced')

            if name == 'adaboost':
                if n_est:
                    self.models['adaboost'] = AdaBoostClassifier(n_estimators=n_est, random_state=self.config['training']['random_state'])
                else:
                    self.models['adaboost'] = AdaBoostClassifier(random_state=self.config['training']['random_state'])
        
        print(f"已初始化模型: {list(self.models.keys())}")
    
    def perform_grid_search(self, X_train, y_train, model_name, task_name):
        """执行网格搜索进行参数调优"""
        print(f"\n开始对 {model_name} 进行参数调优 ({task_name})...")
        if model_name not in self.models:
            raise ValueError(f"未知模型: {model_name}")
        # 从配置中构建 param_grid（向后兼容：使用 models.<name> 下的参数列表）
        model_cfg = self.config.get('models', {}).get(model_name, {})
        param_grid = {}

        if model_name == 'random_forest':
            for k in ('n_estimators', 'max_depth', 'min_samples_split'):
                if k in model_cfg:
                    param_grid[k] = model_cfg[k]

        elif model_name == 'adaboost':
            for k in ('n_estimators', 'learning_rate'):
                if k in model_cfg:
                    param_grid[k] = model_cfg[k]

        # 如果没有在旧配置中找到 param_grid，尝试读取直接给出的 param_grid 字段（新格式）
        if not param_grid:
            param_grid = model_cfg.get('param_grid', {})

        if not param_grid:
            # 没有网格，直接在初始化模型上训练并返回
            self.models[model_name].fit(X_train, y_train)
            return self.models[model_name]

        cv_folds = model_cfg.get('cv_folds', self.config.get('training', {}).get('cv_folds', 5))
        scoring = self.config.get('training', {}).get('scoring_metric', None)

        grid_search = GridSearchCV(estimator=self.models[model_name], param_grid=param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # 保存结果
        self.best_params[f"{task_name}_{model_name}"] = grid_search.best_params_
        self.cv_results[f"{task_name}_{model_name}"] = grid_search.cv_results_
        print(f"最佳参数: {grid_search.best_params_}")
        try:
            print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        except Exception:
            pass
        return grid_search.best_estimator_
    
    def train_with_cross_validation(self, X_train, y_train, model_name, task_name):
        """使用交叉验证训练模型"""

        #新增：数据验证
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("训练数据为空")
    
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"类别数量不足，无法训练模型: {np.unique(y_train)}")
    
        print(f"开始训练 {model_name} - {task_name}")
        print(f"训练数据形状: X={len(X_train)}, y={len(y_train)}")
        print(f"类别分布: {np.bincount(y_train)}")

        if self.config['training']['use_cross_validation']:
            # 执行网格搜索并返回最佳 estimator（perform_grid_search 已训练并返回）
            best_model = self.perform_grid_search(X_train, y_train, model_name, task_name)

        else:
            # 使用默认参数训练
            best_model = self.models[model_name]
            best_model.fit(X_train, y_train)

        return best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name, task_name):
        """评估模型性能（含每类精度/召回/F1/AUC）"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        classes = sorted(np.unique(y_test))

        # 整体指标（加权平均）
        accuracy = accuracy_score(y_test, y_pred)
        precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 分类报告获取每类 precision/recall/f1/support
        report = classification_report(
            y_test,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0
        )

        # 每类准确率（One-vs-Rest 的 (TP+TN)/N）
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        total = cm.sum()
        per_class = {}
        for idx, cls in enumerate(classes):
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            tn = total - tp - fn - fp
            class_acc = (tp + tn) / total if total > 0 else 0.0

            cls_key = str(cls)
            per_class[cls] = {
                'accuracy': class_acc,
                'precision': report.get(cls_key, {}).get('precision', 0.0),
                'recall': report.get(cls_key, {}).get('recall', 0.0),
                'f1_score': report.get(cls_key, {}).get('f1-score', 0.0),
                'support': report.get(cls_key, {}).get('support', 0)
            }

        # 每类 AUC（如有概率输出）
        auc_macro = None
        if y_pred_proba is not None:
            try:
                y_true_bin = label_binarize(y_test, classes=classes)
                if y_pred_proba.shape[1] == len(classes):
                    per_class_auc = roc_auc_score(y_true_bin, y_pred_proba, average=None)
                    auc_macro = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
                    for i, cls in enumerate(classes):
                        per_class[cls]['auc'] = per_class_auc[i]
                else:
                    for cls in classes:
                        per_class[cls]['auc'] = None
            except Exception:
                # 概率无法用于 AUC 时保持 None
                for cls in classes:
                    per_class[cls]['auc'] = None
        else:
            for cls in classes:
                per_class[cls]['auc'] = None

        metrics = {
            # 保持向后兼容的扁平字段
            'accuracy': accuracy,
            'precision': precision_w,
            'recall': recall_w,
            'f1_score': f1_w,
            # 分组字段
            'overall': {
                'accuracy': accuracy,
                'precision_weighted': precision_w,
                'recall_weighted': recall_w,
                'f1_weighted': f1_w,
                'macro_precision': report.get('macro avg', {}).get('precision', 0.0),
                'macro_recall': report.get('macro avg', {}).get('recall', 0.0),
                'macro_f1': report.get('macro avg', {}).get('f1-score', 0.0),
                'auc_macro': auc_macro
            },
            'per_class': per_class
        }

        print(f"\n{model_name} 在 {task_name} 上的表现:")
        print(f"  accuracy: {accuracy:.4f}")
        print(f"  precision (weighted): {precision_w:.4f}")
        print(f"  recall (weighted): {recall_w:.4f}")
        print(f"  f1_score (weighted): {f1_w:.4f}")

        return {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics
        }
    
    def train_all_models(self, features, df, task_columns=None):
        """训练所有模型（默认仅针对黄斑水肿风险分类，集成SMOTE重采样）"""
        if task_columns is None:
            task_columns = ['Risk of macular edema']

        results = {}
        resample_cfg = self.config.get('resampling', {})
        use_smote = resample_cfg.get('use_smote', True)
        smote_type = resample_cfg.get('smote_type', 'borderline')
        smote_random_state = resample_cfg.get('random_state', 42)
        smote_k = resample_cfg.get('k_neighbors', 5)
        extreme_oversample = resample_cfg.get('extreme_oversample', False)
        minority_target_ratio = resample_cfg.get('minority_target_ratio', 3)

        for task_column in task_columns:
            print(f"\n{'='*50}")
            print(f"开始训练 {task_column} 分类模型（SMOTE重采样:{use_smote}）")
            print(f"{'='*50}")

            labels = df[task_column].values

            # 分割数据
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=labels
            )

            # 仅对训练集做SMOTE
            if use_smote:
                print("对训练集进行SMOTE过采样...")
                if smote_type == 'borderline':
                    smote = BorderlineSMOTE(random_state=smote_random_state, k_neighbors=smote_k)
                else:
                    smote = SMOTE(random_state=smote_random_state, k_neighbors=smote_k)
                X_train, y_train = smote.fit_resample(X_train, y_train)

                print(f"SMOTE后训练集样本分布: {np.bincount(y_train)}")

                # 极端过采样：将小类样本数提升到主类的3倍
                if extreme_oversample:
                    from collections import Counter
                    y_counts = Counter(y_train)
                    max_class = max(y_counts, key=lambda k: y_counts[k])
                    for cls in y_counts:
                        if cls == max_class:
                            continue
                        n_target = y_counts[max_class] * minority_target_ratio
                        idxs = np.where(y_train == cls)[0]
                        n_repeat = int(np.ceil(n_target / len(idxs)))
                        X_aug = np.repeat(X_train[idxs], n_repeat, axis=0)[:n_target]
                        y_aug = np.repeat(y_train[idxs], n_repeat, axis=0)[:n_target]
                        X_train = np.concatenate([X_train, X_aug], axis=0)
                        y_train = np.concatenate([y_train, y_aug], axis=0)
                    print(f"极端过采样后训练集样本分布: {np.bincount(y_train)}")

            task_results = {}

            for model_name in self.models.keys():
                print(f"\n训练 {model_name} 模型...")

                # 训练模型
                trained_model = self.train_with_cross_validation(
                    X_train, y_train, model_name, task_column
                )

                # 评估模型
                model_result = self.evaluate_model(
                    trained_model, X_test, y_test, model_name, task_column
                )

                task_results[model_name] = model_result

            results[task_column] = {
                'model_results': task_results,
                'X_test': X_test,
                'y_test': y_test,
                'X_train': X_train,
                'y_train': y_train
            }

        return results
    
    def save_models(self, results, save_dir="models"):
        """保存训练好的模型"""
        for task_name, task_data in results.items():
            for model_name, model_info in task_data['model_results'].items():
                filename = os.path.join(save_dir, f"{task_name}_{model_name}.pkl")
                joblib.dump(model_info['model'], filename)
                print(f"模型已保存: {filename}")
        
        # 保存参数和结果
        results_filename = os.path.join(save_dir, "training_results.pkl")
        joblib.dump({
            'best_params': self.best_params,
            'cv_results': self.cv_results
        }, results_filename)

    #新增：在ModelTrainer类中添加以下方法
    def compare_models(self, results):
        """比较不同模型的性能"""
        comparison_data = []
    
        for task_name, task_data in results.items():
            for model_name, model_info in task_data['model_results'].items():
                metrics = model_info['metrics']
                comparison_data.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall']
                })
    
        comparison_df = pd.DataFrame(comparison_data)
    
        # 找出每个任务的最佳模型
        best_models = {}
        for task in comparison_df['Task'].unique():
            task_df = comparison_df[comparison_df['Task'] == task]
            best_idx = task_df['F1-Score'].idxmax()
            best_models[task] = task_df.loc[best_idx]
    
        print("\n🏆 最佳模型评选:")
        for task, model_info in best_models.items():
            print(f"{task}: {model_info['Model']} (F1: {model_info['F1-Score']:.4f})")
    
        return comparison_df, best_models