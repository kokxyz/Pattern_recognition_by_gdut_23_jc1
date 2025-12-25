import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
import os
import joblib

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.figures_dir = "results/figures"
        self.reports_dir = "results/reports"
        
        # 创建目录
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (10, 8)
    
    def plot_confusion_matrices(self, results, class_names_dict):
        """绘制混淆矩阵"""
        for task_name, task_data in results.items():
            class_names = class_names_dict[task_name]
            
            for model_name, model_info in task_data['model_results'].items():
                y_true = model_info['y_test']
                y_pred = model_info['y_pred']
                
                cm = confusion_matrix(y_true, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Confusion Matrix - {model_name}\n({task_name})')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                filename = os.path.join(
                    self.figures_dir, 
                    f'confusion_matrix_{task_name}_{model_name}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"混淆矩阵已保存: {filename}")
    
    def plot_roc_curves(self, results, class_names_dict):
        """绘制ROC曲线（多分类）"""
        for task_name, task_data in results.items():
            class_names = class_names_dict[task_name]
            n_classes = len(class_names)
            
            plt.figure(figsize=(10, 8))
            
            for model_name, model_info in task_data['model_results'].items():
                if model_info['y_pred_proba'] is not None:
                    y_true = model_info['y_test']
                    y_score = model_info['y_pred_proba']
                    
                    # 二值化标签
                    y_true_bin = label_binarize(y_true, classes=range(n_classes))
                    
                    # 计算每个类别的ROC曲线
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # 计算微观平均ROC曲线
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # 绘制微观平均ROC曲线
                    plt.plot(fpr["micro"], tpr["micro"],
                            label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                            linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {task_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(self.figures_dir, f'roc_curves_{task_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"ROC曲线已保存: {filename}")
    
    def plot_feature_importance(self, results, feature_names, top_n=20):
        """绘制特征重要性"""
        for task_name, task_data in results.items():
            for model_name, model_info in task_data['model_results'].items():
                model = model_info['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:top_n]
                    
                    plt.figure(figsize=(12, 8))
                    plt.title(f'Feature Importances - {model_name}\n({task_name})')
                    plt.bar(range(top_n), importances[indices], color='b', align='center')
                    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
                    plt.xlim([-1, top_n])
                    plt.tight_layout()
                    
                    filename = os.path.join(
                        self.figures_dir, 
                        f'feature_importance_{task_name}_{model_name}.png'
                    )
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"特征重要性图已保存: {filename}")
    
    def plot_learning_curves(self, results):
        """绘制学习曲线：对每个任务使用该任务实际的 X_train 和 y_train（从 results 中读取）。"""
        for task_name, task_data in results.items():
            # 从 trainer 保存的结果中获取用于学习曲线的训练集
            if 'X_train' not in task_data or 'y_train' not in task_data:
                print(f"警告: 任务 {task_name} 中缺少 X_train 或 y_train，跳过学习曲线")
                continue

            X_train = task_data['X_train']
            y_train = task_data['y_train']

            for model_name, model_info in task_data['model_results'].items():
                model = model_info['model']

                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_train, y_train, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy'
                )
                
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                
                plt.figure(figsize=(10, 6))
                plt.grid(True)
                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1, color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
                plt.xlabel("Training examples")
                plt.ylabel("Accuracy")
                plt.title(f"Learning Curve - {model_name}\n({task_name})")
                plt.legend(loc="best")
                plt.tight_layout()
                
                filename = os.path.join(
                    self.figures_dir, 
                    f'learning_curve_{task_name}_{model_name}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"学习曲线已保存: {filename}")
    
    def generate_comprehensive_report(self, results, df):
        """生成综合评估报告（包含每类指标）"""
        overall_rows = []
        per_class_rows = []

        for task_name, task_data in results.items():
            for model_name, model_info in task_data['model_results'].items():
                metrics = model_info['metrics']
                overall_rows.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Accuracy': f"{metrics.get('accuracy', 0.0):.4f}",
                    'Precision(weighted)': f"{metrics.get('precision', 0.0):.4f}",
                    'Recall(weighted)': f"{metrics.get('recall', 0.0):.4f}",
                    'F1-Score(weighted)': f"{metrics.get('f1_score', 0.0):.4f}"
                })

                per_class = metrics.get('per_class', {})
                for cls, cls_metrics in per_class.items():
                    auc_val = cls_metrics.get('auc', None)
                    per_class_rows.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Class': cls,
                        'Accuracy': f"{cls_metrics.get('accuracy', 0.0):.4f}",
                        'Precision': f"{cls_metrics.get('precision', 0.0):.4f}",
                        'Recall': f"{cls_metrics.get('recall', 0.0):.4f}",
                        'F1-Score': f"{cls_metrics.get('f1_score', 0.0):.4f}",
                        'AUC': "N/A" if auc_val is None else f"{auc_val:.4f}",
                        'Support': int(cls_metrics.get('support', 0))
                    })

        overall_df = pd.DataFrame(overall_rows)
        per_class_df = pd.DataFrame(per_class_rows)

        # 保存为CSV
        csv_path = os.path.join(self.reports_dir, "performance_report.csv")
        overall_df.to_csv(csv_path, index=False)

        # 保存为Markdown格式
        md_path = os.path.join(self.reports_dir, "performance_report.md")
        with open(md_path, 'w') as f:
            f.write("# 黄斑水肿风险分类性能报告\n\n")
            f.write("## 整体指标（权重平均）\n\n")
            f.write(overall_df.to_markdown(index=False))
            f.write("\n\n## 按类别指标\n\n")
            if not per_class_df.empty:
                f.write(per_class_df.to_markdown(index=False))
            else:
                f.write("暂无按类别指标\n")

        # 额外保存按类别的 CSV 便于查阅
        per_class_csv = os.path.join(self.reports_dir, "performance_report_per_class.csv")
        per_class_df.to_csv(per_class_csv, index=False)

        print(f"\n性能报告已保存:")
        print(f"CSV格式: {csv_path}")
        print(f"按类别CSV: {per_class_csv}")
        print(f"Markdown格式: {md_path}")

        return overall_df
    
    def evaluate_all_models(self, results, df, feature_names=None):
        """全面评估所有模型"""
        print("\n开始模型评估和可视化...")
        
        # 定义类别名称（仅黄斑水肿风险）
        class_names_dict = {
            'Risk of macular edema': ['Risk 0', 'Risk 1', 'Risk 2']
        }
        
        # 绘制混淆矩阵
        if self.config['evaluation']['plot_confusion_matrix']:
            self.plot_confusion_matrices(results, class_names_dict)
        
        # 绘制ROC曲线
        if self.config['evaluation']['plot_roc_curve']:
            self.plot_roc_curves(results, class_names_dict)
        
        # 绘制特征重要性
        if self.config['evaluation']['plot_feature_importance'] and feature_names is not None:
            self.plot_feature_importance(results, feature_names)
        
        # 绘制学习曲线（plot_learning_curves 会从 results 中读取每个任务的 X_train/y_train）
        if self.config['evaluation']['plot_learning_curve']:
            self.plot_learning_curves(results)
        
        # 生成综合报告
        report_df = self.generate_comprehensive_report(results, df)
        
        print("\n模型评估完成!")
        return report_df