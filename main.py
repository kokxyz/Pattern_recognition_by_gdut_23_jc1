#!/usr/bin/env python3
"""
糖尿病视网膜病变分类主程序
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureProcessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from configs.validator import validate_config
from src.utils import memory_usage, check_gpu_availability
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def setup_environment():
    """设置项目环境"""
    # 创建必要的目录
    directories = [
        'data/annotations_csv',
        'data/processed',
        'models',
        'results/figures',
        'results/reports',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("项目环境设置完成")

def create_config_file():
    """创建配置文件（如果不存在）"""
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        config = {
            'data_paths': {
                'annotations_dir': "data/annotations",
                'annotations_csv_dir': "data/annotations_csv",
                'images_base_dir': "data/images",
                'processed_dir': "data/processed"
            },
            'feature_extraction': {
                'method': "hybrid",
                'deep_learning': {
                    'model_name': "resnet50",
                    'input_size': [224, 224],
                    'feature_layer': "avg_pool"
                },
                'traditional': {
                    'color_bins': 32,
                    'texture_method': "lbp",
                    'surf_features': 500
                }
            },
            'models': {
                'classifiers': ["random_forest", "adaboost"],
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'cv_folds': 5
                },
                'adaboost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': "SAMME.R",
                    'cv_folds': 5
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'use_cross_validation': True,
                'scoring_metric': "f1_weighted"
            },
            'evaluation': {
                'metrics': ["accuracy", "precision", "recall", "f1", "roc_auc"],
                'plot_confusion_matrix': True,
                'plot_roc_curve': True,
                'plot_feature_importance': True,
                'plot_learning_curve': True
            }
        }
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"配置文件已创建: {config_path}")

def main():
    """主程序"""
    print("=" * 60)
    print("糖尿病视网膜病变分类项目")
    print("=" * 60)
    
    #新增：系统检查
    print(check_gpu_availability())
    print(memory_usage())

    # 设置环境
    setup_environment()
    
    # 创建配置文件
    create_config_file()
    
    # 解析命令行参数（支持通过 --epochs 指定训练轮数/迭代次数）
    parser = argparse.ArgumentParser(description='糖尿病视网膜病变分类训练')
    parser.add_argument('--epochs', type=int, help='指定训练轮数或迭代次数（用于可迭代模型或覆盖 n_estimators）')
    args = parser.parse_args()

    # 加载配置
    with open('configs/config.yaml', 'r',encoding='UTF-8') as file:
        config = yaml.safe_load(file)

    # 如果通过命令行传入 epochs，则写入 config，后续 ModelTrainer 会使用该值覆盖相关模型的 n_estimators 等参数
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = int(args.epochs)

    # 注释掉自动回退模型逻辑，允许efficientnetb3等高阶模型正常使用
    # try:
    #     model_name = config.get('feature_extraction', {}).get('deep_learning', {}).get('model_name', '').lower()
    #     supported = ['resnet50', 'vgg16', 'efficientnetb3', 'densenet201']
    #     if model_name == 'efficientnetb3' or model_name not in supported:
    #         old = model_name
    #         config['feature_extraction']['deep_learning']['model_name'] = 'resnet50'
    #         with open('configs/config.yaml', 'w', encoding='utf-8') as f:
    #             yaml.dump(config, f, default_flow_style=False)
    #         print(f"警告: 将特征提取模型从 '{old}' 回退为 'resnet50'，并已保存到 configs/config.yaml 以保证可运行。")
    # except Exception as _e:
    #     # 如果持久化失败，不阻塞流程，仅打印警告
    #     print(f"警告: 在回退模型名时发生错误: {_e}")

    # 配置验证
    if not validate_config(config):
        print("配置文件验证失败，请检查config.yaml")
        return
    
    # 创建配置备份
    #create_backup_config()
    
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 数据预处理
        print("\n1. 数据预处理阶段")
        print("-" * 30)
        
        preprocessor = DataPreprocessor()
        preprocessor.validate_config()

        # 转换XLS到CSV
        preprocessor.convert_xls_to_csv()
        
        # 加载和合并标注数据
        df = preprocessor.load_and_merge_annotations()
        preprocessor.check_data_integrity(df)
        
        # 检查是否已有处理过的数据
        processed_data = preprocessor.load_processed_data("processed_data.npy")
        
        if processed_data is not None:
            print("加载已处理的图像数据...")
            images = processed_data['images']
            failed_images = processed_data['failed_images']
        else:
            # 加载图像
            images, failed_images = preprocessor.load_images(df)
            
            # 保存处理后的数据
            processed_data = {
                'images': images,
                'failed_images': failed_images,
                'timestamp': datetime.now().isoformat()
            }
            preprocessor.save_processed_data(processed_data, "processed_data.npy")
        
        print(f"成功加载 {len(images) - len(failed_images)} 张图像")
        
        # 2. 特征提取
        print("\n2. 特征提取阶段")
        print("-" * 30)
        
        feature_processor = FeatureProcessor(config)
        
        # 检查是否已有提取的特征
        features_data = preprocessor.load_processed_data("extracted_features.npy")
        
        if features_data is not None:
            print("加载已提取的特征...")
            features = features_data['features']
            feature_names = features_data['feature_names']
        else:
            # 提取特征
            features = feature_processor.extract_features_batch(images)
            
            # 降维和标准化
            features_reduced, pca = feature_processor.reduce_dimensionality(features)
            features_normalized, scaler = feature_processor.normalize_features(features_reduced)
            
            features = features_normalized
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            
            # 保存特征
            features_data = {
                'features': features,
                'feature_names': feature_names,
                'pca': pca,
                'scaler': scaler,
                'timestamp': datetime.now().isoformat()
            }
            preprocessor.save_processed_data(features_data, "extracted_features.npy")
            
            # 保存预处理对象
            feature_processor.save_preprocessing_objects('models')
        
        print(f"特征维度: {features.shape}")
        
        # 3. 模型训练
        print("\n3. 模型训练阶段")
        print("-" * 30)
        
        trainer = ModelTrainer(config=config)
        trainer.initialize_models()
        
        # 定义分类任务（仅黄斑水肿风险）
        task_columns = ['Risk of macular edema']
        
        # 训练模型
        results = trainer.train_all_models(features, df, task_columns)
        
        # 保存模型
        trainer.save_models(results)
        
        # 4. 模型评估
        print("\n4. 模型评估阶段")
        print("-" * 30)
        
        evaluator = ModelEvaluator(config)
        
        # 评估模型（学习曲线使用每个任务的 X_train / y_train，已由训练过程保存到 results）
        report_df = evaluator.evaluate_all_models(
            results, df, feature_names
        )
        
        # 5. 生成最终报告
        print("\n5. 生成最终报告")
        print("-" * 30)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        final_report = f"""
        # 糖尿病视网膜病变分类项目最终报告
        
        ## 项目摘要
        - 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
        - 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
        - 运行时长: {duration}
        
        ## 数据集统计
        - 总图像数量: {len(df)}
        - 有效图像数量: {len(images) - len(failed_images)}
        - 失败图像数量: {len(failed_images)}
        
        ## 特征信息
        - 特征提取方法: {config['feature_extraction']['method']}
        - 特征维度: {features.shape[1]}
        
        ## 模型性能总结
        """

        # 将训练过程中的最佳参数与交叉验证分数加入报告（若存在，仅黄斑水肿任务）
        try:
            param_rows = []
            if hasattr(trainer, 'best_params') and trainer.best_params:
                model_names = list(trainer.models.keys()) if hasattr(trainer, 'models') else []
                for key, params in trainer.best_params.items():
                    # key 格式为 f"{task_name}_{model_name}"，model_name 在后缀
                    found_model = None
                    for m in model_names:
                        if key.endswith(m):
                            found_model = m
                            task = key[: - (len(m) + 1)]
                            break
                    if found_model is None:
                        # 回退：尝试以最后一个下划线分割
                        if '_' in key:
                            task, found_model = key.rsplit('_', 1)
                        else:
                            task = key
                            found_model = ''

                    # 从 cv_results 中寻找对应的最佳交叉验证分数
                    best_cv = None
                    cv = trainer.cv_results.get(key) if hasattr(trainer, 'cv_results') else None
                    try:
                        if cv is not None and 'rank_test_score' in cv and 'mean_test_score' in cv:
                            # 找到 rank == 1 的索引
                            ranks = list(cv['rank_test_score'])
                            if 1 in ranks:
                                idx = ranks.index(1)
                                best_cv = float(cv['mean_test_score'][idx])
                            else:
                                # 否则取 mean_test_score 的最大值
                                best_cv = float(max(cv['mean_test_score']))
                        elif cv is not None and 'mean_test_score' in cv:
                            best_cv = float(max(cv['mean_test_score']))
                    except Exception:
                        best_cv = None

                    param_rows.append({
                        'Task': task,
                        'Model': found_model,
                        'Best Params': str(params),
                        'Best CV Score': best_cv
                    })

            if param_rows:
                params_df = pd.DataFrame(param_rows)
                final_report += "\n\n## 最佳超参数与交叉验证分数（按算法与任务拆分）\n\n"
                # 按模型与任务分别生成表格（四个表）
                models_to_show = ['random_forest', 'adaboost']
                # 保证使用训练时的 task_columns 列表顺序
                tasks_to_show = task_columns if 'task_columns' in locals() else sorted(params_df['Task'].unique())

                for model in models_to_show:
                    for task in tasks_to_show:
                        sub = params_df[(params_df['Model'] == model) & (params_df['Task'] == task)]
                        section_title = f"### {model.replace('_',' ').title()} - {task} 最佳超参数与交叉验证分数\n\n"
                        final_report += section_title
                        if not sub.empty:
                            # 只显示 Best Params 与 Best CV Score 两列以保持简洁
                            display_sub = sub[['Best Params', 'Best CV Score']].copy()
                            final_report += display_sub.to_markdown(index=False) + "\n\n"
                        else:
                            final_report += "暂无超参数调优结果或该组合未进行搜索。\n\n"
            else:
                final_report += "\n\n## 最佳超参数与交叉验证分数\n\n暂无超参数调优结果或未启用交叉验证。\n"
        except Exception as e:
            print(f"警告: 在生成最佳参数报告时出错: {e}")

        # 将评估结果按模型分表（random_forest / adaboost）输出为两个 Markdown 表格
        try:
            # 基于 evaluate_all_models 返回的 report_df（测试集指标），以及 results（包含训练/测试集 X/y 和模型），
            # 计算每个模型在训练集与测试集上的性能并生成对比表。
            perf_rows = []

            for task_name, task_data in results.items():
                X_train = task_data.get('X_train')
                y_train = task_data.get('y_train')
                X_test = task_data.get('X_test')
                y_test = task_data.get('y_test')

                for model_name, model_info in task_data['model_results'].items():
                    model = model_info['model']

                    # 训练集性能（更健壮的预测处理）
                    tr_acc = tr_prec = tr_rec = tr_f1 = None
                    if X_train is not None and y_train is not None and len(y_train) > 0:
                        try:
                            X_train_arr = np.asarray(X_train)
                            if X_train_arr.ndim == 1:
                                X_train_arr = X_train_arr.reshape(-1, 1)

                            y_train_pred = None
                            try:
                                y_train_pred = model.predict(X_train_arr)
                            except Exception as e_pred:
                                # 尝试将数据类型转为 float32 再试一次
                                try:
                                    y_train_pred = model.predict(X_train_arr.astype(np.float32))
                                except Exception as e_pred2:
                                    print(f"警告: 在预测训练集时出错 ({task_name} / {model_name}): {e_pred}; {e_pred2}")
                                    y_train_pred = None

                            if y_train_pred is not None:
                                tr_acc = accuracy_score(y_train, y_train_pred)
                                tr_prec = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                                tr_rec = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                                tr_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                        except Exception as e:
                            print(f"警告: 计算训练集性能失败 ({task_name} / {model_name}): {e}")

                    # 测试集性能（优先使用已存在的预测结果），同样做鲁棒处理
                    te_acc = te_prec = te_rec = te_f1 = None
                    if X_test is not None and y_test is not None and len(y_test) > 0:
                        try:
                            X_test_arr = np.asarray(X_test)
                            if X_test_arr.ndim == 1:
                                X_test_arr = X_test_arr.reshape(-1, 1)

                            y_test_pred = model_info.get('y_pred')
                            if y_test_pred is None:
                                try:
                                    y_test_pred = model.predict(X_test_arr)
                                except Exception as e_pred:
                                    try:
                                        y_test_pred = model.predict(X_test_arr.astype(np.float32))
                                    except Exception as e_pred2:
                                        print(f"警告: 在预测测试集时出错 ({task_name} / {model_name}): {e_pred}; {e_pred2}")
                                        y_test_pred = None

                            if y_test_pred is not None:
                                te_acc = accuracy_score(y_test, y_test_pred)
                                te_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                                te_rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                                te_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                        except Exception as e:
                            print(f"警告: 计算测试集性能失败 ({task_name} / {model_name}): {e}")

                    perf_rows.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Set': 'Train',
                        'Accuracy': tr_acc,
                        'Precision': tr_prec,
                        'Recall': tr_rec,
                        'F1-Score': tr_f1
                    })

                    perf_rows.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Set': 'Test',
                        'Accuracy': te_acc,
                        'Precision': te_prec,
                        'Recall': te_rec,
                        'F1-Score': te_f1
                    })

            perf_df = pd.DataFrame(perf_rows)

            # 为每个算法生成分表并追加到最终报告，同时保存 CSV
            reports_dir = os.path.join('results', 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            for algo in ['random_forest', 'adaboost']:
                algo_df = perf_df[perf_df['Model'] == algo]

                if algo_df.empty:
                    final_report += f"\n### {algo.replace('_', ' ').title()} 训练/测试性能对比\n\n暂无 {algo} 的训练/测试对比结果。\n"
                    continue

                # 按任务拆分为多个表（每个算法 × 任务 一个表）
                for task in algo_df['Task'].unique():
                    task_df = algo_df[algo_df['Task'] == task]
                    # 标题示例："### Random Forest - Risk of macular edema 训练/测试性能对比"
                    section_title = f"### {algo.replace('_', ' ').title()} - {task} 训练/测试性能对比\n\n"
                    final_report += "\n" + section_title

                    # 格式化显示表格（保留 NaN -> N/A）
                    display_df = task_df.copy()
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

                    final_report += display_df.to_markdown(index=False)

                    # 保存每个算法-任务对的 CSV，文件名去空格并小写
                    task_slug = task.replace(' ', '_').replace('/', '_').lower()
                    csv_path = os.path.join(reports_dir, f'perf_{algo}_{task_slug}.csv')
                    task_df.to_csv(csv_path, index=False)
                    print(f"已保存 {algo} - {task} 训练/测试对比 CSV: {csv_path}")

        except Exception as e:
            # 如果 process 出错，回退到原始 report_df 显示
            print(f"生成训练/测试对比时出错: {e}")
            final_report += "\n警告：生成训练/测试对比失败，显示测试集整体结果如下：\n\n"
            final_report += report_df.to_markdown(index=False)

        # 追加每个风险等级的详细指标
        try:
            risk_key = 'Risk of macular edema'
            if risk_key in results:
                per_class_rows = []
                for model_name, model_info in results[risk_key]['model_results'].items():
                    for cls, cls_metrics in model_info['metrics'].get('per_class', {}).items():
                        auc_val = cls_metrics.get('auc', None)
                        per_class_rows.append({
                            'Model': model_name,
                            'Class': cls,
                            'Accuracy': cls_metrics.get('accuracy', 0.0),
                          
                            'AUC': float(auc_val) if auc_val is not None else None,

                        })

                if per_class_rows:
                    per_class_df = pd.DataFrame(per_class_rows)
                    display_df = per_class_df.copy()
                    for col in ['Accuracy','AUC']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

                    final_report += "\n\n## 各风险等级指标（Risk of macular edema）\n\n"
                    final_report += display_df.to_markdown(index=False)
        except Exception as e:
            print(f"生成按类别指标表时出错: {e}")

        final_report += "\n\n## 最佳模型\n"
        
        # 找出每个任务的最佳模型
        for task_column in task_columns:
            best_model = None
            best_f1 = 0

            for model_name, model_info in results[task_column]['model_results'].items():
                f1_val = model_info['metrics']['f1_score']
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_model = model_name

            final_report += f"- {task_column}: {best_model} (F1-score: {best_f1:.4f})\n"
        
        # 保存最终报告
        report_path = "results/final_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"最终报告已保存: {report_path}")
        
        print("\n" + "=" * 60)
        print("项目完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"项目执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()