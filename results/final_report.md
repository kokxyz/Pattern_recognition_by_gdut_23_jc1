
        # 糖尿病视网膜病变分类项目最终报告
        
        ## 项目摘要
        - 开始时间: 2025-12-23 17:45:05
        - 结束时间: 2025-12-23 17:46:25
        - 运行时长: 0:01:20.179384
        
        ## 数据集统计
        - 总图像数量: 1200
        - 有效图像数量: 1200
        - 失败图像数量: 0
        
        ## 特征信息
        - 特征提取方法: multi_deep
        - 特征维度: 100
        
        ## 模型性能总结
        

## 最佳超参数与交叉验证分数（按算法与任务拆分）

### Random Forest - Risk of macular edema 最佳超参数与交叉验证分数

| Best Params                                                    |   Best CV Score |
|:---------------------------------------------------------------|----------------:|
| {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |        0.975288 |

### Adaboost - Risk of macular edema 最佳超参数与交叉验证分数

| Best Params                                 |   Best CV Score |
|:--------------------------------------------|----------------:|
| {'learning_rate': 1.0, 'n_estimators': 200} |        0.845734 |


Random Forest - Risk of macular edema 训练/测试性能对比

| Task                  | Model         | Set   |   Accuracy |   Precision |   Recall |   F1-Score |
|:----------------------|:--------------|:------|-----------:|------------:|---------:|-----------:|
| Risk of macular edema | random_forest | Train |     0.9996 |      0.9996 |   0.9996 |     0.9996 |
| Risk of macular edema | random_forest | Test  |     0.8333 |      0.7836 |   0.8333 |     0.7771 |

Adaboost - Risk of macular edema 训练/测试性能对比

| Task                  | Model    | Set   |   Accuracy |   Precision |   Recall |   F1-Score |
|:----------------------|:---------|:------|-----------:|------------:|---------:|-----------:|
| Risk of macular edema | adaboost | Train |     0.8943 |      0.8955 |   0.8943 |     0.8945 |
| Risk of macular edema | adaboost | Test  |     0.7042 |      0.7532 |   0.7042 |     0.7254 |

## 各风险等级指标（Risk of macular edema）

| Model         |   Class |   Accuracy |    AUC |
|:--------------|--------:|-----------:|-------:|
| random_forest |       0 |     0.8333 | 0.7644 |
| random_forest |       1 |     0.9375 | 0.6126 |
| random_forest |       2 |     0.8958 | 0.8026 |
| adaboost      |       0 |     0.7208 | 0.7431 |
| adaboost      |       1 |     0.85   | 0.5615 |
| adaboost      |       2 |     0.8375 | 0.7583 |

## 最佳模型
- Risk of macular edema: random_forest (F1-score: 0.7771)
