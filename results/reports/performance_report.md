# 黄斑水肿风险分类性能报告

## 整体指标（权重平均）

| Task                  | Model         |   Accuracy |   Precision(weighted) |   Recall(weighted) |   F1-Score(weighted) |
|:----------------------|:--------------|-----------:|----------------------:|-------------------:|---------------------:|
| Risk of macular edema | random_forest |     0.8333 |                0.7836 |             0.8333 |               0.7771 |
| Risk of macular edema | adaboost      |     0.7042 |                0.7532 |             0.7042 |               0.7254 |

## 按类别指标

| Task                  | Model         |   Class |   Accuracy |   Precision |   Recall |   F1-Score |    AUC |   Support |
|:----------------------|:--------------|--------:|-----------:|------------:|---------:|-----------:|-------:|----------:|
| Risk of macular edema | random_forest |       0 |     0.8333 |      0.8326 |   0.9949 |     0.9065 | 0.7644 |       195 |
| Risk of macular edema | random_forest |       1 |     0.9375 |      0      |   0      |     0      | 0.6126 |        15 |
| Risk of macular edema | random_forest |       2 |     0.8958 |      0.8571 |   0.2    |     0.3243 | 0.8026 |        30 |
| Risk of macular edema | adaboost      |       0 |     0.7208 |      0.8636 |   0.7795 |     0.8194 | 0.7431 |       195 |
| Risk of macular edema | adaboost      |       1 |     0.85   |      0.0435 |   0.0667 |     0.0526 | 0.5615 |        15 |
| Risk of macular edema | adaboost      |       2 |     0.8375 |      0.3902 |   0.5333 |     0.4507 | 0.7583 |        30 |