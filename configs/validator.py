"""
配置文件验证模块
"""

import os
import yaml
from pathlib import Path

def validate_config(config):
    """验证配置文件的完整性"""
    print("开始验证配置文件...")
    
    # 检查必要路径配置
    required_paths = ['data_paths', 'feature_extraction', 'models', 'training']
    for path in required_paths:
        if path not in config:
            print(f"缺失配置项: {path}")
            return False
    
    # 检查数据路径
    data_paths = config['data_paths']
    for key, path in data_paths.items():
        if key.endswith('_dir') and not os.path.exists(path):
            print(f"路径不存在: {key} = {path}")
            # 创建目录
            os.makedirs(path, exist_ok=True)
            print(f"已创建目录: {path}")
    
    # 检查特征提取配置
    feature_config = config['feature_extraction']
    if feature_config['method'] not in ['deep', 'traditional', 'hybrid', 'pca', 'multi_deep']:
        print("无效的特征提取方法")
        return False
    
    # 检查模型配置
    models_config = config['models']
    if not models_config['classifiers']:
        print("未配置任何分类器")
        return False
    
    print("配置文件验证通过")
    return True



def generate_config_template():
    """生成配置模板"""
    template = {
        'data_paths': {
            'annotations_dir': "data/annotations",
            'annotations_csv_dir': "data/annotations_csv", 
            'images_base_dir': "data/images",
            'processed_dir': "data/processed"
        },
        # ... 其他配置项
    }
    
    with open('configs/config_template.yaml', 'w') as f:
        yaml.dump(template, f, default_flow_style=False)
    
    print("配置模板已生成: configs/config_template.yaml")