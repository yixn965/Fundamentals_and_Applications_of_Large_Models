"""
MVTec AD 数据集预处理脚本
将 MVTec AD 数据集转换为适用于 GRPO 训练的成对数据格式
"""

import os
import json
import glob
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


# ==================== 配置参数 ====================
DATASET_ROOT = r"e:\C盘文件迁移\桌面\大模型基础与应用\dataset"
OUTPUT_FILE = r"e:\C盘文件迁移\桌面\大模型基础与应用\output_dataset.jsonl"
MIN_CONTOUR_AREA = 10  # 最小轮廓面积（像素），过滤噪点
RANDOM_SEED = 42  # 随机种子，确保可复现性

# 设置随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==================== 辅助函数 ====================

def extract_bboxes_from_mask(mask_path: str) -> List[List[int]]:
    """
    从掩膜图像中提取边界框
    
    Args:
        mask_path: 掩膜图像路径
        
    Returns:
        边界框列表，每个边界框格式为 [xmin, ymin, xmax, ymax]
    """
    # 使用 numpy.fromfile 和 cv2.imdecode 读取中文路径的图像
    try:
        # 读取图像文件为字节流
        img_data = np.fromfile(mask_path, dtype=np.uint8)
        # 解码为灰度图像
        mask = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"警告：读取掩膜文件失败 {mask_path}: {e}")
        mask = None
    
    if mask is None:
        print(f"警告：无法读取掩膜文件 {mask_path}")
        return []
    
    # 二值化处理（确保是纯黑白图像）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # 计算轮廓面积，过滤过小的噪点
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        bboxes.append(bbox)
    
    return bboxes


def get_mask_path(test_image_path: str, category_root: str, defect_type: str) -> str:
    """
    根据测试图像路径构造对应的掩膜路径
    
    Args:
        test_image_path: 测试图像路径
        category_root: 类别根目录
        defect_type: 缺陷类型（文件夹名）
        
    Returns:
        掩膜文件路径
    """
    # 获取原始文件名（不含扩展名）
    image_name = Path(test_image_path).stem
    
    # 构造掩膜文件路径：ground_truth/defect_type/xxx_mask.png
    mask_filename = f"{image_name}_mask.png"
    mask_path = os.path.join(category_root, "ground_truth", defect_type, mask_filename)
    
    return mask_path


def process_category(category_path: str) -> List[Dict]:
    """
    处理单个类别的数据
    
    Args:
        category_path: 类别文件夹路径
        
    Returns:
        样本列表
    """
    samples = []
    
    # 获取类别名称
    category_name = os.path.basename(os.path.dirname(category_path))
    
    # 获取所有训练集的正常图像（用于配对）
    train_good_dir = os.path.join(category_path, "train", "good")
    if not os.path.exists(train_good_dir):
        print(f"警告：类别 {category_name} 没有找到 train/good 文件夹")
        return samples
    
    train_good_images = glob.glob(os.path.join(train_good_dir, "*.png"))
    
    if len(train_good_images) == 0:
        print(f"警告：类别 {category_name} 的 train/good 文件夹为空")
        return samples
    
    print(f"处理类别: {category_name}, 训练集正常图像数: {len(train_good_images)}")
    
    # 获取测试集文件夹
    test_dir = os.path.join(category_path, "test")
    if not os.path.exists(test_dir):
        print(f"警告：类别 {category_name} 没有找到 test 文件夹")
        return samples
    
    # 遍历测试集的所有子文件夹（good, defect_type1, defect_type2, ...）
    test_subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for subdir in test_subdirs:
        subdir_path = os.path.join(test_dir, subdir)
        test_images = glob.glob(os.path.join(subdir_path, "*.png"))
        
        print(f"  - 处理子类别: {subdir}, 图像数: {len(test_images)}")
        
        for test_image_path in test_images:
            # 随机选择一张训练集正常图像作为配对
            reference_image_path = random.choice(train_good_images)
            
            # 构造相对路径（相对于工作空间根目录）
            rel_reference_path = os.path.relpath(reference_image_path, 
                                                 start=os.path.dirname(DATASET_ROOT))
            rel_test_path = os.path.relpath(test_image_path, 
                                           start=os.path.dirname(DATASET_ROOT))
            
            # 判断是正常样本还是异常样本
            if subdir == "good":
                # 无异常样本
                sample = {
                    "image_a_path": rel_reference_path.replace('\\', '/'),
                    "image_b_path": rel_test_path.replace('\\', '/'),
                    "label": {
                        "status": "无异常",
                        "changes": []
                    }
                }
            else:
                # 异常样本
                defect_type = subdir
                
                # 获取掩膜路径
                mask_path = get_mask_path(test_image_path, category_path, defect_type)
                
                if not os.path.exists(mask_path):
                    print(f"警告：找不到掩膜文件 {mask_path}，跳过该样本")
                    continue
                
                # 提取边界框
                bboxes = extract_bboxes_from_mask(mask_path)
                
                if len(bboxes) == 0:
                    print(f"警告：掩膜文件 {mask_path} 未提取到有效边界框，跳过该样本")
                    continue
                
                # 构造 changes 列表
                changes = []
                for bbox in bboxes:
                    change = {
                        "bbox": bbox,
                        "description": f"Detected {defect_type} anomaly on the object"
                    }
                    changes.append(change)
                
                sample = {
                    "image_a_path": rel_reference_path.replace('\\', '/'),
                    "image_b_path": rel_test_path.replace('\\', '/'),
                    "label": {
                        "status": "异常",
                        "changes": changes
                    }
                }
            
            samples.append(sample)
    
    return samples


def main():
    """主函数"""
    print("=" * 60)
    print("MVTec AD 数据集预处理脚本")
    print("=" * 60)
    print(f"数据集根目录: {DATASET_ROOT}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"最小轮廓面积: {MIN_CONTOUR_AREA} 像素")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 检查数据集根目录是否存在
    if not os.path.exists(DATASET_ROOT):
        print(f"错误：数据集根目录不存在 {DATASET_ROOT}")
        return
    
    all_samples = []
    
    # 获取所有类别文件夹
    categories = []
    for item in os.listdir(DATASET_ROOT):
        item_path = os.path.join(DATASET_ROOT, item)
        if os.path.isdir(item_path):
            # MVTec 数据集的特殊结构：类别名/类别名/train, test...
            inner_category_path = os.path.join(item_path, item)
            if os.path.exists(inner_category_path):
                categories.append(inner_category_path)
            else:
                # 有些可能直接是类别名/train, test...
                categories.append(item_path)
    
    print(f"\n找到 {len(categories)} 个类别文件夹\n")
    
    # 处理每个类别
    for category_path in categories:
        samples = process_category(category_path)
        all_samples.extend(samples)
        print(f"当前总样本数: {len(all_samples)}\n")
    
    # 保存到 JSONL 文件
    print("=" * 60)
    print(f"总共生成 {len(all_samples)} 个样本")
    print(f"保存到文件: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')
    
    # 统计信息
    normal_count = sum(1 for s in all_samples if s['label']['status'] == '无异常')
    anomaly_count = sum(1 for s in all_samples if s['label']['status'] == '异常')
    
    print("=" * 60)
    print("统计信息:")
    print(f"  - 无异常样本: {normal_count} ({normal_count/len(all_samples)*100:.2f}%)")
    print(f"  - 异常样本: {anomaly_count} ({anomaly_count/len(all_samples)*100:.2f}%)")
    print("=" * 60)
    print("处理完成！")
    
    # 显示几个样本示例
    print("\n样本示例（前3个）:")
    print("-" * 60)
    for i, sample in enumerate(all_samples[:3]):
        print(f"样本 {i+1}:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    main()
