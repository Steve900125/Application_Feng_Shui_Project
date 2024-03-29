import os
import shutil

# 基础路径
base_path = './runs/detect/'

# 遍历基础路径下的所有文件夹和文件
for item in os.listdir(base_path):
    # 构建完整路径
    item_path = os.path.join(base_path, item)
    # 检查是否为文件夹并且名称符合预测的模式
    if os.path.isdir(item_path) and item.startswith('predict'):
        print(f"Deleting folder: {item_path}")
        shutil.rmtree(item_path)
