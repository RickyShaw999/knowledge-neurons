import argparse
from modelscope.hub.snapshot_download import snapshot_download
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-mi","--model_id", type=str)

args = parser.parse_args()


model_id = args.model_id
cache_dir = "/data/wenzhi/llmbase/" + model_id.split('/')[-1]
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print("Download over!")



# 定义源目录和目标目录
source_dir = cache_dir
target_dir = source_dir  # 移动到相同目录

# 遍历源目录及其子目录
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 获取文件的完整路径
        file_path = os.path.join(root, file)
        
        # 跳过目标目录中的文件，避免重复移动
        if root != target_dir:
            # 移动文件到目标目录
            shutil.move(file_path, target_dir)
            
print('Move over!')