import os
from pathlib import Path

def add_underscore_before_last_char(folder_path):
    # 确保路径存在
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    for file_name in os.listdir(folder_path):
        # 只处理 .png 文件
        if file_name.lower().endswith('.png'):
            # 使用 pathlib 处理文件名和后缀
            p = Path(file_name)
            name_stem = p.stem  # 获取文件名（不含后缀），例如 "imageA"
            suffix = p.suffix    # 获取后缀，例如 ".png"

            # 确保主文件名至少有一个字符
            if len(name_stem) > 0:
                # 逻辑：主文件名除最后一个字符外的部分 + 下划线 + 最后一个字符 + 后缀
                new_name = f"{name_stem[:-1]}_{name_stem[-1]}{suffix}"
                
                old_path = os.path.join(folder_path, file_name)
                new_path = os.path.join(folder_path, new_name)

                # 执行重命名
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名: {file_name} -> {new_name}")
                except Exception as e:
                    print(f"跳过 {file_name}: {e}")

if __name__ == "__main__":
    # 修改为你的文件夹路径
    target_folder = "images" 
    add_underscore_before_last_char(target_folder)