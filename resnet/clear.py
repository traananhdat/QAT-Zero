import os
import glob


def delete_seg_txt_files_in_current_directory():
    # 获取当前目录
    current_dir = os.getcwd()

    # 查找当前目录中的所有 seg*.txt 文件
    seg_txt_files = glob.glob(os.path.join(current_dir, "*.jpg"))

    # 删除找到的 seg*.txt 文件
    for file_path in seg_txt_files:
        os.remove(file_path)
        print(f"Deleted: {file_path}")


# 调用函数删除当前目录下的 seg*.txt 文件
delete_seg_txt_files_in_current_directory()
