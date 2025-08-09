import os
import glob


def rename_images(input_folder, output_extension=".jpg"):
    # 获取文件夹中的所有图片文件（包括 .webp）
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    # 如果文件夹为空，打印提示并退出
    if not image_files:
        print("文件夹中没有找到图片文件")
        return

    # 按文件修改时间排序（可选：可以改为按文件名排序）
    image_files.sort(key=os.path.getmtime)

    # 重命名文件
    for index, old_path in enumerate(image_files, 1):
        # 构造新文件名，格式为 00001.jpg, 00002.jpg 等
        new_name = f"{index:04d}{output_extension}"
        new_path = os.path.join(input_folder, new_name)

        # 检查目标文件是否已存在
        if os.path.exists(new_path):
            print(f"警告：{new_name} 已存在，跳过 {os.path.basename(old_path)}")
            continue

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"已将 {os.path.basename(old_path)} 重命名为 {new_name}")

    print(f"完成！总共处理了 {len(image_files)} 个文件")


# 使用示例
input_folder = "D:/Project/Transform_Hometown/fast-neural-style-transfer-master/images/style-images"  # 使用你的文件夹路径
rename_images(input_folder)