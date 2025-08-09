from PIL import Image
import os


def resize_images(input_dir, output_dir, size=(256, 256)):
    """
    批量缩放图片并保存为 JPG

    Args:
        input_dir (str): 输入图片文件夹路径
        output_dir (str): 输出文件夹路径
        size (tuple): 目标尺寸 (width, height)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 开始处理图片：")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 目标尺寸: {size}")

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 错误：输入目录不存在！请检查路径拼写。")
        return

    if not os.path.isdir(input_dir):
        print(f"❌ 错误：输入路径不是一个文件夹。")
        return

    # 获取所有支持的图片文件
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(supported_extensions)
    ]

    if not image_files:
        print(f"⚠️  在输入目录中未找到支持的图片文件（支持格式：{', '.join(supported_extensions)}）")
        return

    print(f"📎 找到 {len(image_files)} 个图片文件，开始处理...")

    processed_count = 0
    failed_count = 0

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        try:
            with Image.open(img_path) as img:
                # 转换为 RGB（处理透明通道如 PNG）
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = img.convert('RGB')
                else:
                    rgb_img = img

                # 缩放图片
                resized_img = rgb_img.resize(size, Image.Resampling.LANCZOS)

                # 构造输出文件名（统一为 .jpg）
                output_name = os.path.splitext(img_name)[0] + ".jpg"
                output_path = os.path.join(output_dir, output_name)

                # 保存为 JPG 格式
                resized_img.save(output_path, "JPEG", quality=95)
                print(f"✅ 已保存: {output_name}")
                processed_count += 1

        except Exception as e:
            print(f"❌ 处理失败: {img_name} -> 错误: {e}")
            failed_count += 1

    print(f"\n✅ 处理完成！共处理 {processed_count} 张，失败 {failed_count} 张。")


if __name__ == "__main__":
    # === 请在这里修改你的路径 ===
    input_dir = r"D:/Project/Transform_Hometown/fast-neural-style-transfer-master/images/style-images"
    output_dir = r"D:/Project/Transform_Hometown/fast-neural-style-transfer-master/images/style-images"

    # 可选：自定义尺寸
    target_size = (256, 256)

    resize_images(input_dir, output_dir, size=target_size)