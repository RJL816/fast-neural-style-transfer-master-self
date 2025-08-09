import cv2
import os


def video_to_frames(video_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频
    video = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频文件名（不含扩展名）用于命名图片
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    while True:
        # 读取一帧
        success, frame = video.read()

        # 如果没有读取到帧，退出循环
        if not success:
            break

        # 构造输出图片路径
        frame_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.jpg")

        # 保存帧为图片
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # 释放视频对象
    video.release()
    print(f"成功提取 {frame_count} 帧，保存至 {output_folder}")


# 使用示例
video_path = "D:/Project/Transform_Hometown/fast-neural-style-transfer-master/video1.mp4"  # 替换为你的视频文件路径
output_folder = "D:/Project/Transform_Hometown/fast-neural-style-transfer-master/my_dataset/images_raw"  # 替换为你的输出文件夹路径
video_to_frames(video_path, output_folder)