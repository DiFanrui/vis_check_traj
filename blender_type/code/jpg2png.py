from PIL import Image
import os

image_folder = 'blender_type/images'

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        path_jpg = os.path.join(image_folder, filename)
        path_png = os.path.join(image_folder, os.path.splitext(filename)[0] + '.png')

        with Image.open(path_jpg) as img:
            img = img.convert('RGBA')  # ⭐ 添加这一行
            img.save(path_png)

        print(f"Converted: {filename} -> {path_png}")
