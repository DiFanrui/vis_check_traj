# 文件路径
output_file = "depth.txt"

# 开始时间戳
start_timestamp = 0

# 时间间隔（每张图片间隔1秒）
interval = 1

# 打开文件写入
with open(output_file, "w") as file:
    # 写入文件头
    file.write("# depth maps\n")
    file.write("# file: 'rgbd_dataset_freiburg3_long_office_household.bag'\n")
    file.write("# timestamp filename\n")
    
    # 写入每一行的时间戳和文件名
    for i in range(400):
        timestamp = start_timestamp + i * interval
        filename = f"depth/{i:03d}_depth.png"
        file.write(f"{timestamp:.6f} {filename}\n")

print(f"文件已生成：{output_file}")