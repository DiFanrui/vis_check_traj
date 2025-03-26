import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Extract the number part from the filename
            number_str = filename.split('.')[0]
            # Create the new filename with leading zeros
            new_filename = f"{int(number_str):08d}.jpg"
            # Construct full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")

if __name__ == "__main__":
    images_directory = "./tum_stable_seq_005_part_1_dif_1/images"  # Change this to your images directory path
    rename_files(images_directory)