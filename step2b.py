
import os

# Path ke folder dataset
dataset_path = 'C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/dataset/'

# Path ke folder cutting
cutting_path = os.path.join(dataset_path, 'gun')
# List semua file di dalam folder cutting
cutting_files = os.listdir(cutting_path)

# Path ke folder non_cutting
non_cutting_path = os.path.join(dataset_path, 'nongun')
# List semua file di dalam folder non_cutting
non_cutting_files = os.listdir(non_cutting_path)

# Menampilkan list nama file di folder cutting
print("Files in cutting folder:", cutting_files)

# Menampilkan list nama file di folder non_cutting
print("Files in non_cutting folder:", non_cutting_files)

import pandas as pd
import os

df = pd.read_csv('C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/keypoints.csv')

# Path ke folder dataset
dataset_path = 'C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/dataset/'
cutting_path = os.path.join(dataset_path, 'gun')
non_cutting_path = os.path.join(dataset_path, 'nongun')


# Fungsi untuk menentukan label berdasarkan nama file
def get_label(image_name, cutting_path, non_cutting_path):
    if image_name in os.listdir(cutting_path):
        return 'gun'
    elif image_name in os.listdir(non_cutting_path):
        return 'nongun'
    else:
        return None  # Tidak dapat menemukan file di kedua folder

# Menambahkan kolom label berdasarkan nama folder
df['label'] = df['image_name'].apply(lambda x: get_label(x, cutting_path, non_cutting_path))
df.to_csv(f'{dataset_path}dataset.csv', index=False)
df
