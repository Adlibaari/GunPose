import pandas as pd
import os

df = pd.read_csv('/path/to/your/directory/level2/keypoints.csv')

# Path to dataset folder 
dataset_path = '/path/to/your/directory/level2/dataset/'
pose_path = os.path.join(dataset_path, 'pose')
non_pose_path = os.path.join(dataset_path, 'nonpose')

# Path to pose folder
pose_path = os.path.join(dataset_path, 'pose')
# List all files in pose folder
pose_files = os.listdir(pose_path)

# Path to non-pose folder
non_pose_path = os.path.join(dataset_path, 'nonpose')
# List all files in non-pose folder
non_pose_files = os.listdir(non_pose_path)

# Display all file names in pose folder
print("Files in pose folder:", pose_files)

# Display all file names in non-pose folder
print("Files in non_pose folder:", non_pose_files)

# function to determine label based on name file
def get_label(image_name, pose_path, non_pose_path):
    if image_name in os.listdir(pose_path):
        return 'pose'
    elif image_name in os.listdir(non_pose_path):
        return 'nonpose'
    else:
        return None  # Not found in both folders

# Adds label column based on the folder name (pose or nonpose)
df['label'] = df['image_name'].apply(lambda x: get_label(x, pose_path, non_pose_path))
df.to_csv(f'{dataset_path}dataset.csv', index=False)
