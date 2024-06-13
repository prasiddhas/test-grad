import pandas as pd

def prepare_dataset(frames_folder, label):
    data = []
    for frame_filename in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_filename)
        keypoints = extract_skeleton_keypoints(frame_path)
        if keypoints is not None:
            data.append(np.append(keypoints, label))
    return pd.DataFrame(data)

right_steps_data = prepare_dataset('right_steps_frames', 1)
wrong_steps_data = prepare_dataset('wrong_steps_frames', 0)

# Combine and shuffle the dataset
dataset = pd.concat([right_steps_data, wrong_steps_data]).sample(frac=1).reset_index(drop=True)

# Split features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
