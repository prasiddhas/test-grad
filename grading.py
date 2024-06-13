def grade_video(video_path):
    # Extract frames and skeleton keypoints
    keypoints_list = []
    extract_frames(video_path, 'temp_frames')
    for frame_filename in os.listdir('temp_frames'):
        frame_path = os.path.join('temp_frames', frame_filename)
        keypoints = extract_skeleton_keypoints(frame_path)
        if keypoints is not None:
            keypoints_list.append(keypoints)
    
    if not keypoints_list:
        return None

    keypoints_tensor = torch.tensor(keypoints_list, dtype=torch.float32)
    
    # Predict labels for each frame
    with torch.no_grad():
        outputs = model(keypoints_tensor)
        predicted_labels = (outputs > 0.5).float().numpy()
    
    # Calculate the percentage of right and wrong poses
    right_percentage = np.mean(predicted_labels)
    wrong_percentage = 1 - right_percentage
    
    # Clean up temporary frames
    for frame_filename in os.listdir('temp_frames'):
        os.remove(os.path.join('temp_frames', frame_filename))
    
    return right_percentage, wrong_percentage

# Example usage
right_percentage, wrong_percentage = grade_video('path_to_new_video.mp4')
print(f'Right Pose Percentage: {right_percentage * 100:.2f}%')
print(f'Wrong Pose Percentage: {wrong_percentage * 100:.2f}%')
