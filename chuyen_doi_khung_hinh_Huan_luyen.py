import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import pickle
def extract_color_histogram(frame, bins=16):
    """ Trích xuất histogram màu sắc từ khung hình dưới dạng HSV. """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins]*3, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
def extract_lbp_features(frame, radius=3, n_points=24, method='uniform'):
    """ Trích xuất đặc trưng Local Binary Patterns từ khung hình. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    # Điều chỉnh số bin ở đây
    hist, _ = np.histogram(lbp.ravel(), bins=50, range=(0, n_points + 2)) # Giảm bin xuống
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def process_video(video_path, sample_rate=30, bins=16, radius=3, n_points=24):
    cap = cv2.VideoCapture(video_path)
    color_features = []
    texture_features = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_rate == 0:
            color_hist = extract_color_histogram(frame, bins=bins)
            lbp_hist = extract_lbp_features(frame, radius=radius, n_points=n_points)
            color_features.append(color_hist)
            texture_features.append(lbp_hist)
        frame_index += 1

    cap.release()
    # Đảm bảo làm phẳng và trung bình mọi thứ một cách thích hợp
    if color_features:
        color_features = np.mean(np.array(color_features), axis=0)
    if texture_features:
        texture_features = np.mean(np.array(texture_features), axis=0)
    return color_features, texture_features
def save_features(features, file_name):
    """ Lưu đặc trưng ra file. """
    with open(file_name, 'wb') as f:
        pickle.dump(features, f)


import os


def process_all_videos(directory_path, sample_rate=30):
    """Process all video files in the directory and save the .pkl files in a 'data' subdirectory."""
    # Create a directory for the .pkl files if it doesn't exist
    data_directory = os.path.join('/kaggle/working/', 'data')
    os.makedirs(data_directory, exist_ok=True)

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.mp4'):  # Check if the file is a video
            video_path = os.path.join(directory_path, file_name)
            color_features, texture_features = process_video(video_path, sample_rate)

            # Save the features in the 'data' subdirectory
            save_features(color_features, os.path.join(data_directory, f'{file_name}_color_features.pkl'))
            save_features(texture_features, os.path.join(data_directory, f'{file_name}_texture_features.pkl'))
            print(f"Đã trích xuất và lưu đặc trưng cho video {file_name}")
directory_path = '/kaggle/input/gogogomeomeomeo/cho-meo'
process_all_videos(directory_path,sample_rate=30)