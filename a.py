import numpy as np
from scipy.spatial import distance
import cv2
from skimage.feature import local_binary_pattern
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

def load_features(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def extract_color_histogram(frame, bins=16):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins]*3, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(frame, radius=3, n_points=24, method='uniform'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
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



def compare_features(input_features, features_dict):
    input_color_features, input_texture_features = input_features
    similarities = {}
    for video_name, (color_features, texture_features) in features_dict.items():
        # Kiểm tra và điều chỉnh kích thước vector nếu cần
        print(
            f"Dimension mismatch in color features for video {video_name}: expected {input_color_features.shape}, got {color_features.shape}")

        print(
            f"Dimension mismatch in texture features for video {video_name}: expected {input_texture_features.shape}, got {texture_features.shape}")

        # Tính toán độ tương đồng
        color_similarity = distance.cosine(input_color_features, color_features)
        texture_similarity = distance.cosine(input_texture_features, texture_features)
        similarities[video_name] = color_similarity + texture_similarity

    return similarities

def load_all_features(data_directory):
    features_dict = {}
    for file_name in os.listdir(data_directory):
        if file_name.endswith('_color_features.pkl'):
            video_name = file_name[:-19]
            color_features = load_features(os.path.join(data_directory, f'{video_name}_color_features.pkl')).flatten()
            texture_features = load_features(os.path.join(data_directory, f'{video_name}_texture_features.pkl')).flatten()
            features_dict[video_name] = (color_features, texture_features)
    return features_dict

def find_most_similar_videos(input_video_path, data_directory, top_k=5):
    input_features = process_video(input_video_path)
    features_dict = load_all_features(data_directory)
    similarities = compare_features(input_features, features_dict)
    most_similar_videos = sorted(similarities, key=similarities.get)[:top_k]
    return most_similar_videos

input_video_path = 'D:/pythonProject1/cho_den/1.mp4'
data_directory = 'D:/pythonProject1/results/data'
most_similar_videos = find_most_similar_videos(input_video_path, data_directory, top_k=5)
print(f'The top 5 most similar videos to {input_video_path} are {most_similar_videos}')
