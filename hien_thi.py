import cv2

def play_video(video_path):
    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video")
        return

    # Lặp qua các khung hình trong video và phát chúng
    while True:
        ret, frame = cap.read()
        if ret:
            # Hiển thị khung hình
            cv2.imshow('Video', frame)

            # Nhấn phím 'q' để thoát
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# Đường dẫn của video
video_path = r'D:\OneDrive\Desktop\DATA_video\cho_den\2.mp4'

# Phát video
play_video(video_path)
