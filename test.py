import os
import cv2

if __name__ == "__main__":
    videos_dir = 'data'

    for filename in os.listdir(videos_dir):
        cap = cv2.VideoCapture(os.path.join(videos_dir, filename))

        while (cap.isOpened()):
            ret, frame = cap.read()
            print(frame, ret)
            if ret:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
