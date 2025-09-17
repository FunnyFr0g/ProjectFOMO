import cv2
import numpy as np


# Параметры для vid1:
# bg_subtractor = AlignedBackgroundSubtractor(
#     buffer_size=5,
#     threshold=10,
#     min_contour_area=3,
#     max_contour_area=1000
# )
# def align_images(self, img1, img2):
#     """Выравнивание img2 относительно img1 с помощью ORB + гомографии"""
#     min_match_count = 10
#       ransac_thr = 3.0

# kernel = np.ones((3, 3), np.uint8)




class AlignedBackgroundSubtractor:
    def __init__(self, buffer_size=5, threshold=30, min_contour_area=5, max_contour_area=75):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.frame_buffer = []
        self.raw_frame_buffer = []
        self.background = None
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.mask = None

    def align_images(self, img1, img2):
        """Выравнивание img2 относительно img1 с помощью ORB + гомографии"""
        min_match_count = 10
        use_SIFT = False


        if use_SIFT:
            # Initiate SIFT detector
            sift = self.sift
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
        else:
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)



        if des1 is None or des2 is None or len(des1) < min_match_count or len(des2) < min_match_count:
            print(f'мало KP, пропускаем')
            return img2

        if use_SIFT:
            ## SIFT ###
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        else:
            ### ORB ###
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches) * 0.15)]



        if len(good_matches) < min_match_count:
            print(f'мало KP, пропускаем')
            return img2

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

        if H is None:
            print(f'Вырожденная матрица, пропускаем')
            return img2

        aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        return aligned

    def update_background(self, frame):
        """Обновление фона с выравниванием кадров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
            self.raw_frame_buffer.pop(0)

        if self.frame_buffer:
            self.raw_frame_buffer.append(gray)
            self.frame_buffer = list([self.align_images(gray, b_im) for b_im in self.raw_frame_buffer])
            # for i in range(len(self.frame_buffer)):        # Выровнять все старые по новому кадру
            #     self.frame_buffer[i] = self.align_images(self.raw_frame_buffer[i], gray)

            # aligned = self.align_images(self.frame_buffer[-1], gray)
            # self.frame_buffer.append(aligned)

        else:
            self.frame_buffer.append(gray)
            self.raw_frame_buffer.append(gray)

        if len(self.frame_buffer) == self.buffer_size:
            self.background = np.median(self.frame_buffer, axis=0).astype(np.uint8)

    def detect_objects(self, frame):
        """Обнаружение объектов с вычитанием фона"""
        if self.background is None:
            return frame, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.background)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        self.mask = thresh

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area <= area <= self.max_contour_area:
                filtered_contours.append(cnt)

        bboxes = []
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)  # Получаем площадь контура
            bboxes.append((x, y, w, h, area))  # Добавляем площадь в информацию о bbox

            # Рисуем прямоугольник
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Добавляем текст с площадью над прямоугольником
            cv2.putText(frame, f"Area: {area:.1f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame, bboxes


def main(video_path, output_path, start_frame=0, save_vid=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 60

    if save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID' для AVI MP4v для mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    bg_subtractor = AlignedBackgroundSubtractor(
        buffer_size=5,
        threshold=30,
        min_contour_area=3,
        max_contour_area=250
    )

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            print(f'{frame_count / 75 :.2f}s')

            frame_count += 1
            if frame_count < start_frame:
                continue

            bg_subtractor.update_background(frame)
            processed_frame, bboxes = bg_subtractor.detect_objects(frame.copy())
            # processed_frame, bboxes = frame.copy(), []

            # Отображение номера кадра
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Object Detection', processed_frame)

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, bg_subtractor.background)
                # cv2.imshow('Object Detection (mask)', bg_subtractor.mask)
                cv2.imshow('Object Detection (diff)', diff)

                mask_bgr = cv2.cvtColor(bg_subtractor.mask, cv2.COLOR_GRAY2BGR)
                combined_frame = np.hstack((processed_frame, mask_bgr))
            except Exception as e:
                cv2.imshow('Object Detection (diff)', processed_frame)
                combined_frame = np.hstack((processed_frame, frame))
                print(e)

            if save_vid:
                out.write(combined_frame)
                # out.write(frame)
            cv2.imshow('combined_frame', combined_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Пробел - пауза и показ буфера
            paused = True
            print("Пауза. Показываю выравненные кадры из буфера...")
            for i, buf_frame in enumerate(bg_subtractor.frame_buffer):
                cv2.imshow(f'Aligned Frame {i}', buf_frame)
        elif key == 13 or key == ord('w'):  # Enter - продолжение
            paused = False
            # Закрываем все окна с буферными кадрами
            for i in range(len(bg_subtractor.frame_buffer)):
                cv2.destroyWindow(f'Aligned Frame {i}')
            print("Продолжение воспроизведения...")

    if save_vid:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fps = 60
    video_path = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\2025-06-05 14-31-20.mp4"
    output_path = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\vid2_det.mp4"
    main(video_path, output_path=output_path, start_frame=0, save_vid=True)