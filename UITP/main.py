import cv2
import mediapipe as mp
import time
from subprocess import call
import numpy as np

INDEX_FINGER_IDX = 8
THUMB_IDX = 4
VOLUME_UPDATE_INTERVAL = 15

# Открываем камеру
videoCap = cv2.VideoCapture(0)
lastFrameTime = 0
frame = 0
max_diff = 0
min_diff = 100000
handSolution = mp.solutions.hands
hands = handSolution.Hands()
while True:
    frame += 1

    # Чтение изоображения
    success, img = videoCap.read()

    # Вывод изоображения в отдельном окне (только удачные попытки)
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Вычисления ФПС
        thisFrameTime = time.time()
        fps = 1 / (thisFrameTime - lastFrameTime)
        lastFrameTime = thisFrameTime

        # ФПС изоображения
        cv2.putText(img, f'FPS:{int(fps)}',
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Распознавание рук на изоображении

        recHands = hands.process(img)
        if recHands.multi_hand_landmarks:
            for hand in recHands.multi_hand_landmarks:

                # Рисуем точки на каждом сегменте изоображения для облегчения визуализации

                for datapoint_id, point in enumerate(hand.landmark):
                    h, w, c = img.shape
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(img, (x, y),
                               10, (255, 0, 255)
                               , cv2.FILLED)
            if frame % VOLUME_UPDATE_INTERVAL == 0:
                thumb_y = hand.landmark[THUMB_IDX].y
                index_y = hand.landmark[INDEX_FINGER_IDX].y
                distance = thumb_y * h - index_y * h

                # Откалибровать минимальную и максимальную разницу

                min_diff = np.minimum(distance + 50, min_diff)
                max_diff = np.maximum(distance, max_diff)

                #change volume
                call(["osascript -e 'set volume output volume {}'"
                      .format(np.clip(
                              (distance/(max_diff - min_diff)*100),
                                      0, 100))], shell=True)
                frame = 0
        cv2.imshow("CamOutput", img)
        cv2.waitKey(1)