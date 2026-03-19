import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # ggf. 1 oder 2 testen
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-Fix

if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden. Anderen Index probieren (1/2) oder Berechtigung prüfen.")

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mpDraw = mp.solutions.drawing_utils

pTime = time.time()

tips = {4, 8, 12, 16, 20}

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Kamera liefert kein Bild")
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = img.shape

            for idx, lm in enumerate(handLms.landmark):
                if idx in tips:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    dt = cTime - pTime
    fps = 1 / dt if dt > 0 else 0
    pTime = cTime

    cv2.putText(img, f"{int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()