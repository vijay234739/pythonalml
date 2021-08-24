import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
handlist = []
v_4_cx = 0
v_8_cx = 0
v_12_cx = 0
v_16_cx = 0
v_20_cx = 0
v_4_cy = 0
h = 0
w = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        text = "NA"

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                assert isinstance(hand_landmarks.landmark, object)
                for id, landmks in enumerate(hand_landmarks.landmark):
                    # print(id, landmks)
                    h, w, c = image.shape
                    #print("windwow")
                    #print(h)
                    #print(w)
                    #print("end")
                    cx, cy = int(landmks.x * w), int(landmks.y * h)
                    handlist.append([id, cx, cy])
                    if id == 4:
                        v_4_cx = cx
                        v_4_cy = cy
                    if id == 8:
                        v_8_cx = cx

                    if id == 12:
                        v_12_cx = cx

                    if id == 16:
                        v_16_cx = cx

                    if id == 20:
                        v_20_cx = cx

                    if id == 0:
                        v_0_cx = cx
                        v_0_cy = cy

                    if v_4_cy > v_0_cy and (v_8_cx - v_12_cx) < 50 and (v_16_cx - v_20_cx) < 50:
                        text = "Not Good"
                    if v_4_cy < v_0_cy and (v_8_cx - v_12_cx) < 50 and (v_16_cx - v_20_cx) < 50:
                        text = "Good"
                        #print(v_4_cy)
                        #print(v_0_cy)
                    # if handlist[4][cx]= handlist[4][cx] :
                    # cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(image, text, (50, 120), cv2.FONT_HERSHEY_PLAIN , 2, (255,0,0))
        cv2.imshow('Thumbsup', image)
        # print(handlist)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
