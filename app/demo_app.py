import time, cv2, numpy as np, mediapipe as mp, pyttsx3

STABLE_FRAMES = 5   # frames a letter must hold before it's accepted
MIN_CONF = 0.7      # ignore low-confidence predictions
GAP_FRAMES = 15     # frames with no hand = end of word

def predict(pts):
    """Swap in the real model. Input: (21, 3) landmark array. Output: (letter, confidence)."""
    return None, 0.0

tts = pyttsx3.init()
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

text, cand, count, last, no_hand = "", None, 0, None, 0
t_prev = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    letter, conf = None, 0.0
    if res.multi_hand_landmarks:
        no_hand = 0
        lm = res.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        letter, conf = predict(np.array([[p.x, p.y, p.z] for p in lm.landmark]))
    else:
        no_hand += 1
        if no_hand == GAP_FRAMES and text and not text.endswith(" "):
            tts.say(text.split()[-1]); tts.runAndWait()   # speak the finished word
            text += " "; last = None

    if letter and conf >= MIN_CONF:
        count = count + 1 if letter == cand else 1
        cand = letter
        if count == STABLE_FRAMES and letter != last:     # debounce: accept once
            text += letter; last = letter
    else:
        cand, count = None, 0

    now = time.perf_counter(); fps = 1 / (now - t_prev); t_prev = now
    cv2.putText(frame, f"{cand or '-'} {conf:.2f}  FPS {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, text[-30:], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("ASL demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()