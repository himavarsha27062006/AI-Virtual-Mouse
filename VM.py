import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import numpy as np

# ─────────────────────────────────────────────
#  Init
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_face   = mp.solutions.face_mesh

hands     = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

# ─────────────────────────────────────────────
#  Mouse state
# ─────────────────────────────────────────────
prev_x, prev_y = 0, 0
dragging       = False

# ─────────────────────────────────────────────
#  Eye / screenshot state
# ─────────────────────────────────────────────
last_click      = 0
last_screenshot = 0
show_ss_text    = 0
flash           = 0

LEFT_EYE_TOP      = 159
LEFT_EYE_BOTTOM   = 145
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOTTOM  = 374

EYE_CLOSED_THRESHOLD  = 0.010
WINK_CLOSED_THRESHOLD = 0.012
WINK_OPEN_THRESHOLD   = 0.018

# ─────────────────────────────────────────────
#  Zoom state
# ─────────────────────────────────────────────
zoom_locked    = False
zoom_active    = False
prev_zoom_dist = None
last_zoom      = 0
zoom_cooldown  = 0.08
zoom_threshold = 0.01
zoom_label     = ""
show_zoom_text = 0

# ─────────────────────────────────────────────
#  Crossed fingers gesture state  (notepad)
# ─────────────────────────────────────────────
last_crossed      = 0
crossed_cooldown  = 3.0
show_crossed_text = 0
crossed_label     = ""

# ─────────────────────────────────────────────
#  OK sign gesture state  (voice typing)
# ─────────────────────────────────────────────
last_ok      = 0
ok_cooldown  = 3.0
show_ok_text = 0
ok_label     = ""

# ─────────────────────────────────────────────
#  Cursor movement config
# ─────────────────────────────────────────────
CONTROL_LEFT   = 0.20
CONTROL_RIGHT  = 0.80
CONTROL_TOP    = 0.20
CONTROL_BOTTOM = 0.80

DEAD_ZONE   = 0.008
ACCEL_POWER = 1.6
MAX_SPEED   = 0.04

ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA       = 0.07
ONE_EURO_D_CUTOFF   = 1.0
FRAME_DT            = 1 / 30

# ─────────────────────────────────────────────
#  One Euro Filter state
# ─────────────────────────────────────────────
_prev_x_filt = None
_prev_y_filt = None
_dx_hat      = 0.0
_dy_hat      = 0.0

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def dist(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def eye_gap(landmarks, top_idx, bottom_idx):
    return dist(landmarks[top_idx], landmarks[bottom_idx])

def is_fist(lm):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    return all(lm[tip].y > lm[pip].y for tip, pip in zip(finger_tips, finger_pips))

def is_open_hand(lm):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    return all(lm[tip].y < lm[pip].y for tip, pip in zip(finger_tips, finger_pips))

def _euro_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)

def one_euro_step(x, prev_x, dx_hat,
                  min_cutoff=ONE_EURO_MIN_CUTOFF,
                  beta=ONE_EURO_BETA,
                  d_cutoff=ONE_EURO_D_CUTOFF,
                  dt=FRAME_DT):
    dx         = (x - prev_x) / dt
    a_d        = _euro_alpha(d_cutoff, dt)
    dx_hat_new = a_d * dx + (1.0 - a_d) * dx_hat
    cutoff     = min_cutoff + beta * abs(dx_hat_new)
    a          = _euro_alpha(cutoff, dt)
    x_filt     = a * x + (1.0 - a) * prev_x
    return x_filt, dx_hat_new

def map_to_screen(raw, lo, hi, screen_size):
    clamped    = max(lo, min(hi, raw))
    normalised = (clamped - lo) / (hi - lo)
    return normalised * screen_size

def apply_accel(delta_norm, dead_zone, power, max_speed):
    if abs(delta_norm) < dead_zone:
        return 0.0
    clamped = max(-max_speed, min(max_speed, delta_norm))
    sign    = 1.0 if clamped > 0 else -1.0
    return sign * (abs(clamped) / max_speed) ** power * max_speed

def is_crossed_fingers(lm):
    # Index and middle raised and crossed (tips close together),
    # ring and pinky curled, thumb relaxed
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    tips_close = abs(lm[8].x - lm[12].x) < 0.04
    return index_up and middle_up and ring_down and pinky_down and tips_close

def is_ok_sign(lm):
    # Thumb tip and index tip pinched together, other three fingers extended
    thumb_index_pinch = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.05
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    return thumb_index_pinch and middle_up and ring_up and pinky_up

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    if hand_results.multi_hand_landmarks:

        num_hands = len(hand_results.multi_hand_landmarks)
        all_lm    = [h.landmark for h in hand_results.multi_hand_landmarks]

        for hand in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        if num_hands == 2:

            if dragging:
                pyautogui.mouseUp()
                dragging = False

            lm0, lm1 = all_lm[0], all_lm[1]
            fist0    = is_fist(lm0)
            fist1    = is_fist(lm1)
            open0    = is_open_hand(lm0)
            open1    = is_open_hand(lm1)

            if fist0 and fist1:
                if zoom_active:
                    zoom_active    = False
                    zoom_locked    = True
                    prev_zoom_dist = None
                    zoom_label     = "ZOOM STOPPED"
                    show_zoom_text = time.time()

                for lm in [lm0, lm1]:
                    cx = int(lm[9].x * w_img)
                    cy = int(lm[9].y * h_img)
                    cv2.circle(img, (cx, cy), 30, (0, 0, 255), 3)

            elif open0 and open1:
                zoom_locked = False
                zoom_active = True

                w0 = lm0[0]
                w1 = lm1[0]
                curr_zoom_dist = dist(w0, w1)

                pt0 = (int(w0.x * w_img), int(w0.y * h_img))
                pt1 = (int(w1.x * w_img), int(w1.y * h_img))
                cv2.line(img, pt0, pt1, (255, 165, 0), 2)
                cv2.circle(img, pt0, 10, (255, 165, 0), -1)
                cv2.circle(img, pt1, 10, (255, 165, 0), -1)

                if prev_zoom_dist is not None and not zoom_locked:
                    delta = curr_zoom_dist - prev_zoom_dist
                    if time.time() - last_zoom > zoom_cooldown:
                        if delta > zoom_threshold:
                            pyautogui.hotkey('ctrl', '+')
                            zoom_label     = "ZOOM IN  +"
                            show_zoom_text = time.time()
                            last_zoom      = time.time()
                        elif delta < -zoom_threshold:
                            pyautogui.hotkey('ctrl', '-')
                            zoom_label     = "ZOOM OUT -"
                            show_zoom_text = time.time()
                            last_zoom      = time.time()

                prev_zoom_dist = curr_zoom_dist

            else:
                prev_zoom_dist = None

        else:
            zoom_active    = False
            zoom_locked    = False
            prev_zoom_dist = None

            lm        = all_lm[0]
            index_tip = lm[8]
            thumb_tip = lm[4]

            target_x = map_to_screen(index_tip.x, CONTROL_LEFT, CONTROL_RIGHT, screen_w)
            target_y = map_to_screen(index_tip.y, CONTROL_TOP,  CONTROL_BOTTOM, screen_h)

            if _prev_x_filt is None:
                _prev_x_filt = target_x
                _prev_y_filt = target_y

            target_x, _dx_hat = one_euro_step(target_x, _prev_x_filt, _dx_hat)
            target_y, _dy_hat = one_euro_step(target_y, _prev_y_filt, _dy_hat)

            delta_norm_x = (target_x - _prev_x_filt) / screen_w
            delta_norm_y = (target_y - _prev_y_filt) / screen_h

            z_scale   = max(0.6, min(1.4, 1.0 - lm[0].z * 3))
            eff_power = ACCEL_POWER * z_scale

            accel_x = apply_accel(delta_norm_x, DEAD_ZONE, eff_power, MAX_SPEED) * screen_w
            accel_y = apply_accel(delta_norm_y, DEAD_ZONE, eff_power, MAX_SPEED) * screen_h

            index_up  = lm[8].y  < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up   = lm[16].y < lm[14].y
            pinky_up  = lm[20].y < lm[18].y

            precision_mode = (pinky_up and not index_up
                              and not middle_up and not ring_up)
            if precision_mode:
                accel_x *= 0.35
                accel_y *= 0.35
                cv2.putText(img, "PRECISION MODE", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)

            curr_x = max(0, min(screen_w - 1, _prev_x_filt + accel_x))
            curr_y = max(0, min(screen_h - 1, _prev_y_filt + accel_y))

            pyautogui.moveTo(curr_x, curr_y)
            _prev_x_filt   = curr_x
            _prev_y_filt   = curr_y
            prev_x, prev_y = curr_x, curr_y

            if dist(thumb_tip, index_tip) < 0.03:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            if index_up and middle_up and not ring_up and not pinky_up:
                pyautogui.scroll(60)
            elif not index_up and not middle_up and not ring_up and not pinky_up:
                pyautogui.scroll(-60)

            if is_crossed_fingers(lm) and time.time() - last_crossed > crossed_cooldown:
                os.startfile('notepad.exe')
                crossed_label     = "NOTEPAD OPENED"
                show_crossed_text = time.time()
                last_crossed      = time.time()

            if is_ok_sign(lm) and time.time() - last_ok > ok_cooldown:
                pyautogui.hotkey('win', 'h')
                ok_label     = "VOICE TYPING ON"
                show_ok_text = time.time()
                last_ok      = time.time()


    else:
        _prev_x_filt   = None
        _prev_y_filt   = None
        _dx_hat        = 0.0
        _dy_hat        = 0.0
        prev_zoom_dist = None
        zoom_active    = False
        zoom_locked    = False

    if face_results.multi_face_landmarks:

        for face in face_results.multi_face_landmarks:

            flm = face.landmark

            left_ear  = eye_gap(flm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM)
            right_ear = eye_gap(flm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

            left_closed  = left_ear  < WINK_CLOSED_THRESHOLD
            right_closed = right_ear < WINK_CLOSED_THRESHOLD
            left_open    = left_ear  > WINK_OPEN_THRESHOLD
            right_open   = right_ear > WINK_OPEN_THRESHOLD

            if left_closed and right_closed:
                if time.time() - last_click > 1:
                    pyautogui.click()
                    last_click = time.time()

            elif left_closed and right_open:
                if time.time() - last_screenshot > 2:
                    folder   = os.path.join(os.getcwd(), "screenshots")
                    os.makedirs(folder, exist_ok=True)
                    filename = os.path.join(folder, f"screenshot_{int(time.time())}.png")
                    pyautogui.screenshot(filename)
                    show_ss_text    = time.time()
                    flash           = 1
                    last_screenshot = time.time()

            elif right_closed and left_open:
                if time.time() - last_screenshot > 2:
                    folder   = os.path.join(os.getcwd(), "screenshots")
                    os.makedirs(folder, exist_ok=True)
                    filename = os.path.join(folder, f"screenshot_{int(time.time())}.png")
                    pyautogui.screenshot(filename)
                    show_ss_text    = time.time()
                    flash           = 1
                    last_screenshot = time.time()

    # ── HUD ───────────────────────────────────

    if time.time() - show_zoom_text < 0.8:
        color = (0, 200, 255) if "IN"  in zoom_label else \
                (0, 80,  255) if "OUT" in zoom_label else \
                (0, 0,   220)
        cv2.putText(img, zoom_label, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if zoom_active:
        cv2.putText(img, "ZOOM MODE  |  fist both hands to stop",
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 165, 0), 2)

    if time.time() - show_ss_text < 1:
        cv2.putText(img, "SCREENSHOT TAKEN", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if time.time() - show_crossed_text < 1.0:
        cv2.putText(img, crossed_label, (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 100), 2)

    if time.time() - show_ok_text < 1.0:
        cv2.putText(img, ok_label, (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    if flash == 1:
        white = np.ones_like(img) * 255
        img   = cv2.addWeighted(img, 0.3, white, 0.7, 0)
        if time.time() - show_ss_text > 0.2:
            flash = 0

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
