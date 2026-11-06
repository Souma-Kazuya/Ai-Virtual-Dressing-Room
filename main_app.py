import os
import time
import cv2
import mediapipe as mp
import numpy as np
import json
import base64

# ---------------------- Config ---------------------- #
CAM_WIDTH = 1280
CAM_HEIGHT = 720
WIN_NAME = "AI Virtual Try-On - JARVIS HUD"

CATEGORY_BUTTON_SIZE = (250, 140)
BUTTON_SIZE = 100
THUMBNAIL_HEIGHT = 150
SHIRT_RATIO_H_W = 581 / 440
THUMBNAIL_WIDTH = int(THUMBNAIL_HEIGHT / SHIRT_RATIO_H_W)
THUMBNAIL_GAP = 30
BOTTOM_BAR_HEIGHT = THUMBNAIL_HEIGHT + (THUMBNAIL_GAP * 2)

PICTURE_COOLDOWN = 3.0
HOVER_SELECTION_DELAY = 1.5

# HUD Colors
HUD_CYAN = (255, 255, 0)
HUD_GOLD = (0, 215, 255)
HUD_RED = (0, 0, 255)

# ---------------------- MediaPipe Setup ---------------------- #
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.hands as mp_hands
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ---------------------- Helper Functions ---------------------- #
class LandmarkSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_pts = None

    def smooth(self, pts):
        if self.prev_pts is None:
            self.prev_pts = pts
            return pts
        smoothed = self.alpha * pts + (1 - self.alpha) * self.prev_pts
        self.prev_pts = smoothed
        return smoothed

smoother = LandmarkSmoother(alpha=0.3)

def warp_and_overlay(img_back, img_front, points_back, points_front):
    h_b, w_b = img_back.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32(points_front), np.float32(points_back))
    warped_shirt = cv2.warpPerspective(img_front, matrix, (w_b, h_b), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    if warped_shirt.shape[2] == 4:
        alpha = warped_shirt[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]
        img_back[:] = (1.0 - alpha) * img_back + alpha * warped_shirt[:, :, :3]
    return img_back

def overlay_png(img_back, img_front, pos=(0, 0)):
    if img_front is None: return img_back
    x, y = map(int, pos)
    hf, wf = img_front.shape[:2]
    hb, wb = img_back.shape[:2]
    x1, x2 = max(0, x), min(wb, x + wf)
    y1, y2 = max(0, y), min(hb, y + hf)
    x1f, x2f = max(0, -x), min(wf, wb - x)
    y1f, y2f = max(0, -y), min(hf, hb - y)
    if x1 >= x2 or y1 >= y2: return img_back
    front_roi = img_front[y1f:y2f, x1f:x2f]
    back_roi = img_back[y1:y2, x1:x2]
    if front_roi.shape[2] == 4:
        alpha = front_roi[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]
        img_back[y1:y2, x1:x2] = (1.0 - alpha) * back_roi + alpha * front_roi[:, :, :3]
    else:
        img_back[y1:y2, x1:x2] = front_roi
    return img_back

def load_ui_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        p = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        p[:, :, :3] = (100, 100, 100)
        p[:, :, 3] = 255
        return p
    return cv2.resize(img, size)

def load_shirts_data(folder):
    if not os.path.exists(folder): return [], []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    thumbnails = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        if img is not None:
            thumbnails.append(cv2.resize(img, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)))
        else:
            thumbnails.append(np.zeros((THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 4), dtype=np.uint8))
    return files, thumbnails

def is_peace_sign(hl):
    return hl.landmark[8].y < hl.landmark[6].y and hl.landmark[12].y < hl.landmark[10].y and \
           hl.landmark[16].y > hl.landmark[14].y and hl.landmark[20].y > hl.landmark[18].y

# ---------------------- Nano Banana Engine ---------------------- #
class NanoBananaEngine:
    def __init__(self):
        self.enabled = False
        self.client = None
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.initialize()

    def initialize(self, api_key=None):
        if api_key: self.api_key = api_key
        if self.api_key and self.api_key != "YOUR_API_KEY_HERE":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.enabled = True
                print("Nano Banana: AI Engine Ready.")
            except: pass

    def generate_tryon(self, person_img, cloth_img):
        if not self.enabled: return None
        try:
            from google.genai import types
            _, p_enc = cv2.imencode('.jpg', person_img)
            _, c_enc = cv2.imencode('.jpg', cloth_img)
            # The 'No Prompt' neural alignment simulation
            return None 
        except: return None

# ---------------------- Main Logic ---------------------- #
class TryOnApp:
    def __init__(self):
        self.nano_engine = NanoBananaEngine()
        self.user_wardrobe_path = "assets/user_wardrobe"
        os.makedirs(self.user_wardrobe_path, exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.mouse_pos = [0, 0]
        cv2.setMouseCallback(WIN_NAME, lambda e,x,y,f,p: setattr(self, 'mouse_pos', [x,y]))
        self.exit_btn = load_ui_image("assets/ui/exit.png", (BUTTON_SIZE, BUTTON_SIZE))
        self.cam_btn = load_ui_image("assets/ui/camera.png", (BUTTON_SIZE, BUTTON_SIZE))
        self.back_btn = load_ui_image("assets/ui/back.png", (BUTTON_SIZE, BUTTON_SIZE))
        self.categories = ["casual", "formal", "summer", "winter", "personal"]
        self.cat_imgs = [load_ui_image(f"assets/categories/{c}.png", CATEGORY_BUTTON_SIZE) for c in self.categories[:-1]]
        p_img = np.zeros((CATEGORY_BUTTON_SIZE[1], CATEGORY_BUTTON_SIZE[0], 4), dtype=np.uint8)
        cv2.putText(p_img, "PERSONAL", (30, 80), 0, 1, (255, 255, 0), 2)
        self.cat_imgs.append(p_img)
        self.hover_timer = 0
        self.hovered_item = None
        self.start_time = time.time()

    def check_hover(self, item_id):
        if self.hovered_item != item_id:
            self.hovered_item, self.hover_timer = item_id, time.time()
            return False
        return time.time() - self.hover_timer >= HOVER_SELECTION_DELAY

    def draw_glass_button(self, frame, x, y, w, h, text, color, hover=False):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, 0.6 if hover else 0.4, frame, 0.4 if hover else 0.6, 0, frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x+15, y+h//2+10), 0, 0.7, (255, 255, 255), 1)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        t = time.time() - self.start_time
        cv2.line(frame, (20, 20), (60, 20), HUD_CYAN, 2)
        cv2.line(frame, (20, 20), (20, 60), HUD_CYAN, 2)
        cv2.line(frame, (w-20, 20), (w-60, 20), HUD_CYAN, 2)
        cv2.line(frame, (w-20, 20), (w-20, 60), HUD_CYAN, 2)
        sy = int((np.sin(t * 2) + 1) / 2 * h)
        cv2.line(frame, (0, sy), (w, sy), HUD_CYAN, 1)
        cv2.putText(frame, "SYSTEM: ONLINE", (w//2-70, 30), 0, 0.6, HUD_GOLD, 2)

    def get_interaction(self, res_hands, w, h):
        c = (self.mouse_pos[0], self.mouse_pos[1])
        p = False
        if res_hands.multi_hand_landmarks:
            hl = res_hands.multi_hand_landmarks[0]
            c = (int(hl.landmark[8].x * w), int(hl.landmark[8].y * h))
            p = is_peace_sign(hl)
        return c, p

    def set_key_popup(self):
        k = ""
        while True:
            s, f = self.cap.read()
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            overlay = f.copy()
            cv2.rectangle(overlay, (w//2-300, h//2-50), (w//2+300, h//2+50), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)
            cv2.rectangle(f, (w//2-300, h//2-50), (w//2+300, h//2+50), HUD_CYAN, 2)
            cv2.putText(f, "ENTER API KEY:", (w//2-280, h//2-70), 0, 0.8, HUD_GOLD, 2)
            cv2.putText(f, k + "_", (w//2-280, h//2+10), 0, 0.7, HUD_CYAN, 2)
            cv2.imshow(WIN_NAME, f)
            key = cv2.waitKey(1) & 0xFF
            if key == 13: self.nano_engine.initialize(k); break
            elif key == 27: break
            elif key == 8: k = k[:-1]
            elif key != 255: k += chr(key)

    def home_screen(self):
        while True:
            s, f = self.cap.read()
            if not s: return False
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            rh = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            c, _ = self.get_interaction(rh, w, h)
            self.draw_hud(f)
            cx, cy, r = w//2, h//2+50, 80
            hv = np.linalg.norm(np.array([cx, cy]) - np.array(c)) < r
            cv2.circle(f, (cx, cy), r, HUD_CYAN if not hv else HUD_GOLD, 2)
            cv2.putText(f, "START", (cx-45, cy+15), 0, 1, (255, 255, 255), 2)
            self.draw_glass_button(f, 20, 20, 150, 50, "SET API", HUD_GOLD, 20<c[0]<170 and 20<c[1]<70)
            if 20<c[0]<170 and 20<c[1]<70 and self.check_hover("set_key"): self.set_key_popup()
            if hv and self.check_hover("start"): return True
            cv2.drawMarker(f, c, HUD_CYAN, 0, 20, 2)
            cv2.imshow(WIN_NAME, f)
            if cv2.waitKey(1) & 0xFF == ord('q'): return False

    def select_gender(self):
        while True:
            s, f = self.cap.read()
            if not s: return None
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            rh = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            c, _ = self.get_interaction(rh, w, h)
            self.draw_glass_button(f, w//2-400, h//2-90, 350, 180, "MALE", (200, 100, 0), w//2-400<c[0]<w//2-50)
            self.draw_glass_button(f, w//2+50, h//2-90, 350, 180, "FEMALE", (100, 0, 200), w//2+50<c[0]<w//2+400)
            if h//2-90 < c[1] < h//2+90:
                if w//2-400 < c[0] < w//2-50 and self.check_hover("male"): return "male"
                if w//2+50 < c[0] < w//2+400 and self.check_hover("female"): return "female"
            cv2.drawMarker(f, c, HUD_CYAN, 0, 20, 2)
            cv2.imshow(WIN_NAME, f)
            if cv2.waitKey(1) & 0xFF == ord('q'): return None

    def capture_mode(self):
        while True:
            s, f = self.cap.read()
            if not s: return None
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            bx, by, bw, bh = w//2-250, h//2-300, 500, 600
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), HUD_CYAN, 2)
            cv2.putText(f, "PEACE TO CAPTURE", (bx, by-20), 0, 0.7, HUD_GOLD, 2)
            rh = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if rh.multi_hand_landmarks and is_peace_sign(rh.multi_hand_landmarks[0]):
                roi = f[by:by+bh, bx:bx+bw]
                p = os.path.join(self.user_wardrobe_path, f"u_{int(time.time())}.png")
                cv2.imwrite(p, roi); return p
            cv2.imshow(WIN_NAME, f)
            if cv2.waitKey(1) & 0xFF == ord('q'): return None

    def category_screen(self, gender):
        while True:
            s, f = self.cap.read()
            if not s: return None
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            rh = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            c, _ = self.get_interaction(rh, w, h)
            for i, cat in enumerate(self.categories):
                x, y = (50 if i%2==0 else w-300, 150 + (i//2)*190)
                f = overlay_png(f, self.cat_imgs[i], (x, y))
                if x < c[0] < x+250 and y < c[1] < y+140:
                    cv2.rectangle(f, (x, y), (x+250, y+140), HUD_GOLD, 3)
                    if self.check_hover(cat):
                        if cat == "personal": self.capture_mode()
                        return cat
            cv2.drawMarker(f, c, HUD_CYAN, 0, 20, 2)
            cv2.imshow(WIN_NAME, f)
            if cv2.waitKey(1) & 0xFF == ord('q'): return "exit"

    def try_on_screen(self, gender, category):
        folder = self.user_wardrobe_path if category=="personal" else os.path.join("assets/clothes", gender, category)
        files, thumbs = load_shirts_data(folder)
        if not files: return "back"
        idx, neural, ai_f, s_o, s_v = 0, False, None, 0, 0
        curr = cv2.imread(os.path.join(folder, files[idx]), cv2.IMREAD_UNCHANGED)
        while True:
            s, f = self.cap.read()
            if not s: return "exit"
            f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rp = pose.process(img_rgb)
            rh = hands.process(img_rgb)
            c, p = self.get_interaction(rh, w, h)
            ax, ay = w-120, 250
            self.draw_glass_button(f, ax, ay, 100, 100, "AI", (0,255,0) if neural else (100,100,100), ax<c[0]<ax+100 and ay<c[1]<ay+100)
            if ax<c[0]<ax+100 and ay<c[1]<ay+100 and self.check_hover("ai"):
                neural = not neural
                if neural and self.nano_engine.enabled: ai_f = self.nano_engine.generate_tryon(f, curr)
            if neural and ai_f is not None: f = ai_f
            elif rp.pose_landmarks:
                lm = rp.pose_landmarks.landmark
                p11, p12 = np.array([lm[11].x*w, lm[11].y*h]), np.array([lm[12].x*w, lm[12].y*h])
                p23, p24 = np.array([lm[23].x*w, lm[23].y*h]), np.array([lm[24].x*w, lm[24].y*h])
                sw = np.linalg.norm(p11-p12)
                if sw > 0:
                    src = np.float32([p12+[-sw*0.1,-sw*0.1], p11+[sw*0.1,-sw*0.1], p24+[-sw*0.1,sw*0.1], p23+[sw*0.1,sw*0.1]])
                    dst = np.float32([[0,0], [curr.shape[1],0], [0,curr.shape[0]], [curr.shape[1],curr.shape[0]]])
                    f = warp_and_overlay(f, curr, src, dst)
            ov = f.copy()
            cv2.rectangle(ov, (0, h-BOTTOM_BAR_HEIGHT), (w, h), (20,20,20), -1)
            cv2.addWeighted(ov, 0.6, f, 0.4, 0, f)
            if c[1] > h-BOTTOM_BAR_HEIGHT: s_v += (c[0]-w//2)*0.05
            s_o = max(0, min(s_o+s_v, (len(thumbs)-1)*(THUMBNAIL_WIDTH+THUMBNAIL_GAP)))
            s_v *= 0.8
            for i, th in enumerate(thumbs):
                tx = int(w/2 - THUMBNAIL_WIDTH/2 + i*(THUMBNAIL_WIDTH+THUMBNAIL_GAP) - s_o)
                if -THUMBNAIL_WIDTH < tx < w:
                    f = overlay_png(f, th, (tx, h-THUMBNAIL_HEIGHT-THUMBNAIL_GAP))
                    if i == idx: cv2.rectangle(f, (tx-5, h-THUMBNAIL_HEIGHT-THUMBNAIL_GAP-5), (tx+THUMBNAIL_WIDTH+5, h-THUMBNAIL_GAP+5), HUD_GOLD, 3)
                    if tx < c[0] < tx+THUMBNAIL_WIDTH and h-BOTTOM_BAR_HEIGHT < c[1] < h and self.check_hover(f"s_{i}"):
                        idx, curr = i, cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_UNCHANGED)
            f = overlay_png(f, self.exit_btn, (w-BUTTON_SIZE-20, 20))
            f = overlay_png(f, self.cam_btn, (w-BUTTON_SIZE-20, 140))
            f = overlay_png(f, self.back_btn, (20, 20))
            if 20 < c[1] < 120:
                if w-120 < c[0] < w-20 and self.check_hover("exit"): return "exit"
                if 20 < c[0] < 120 and self.check_hover("back"): return "back"
            if w-120 < c[0] < w-20 and 140 < c[1] < 240 and (self.check_hover("cam") or p):
                os.makedirs("assets/pictures", exist_ok=True); cv2.imwrite(f"assets/pictures/p_{int(time.time())}.png", f)
            cv2.drawMarker(f, c, HUD_CYAN, 0, 20, 2)
            cv2.imshow(WIN_NAME, f)
            if cv2.waitKey(1) & 0xFF == ord('q'): return "exit"

    def run(self):
        while True:
            if not self.home_screen(): break
            g = self.select_gender()
            if g:
                while True:
                    cat = self.category_screen(g)
                    if cat == "exit": return
                    if cat and self.try_on_screen(g, cat) == "exit": return
        self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    app = TryOnApp(); app.run()
