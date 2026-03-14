import os

import cv2
import numpy as np
import torch
import mediapipe as mp
import pickle
import time
from src.model import BiLSTMAttentionModel

# ================= Config =================
# 1. 获取 inference.py 所在的文件夹 (.../ict/src)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (往上走一级 -> .../ict)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)

# 3. 使用 os.path.join 拼接路径 (这才是最规范的写法，Windows/Mac 通用)
# 这样无论你在哪里运行 main.py，它都能精准找到文件
MODEL_PATH = os.path.join(PROJECT_ROOT, "result", "checkpoints", "best_model_300.pth")
DICT_PATH  = os.path.join(PROJECT_ROOT, "data", "idx2name.pkl")
MEAN_PATH  = os.path.join(PROJECT_ROOT, "data", "global_mean_300_double_vel.npy")
STD_PATH   = os.path.join(PROJECT_ROOT, "data", "global_std_300_double_vel.npy")

INPUT_SIZE = 268
SEQ_LEN = 64
HIDDEN_SIZE = 512
NUM_CLASSES = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 1) 特征转换：双重相对坐标 + 速度  → 268维
# ==========================================================
def to_double_relative_with_velocity(data):
    T = data.shape[0]
    pose = data[:, 0:50].reshape(T, 25, 2)
    lh   = data[:, 50:92].reshape(T, 21, 2)
    rh   = data[:, 92:134].reshape(T, 21, 2)

    nose = pose[:, 0:1, :]
    l_wrist = lh[:, 0:1, :]
    r_wrist = rh[:, 0:1, :]

    pose_rel = pose - nose
    lh_rel = lh - l_wrist
    rh_rel = rh - r_wrist

    pose_d = np.diff(pose_rel, axis=0)
    lh_d   = np.diff(lh_rel, axis=0)
    rh_d   = np.diff(rh_rel, axis=0)

    pose_d = np.concatenate([np.zeros_like(pose_d[:1]), pose_d], axis=0)
    lh_d   = np.concatenate([np.zeros_like(lh_d[:1]), lh_d], axis=0)
    rh_d   = np.concatenate([np.zeros_like(rh_d[:1]), rh_d], axis=0)

    final_feat = np.concatenate([
        pose_rel.reshape(T, -1),
        lh_rel.reshape(T, -1),
        rh_rel.reshape(T, -1),
        pose_d.reshape(T, -1),
        lh_d.reshape(T, -1),
        rh_d.reshape(T, -1)
    ], axis=1)

    return final_feat

# ==========================================================
# 2) EMA 平滑类
# ==========================================================
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if current_landmarks is None:
            self.prev_landmarks = None
            return None
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        smoothed = self.alpha * current_landmarks + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed
        return smoothed

# ==========================================================
# 3) 实时手语识别
# ==========================================================
class RealTimeSignRecognizer:
    def __init__(self):
        print(f"Loading model on {DEVICE}...")

        self.model = BiLSTMAttentionModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout=0.0
        ).to(DEVICE)

        state = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()
        print("Model loaded.")

        with open(DICT_PATH, 'rb') as f:
            self.idx2name = pickle.load(f)

        self.mean = np.load(MEAN_PATH)
        self.std  = np.load(STD_PATH)
        print(f"Loaded normalization shapes: mean={self.mean.shape}, std={self.std.shape}")

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.smoother = LandmarkSmoother(alpha=0.6)
        self.recording = False
        self.frame_buffer = []

        self.pred_text = "Waiting..."
        self.conf_text = ""
        self.frame_text = ""
        self.fps_text = "0"
        self.prev_time = time.time()

    # ---------------- Feature extraction ----------------
    def extract_features(self, results):
        row = []

        if results.pose_landmarks:
            for i in range(25):
                lm = results.pose_landmarks.landmark[i]
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 50)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)

        return np.array(row, dtype=np.float32)

    # ---------------- Preprocess for LSTM ----------------
    def preprocess(self, seq):
        if len(seq) < 5:
            return None

        raw = np.array(seq, dtype=np.float32)
        feat268 = to_double_relative_with_velocity(raw)
        feat268 = (feat268 - self.mean) / self.std

        idx = np.linspace(0, len(feat268)-1, SEQ_LEN)
        idx = np.round(idx).astype(int)
        final = feat268[idx]

        return torch.from_numpy(final).unsqueeze(0).float().to(DEVICE)

    # ---------------- Predict ----------------
    def predict(self):
        inp = self.preprocess(self.frame_buffer)
        if inp is None:
            self.pred_text = "Too Short"
            return

        with torch.no_grad():
            out = self.model(inp)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)

        self.pred_text = self.idx2name[idx.item()]
        self.conf_text = f"{conf.item():.2%}"
        self.frame_text = str(len(self.frame_buffer))
        print(f"[PRED] {self.pred_text}, Conf={conf.item():.4f}, Frames={len(self.frame_buffer)}")

    # ---------------- Draw ----------------
    def draw_landmarks(self, frame, results):
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    # ---------------- Main loop ----------------
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return

        cv2.namedWindow("Sign Language AI", cv2.WINDOW_NORMAL)

        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.recording:
                    self.recording = True
                    self.frame_buffer = []
                    self.pred_text = "..."
                else:
                    self.recording = False
                    self.predict()

        cv2.setMouseCallback("Sign Language AI", mouse_click)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()
            fps = 1 / (now - self.prev_time)
            self.prev_time = now
            self.fps_text = f"{fps:.1f}"

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            self.draw_landmarks(frame, results)

            if self.recording:
                raw_feat = self.extract_features(results)
                smooth_feat = self.smoother.smooth(raw_feat)
                if smooth_feat is not None:
                    self.frame_buffer.append(smooth_feat)
                cv2.putText(frame, f"REC {len(self.frame_buffer)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self.smoother.prev_landmarks = None
                cv2.putText(frame, "Click to Record", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Info overlay
            cv2.rectangle(frame, (0, 400), (640, 480), (40, 40, 40), -1)
            cv2.putText(frame, f"Result: {self.pred_text}", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Conf: {self.conf_text}", (20, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {self.frame_text}", (230, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps_text}", (420, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Sign Language AI", frame)
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("Sign Language AI", 0) < 0:  # ESC 或关闭窗口
                break

        cap.release()
        cv2.destroyAllWindows()


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    print("[DEBUG] Starting RealTimeSignRecognizer...")
    try:
        app = RealTimeSignRecognizer()
        print("[DEBUG] Initialized successfully.")
    except Exception as e:
        print("[ERROR] Initialization failed:", e)
        raise

    try:
        app.run()
    except Exception as e:
        print("[ERROR] Runtime error:", e)
