#!/usr/bin/env python3
"""
Offline video -> hand landmarks DB -> 3D rendering video pipeline.

This is a Python refactor of the reusable parts from:
- Test.cs
- KalmanFilter.cs
- Vector3KalmanFilter.cs
- HandLandmarkFilter.cs

It does not modify the original Unity/C# code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandLandmarkData:
    id: int
    x: float
    y: float
    z: float


@dataclass
class HandData:
    hand_index: int
    hand_type: str
    bound_area: float
    hand_gesture: str
    landmarks: List[HandLandmarkData]


@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


class KalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self.q = process_noise
        self.r = measurement_noise
        self.x = 0.0
        self.p = 1.0
        self.k = 0.0

    def update(self, measurement: float) -> float:
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1.0 - self.k) * self.p
        return self.x

    def set_state(self, state: float, covariance: float) -> None:
        self.x = state
        self.p = covariance


class Vector3KalmanFilter:
    def __init__(self, process_noise: float = 0.008, measurement_noise: float = 0.03) -> None:
        self.x_filter = KalmanFilter(process_noise, measurement_noise)
        self.y_filter = KalmanFilter(process_noise, measurement_noise)
        self.z_filter = KalmanFilter(process_noise, measurement_noise)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.x_filter.update(float(measurement[0])),
                self.y_filter.update(float(measurement[1])),
                self.z_filter.update(float(measurement[2])),
            ],
            dtype=np.float32,
        )

    def set_state(self, state: np.ndarray, covariance: float) -> None:
        self.x_filter.set_state(float(state[0]), covariance)
        self.y_filter.set_state(float(state[1]), covariance)
        self.z_filter.set_state(float(state[2]), covariance)


class HandLandmarkFilter:
    LANDMARK_COUNT = 21

    def __init__(self, process_noise: float = 0.008, measurement_noise: float = 0.03) -> None:
        self.left_filters = [Vector3KalmanFilter(process_noise, measurement_noise) for _ in range(self.LANDMARK_COUNT)]
        self.right_filters = [Vector3KalmanFilter(process_noise, measurement_noise) for _ in range(self.LANDMARK_COUNT)]
        self.initialized_left = False
        self.initialized_right = False

    def filter_left_hand_landmarks(self, left_hand: List[HandLandmarkData], z_scale: float) -> None:
        self._filter_hand(left_hand, self.left_filters, "left", z_scale)

    def filter_right_hand_landmarks(self, right_hand: List[HandLandmarkData], z_scale: float) -> None:
        self._filter_hand(right_hand, self.right_filters, "right", z_scale)

    def _filter_hand(
        self,
        hand: List[HandLandmarkData],
        filters: List[Vector3KalmanFilter],
        hand_side: str,
        z_scale: float,
    ) -> None:
        if hand is None:
            return

        if len(hand) < self.LANDMARK_COUNT:
            return

        if hand_side == "left" and not self.initialized_left:
            for i in range(self.LANDMARK_COUNT):
                pos = np.array([hand[i].x, hand[i].y, hand[i].z * z_scale], dtype=np.float32)
                filters[i].set_state(pos, 1.0)
            self.initialized_left = True

        if hand_side == "right" and not self.initialized_right:
            for i in range(self.LANDMARK_COUNT):
                pos = np.array([hand[i].x, hand[i].y, hand[i].z * z_scale], dtype=np.float32)
                filters[i].set_state(pos, 1.0)
            self.initialized_right = True

        for i in range(self.LANDMARK_COUNT):
            raw_pos = np.array([hand[i].x, hand[i].y, hand[i].z * z_scale], dtype=np.float32)
            filtered = filters[i].update(raw_pos)
            hand[i].x = float(filtered[0])
            hand[i].y = float(filtered[1])
            hand[i].z = float(filtered[2] / z_scale)


def get_mysql_connection(cfg: MySQLConfig, with_database: bool = True):
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("pymysql is required: pip install pymysql") from exc

    conn_kwargs = {
        "host": cfg.host,
        "port": cfg.port,
        "user": cfg.user,
        "password": cfg.password,
        "charset": "utf8mb4",
        "autocommit": False,
    }
    if with_database:
        conn_kwargs["database"] = cfg.database
    return pymysql.connect(**conn_kwargs)


def init_db(cfg: MySQLConfig):
    root_conn = get_mysql_connection(cfg, with_database=False)
    root_cur = root_conn.cursor()
    root_cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    root_cur.close()
    root_conn.close()

    conn = get_mysql_connection(cfg, with_database=True)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hand_landmarks (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            video_id VARCHAR(255) NOT NULL,
            frame_index INT NOT NULL,
            timestamp_ms INT NOT NULL,
            hand_type VARCHAR(16) NOT NULL,
            hand_index INT NOT NULL,
            bound_area DOUBLE NOT NULL,
            hand_gesture VARCHAR(64) NOT NULL,
            landmarks_json LONGTEXT NOT NULL
        );
        """
    )
    try:
        cur.execute("CREATE INDEX idx_video_frame ON hand_landmarks(video_id, frame_index);")
    except Exception:
        # Index may already exist.
        pass
    conn.commit()
    return conn


def compute_bound_area(landmarks: List[HandLandmarkData]) -> float:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    if not xs or not ys:
        return 0.0
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def extract_to_db(video_path: Path, mysql_cfg: MySQLConfig, video_id: str, max_hands: int = 2) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for extract mode: pip install mediapipe") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    conn = init_db(mysql_cfg)
    cur = conn.cursor()
    cur.execute("DELETE FROM hand_landmarks WHERE video_id = %s", (video_id,))
    conn.commit()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0
    total_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_index, (hand_lms, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                hand_type = handedness.classification[0].label

                landmarks: List[HandLandmarkData] = []
                for i, lm in enumerate(hand_lms.landmark):
                    landmarks.append(
                        HandLandmarkData(
                            id=i,
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                        )
                    )

                bound_area = compute_bound_area(landmarks)
                hand_gesture = "unknown"
                landmarks_json = json.dumps([p.__dict__ for p in landmarks], ensure_ascii=False)

                cur.execute(
                    """
                    INSERT INTO hand_landmarks(
                        video_id, frame_index, timestamp_ms, hand_type, hand_index,
                        bound_area, hand_gesture, landmarks_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        video_id,
                        frame_index,
                        timestamp_ms,
                        hand_type,
                        hand_index,
                        bound_area,
                        hand_gesture,
                        landmarks_json,
                    ),
                )
                total_written += 1

        if frame_index % 200 == 0:
            conn.commit()
        frame_index += 1

    conn.commit()
    conn.close()
    hands.close()
    cap.release()
    print(f"[extract] frames={frame_index}, rows={total_written}, video_id={video_id}")


def load_frame_data(mysql_cfg: MySQLConfig, video_id: str) -> Dict[int, List[HandData]]:
    conn = get_mysql_connection(mysql_cfg, with_database=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT frame_index, hand_index, hand_type, bound_area, hand_gesture, landmarks_json
        FROM hand_landmarks
        WHERE video_id = %s
        ORDER BY frame_index ASC, hand_index ASC
        """,
        (video_id,),
    )

    frame_data: Dict[int, List[HandData]] = {}
    for frame_index, hand_index, hand_type, bound_area, hand_gesture, landmarks_json in cur.fetchall():
        landmarks_raw = json.loads(landmarks_json)
        landmarks = [
            HandLandmarkData(
                id=int(p["id"]),
                x=float(p["x"]),
                y=float(p["y"]),
                z=float(p["z"]),
            )
            for p in landmarks_raw
        ]
        hand = HandData(
            hand_index=int(hand_index),
            hand_type=str(hand_type),
            bound_area=float(bound_area),
            hand_gesture=str(hand_gesture),
            landmarks=landmarks,
        )
        frame_data.setdefault(int(frame_index), []).append(hand)

    conn.close()
    return frame_data


def render_3d_panel(
    hands: List[HandData],
    landmark_filter: Optional[HandLandmarkFilter],
    z_scale: float,
    xy_divisor: float,
    y_offset: float,
    left_x_offset: float,
    right_x_offset: float,
    fig: plt.Figure,
    ax: plt.Axes,
) -> np.ndarray:
    ax.clear()
    ax.set_title("Offline 3D Hand Render")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-2.0, 2.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)

    for hand in hands:
        if hand.hand_type == "Left":
            if landmark_filter is not None:
                landmark_filter.filter_left_hand_landmarks(hand.landmarks, z_scale=z_scale)
            color = "#2ca02c"
            x_offset = left_x_offset
        else:
            if landmark_filter is not None:
                landmark_filter.filter_right_hand_landmarks(hand.landmarks, z_scale=z_scale)
            color = "#d62728"
            x_offset = right_x_offset

        points = np.zeros((len(hand.landmarks), 3), dtype=np.float32)
        for i, p in enumerate(hand.landmarks):
            points[i, 0] = p.x / xy_divisor + x_offset
            points[i, 1] = -p.y / xy_divisor - y_offset
            points[i, 2] = p.z * z_scale

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=16)
        for a, b in HAND_CONNECTIONS:
            ax.plot(
                [points[a, 0], points[b, 0]],
                [points[a, 1], points[b, 1]],
                [points[a, 2], points[b, 2]],
                color=color,
                linewidth=2.0,
            )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def render_video(
    video_path: Path,
    mysql_cfg: MySQLConfig,
    video_id: str,
    output_path: Path,
    z_scale: float = 8.0,
    xy_divisor: float = 1.5,
    y_offset: float = 0.3,
    left_x_offset: float = -0.15,
    right_x_offset: float = 0.15,
    use_kalman: bool = True,
    left_hand_model: Optional[Path] = None,
    right_hand_model: Optional[Path] = None,
    sphere_prefab: Optional[Path] = None,
) -> None:
    frame_data = load_frame_data(mysql_cfg, video_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width * 2
    out_height = height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout(pad=0.2)

    filter_obj = HandLandmarkFilter(0.008, 0.03) if use_kalman else None

    if left_hand_model and not left_hand_model.exists():
        print(f"[warn] LeftHand.fbx not found: {left_hand_model}")
    if right_hand_model and not right_hand_model.exists():
        print(f"[warn] RightHand.fbx not found: {right_hand_model}")
    if sphere_prefab and not sphere_prefab.exists():
        print(f"[warn] Sphere.prefab not found: {sphere_prefab}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hands = frame_data.get(frame_index, [])
        panel = render_3d_panel(
            hands=hands,
            landmark_filter=filter_obj,
            z_scale=z_scale,
            xy_divisor=xy_divisor,
            y_offset=y_offset,
            left_x_offset=left_x_offset,
            right_x_offset=right_x_offset,
            fig=fig,
            ax=ax,
        )
        panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        combined = np.hstack([frame, panel])
        writer.write(combined)
        frame_index += 1

    cap.release()
    writer.release()
    plt.close(fig)
    print(f"[render] frames={frame_index}, out={output_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()
    video_id = args.video_id or video_path.stem
    mysql_cfg = MySQLConfig(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
    )

    if args.mode in ("extract", "pipeline"):
        extract_to_db(video_path, mysql_cfg, video_id, max_hands=args.max_hands)

    if args.mode in ("render", "pipeline"):
        render_video(
            video_path=video_path,
            mysql_cfg=mysql_cfg,
            video_id=video_id,
            output_path=output_path,
            z_scale=args.z_scale,
            xy_divisor=args.xy_divisor,
            y_offset=args.y_offset,
            left_x_offset=args.left_x_offset,
            right_x_offset=args.right_x_offset,
            use_kalman=not args.disable_kalman,
            left_hand_model=Path(args.left_hand_model).resolve() if args.left_hand_model else None,
            right_hand_model=Path(args.right_hand_model).resolve() if args.right_hand_model else None,
            sphere_prefab=Path(args.sphere_prefab).resolve() if args.sphere_prefab else None,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline video hand processing and 3D render pipeline."
    )
    parser.add_argument("--mode", choices=["extract", "render", "pipeline"], default="pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="offline_render.mp4", help="Rendered output video path")
    parser.add_argument("--video-id", default=None, help="Video id in database; default uses input video stem")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", default="")
    parser.add_argument("--mysql-database", default="sign_language")

    parser.add_argument("--z-scale", type=float, default=8.0, help="Depth scale (Test.cs used *4, here default is *8)")
    parser.add_argument("--xy-divisor", type=float, default=1.5, help="X/Y scale divisor")
    parser.add_argument("--y-offset", type=float, default=0.3)
    parser.add_argument("--left-x-offset", type=float, default=-0.15)
    parser.add_argument("--right-x-offset", type=float, default=0.15)
    parser.add_argument("--disable-kalman", action="store_true")

    parser.add_argument("--left-hand-model", default=None, help="LeftHand.fbx path (asset check)")
    parser.add_argument("--right-hand-model", default=None, help="RightHand.fbx path (asset check)")
    parser.add_argument("--sphere-prefab", default=None, help="Sphere.prefab path (asset check)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
