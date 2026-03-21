#!/usr/bin/env python3
"""
High-detail Panda3D skin-mesh hand viewer.

It consumes Unity-compatible per-frame GestureData JSONL:
{"hand_count": N, "hands": [...]}

Render style:
- Full mesh-like surface using skinned primitives:
  - Joint spheres
  - Bone capsules (cylinder + 2 end-caps)

Interaction:
- Left mouse drag: 360 orbit
- Mouse wheel: zoom
- Space: play / pause
- Left/Right arrow: frame stepping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Material,
    NodePath,
    Point3,
    Vec3,
    Vec4,
    WindowProperties,
)


HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


JOINT_RADIUS = {
    0: 1.25,
    1: 1.06, 2: 0.98, 3: 0.90, 4: 0.80,
    5: 1.10, 6: 0.95, 7: 0.84, 8: 0.72,
    9: 1.10, 10: 0.98, 11: 0.86, 12: 0.72,
    13: 1.04, 14: 0.92, 15: 0.82, 16: 0.70,
    17: 1.02, 18: 0.90, 19: 0.80, 20: 0.68,
}


class HandSkinMesh:
    def __init__(
        self,
        parent: NodePath,
        loader,
        skin_color: Vec4,
        base_joint_radius: float,
        bone_radius_scale: float,
    ) -> None:
        self.root = parent.attachNewNode("hand_skin")
        self.loader = loader
        self.base_joint_radius = base_joint_radius
        self.bone_radius_scale = bone_radius_scale

        self.mat = Material()
        self.mat.setShininess(68.0)
        self.mat.setSpecular((0.36, 0.34, 0.34, 1.0))

        self.joints: List[NodePath] = []
        for i in range(21):
            s = loader.loadModel("models/misc/sphere")
            s.reparentTo(self.root)
            s.setColor(skin_color)
            s.setMaterial(self.mat, 1)
            self.joints.append(s)

        self.bones: List[Tuple[int, int, NodePath, NodePath, NodePath]] = []
        for a, b in HAND_CONNECTIONS:
            cyl = loader.loadModel("models/misc/cylinder")
            cap_a = loader.loadModel("models/misc/sphere")
            cap_b = loader.loadModel("models/misc/sphere")
            for n in (cyl, cap_a, cap_b):
                n.reparentTo(self.root)
                n.setColor(skin_color)
                n.setMaterial(self.mat, 1)
            self.bones.append((a, b, cyl, cap_a, cap_b))

    def hide(self) -> None:
        self.root.hide()

    def show(self) -> None:
        self.root.show()

    def update(self, points: List[Vec3]) -> None:
        for i, p in enumerate(points):
            r = self.base_joint_radius * JOINT_RADIUS.get(i, 0.8)
            self.joints[i].setPos(p)
            self.joints[i].setScale(r)

        for a, b, cyl, cap_a, cap_b in self.bones:
            p0 = points[a]
            p1 = points[b]
            v = p1 - p0
            length = max(1e-4, v.length())

            # Blend thickness by endpoint size for smoother skin silhouette.
            ra = self.base_joint_radius * JOINT_RADIUS.get(a, 0.8) * self.bone_radius_scale
            rb = self.base_joint_radius * JOINT_RADIUS.get(b, 0.8) * self.bone_radius_scale
            r = (ra + rb) * 0.5

            mid = p0 + v * 0.5
            cyl.setPos(mid)
            cyl.lookAt(p1)
            # Panda misc cylinder is unit-height around Z axis.
            cyl.setScale(r, r, length * 0.5)

            cap_a.setPos(p0)
            cap_b.setPos(p1)
            cap_a.setScale(r * 0.98)
            cap_b.setScale(r * 0.98)


class SkinMeshViewer(ShowBase):
    def __init__(
        self,
        stream_path: Path,
        fps: float,
        z_scale: float,
        xy_divisor: float,
        y_offset: float,
        left_x_offset: float,
        right_x_offset: float,
        base_joint_radius: float,
        bone_radius_scale: float,
    ) -> None:
        super().__init__()
        self.disableMouse()

        self.frames = self._load_frames(stream_path)
        if not self.frames:
            raise RuntimeError(f"No frames found in {stream_path}")

        self.play_fps = max(1.0, fps)
        self.play_interval = 1.0 / self.play_fps
        self.time_acc = 0.0
        self.current_frame = 0
        self.is_playing = True

        self.z_scale = z_scale
        self.xy_divisor = xy_divisor
        self.y_offset = y_offset
        self.left_x_offset = left_x_offset
        self.right_x_offset = right_x_offset

        self._init_window()
        self._init_lights()
        self._init_camera()

        self.scene_root = self.render.attachNewNode("scene_root")
        self.left_mesh = HandSkinMesh(
            self.scene_root, self.loader, Vec4(0.88, 0.66, 0.58, 1.0), base_joint_radius, bone_radius_scale
        )
        self.right_mesh = HandSkinMesh(
            self.scene_root, self.loader, Vec4(0.88, 0.66, 0.58, 1.0), base_joint_radius, bone_radius_scale
        )
        self.left_mesh.hide()
        self.right_mesh.hide()

        self._init_controls()
        self.taskMgr.add(self._play_task, "play_task")
        self._apply_frame(self.current_frame)

    def _load_frames(self, path: Path) -> List[Dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _init_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Panda3D Hand Skin Mesh Viewer")
        props.setSize(1500, 960)
        self.win.requestProperties(props)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.03, 0.03, 0.04, 1.0)

    def _init_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.28, 0.28, 0.30, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        key = DirectionalLight("key")
        key.setColor(Vec4(0.95, 0.93, 0.90, 1.0))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-30, -35, 0)
        self.render.setLight(key_np)

        rim = DirectionalLight("rim")
        rim.setColor(Vec4(0.38, 0.45, 0.58, 1.0))
        rim_np = self.render.attachNewNode(rim)
        rim_np.setHpr(145, -20, 0)
        self.render.setLight(rim_np)

    def _init_camera(self) -> None:
        self.camera_pivot = self.render.attachNewNode("camera_pivot")
        self.camera.reparentTo(self.camera_pivot)
        self.camera.setPos(0, -3.8, 1.25)
        self.camera.lookAt(0, 0, 0)

    def _init_controls(self) -> None:
        self.accept("space", self._toggle_play)
        self.accept("arrow_right", self._step_next)
        self.accept("arrow_left", self._step_prev)
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("escape", self.userExit)

        self.dragging = False
        self.last_mouse = Point3(0, 0, 0)
        self.accept("mouse1", self._drag_start)
        self.accept("mouse1-up", self._drag_end)
        self.taskMgr.add(self._drag_task, "drag_task")

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing

    def _step_next(self) -> None:
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _step_prev(self) -> None:
        self.current_frame = (self.current_frame - 1 + len(self.frames)) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _zoom_in(self) -> None:
        self.camera.setY(self.camera, 0.22)

    def _zoom_out(self) -> None:
        self.camera.setY(self.camera, -0.22)

    def _drag_start(self) -> None:
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = Point3(m.getX(), m.getY(), 0)
            self.dragging = True

    def _drag_end(self) -> None:
        self.dragging = False

    def _drag_task(self, task: Task):
        if not self.dragging or not self.mouseWatcherNode.hasMouse():
            return Task.cont
        m = self.mouseWatcherNode.getMouse()
        dx = m.getX() - self.last_mouse.x
        dy = m.getY() - self.last_mouse.y
        self.last_mouse = Point3(m.getX(), m.getY(), 0)

        self.camera_pivot.setH(self.camera_pivot.getH() - dx * 135.0)
        new_p = self.camera_pivot.getP() + dy * 95.0
        self.camera_pivot.setP(max(-88.0, min(88.0, new_p)))
        return Task.cont

    def _play_task(self, task: Task):
        if not self.is_playing:
            return Task.cont
        self.time_acc += globalClock.getDt()
        while self.time_acc >= self.play_interval:
            self.time_acc -= self.play_interval
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self._apply_frame(self.current_frame)
        return Task.cont

    def _map_landmark(self, lm: Dict, hand_type: str) -> Vec3:
        x_offset = self.left_x_offset if hand_type == "Left" else self.right_x_offset
        x = float(lm["x"]) / self.xy_divisor + x_offset
        y = -float(lm["y"]) / self.xy_divisor - self.y_offset
        z = float(lm["z"]) * self.z_scale
        return Vec3(x, y, z)

    def _apply_frame(self, idx: int) -> None:
        frame = self.frames[idx]
        hands = frame.get("hands", [])

        left = None
        right = None
        for hand in hands:
            htype = hand.get("hand_type")
            if htype == "Left":
                left = hand
            elif htype == "Right":
                right = hand

        self._update_hand(self.left_mesh, left, "Left")
        self._update_hand(self.right_mesh, right, "Right")

    def _update_hand(self, mesh: HandSkinMesh, hand_data: Dict | None, hand_type: str) -> None:
        if hand_data is None:
            mesh.hide()
            return
        lms = hand_data.get("landmarks", [])
        if len(lms) < 21:
            mesh.hide()
            return

        points = [self._map_landmark(lms[i], hand_type) for i in range(21)]
        mesh.show()
        mesh.update(points)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="High-detail Panda3D skin mesh hand viewer")
    p.add_argument("--stream", required=True, help="unity_gesture_stream.jsonl path")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--z-scale", type=float, default=8.0)
    p.add_argument("--xy-divisor", type=float, default=1.5)
    p.add_argument("--y-offset", type=float, default=0.3)
    p.add_argument("--left-x-offset", type=float, default=-0.15)
    p.add_argument("--right-x-offset", type=float, default=0.15)
    p.add_argument("--base-joint-radius", type=float, default=0.020)
    p.add_argument("--bone-radius-scale", type=float, default=0.78)
    return p


def main() -> None:
    args = build_parser().parse_args()
    app = SkinMeshViewer(
        stream_path=Path(args.stream).resolve(),
        fps=args.fps,
        z_scale=args.z_scale,
        xy_divisor=args.xy_divisor,
        y_offset=args.y_offset,
        left_x_offset=args.left_x_offset,
        right_x_offset=args.right_x_offset,
        base_joint_radius=args.base_joint_radius,
        bone_radius_scale=args.bone_radius_scale,
    )
    app.run()


if __name__ == "__main__":
    main()
