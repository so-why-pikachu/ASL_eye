#!/usr/bin/env python3
"""
Panda3D hand renderer for Unity-compatible GestureData JSONL stream.

Input JSONL format (one frame per line):
{
  "hand_count": 1,
  "hands": [...]
}
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
    LineSegs,
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


class PandaHandViewer(ShowBase):
    def __init__(
        self,
        stream_path: Path,
        fps: float,
        z_scale: float,
        xy_divisor: float,
        y_offset: float,
        left_x_offset: float,
        right_x_offset: float,
        joint_scale: float,
        bone_thickness: float,
    ) -> None:
        super().__init__()

        self.disableMouse()
        self.stream_path = stream_path
        self.frames = self._load_frames(stream_path)
        if not self.frames:
            raise RuntimeError(f"No frames in stream: {stream_path}")

        self.play_fps = fps
        self.play_interval = 1.0 / max(1.0, fps)
        self.time_acc = 0.0
        self.current_frame = 0
        self.is_playing = True

        self.z_scale = z_scale
        self.xy_divisor = xy_divisor
        self.y_offset = y_offset
        self.left_x_offset = left_x_offset
        self.right_x_offset = right_x_offset
        self.joint_scale = joint_scale
        self.bone_thickness = bone_thickness

        self._init_window()
        self._init_scene()
        self._init_interaction()

        self.taskMgr.add(self._update_playback, "update_playback")
        self._apply_frame(self.current_frame)

    def _load_frames(self, stream_path: Path) -> List[Dict]:
        lines = stream_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _init_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Panda3D Hand Viewer - 360 Rotate / Zoom")
        props.setSize(1400, 900)
        self.win.requestProperties(props)

    def _init_scene(self) -> None:
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.04, 0.04, 0.06, 1.0)

        self.root = self.render.attachNewNode("root")
        self.camera_pivot = self.root.attachNewNode("camera_pivot")
        self.camera.reparentTo(self.camera_pivot)
        self.camera.setPos(0, -3.2, 1.2)
        self.camera.lookAt(0, 0, 0)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.30, 0.30, 0.34, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        key = DirectionalLight("key")
        key.setColor(Vec4(0.95, 0.95, 0.95, 1.0))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-35, -40, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.45, 0.48, 0.55, 1.0))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(120, -20, 0)
        self.render.setLight(fill_np)

        self.left_node = self.root.attachNewNode("left_hand")
        self.right_node = self.root.attachNewNode("right_hand")
        self.left_node.hide()
        self.right_node.hide()

        self.left_joints = self._build_joint_nodes(self.left_node, Vec4(0.86, 0.62, 0.50, 1.0))
        self.right_joints = self._build_joint_nodes(self.right_node, Vec4(0.86, 0.62, 0.50, 1.0))

    def _build_joint_nodes(self, parent: NodePath, color: Vec4) -> List[NodePath]:
        nodes: List[NodePath] = []
        for i in range(21):
            sphere = self.loader.loadModel("models/misc/sphere")
            sphere.reparentTo(parent)
            sphere.setScale(self.joint_scale)
            sphere.setColor(color)
            mat = Material()
            mat.setShininess(45.0)
            mat.setSpecular((0.35, 0.35, 0.35, 1))
            sphere.setMaterial(mat, 1)
            sphere.setName(f"joint_{i}")
            nodes.append(sphere)
        return nodes

    def _init_interaction(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("space", self._toggle_play)
        self.accept("arrow_right", self._next_frame)
        self.accept("arrow_left", self._prev_frame)
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)

        self.dragging = False
        self.last_mouse = Point3(0, 0, 0)
        self.accept("mouse1", self._start_drag)
        self.accept("mouse1-up", self._stop_drag)

        self.taskMgr.add(self._update_mouse_drag, "update_mouse_drag")

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing

    def _next_frame(self) -> None:
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _prev_frame(self) -> None:
        self.current_frame = (self.current_frame - 1 + len(self.frames)) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _zoom_in(self) -> None:
        self.camera.setY(self.camera, 0.18)

    def _zoom_out(self) -> None:
        self.camera.setY(self.camera, -0.18)

    def _start_drag(self) -> None:
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = Point3(m.getX(), m.getY(), 0)
            self.dragging = True

    def _stop_drag(self) -> None:
        self.dragging = False

    def _update_mouse_drag(self, task: Task):
        if not self.dragging or not self.mouseWatcherNode.hasMouse():
            return Task.cont
        m = self.mouseWatcherNode.getMouse()
        dx = m.getX() - self.last_mouse.x
        dy = m.getY() - self.last_mouse.y
        self.last_mouse = Point3(m.getX(), m.getY(), 0)
        self.camera_pivot.setH(self.camera_pivot.getH() - dx * 140.0)
        new_p = self.camera_pivot.getP() + dy * 90.0
        self.camera_pivot.setP(max(-89.0, min(89.0, new_p)))
        return Task.cont

    def _update_playback(self, task: Task):
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

    def _apply_frame(self, frame_idx: int) -> None:
        frame = self.frames[frame_idx]
        hands = frame.get("hands", [])

        left_data = None
        right_data = None
        for hand in hands:
            if hand.get("hand_type") == "Left":
                left_data = hand
            elif hand.get("hand_type") == "Right":
                right_data = hand

        self._update_hand(self.left_node, self.left_joints, left_data, "Left")
        self._update_hand(self.right_node, self.right_joints, right_data, "Right")

    def _update_hand(
        self,
        hand_root: NodePath,
        joints: List[NodePath],
        hand_data: Dict | None,
        hand_type: str,
    ) -> None:
        old_lines = hand_root.find("**/bones")
        if not old_lines.isEmpty():
            old_lines.removeNode()

        if hand_data is None:
            hand_root.hide()
            return

        lms = hand_data.get("landmarks", [])
        if len(lms) < 21:
            hand_root.hide()
            return

        hand_root.show()
        points: List[Vec3] = []
        for i in range(21):
            pos = self._map_landmark(lms[i], hand_type)
            joints[i].setPos(pos)
            points.append(pos)

        segs = LineSegs("bones")
        segs.setThickness(self.bone_thickness)
        segs.setColor(0.92, 0.80, 0.73, 1.0)
        for a, b in HAND_CONNECTIONS:
            segs.moveTo(points[a])
            segs.drawTo(points[b])
        hand_root.attachNewNode(segs.create())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Panda3D 3D hand viewer for Unity GestureData stream")
    p.add_argument("--stream", required=True, help="Path to unity_gesture_stream.jsonl")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--z-scale", type=float, default=8.0)
    p.add_argument("--xy-divisor", type=float, default=1.5)
    p.add_argument("--y-offset", type=float, default=0.3)
    p.add_argument("--left-x-offset", type=float, default=-0.15)
    p.add_argument("--right-x-offset", type=float, default=0.15)
    p.add_argument("--joint-scale", type=float, default=0.022)
    p.add_argument("--bone-thickness", type=float, default=4.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    app = PandaHandViewer(
        stream_path=Path(args.stream).resolve(),
        fps=args.fps,
        z_scale=args.z_scale,
        xy_divisor=args.xy_divisor,
        y_offset=args.y_offset,
        left_x_offset=args.left_x_offset,
        right_x_offset=args.right_x_offset,
        joint_scale=args.joint_scale,
        bone_thickness=args.bone_thickness,
    )
    app.run()


if __name__ == "__main__":
    main()
