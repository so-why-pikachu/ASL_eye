#!/usr/bin/env python3
"""
Axis-aware FBX skinning-driven Panda3D hand viewer.
Uses hierarchical swing-twist style quaternion driving without offset JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Filename,
    Material,
    NodePath,
    Point3,
    Quat,
    Vec3,
    Vec4,
    WindowProperties,
)


def _to_model_path(p: Path) -> str:
    return Filename.fromOsSpecific(str(p.resolve())).getFullpath()


def _finger(prefix: str, name: str) -> List[str]:
    if name == "Thumb":
        return [
            f"{prefix}_{name}Metacarpal",
            f"{prefix}_{name}Proximal",
            f"{prefix}_{name}Distal",
        ]
    return [
        f"{prefix}_{name}Metacarpal",
        f"{prefix}_{name}Proximal",
        f"{prefix}_{name}Intermediate",
        f"{prefix}_{name}Distal",
    ]


LANDMARK_TO_BONE_SEQ_LEFT = {
    "wrist": (0, "L_Wrist", 9),
    "thumb": [
        (1, _finger("L", "Thumb")[0], 2),
        (2, _finger("L", "Thumb")[1], 3),
        (3, _finger("L", "Thumb")[2], 4),
    ],
    "index": [
        (0, _finger("L", "Index")[0], 5),
        (5, _finger("L", "Index")[1], 6),
        (6, _finger("L", "Index")[2], 7),
        (7, _finger("L", "Index")[3], 8),
    ],
    "middle": [
        (0, _finger("L", "Middle")[0], 9),
        (9, _finger("L", "Middle")[1], 10),
        (10, _finger("L", "Middle")[2], 11),
        (11, _finger("L", "Middle")[3], 12),
    ],
    "ring": [
        (0, _finger("L", "Ring")[0], 13),
        (13, _finger("L", "Ring")[1], 14),
        (14, _finger("L", "Ring")[2], 15),
        (15, _finger("L", "Ring")[3], 16),
    ],
    "little": [
        (0, _finger("L", "Little")[0], 17),
        (17, _finger("L", "Little")[1], 18),
        (18, _finger("L", "Little")[2], 19),
        (19, _finger("L", "Little")[3], 20),
    ],
}

LANDMARK_TO_BONE_SEQ_RIGHT = {
    "wrist": (0, "R_Wrist", 9),
    "thumb": [
        (1, _finger("R", "Thumb")[0], 2),
        (2, _finger("R", "Thumb")[1], 3),
        (3, _finger("R", "Thumb")[2], 4),
    ],
    "index": [
        (0, _finger("R", "Index")[0], 5),
        (5, _finger("R", "Index")[1], 6),
        (6, _finger("R", "Index")[2], 7),
        (7, _finger("R", "Index")[3], 8),
    ],
    "middle": [
        (0, _finger("R", "Middle")[0], 9),
        (9, _finger("R", "Middle")[1], 10),
        (10, _finger("R", "Middle")[2], 11),
        (11, _finger("R", "Middle")[3], 12),
    ],
    "ring": [
        (0, _finger("R", "Ring")[0], 13),
        (13, _finger("R", "Ring")[1], 14),
        (14, _finger("R", "Ring")[2], 15),
        (15, _finger("R", "Ring")[3], 16),
    ],
    "little": [
        (0, _finger("R", "Little")[0], 17),
        (17, _finger("R", "Little")[1], 18),
        (18, _finger("R", "Little")[2], 19),
        (19, _finger("R", "Little")[3], 20),
    ],
}


class FBXHandRig:
    def __init__(
        self,
        actor: Actor,
        actor_root: NodePath,
        is_left: bool,
        pos_smooth: float,
        rot_smooth: float,
        wrist_stability: float,
        model_scale: float,
        landmark_scale: float,
        root_motion_scale: float,
    ) -> None:
        self.actor = actor
        self.actor_root = actor_root
        self.is_left = is_left
        self.pos_smooth = max(0.0, min(0.98, pos_smooth))
        self.rot_smooth = max(0.0, min(0.98, rot_smooth))
        self.wrist_stability = max(0.0, min(0.98, wrist_stability))
        self.model_scale = max(0.001, model_scale)
        self.landmark_scale = max(0.1, landmark_scale)
        self.root_motion_scale = max(0.0, root_motion_scale)

        self.mapping = LANDMARK_TO_BONE_SEQ_LEFT if is_left else LANDMARK_TO_BONE_SEQ_RIGHT
        self.joints: Dict[str, NodePath] = {}
        self.rest_quat: Dict[str, Quat] = {}
        self.prev_hpr: Dict[str, Vec3] = {}
        self.local_fwd: Dict[str, Vec3] = {}

        self.smoothed_pts: List[Vec3] = []
        self.root_pos = Vec3(0, 0, 0)
        self.palm_up_smooth = Vec3(0, 1, 0)
        self.palm_normal_smooth = Vec3(0, 0, 1)

        all_bones = set()
        all_bones.add(self.mapping["wrist"][1])
        for k in ("thumb", "index", "middle", "ring", "little"):
            for _, name, _ in self.mapping[k]:
                all_bones.add(name)

        for bone in sorted(all_bones):
            node = actor.controlJoint(None, "modelRoot", bone)
            if node is not None and not node.isEmpty():
                self.joints[bone] = node

        self.actor.reparentTo(self.actor_root)
        self.actor.setScale(1.0)
        self.actor_root.setScale(self.model_scale)

        for bone, node in self.joints.items():
            q = Quat(node.getQuat(self.actor_root))
            self.rest_quat[bone] = q
            hpr = node.getHpr(self.actor_root)
            self.prev_hpr[bone] = Vec3(hpr.x, hpr.y, hpr.z)

        self._init_local_forward_axes()
        self._init_rest_palm_frame()

    def _init_local_forward_axes(self) -> None:
        # Infer each bone's forward axis from rest-pose child direction.
        for bone, node in self.joints.items():
            if node.getNumChildren() <= 0:
                continue
            child = node.getChild(0)
            d = child.getPos(self.actor_root) - node.getPos(self.actor_root)
            if d.lengthSquared() < 1e-8:
                continue
            d.normalize()
            inv_rest = Quat(self.rest_quat[bone])
            inv_rest.invertInPlace()
            local = inv_rest.xform(d)
            if local.lengthSquared() < 1e-8:
                continue
            local.normalize()
            self.local_fwd[bone] = local

        # Fallback for distal bones without direct child in controlled chain.
        for bone in self.joints.keys():
            if bone in self.local_fwd:
                continue
            parent = bone.replace("Distal", "Intermediate").replace("ThumbDistal", "ThumbProximal")
            if parent in self.local_fwd:
                self.local_fwd[bone] = Vec3(self.local_fwd[parent])
            else:
                self.local_fwd[bone] = Vec3(0, 0, 1)

    def _init_rest_palm_frame(self) -> None:
        wrist_name = self.mapping["wrist"][1]
        mid_meta = self.mapping["middle"][0][1]
        idx_meta = self.mapping["index"][0][1]
        lit_meta = self.mapping["little"][0][1]

        w_pos = self.joints[wrist_name].getPos(self.actor_root)
        m_pos = self.joints[mid_meta].getPos(self.actor_root)
        i_pos = self.joints[idx_meta].getPos(self.actor_root)
        l_pos = self.joints[lit_meta].getPos(self.actor_root)

        rest_up = m_pos - w_pos
        if rest_up.lengthSquared() < 1e-8:
            rest_up = Vec3(0, 1, 0)
        else:
            rest_up.normalize()

        rest_side = l_pos - i_pos
        if rest_side.lengthSquared() < 1e-8:
            rest_side = Vec3(1, 0, 0)
        else:
            rest_side.normalize()

        rest_norm = rest_side.cross(rest_up)
        if rest_norm.lengthSquared() < 1e-8:
            rest_norm = Vec3(0, 0, 1)
        else:
            rest_norm.normalize()
        if self.is_left:
            rest_norm = -rest_norm

        self.rest_dummy = self.actor_root.attachNewNode("rest_dummy")
        self.curr_dummy = self.actor_root.attachNewNode("curr_dummy")
        self.rest_dummy.setPos(0, 0, 0)
        self.rest_dummy.lookAt(self.rest_dummy.getPos() + rest_up, rest_norm)
        self.rest_palm_q = Quat(self.rest_dummy.getQuat())

    def set_visible(self, visible: bool) -> None:
        if visible:
            self.actor_root.show()
        else:
            self.actor_root.hide()

    def drive_from_landmarks(self, landmarks: List[Dict], x_offset: float) -> bool:
        if len(landmarks) < 21:
            return False
        self.set_visible(True)

        pts_raw = [self._lm_to_local(lm, x_offset) for lm in landmarks]
        if not self.smoothed_pts:
            self.smoothed_pts = list(pts_raw)
        else:
            a = 1.0 - self.pos_smooth
            for i, p in enumerate(pts_raw):
                self.smoothed_pts[i] = self._lerp_vec3(self.smoothed_pts[i], p, a)
        pts = self.smoothed_pts

        target_root = pts[0] * self.root_motion_scale
        self.root_pos = self._lerp_vec3(self.root_pos, target_root, 1.0 - self.pos_smooth)
        self.actor_root.setPos(self.root_pos)
        self.actor_root.setScale(self.model_scale)

        curr_up = pts[9] - pts[0]
        if curr_up.lengthSquared() < 1e-8:
            curr_up = Vec3(0, 1, 0)
        else:
            curr_up.normalize()

        curr_side = pts[17] - pts[5]
        if curr_side.lengthSquared() < 1e-8:
            curr_side = Vec3(1, 0, 0)
        else:
            curr_side.normalize()

        curr_norm = curr_side.cross(curr_up)
        if curr_norm.lengthSquared() < 1e-8:
            curr_norm = Vec3(0, 0, 1)
        else:
            curr_norm.normalize()
        if self.is_left:
            curr_norm = -curr_norm

        ws = 1.0 - self.wrist_stability
        self.palm_up_smooth = self._lerp_vec3(self.palm_up_smooth, curr_up, ws)
        self.palm_normal_smooth = self._lerp_vec3(self.palm_normal_smooth, curr_norm, ws)
        if self.palm_up_smooth.lengthSquared() > 1e-8:
            self.palm_up_smooth.normalize()
        else:
            self.palm_up_smooth = Vec3(0, 1, 0)
        if self.palm_normal_smooth.lengthSquared() > 1e-8:
            self.palm_normal_smooth.normalize()
        else:
            self.palm_normal_smooth = Vec3(0, 0, 1)

        self.curr_dummy.setPos(0, 0, 0)
        self.curr_dummy.lookAt(self.curr_dummy.getPos() + self.palm_up_smooth, self.palm_normal_smooth)
        curr_palm_q = Quat(self.curr_dummy.getQuat())
        r_hand = curr_palm_q * self.rest_palm_q.conjugate()

        wrist_name = self.mapping["wrist"][1]
        self._drive_bone(wrist_name, r_hand, pts[0], pts[9])
        for chain_name in ("thumb", "index", "middle", "ring", "little"):
            for lm_idx, bone_name, child_idx in self.mapping[chain_name]:
                self._drive_bone(bone_name, r_hand, pts[lm_idx], pts[child_idx])
        return True

    def _drive_bone(self, bone_name: str, r_hand: Quat, p: Vec3, c: Vec3) -> None:
        joint = self.joints.get(bone_name)
        if joint is None:
            return
        target_dir = c - p
        if target_dir.lengthSquared() < 1e-8:
            return
        target_dir.normalize()

        base_q = r_hand * self.rest_quat[bone_name]
        curr_fwd = base_q.xform(self.local_fwd.get(bone_name, Vec3(0, 0, 1)))
        if curr_fwd.lengthSquared() < 1e-8:
            return
        curr_fwd.normalize()
        swing_q = self._shortest_arc(curr_fwd, target_dir)
        target_q = swing_q * base_q

        alpha = 1.0 - self.rot_smooth
        if "Wrist" in bone_name:
            alpha *= (1.0 - 0.7 * self.wrist_stability)
        alpha = max(0.02, min(1.0, alpha))

        target_hpr = target_q.getHpr()
        prev_hpr = self.prev_hpr.get(bone_name, target_hpr)
        smoothed_hpr = Vec3(
            self._lerp_angle(prev_hpr.x, target_hpr.x, alpha),
            self._lerp_angle(prev_hpr.y, target_hpr.y, alpha),
            self._lerp_angle(prev_hpr.z, target_hpr.z, alpha),
        )
        self.prev_hpr[bone_name] = smoothed_hpr
        smoothed_q = Quat()
        smoothed_q.setHpr(smoothed_hpr)
        joint.setQuat(self.actor_root, smoothed_q)

    def _lm_to_local(self, lm: Dict, x_offset: float) -> Vec3:
        x = (float(lm["x"]) - 0.5) * self.landmark_scale
        z = (-(float(lm["y"]) - 0.5)) * self.landmark_scale
        y = float(lm["z"]) * self.landmark_scale
        if self.is_left:
            x = x + x_offset
        else:
            x = -x - x_offset
        return Vec3(x, y, z)

    @staticmethod
    def _shortest_arc(v1: Vec3, v2: Vec3) -> Quat:
        a = Vec3(v1)
        b = Vec3(v2)
        if a.lengthSquared() < 1e-8 or b.lengthSquared() < 1e-8:
            return Quat.identQuat()
        a.normalize()
        b.normalize()
        cross = a.cross(b)
        dot = max(-1.0, min(1.0, a.dot(b)))
        if cross.lengthSquared() < 1e-8:
            if dot > 0.0:
                return Quat.identQuat()
            perp = Vec3(1, 0, 0) if abs(a.x) < 0.9 else Vec3(0, 1, 0)
            axis = a.cross(perp)
            if axis.lengthSquared() < 1e-8:
                axis = Vec3(0, 0, 1)
            else:
                axis.normalize()
            q180 = Quat()
            q180.setFromAxisAngle(180.0, axis)
            return q180
        cross.normalize()
        q = Quat()
        q.setFromAxisAngle(math.degrees(math.acos(dot)), cross)
        return q

    @staticmethod
    def _delta_angle(a: float, b: float) -> float:
        d = a - b
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        return d

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        return a + FBXHandRig._delta_angle(b, a) * t

    @staticmethod
    def _lerp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
        return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)


class FBXSkinningViewer(ShowBase):
    def __init__(
        self,
        stream_path: Path,
        left_fbx: Path,
        right_fbx: Path,
        fps: float,
        hand_separation: float,
        hide_after_miss: int,
        model_scale: float,
        landmark_scale: float,
        root_motion_scale: float,
        pos_smooth: float,
        rot_smooth: float,
        wrist_stability: float,
    ) -> None:
        super().__init__()
        self.disableMouse()
        self.frames = [json.loads(l) for l in stream_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.frames:
            raise RuntimeError(f"No frame data in {stream_path}")

        self.play_interval = 1.0 / max(1.0, fps)
        self.time_acc = 0.0
        self.current_frame = 0
        self.playing = True
        self.hand_separation = hand_separation
        self.hide_after_miss = max(1, hide_after_miss)
        self.freeze_on_miss = True
        self.left_miss = 0
        self.right_miss = 0
        self.cam_y_min = -6.0
        self.cam_y_max = -0.20

        self._init_window()
        self._init_scene()

        self.left_actor = Actor(_to_model_path(left_fbx))
        self.right_actor = Actor(_to_model_path(right_fbx))
        self.left_rig = FBXHandRig(
            self.left_actor,
            self.render.attachNewNode("left_actor_root"),
            is_left=True,
            pos_smooth=pos_smooth,
            rot_smooth=rot_smooth,
            wrist_stability=wrist_stability,
            model_scale=model_scale,
            landmark_scale=landmark_scale,
            root_motion_scale=root_motion_scale,
        )
        self.right_rig = FBXHandRig(
            self.right_actor,
            self.render.attachNewNode("right_actor_root"),
            is_left=False,
            pos_smooth=pos_smooth,
            rot_smooth=rot_smooth,
            wrist_stability=wrist_stability,
            model_scale=model_scale,
            landmark_scale=landmark_scale,
            root_motion_scale=root_motion_scale,
        )
        self.left_rig.set_visible(False)
        self.right_rig.set_visible(False)

        self._init_controls()
        self.taskMgr.add(self._update_task, "update_task")
        self._apply_frame(0)

    def _init_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Panda3D FBX Skinning Hand Viewer")
        props.setSize(1520, 960)
        self.win.requestProperties(props)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.03, 0.03, 0.04, 1.0)
        self.camLens.setNearFar(0.02, 200.0)

    def _init_scene(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.33, 0.33, 0.35, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        key = DirectionalLight("key")
        key.setColor(Vec4(0.96, 0.95, 0.93, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-30, -35, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.42, 0.47, 0.56, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(130, -18, 0)
        self.render.setLight(fill_np)

        mat = Material()
        mat.setShininess(70.0)
        mat.setSpecular((0.34, 0.34, 0.34, 1.0))
        self.render.setMaterial(mat, 1)

        self.cam_pivot = self.render.attachNewNode("cam_pivot")
        self.camera.reparentTo(self.cam_pivot)
        self.camera.setPos(0, -3.5, 1.1)
        self.camera.lookAt(0, 0, 0)

    def _init_controls(self) -> None:
        self.accept("space", self._toggle_play)
        self.accept("arrow_right", self._step_next)
        self.accept("arrow_left", self._step_prev)
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("=", self._zoom_in)
        self.accept("+", self._zoom_in)
        self.accept("-", self._zoom_out)
        self.accept("_", self._zoom_out)
        self.accept("r", self._reset_camera)
        self.accept("escape", self.userExit)

        self.dragging = False
        self.last_mouse = Point3(0, 0, 0)
        self.accept("mouse1", self._drag_start)
        self.accept("mouse1-up", self._drag_end)
        self.taskMgr.add(self._drag_task, "drag_task")

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    def _step_next(self) -> None:
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _step_prev(self) -> None:
        self.current_frame = (self.current_frame - 1 + len(self.frames)) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _zoom_in(self) -> None:
        y = self.camera.getY() + 0.28
        self.camera.setY(min(self.cam_y_max, y))

    def _zoom_out(self) -> None:
        y = self.camera.getY() - 0.28
        self.camera.setY(max(self.cam_y_min, y))

    def _reset_camera(self) -> None:
        self.cam_pivot.setHpr(0, 0, 0)
        self.camera.setPos(0, -3.5, 1.1)
        self.camera.lookAt(0, 0, 0)

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
        self.cam_pivot.setH(self.cam_pivot.getH() - dx * 135.0)
        self.cam_pivot.setP(max(-88.0, min(88.0, self.cam_pivot.getP() + dy * 90.0)))
        return Task.cont

    def _update_task(self, task: Task):
        if not self.playing:
            return Task.cont
        self.time_acc += globalClock.getDt()
        while self.time_acc >= self.play_interval:
            self.time_acc -= self.play_interval
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self._apply_frame(self.current_frame)
        return Task.cont

    def _apply_frame(self, frame_idx: int) -> None:
        frame = self.frames[frame_idx]
        hands = frame.get("hands", [])
        left = None
        right = None
        for h in hands:
            ht = h.get("hand_type")
            if ht == "Left":
                left = h
            elif ht == "Right":
                right = h

        if left is None:
            self.left_miss += 1
            if not self.freeze_on_miss and self.left_miss >= self.hide_after_miss:
                self.left_rig.set_visible(False)
        else:
            ok = self.left_rig.drive_from_landmarks(left.get("landmarks", []), x_offset=-self.hand_separation)
            if ok:
                self.left_miss = 0
            else:
                self.left_miss += 1
                if not self.freeze_on_miss and self.left_miss >= self.hide_after_miss:
                    self.left_rig.set_visible(False)

        if right is None:
            self.right_miss += 1
            if not self.freeze_on_miss and self.right_miss >= self.hide_after_miss:
                self.right_rig.set_visible(False)
        else:
            ok = self.right_rig.drive_from_landmarks(right.get("landmarks", []), x_offset=self.hand_separation)
            if ok:
                self.right_miss = 0
            else:
                self.right_miss += 1
                if not self.freeze_on_miss and self.right_miss >= self.hide_after_miss:
                    self.right_rig.set_visible(False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FBX skinning-driven Panda3D hand viewer")
    p.add_argument("--stream", required=True, help="unity_gesture_stream.jsonl path")
    p.add_argument("--left-fbx", default="../new_sign_language/LeftHand.fbx")
    p.add_argument("--right-fbx", default="../new_sign_language/RightHand.fbx")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--hand-separation", type=float, default=0.22)
    p.add_argument("--hide-after-miss", type=int, default=8)
    p.add_argument("--model-scale", type=float, default=0.018)
    p.add_argument("--landmark-scale", type=float, default=13.5)
    p.add_argument("--root-motion-scale", type=float, default=0.18)
    p.add_argument("--pos-smooth", type=float, default=0.72)
    p.add_argument("--rot-smooth", type=float, default=0.82)
    p.add_argument("--wrist-stability", type=float, default=0.65)
    return p


def main() -> None:
    args = build_parser().parse_args()
    app = FBXSkinningViewer(
        stream_path=Path(args.stream).resolve(),
        left_fbx=Path(args.left_fbx).resolve(),
        right_fbx=Path(args.right_fbx).resolve(),
        fps=args.fps,
        hand_separation=args.hand_separation,
        hide_after_miss=args.hide_after_miss,
        model_scale=args.model_scale,
        landmark_scale=args.landmark_scale,
        root_motion_scale=args.root_motion_scale,
        pos_smooth=args.pos_smooth,
        rot_smooth=args.rot_smooth,
        wrist_stability=args.wrist_stability,
    )
    app.run()


if __name__ == "__main__":
    main()
