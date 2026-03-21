#!/usr/bin/env python3
"""
True FBX skinning-driven Panda3D hand viewer.

This viewer loads:
- ../new_sign_language/LeftHand.fbx
- ../new_sign_language/RightHand.fbx

Then drives skeleton joints per frame from Unity-compatible GestureData JSONL stream.
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
    # Convert OS path into Panda-native absolute path (handles Windows drives).
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
    "thumb": [(1, _finger("L", "Thumb")[0], 2), (2, _finger("L", "Thumb")[1], 3), (3, _finger("L", "Thumb")[2], 4)],
    "index": [(0, _finger("L", "Index")[0], 5), (5, _finger("L", "Index")[1], 6), (6, _finger("L", "Index")[2], 7), (7, _finger("L", "Index")[3], 8)],
    "middle": [(0, _finger("L", "Middle")[0], 9), (9, _finger("L", "Middle")[1], 10), (10, _finger("L", "Middle")[2], 11), (11, _finger("L", "Middle")[3], 12)],
    "ring": [(0, _finger("L", "Ring")[0], 13), (13, _finger("L", "Ring")[1], 14), (14, _finger("L", "Ring")[2], 15), (15, _finger("L", "Ring")[3], 16)],
    "little": [(0, _finger("L", "Little")[0], 17), (17, _finger("L", "Little")[1], 18), (18, _finger("L", "Little")[2], 19), (19, _finger("L", "Little")[3], 20)],
}

LANDMARK_TO_BONE_SEQ_RIGHT = {
    "wrist": (0, "R_Wrist", 9),
    "thumb": [(1, _finger("R", "Thumb")[0], 2), (2, _finger("R", "Thumb")[1], 3), (3, _finger("R", "Thumb")[2], 4)],
    "index": [(0, _finger("R", "Index")[0], 5), (5, _finger("R", "Index")[1], 6), (6, _finger("R", "Index")[2], 7), (7, _finger("R", "Index")[3], 8)],
    "middle": [(0, _finger("R", "Middle")[0], 9), (9, _finger("R", "Middle")[1], 10), (10, _finger("R", "Middle")[2], 11), (11, _finger("R", "Middle")[3], 12)],
    "ring": [(0, _finger("R", "Ring")[0], 13), (13, _finger("R", "Ring")[1], 14), (14, _finger("R", "Ring")[2], 15), (15, _finger("R", "Ring")[3], 16)],
    "little": [(0, _finger("R", "Little")[0], 17), (17, _finger("R", "Little")[1], 18), (18, _finger("R", "Little")[2], 19), (19, _finger("R", "Little")[3], 20)],
}


_PX = Vec3(1, 0, 0)
_NX = Vec3(-1, 0, 0)
_PY = Vec3(0, 1, 0)
_PZ = Vec3(0, 0, 1)
_NZ = Vec3(0, 0, -1)

LEFT_FORWARD_AXIS: Dict[str, Vec3] = {
    "L_Wrist": _PZ,
    "L_ThumbMetacarpal": _NZ,
    "L_ThumbProximal": _PZ,
    "L_ThumbDistal": _PZ,
    "L_IndexMetacarpal": _PZ,
    "L_IndexProximal": _NZ,
    "L_IndexIntermediate": _NZ,
    "L_IndexDistal": _NZ,
    "L_MiddleMetacarpal": _PZ,
    "L_MiddleProximal": _NZ,
    "L_MiddleIntermediate": _NZ,
    "L_MiddleDistal": _NZ,
    "L_RingMetacarpal": _PZ,
    "L_RingProximal": _NZ,
    "L_RingIntermediate": _NZ,
    "L_RingDistal": _NZ,
    "L_LittleMetacarpal": _NX,
    "L_LittleProximal": _NZ,
    "L_LittleIntermediate": _NZ,
    "L_LittleDistal": _NZ,
}

RIGHT_FORWARD_AXIS: Dict[str, Vec3] = {
    "R_Wrist": _PZ,
    "R_ThumbMetacarpal": _NZ,
    "R_ThumbProximal": _PY,
    "R_ThumbDistal": _PY,
    "R_IndexMetacarpal": _PZ,
    "R_IndexProximal": _NZ,
    "R_IndexIntermediate": _NZ,
    "R_IndexDistal": _NZ,
    "R_MiddleMetacarpal": _PZ,
    "R_MiddleProximal": _NZ,
    "R_MiddleIntermediate": _NZ,
    "R_MiddleDistal": _NZ,
    "R_RingMetacarpal": _PZ,
    "R_RingProximal": _NZ,
    "R_RingIntermediate": _NZ,
    "R_RingDistal": _NZ,
    "R_LittleMetacarpal": _PX,
    "R_LittleProximal": _NZ,
    "R_LittleIntermediate": _NZ,
    "R_LittleDistal": _NZ,
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
        bone_offsets: Dict[str, Tuple[float, float, float]] | None = None,
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
        self.joints: Dict[str, NodePath] = {}
        self.rest_hpr: Dict[str, Vec3] = {}
        self.prev_hpr: Dict[str, Vec3] = {}
        self.rest_quat: Dict[str, Quat] = {}
        self.prev_quat: Dict[str, Quat] = {}
        self.smoothed_pts: List[Vec3] = []
        self.root_pos = Vec3(0, 0, 0)
        self._aim = self.actor_root.attachNewNode("aim_tmp")
        self.bone_offsets = bone_offsets or {}
        self.palm_up_smooth = Vec3(0, 1, 0)
        self.palm_normal_smooth = Vec3(0, 0, 1)
        self.forward_axes = LEFT_FORWARD_AXIS if is_left else RIGHT_FORWARD_AXIS

        all_bones = set()
        mapping = LANDMARK_TO_BONE_SEQ_LEFT if is_left else LANDMARK_TO_BONE_SEQ_RIGHT
        all_bones.add(mapping["wrist"][1])
        for k in ("thumb", "index", "middle", "ring", "little"):
            for _, name, _ in mapping[k]:
                all_bones.add(name)

        for bone in sorted(all_bones):
            node = actor.controlJoint(None, "modelRoot", bone)
            if node is not None and not node.isEmpty():
                self.joints[bone] = node

        self.actor.reparentTo(self.actor_root)
        self.actor.setScale(1.0)
        self.actor_root.setScale(self.model_scale)

        for bone, node in self.joints.items():
            hpr = node.getHpr(self.actor_root)
            self.rest_hpr[bone] = Vec3(hpr.x, hpr.y, hpr.z)
            self.prev_hpr[bone] = Vec3(hpr.x, hpr.y, hpr.z)
            q = Quat()
            q.setHpr(self.rest_hpr[bone])
            self.rest_quat[bone] = Quat(q)
            self.prev_quat[bone] = Quat(q)

        self.limits = self._build_limits()
        prefix = "L" if self.is_left else "R"
        self.finger_chains = {
            "thumb": [f"{prefix}_ThumbMetacarpal", f"{prefix}_ThumbProximal", f"{prefix}_ThumbDistal"],
            "index": [f"{prefix}_IndexProximal", f"{prefix}_IndexIntermediate", f"{prefix}_IndexDistal"],
            "middle": [f"{prefix}_MiddleProximal", f"{prefix}_MiddleIntermediate", f"{prefix}_MiddleDistal"],
            "ring": [f"{prefix}_RingProximal", f"{prefix}_RingIntermediate", f"{prefix}_RingDistal"],
            "little": [f"{prefix}_LittleProximal", f"{prefix}_LittleIntermediate", f"{prefix}_LittleDistal"],
        }

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
            for i, p in enumerate(pts_raw):
                self.smoothed_pts[i] = self._lerp_vec3(self.smoothed_pts[i], p, 1.0 - self.pos_smooth)
        pts = self.smoothed_pts

        wrist = pts[0]
        target_root = wrist * self.root_motion_scale
        self.root_pos = self._lerp_vec3(self.root_pos, target_root, 1.0 - self.pos_smooth)
        self.actor_root.setPos(self.root_pos)
        self.actor_root.setScale(self.model_scale)

        palm_up = (pts[9] - pts[0]).normalized()
        palm_side = (pts[17] - pts[5]).normalized()
        palm_normal = palm_side.cross(palm_up)
        if palm_normal.lengthSquared() < 1e-8:
            palm_normal = Vec3(0, 0, 1)
        else:
            palm_normal.normalize()
        if self.is_left:
            palm_normal = -palm_normal

        # Local palm stabilizer: smooth palm frame vectors before wrist orientation.
        self.palm_up_smooth = self._lerp_vec3(self.palm_up_smooth, palm_up, 1.0 - self.wrist_stability)
        self.palm_normal_smooth = self._lerp_vec3(
            self.palm_normal_smooth, palm_normal, 1.0 - self.wrist_stability
        )
        if self.palm_up_smooth.lengthSquared() > 1e-8:
            self.palm_up_smooth.normalize()
        else:
            self.palm_up_smooth = Vec3(0, 1, 0)
        if self.palm_normal_smooth.lengthSquared() > 1e-8:
            self.palm_normal_smooth.normalize()
        else:
            self.palm_normal_smooth = Vec3(0, 0, 1)

        # Wrist orientation from palm plane (use middle root as forward anchor).
        wrist_name = (LANDMARK_TO_BONE_SEQ_LEFT if self.is_left else LANDMARK_TO_BONE_SEQ_RIGHT)["wrist"][1]
        if wrist_name in self.joints:
            self._drive_joint(wrist_name, pts[0], pts[9], self.palm_normal_smooth)

        mapping = LANDMARK_TO_BONE_SEQ_LEFT if self.is_left else LANDMARK_TO_BONE_SEQ_RIGHT
        for chain_name in ("thumb", "index", "middle", "ring", "little"):
            for lm_idx, bone_name, child_idx in mapping[chain_name]:
                joint = self.joints.get(bone_name)
                if joint is None:
                    continue
                p = pts[lm_idx]
                c = pts[child_idx]
                v = c - p
                if v.lengthSquared() < 1e-8:
                    continue
                self._drive_joint(bone_name, p, c, self.palm_normal_smooth)

        # Mild anatomical coupling improves finger progression smoothness.
        self._enforce_finger_coupling()
        return True

    def _lm_to_local(self, lm: Dict, x_offset: float) -> Vec3:
        # Correct coordinate mapping:
        # MediaPipe/Unity: X=left-right, Y=up-down, Z=depth
        # Panda3D(Z-up):   X=left-right, Y=depth,   Z=up-down
        x = (float(lm["x"]) - 0.5) * self.landmark_scale
        z = (-(float(lm["y"]) - 0.5)) * self.landmark_scale
        y = float(lm["z"]) * self.landmark_scale

        if self.is_left:
            x = x + x_offset
        else:
            x = -x - x_offset
        return Vec3(x, y, z)

    def _build_limits(self) -> Dict[str, Tuple[float, float, float]]:
        limits: Dict[str, Tuple[float, float, float]] = {}
        for name in self.joints.keys():
            if "Wrist" in name:
                limits[name] = (32.0, 30.0, 30.0)
            elif "Metacarpal" in name:
                limits[name] = (22.0, 28.0, 20.0)
            elif "Proximal" in name:
                limits[name] = (18.0, 45.0, 16.0)
            elif "Intermediate" in name:
                limits[name] = (12.0, 55.0, 10.0)
            elif "Distal" in name:
                limits[name] = (10.0, 45.0, 8.0)
            else:
                limits[name] = (24.0, 40.0, 24.0)
        return limits

    def _drive_joint(self, bone_name: str, joint_pos: Vec3, target_pos: Vec3, up_vec: Vec3) -> None:
        joint = self.joints.get(bone_name)
        if joint is None:
            return
        direction = target_pos - joint_pos
        if direction.lengthSquared() < 1e-10:
            return
        direction.normalize()

        rest_q = self.rest_quat.get(bone_name)
        if rest_q is None:
            return
        local_fwd = self.forward_axes.get(bone_name, _PZ)
        rest_fwd = rest_q.xform(local_fwd)
        rot_q = self._quat_align(rest_fwd, direction, up_vec)
        target_q = rot_q * rest_q

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
        self.prev_quat[bone_name] = Quat(smoothed_q)
        joint.setQuat(self.actor_root, smoothed_q)

    @staticmethod
    def _quat_align(from_vec: Vec3, to_vec: Vec3, up_hint: Vec3) -> Quat:
        f = Vec3(from_vec)
        t = Vec3(to_vec)
        if f.lengthSquared() < 1e-10 or t.lengthSquared() < 1e-10:
            return Quat.identQuat()
        f.normalize()
        t.normalize()
        dot = max(-1.0, min(1.0, f.dot(t)))
        axis = f.cross(t)
        if axis.lengthSquared() < 1e-10:
            if dot > 0.0:
                return Quat.identQuat()
            perp = Vec3(up_hint)
            if perp.lengthSquared() < 1e-10:
                perp = Vec3(0, 0, 1)
            perp.normalize()
            if abs(perp.dot(f)) > 0.99:
                perp = Vec3(1, 0, 0) if abs(f.x) < 0.9 else Vec3(0, 1, 0)
            axis = f.cross(perp)
            if axis.lengthSquared() < 1e-10:
                axis = Vec3(0, 0, 1)
            axis.normalize()
            q180 = Quat()
            q180.setFromAxisAngle(180.0, axis)
            return q180
        axis.normalize()
        angle_deg = math.degrees(math.acos(dot))
        q = Quat()
        q.setFromAxisAngle(angle_deg, axis)
        return q

    def _enforce_finger_coupling(self) -> None:
        # We use pitch (Y) as primary flexion axis in this rig.
        for finger, chain in self.finger_chains.items():
            valid = [b for b in chain if b in self.prev_hpr and b in self.rest_hpr and b in self.joints]
            if len(valid) < 3:
                continue

            # For thumb use softer coupling than other fingers.
            if finger == "thumb":
                min2, max2 = 0.25, 0.70
                min3, max3 = 0.20, 0.60
            else:
                min2, max2 = 0.35, 0.80
                min3, max3 = 0.25, 0.70

            b1, b2, b3 = valid[0], valid[1], valid[2]
            r1 = self.rest_hpr[b1].y
            r2 = self.rest_hpr[b2].y
            r3 = self.rest_hpr[b3].y

            d1 = self._delta_angle(self.prev_hpr[b1].y, r1)
            d2 = self._delta_angle(self.prev_hpr[b2].y, r2)
            d3 = self._delta_angle(self.prev_hpr[b3].y, r3)

            sign = 1.0 if d1 >= 0 else -1.0
            a1 = abs(d1)
            a2_raw = abs(d2)
            a3_raw = abs(d3)
            a2 = max(a1 * min2, min(a1 * max2, a2_raw))
            a3 = max(a2 * min3, min(a2 * max3, a3_raw))

            target2 = r2 + sign * a2
            target3 = r3 + sign * a3

            lim2 = self.limits.get(b2, (25.0, 25.0, 25.0))[1]
            lim3 = self.limits.get(b3, (25.0, 25.0, 25.0))[1]

            target2 = self._clamp_around_rest(target2, r2, lim2)
            target3 = self._clamp_around_rest(target3, r3, lim3)

            y2 = self._lerp_angle(self.prev_hpr[b2].y, target2, 0.25)
            y3 = self._lerp_angle(self.prev_hpr[b3].y, target3, 0.25)

            hpr2 = self.prev_hpr[b2]
            hpr3 = self.prev_hpr[b3]
            hpr2 = Vec3(hpr2.x, y2, hpr2.z)
            hpr3 = Vec3(hpr3.x, y3, hpr3.z)
            self.prev_hpr[b2] = hpr2
            self.prev_hpr[b3] = hpr3
            q2 = Quat()
            q3 = Quat()
            q2.setHpr(hpr2)
            q3.setHpr(hpr3)
            self.prev_quat[b2] = q2
            self.prev_quat[b3] = q3
            self.joints[b2].setQuat(self.actor_root, q2)
            self.joints[b3].setQuat(self.actor_root, q3)

    @staticmethod
    def _clamp_around_rest(val: float, rest: float, lim: float) -> float:
        delta = FBXHandRig._delta_angle(val, rest)
        delta = max(-lim, min(lim, delta))
        return rest + delta

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


class OffsetCalibrator(ShowBase):
    def __init__(
        self,
        left_fbx: Path,
        right_fbx: Path,
    ) -> None:
        super().__init__()
        self.disableMouse()
        self.left_actor = Actor(_to_model_path(left_fbx))
        self.right_actor = Actor(_to_model_path(right_fbx))
        self.left_root = self.render.attachNewNode("left_calib")
        self.right_root = self.render.attachNewNode("right_calib")
        self.left_actor.reparentTo(self.left_root)
        self.right_actor.reparentTo(self.right_root)
        self.aim = self.render.attachNewNode("calib_aim")

    def _calibrate_side(self, actor: Actor, root: NodePath, is_left: bool) -> Dict[str, Tuple[float, float, float]]:
        mapping = LANDMARK_TO_BONE_SEQ_LEFT if is_left else LANDMARK_TO_BONE_SEQ_RIGHT

        def get_joint(bone_name: str) -> NodePath | None:
            j = actor.controlJoint(None, "modelRoot", bone_name)
            if j is None or j.isEmpty():
                return None
            return j

        def get_pos(bone_name: str) -> Vec3 | None:
            j = get_joint(bone_name)
            if j is None:
                return None
            return j.getPos(root)

        p_w = get_pos(mapping["wrist"][1])
        p_mid = get_pos(mapping["middle"][0][1])
        p_idx = get_pos(mapping["index"][0][1])
        p_lit = get_pos(mapping["little"][0][1])
        if not all((p_w, p_mid, p_idx, p_lit)):
            return {}

        palm_up = (p_mid - p_w).normalized()
        palm_side = (p_lit - p_idx).normalized()
        palm_normal = palm_side.cross(palm_up)
        if palm_normal.lengthSquared() < 1e-8:
            palm_normal = Vec3(0, 0, 1)
        else:
            palm_normal.normalize()
        if is_left:
            palm_normal = -palm_normal

        child_map: Dict[str, str] = {mapping["wrist"][1]: mapping["middle"][0][1]}
        for chain_name in ("thumb", "index", "middle", "ring", "little"):
            chain = mapping[chain_name]
            for i in range(len(chain) - 1):
                child_map[chain[i][1]] = chain[i + 1][1]

        offsets: Dict[str, Tuple[float, float, float]] = {}
        for bone, child_bone in child_map.items():
            p = get_pos(bone)
            c = get_pos(child_bone)
            j = get_joint(bone)
            if p is None or c is None or j is None:
                continue
            if (c - p).lengthSquared() < 1e-8:
                continue

            self.aim.setPos(root, p)
            self.aim.lookAt(root, c, palm_normal)

            dummy = root.attachNewNode("dummy")
            dummy.setHpr(root, j.getHpr(root))
            lhpr = dummy.getHpr(self.aim)
            offsets[bone] = (round(lhpr.x, 3), round(lhpr.y, 3), round(lhpr.z, 3))
            dummy.removeNode()

        return offsets

    def calibrate(self) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        left_offsets = self._calibrate_side(self.left_actor, self.left_root, True)
        right_offsets = self._calibrate_side(self.right_actor, self.right_root, False)
        return {"left": left_offsets, "right": right_offsets}


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
        left_bone_offsets: Dict[str, Tuple[float, float, float]],
        right_bone_offsets: Dict[str, Tuple[float, float, float]],
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
            bone_offsets=left_bone_offsets,
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
            bone_offsets=right_bone_offsets,
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
        self.accept("=", self._zoom_in)   # keyboard zoom in
        self.accept("+", self._zoom_in)
        self.accept("-", self._zoom_out)  # keyboard zoom out
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
    p.add_argument("--stream", required=False, help="unity_gesture_stream.jsonl path")
    p.add_argument("--left-fbx", default="../new_sign_language/LeftHand.fbx")
    p.add_argument("--right-fbx", default="../new_sign_language/RightHand.fbx")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--hand-separation", type=float, default=0.22)
    p.add_argument("--hide-after-miss", type=int, default=8, help="Hide hand after N consecutive missing frames")
    p.add_argument("--model-scale", type=float, default=0.018, help="Global FBX model scale")
    p.add_argument("--landmark-scale", type=float, default=13.5, help="Landmark world scale for rotation solve")
    p.add_argument("--root-motion-scale", type=float, default=0.18, help="How much wrist translation moves whole hand")
    p.add_argument("--pos-smooth", type=float, default=0.72, help="0-0.98, higher means smoother positions")
    p.add_argument("--rot-smooth", type=float, default=0.82, help="0-0.98, higher means smoother joint rotations")
    p.add_argument("--wrist-stability", type=float, default=0.65, help="0-0.98, higher means steadier wrist/palm orientation")
    p.add_argument("--offset-config", default=None, help="Optional JSON path for per-bone HPR offsets")
    p.add_argument("--calibrate-output", default=None, help="If set, runs FBX rest-pose calibration and exits")
    return p


def _default_bone_offsets() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    # Safe default: no correction until a calibrated config is provided.
    return {"left": {}, "right": {}}


def _load_bone_offsets(path: Path | None) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    cfg = _default_bone_offsets()
    if path is None:
        print("[offset] no --offset-config provided; using ZERO offsets. Run --calibrate-output first.")
        return cfg
    if not path.exists():
        print(f"[offset] config not found: {path}; using ZERO offsets.")
        return cfg
    data = json.loads(path.read_text(encoding="utf-8"))
    for side in ("left", "right"):
        src = data.get(side, {})
        for bone, hpr in src.items():
            if isinstance(hpr, (list, tuple)) and len(hpr) == 3:
                cfg[side][bone] = (float(hpr[0]), float(hpr[1]), float(hpr[2]))
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    left_fbx = Path(args.left_fbx).resolve()
    right_fbx = Path(args.right_fbx).resolve()

    if args.calibrate_output:
        from panda3d.core import loadPrcFileData

        loadPrcFileData("", "window-type none")
        calib = OffsetCalibrator(
            left_fbx=left_fbx,
            right_fbx=right_fbx,
        )
        offsets = calib.calibrate()
        out = Path(args.calibrate_output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(offsets, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[calibrate] wrote offsets: {out}")
        return

    if not args.stream:
        raise SystemExit("viewer mode requires --stream when --calibrate-output is not set")
    stream_path = Path(args.stream).resolve()

    offsets = _load_bone_offsets(Path(args.offset_config).resolve() if args.offset_config else None)
    app = FBXSkinningViewer(
        stream_path=stream_path,
        left_fbx=left_fbx,
        right_fbx=right_fbx,
        fps=args.fps,
        hand_separation=args.hand_separation,
        hide_after_miss=args.hide_after_miss,
        model_scale=args.model_scale,
        landmark_scale=args.landmark_scale,
        root_motion_scale=args.root_motion_scale,
        pos_smooth=args.pos_smooth,
        rot_smooth=args.rot_smooth,
        wrist_stability=args.wrist_stability,
        left_bone_offsets=offsets["left"],
        right_bone_offsets=offsets["right"],
    )
    app.run()


if __name__ == "__main__":
    main()
