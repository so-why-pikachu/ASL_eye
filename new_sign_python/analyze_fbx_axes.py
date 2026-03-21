#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename, Vec3


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


class App(ShowBase):
    def __init__(self) -> None:
        super().__init__(windowType="none")
        self.disableMouse()


def _axis_score(joint, root, child_pos: Vec3) -> List[Tuple[str, float]]:
    target = child_pos - joint.getPos(root)
    if target.lengthSquared() < 1e-10:
        return []
    target.normalize()
    # Panda basis in joint local to root space.
    x = joint.getQuat(root).xform(Vec3(1, 0, 0))
    y = joint.getQuat(root).xform(Vec3(0, 1, 0))
    z = joint.getQuat(root).xform(Vec3(0, 0, 1))
    axes = [
        ("+X", x.dot(target)),
        ("-X", (-x).dot(target)),
        ("+Y", y.dot(target)),
        ("-Y", (-y).dot(target)),
        ("+Z", z.dot(target)),
        ("-Z", (-z).dot(target)),
    ]
    axes.sort(key=lambda t: t[1], reverse=True)
    return axes


def analyze_side(actor: Actor, root, is_left: bool) -> None:
    mapping = LANDMARK_TO_BONE_SEQ_LEFT if is_left else LANDMARK_TO_BONE_SEQ_RIGHT
    child_map: Dict[str, str] = {mapping["wrist"][1]: mapping["middle"][0][1]}
    for chain_name in ("thumb", "index", "middle", "ring", "little"):
        chain = mapping[chain_name]
        for i in range(len(chain) - 1):
            child_map[chain[i][1]] = chain[i + 1][1]

    print("LEFT" if is_left else "RIGHT")
    for bone, child_bone in child_map.items():
        j = actor.controlJoint(None, "modelRoot", bone)
        c = actor.controlJoint(None, "modelRoot", child_bone)
        if j is None or c is None or j.isEmpty() or c.isEmpty():
            continue
        axes = _axis_score(j, root, c.getPos(root))
        if not axes:
            continue
        best = axes[0]
        print(f"{bone:24s} -> {child_bone:24s} | best={best[0]:>2s} dot={best[1]:.3f} | next={axes[1][0]}:{axes[1][1]:.3f}")


def main() -> None:
    app = App()
    left_fbx = Path(r"f:\sign_language\new_sign_language\LeftHand.fbx")
    right_fbx = Path(r"f:\sign_language\new_sign_language\RightHand.fbx")

    left_actor = Actor(_to_model_path(left_fbx))
    right_actor = Actor(_to_model_path(right_fbx))
    left_root = app.render.attachNewNode("left_root")
    right_root = app.render.attachNewNode("right_root")
    left_actor.reparentTo(left_root)
    right_actor.reparentTo(right_root)

    analyze_side(left_actor, left_root, True)
    print()
    analyze_side(right_actor, right_root, False)

    app.userExit()


if __name__ == "__main__":
    main()
