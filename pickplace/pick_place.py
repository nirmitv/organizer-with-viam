"""Manipulation block: pick up an object, drop it off.

Contract with the rest of the team:
- All poses are in the "world" frame, millimeters, Viam orientation vectors.
- pick(motion, gripper, obj_pose): grasp the object at that pose (top-down,
  yawed to obj_pose.theta — the block's long-axis angle in degrees, so the
  jaws close across the short side), then lift to carry height.
- place(motion, gripper, drop_pose): move overhead and release — the object
  drops in; the arm never enters the drawer.
- Obstacles are someone else's problem: callers pass a WorldState and motion
  plans around it. Without one, moves still plan but avoid nothing.
"""

import asyncio

from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame, WorldState
from viam.services.motion import MotionClient

ARM_NAME = "arm-1"
WORLD_FRAME = "world"
APPROACH_OFFSET_MM = 80  # hover height above the grasp pose before descending
LIFT_OFFSET_MM = 150  # lift above the grasp pose after grabbing; size so transit clears the drawer rim
SETTLE_S = 1.0  # pause between steps so each motion/gripper action fully settles


def _above(pose: Pose, offset_mm: float) -> Pose:
    return Pose(
        x=pose.x, y=pose.y, z=pose.z + offset_mm,
        o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta,
    )


async def _move(motion: MotionClient, pose: Pose, world_state: WorldState | None) -> None:
    moved = await motion.move(
        component_name=ARM_NAME,
        destination=PoseInFrame(reference_frame=WORLD_FRAME, pose=pose),
        world_state=world_state,
    )
    if not moved:
        raise RuntimeError(f"motion.move could not reach x={pose.x} y={pose.y} z={pose.z}")


async def pick(
    motion: MotionClient,
    gripper: Gripper,
    obj_pose: Pose,
    world_state: WorldState | None = None,
) -> bool:
    """Approach from above, descend, grab, lift to carry height. Returns True if the gripper caught something."""
    # Top-down grasp, yawed by obj_pose.theta so the jaws close across a
    # rectangular block's short side (theta = block's long-axis angle, degrees).
    grasp = Pose(x=obj_pose.x, y=obj_pose.y, z=obj_pose.z, o_x=0, o_y=0, o_z=-1, theta=obj_pose.theta)

    await gripper.open()
    await asyncio.sleep(SETTLE_S)
    await _move(motion, _above(grasp, APPROACH_OFFSET_MM), world_state)
    await asyncio.sleep(SETTLE_S)
    await _move(motion, grasp, world_state)
    await asyncio.sleep(SETTLE_S)
    grabbed = await gripper.grab()
    await asyncio.sleep(SETTLE_S)
    if grabbed:
        await _move(motion, _above(grasp, LIFT_OFFSET_MM), world_state)
    else:
        await gripper.open()
        await asyncio.sleep(SETTLE_S)
        await _move(motion, _above(grasp, APPROACH_OFFSET_MM), world_state)
    return grabbed


async def place(
    motion: MotionClient,
    gripper: Gripper,
    drop_pose: Pose,
    world_state: WorldState | None = None,
) -> None:
    """Move to the overhead drop pose and release — the object falls into the drawer."""
    await _move(motion, drop_pose, world_state)
    await asyncio.sleep(SETTLE_S)
    await gripper.open()
