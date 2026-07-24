#!/usr/bin/env python3
"""Detect red and green blocks on robot8 (South Park Commons / viam-events), grab, and sort them.

3D localization comes from the "segment-red"/"segment-green" vision services
(viam:vision:detections-to-segments), which project detect-red/detect-green's 2D boxes into 3D
using cam-1's depth data.

Requires ROBOT8_API_KEY_ID and ROBOT8_API_KEY in the environment (see .env.example).

Usage:
    python3 detect_and_grab.py              # dry run: detect + print the planned moves, no motion
    python3 detect_and_grab.py --execute    # actually grab one red block and one green block, sorting
                                             # each to its own drop-off position
"""
import argparse
import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

ROBOT_ADDRESS = "robot8-main.ag9khwy6jn.viam.cloud"
CAMERA_NAME = "cam-1"
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
MOTION_SERVICE = "builtin"

APPROACH_HEIGHT_MM = 120.0  # hover height above the table before/after grasping
TOP_DOWN_ORIENTATION = dict(o_x=0, o_y=0, o_z=-1, theta=-90)  # matches cam-1/arm-1's current top-down pose

# gripper-1's frame origin (what motion.move actually drives) is the mount point, 105mm from arm-1 -
# the fingertips are a further ~2in beyond that along the same (downward-pointing) axis. Since our
# orientation is pure top-down with no tilt, that offset only affects world Z: every Z we send to
# gripper-1 must be raised by this much so the fingertips (not the mount point) land on target.
GRIPPER_TIP_OFFSET_MM = 50.8

# forces gripper-1 to travel in a straight line between poses instead of letting the planner pick an
# arced joint-space path (which was swinging the arm up before the descend-and-grab move).
LINE_TOLERANCE_MM = 5.0
STRAIGHT_LINE_CONSTRAINTS = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=LINE_TOLERANCE_MM)])

# pause after the descend so the arm is fully at rest at the grasp point before the gripper closes
GRASP_SETTLE_SECONDS = 1.0

# ufactory gripper speed range is 1-5000 (firmware default otherwise); raise it so closing is quicker
GRIPPER_CLOSE_SPEED = 4000
# pause after the gripper closes so it's fully gripping before the arm starts lifting
POST_GRAB_SETTLE_SECONDS = 0.5

# known-good pose for arm-1 that gives cam-1 a clear top-down view of the table before detecting
START_POSE = Pose(
    x=-22.225737438601186,
    y=307.04960550268675,
    z=558.0385211819946,
    o_x=0.009485579527273146,
    o_y=0,
    o_z=-0.9999550108785054,
    theta=-92.35331845194332,
)

# per-color vision service to localize it, and drop-off pose for gripper-1 (its own frame, not
# fingertip-adjusted) where a grabbed block of that color is released
COLORS = [
    {
        "name": "red",
        "segment_service": "segment-red",
        "release_pose": Pose(x=62.7, y=330.9, z=200.0, **TOP_DOWN_ORIENTATION),
    },
    {
        "name": "green",
        "segment_service": "segment-green",
        "release_pose": Pose(x=-16.0, y=330.9, z=200.0, **TOP_DOWN_ORIENTATION),
    },
]

# table obstacle bounds in world frame (from robot config), used as a sanity check on computed targets
TABLE_X_RANGE = (-250.0, 1750.0)
TABLE_Y_RANGE = (0.0, 2000.0)

# GetObjectPointClouds occasionally drops the connection over this network path; retry rather than fail outright.
SEGMENT_RETRIES = 4


async def connect() -> RobotClient:
    key_id = os.environ["ROBOT8_API_KEY_ID"]
    key = os.environ["ROBOT8_API_KEY"]
    opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


async def goto_start(robot: RobotClient, execute: bool) -> None:
    motion = MotionClient.from_robot(robot, MOTION_SERVICE)
    destination = PoseInFrame(reference_frame="world", pose=START_POSE)
    print(f"plan: move arm-1 to start pose {START_POSE}")
    if not execute:
        print("dry run only, no motion sent (pass --execute to actually move the arm)")
        return
    await motion.move(component_name=ARM_NAME, destination=destination)


async def get_segments(vision: VisionClient):
    last_err = None
    for _ in range(SEGMENT_RETRIES):
        try:
            return await vision.get_object_point_clouds(CAMERA_NAME, timeout=30)
        except Exception as e:
            last_err = e
            await asyncio.sleep(1.0)
    raise last_err


async def find_block(robot: RobotClient, segment_service: str):
    vision = VisionClient.from_robot(robot, segment_service)
    objects = await get_segments(vision)
    if not objects:
        return None

    # pick the largest 3D box (by volume) among the detected segments
    best_geom = None
    best_volume = -1.0
    for obj in objects:
        for g in obj.geometries.geometries:
            if g.WhichOneof("geometry_type") == "box":
                dims = g.box.dims_mm
                volume = dims.x * dims.y * dims.z
            else:
                volume = 0.0
            if volume > best_volume:
                best_volume = volume
                best_geom = g

    center_in_cam = PoseInFrame(reference_frame=CAMERA_NAME, pose=best_geom.center)
    world = await robot.transform_pose(center_in_cam, "world")
    return world.pose.x, world.pose.y, world.pose.z


def top_down_pose(x: float, y: float, z: float) -> Pose:
    return Pose(x=x, y=y, z=z, **TOP_DOWN_ORIENTATION)


async def grab_at(robot: RobotClient, x: float, y: float, z: float, execute: bool) -> None:
    if not (TABLE_X_RANGE[0] <= x <= TABLE_X_RANGE[1] and TABLE_Y_RANGE[0] <= y <= TABLE_Y_RANGE[1]):
        raise RuntimeError(f"computed target ({x:.1f}, {y:.1f}) is outside the table bounds, refusing to move")

    motion = MotionClient.from_robot(robot, MOTION_SERVICE)
    gripper = Gripper.from_robot(robot, GRIPPER_NAME)

    # targets are fingertip heights; gripper-1's frame origin sits GRIPPER_TIP_OFFSET_MM above the
    # fingertips, so the commanded component pose must be raised by that much to compensate.
    hover_origin_z = APPROACH_HEIGHT_MM + GRIPPER_TIP_OFFSET_MM
    grasp_origin_z = z + GRIPPER_TIP_OFFSET_MM

    hover = PoseInFrame(reference_frame="world", pose=top_down_pose(x, y, hover_origin_z))
    grasp = PoseInFrame(reference_frame="world", pose=top_down_pose(x, y, grasp_origin_z))

    print(f"plan: open gripper -> hover, fingertip at ({x:.1f}, {y:.1f}, {APPROACH_HEIGHT_MM}) "
          f"[gripper-1 origin z={hover_origin_z:.1f}] "
          f"-> descend, fingertip at ({x:.1f}, {y:.1f}, {z:.1f}) [gripper-1 origin z={grasp_origin_z:.1f}] "
          f"-> close gripper -> retract to hover")

    if not execute:
        print("dry run only, no motion sent (pass --execute to actually move the arm)")
        return

    await gripper.open()
    await gripper.do_command({"set_gripper_speed": GRIPPER_CLOSE_SPEED})
    await motion.move(component_name=GRIPPER_NAME, destination=hover)
    await motion.move(component_name=GRIPPER_NAME, destination=grasp, constraints=STRAIGHT_LINE_CONSTRAINTS)
    # let the arm come fully to rest at the grasp point so it doesn't close while still settling upward
    await asyncio.sleep(GRASP_SETTLE_SECONDS)
    grabbed = await gripper.grab()
    print("grabbed:", grabbed)
    # give the fingers a moment to fully close/settle on the block before the arm starts lifting
    await asyncio.sleep(POST_GRAB_SETTLE_SECONDS)
    await motion.move(component_name=GRIPPER_NAME, destination=hover, constraints=STRAIGHT_LINE_CONSTRAINTS)


async def release_at(robot: RobotClient, release_pose: Pose, execute: bool) -> None:
    motion = MotionClient.from_robot(robot, MOTION_SERVICE)
    gripper = Gripper.from_robot(robot, GRIPPER_NAME)

    destination = PoseInFrame(reference_frame="world", pose=release_pose)
    print(f"plan: move gripper-1 to release pose {release_pose} -> open gripper")

    if not execute:
        print("dry run only, no motion sent (pass --execute to actually move the arm)")
        return

    await motion.move(component_name=GRIPPER_NAME, destination=destination)
    await gripper.open()
    print("released")


async def sort_color(robot: RobotClient, color: dict, execute: bool) -> None:
    name = color["name"]
    await goto_start(robot, execute)
    result = await find_block(robot, color["segment_service"])
    if result is None:
        print(f"no {name} blocks detected")
        return
    x, y, z = result
    print(f"target {name} block: world=({x:.1f},{y:.1f},{z:.1f})mm")
    await grab_at(robot, x, y, z, execute)
    await release_at(robot, color["release_pose"], execute)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually move the arm/gripper; omit for a dry run")
    args = parser.parse_args()

    robot = await connect()
    try:
        for color in COLORS:
            await sort_color(robot, color, args.execute)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
