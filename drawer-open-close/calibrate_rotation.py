"""Eye-in-hand rotation calibration for cam-1 relative to arm-1.

Camera and gripper are both mounted on the arm and move together. As
long as the arm's orientation stays constant, the only thing we need
to relate "a point the camera sees" to "where the arm should move" is
a single fixed rotation matrix R such that:

    world_frame_vector = R @ camera_frame_vector

We solve for R by hovering the arm at several nearby poses (same
orientation each time, small XYZ offsets from the current position),
observing ONE fixed physical point (a drawer knob) at each pose, and
fitting a best-fit rotation between:
  - the arm's own end position at each pose (ground truth, from
    arm.get_end_position())
  - the knob's position in camera-frame coordinates at each pose
    (from depth + intrinsics backprojection)
via the Kabsch/orthogonal Procrustes algorithm.

IMPORTANT: only the rotation R is meaningful/saved here. The fitted
translation is a nuisance parameter (it's entangled with the unknown
absolute position of the reference knob and the unknown camera-to-
gripper mounting offset) and must NOT be used as a camera-to-gripper
offset. Locating a new target for an actual grasp should use R plus
iterative visual correction (re-observe and re-move), not a single
open-loop jump computed from this translation.

SAFETY: every waypoint is a small offset (<= 30mm) from the arm's
current position, same orientation throughout. This DOES move the
arm. Confirm the area around the drawer chest is clear before running.
"""
import asyncio
import os

import numpy as np
import cv2

from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials
from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.services.motion import MotionClient
from viam.gen.common.v1.common_pb2 import PoseInFrame, Pose

ROBOT_ADDRESS = os.environ.get("ROBOT_ADDRESS", "robot8-main.ag9khwy6jn.viam.cloud")
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
API_KEY = os.environ["VIAM_API_KEY"]

# from cam-1 get_properties()
FX, FY, CX, CY = 910.4447021484375, 909.48077392578125, 653.95556640625, 375.75393676757812

LOWER_RED_1 = np.array([0, 100, 80])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 80])
UPPER_RED_2 = np.array([180, 255, 255])

# small, safe offsets (mm) from the current arm position, same orientation.
# reference knob used throughout: index 1 = middle knob (drawer 2), most
# likely to stay in view across all of these small moves.
REFERENCE_KNOB_INDEX = 1
WAYPOINT_OFFSETS = [
    (0, 0, 0),
    (30, 0, 0),
    (-30, 0, 0),
    (0, 30, 0),
    (0, -30, 0),
    (0, 0, 25),
]


async def connect() -> RobotClient:
    opts = RobotClient.Options(
        dial_options=DialOptions(
            credentials=Credentials(type="api-key", payload=API_KEY),
            auth_entity=API_KEY_ID,
        ),
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


def fast_depth_array(depth_img):
    """Vectorized replacement for NamedImage.bytes_to_depth_array(), which
    decodes via a pure-Python nested loop slow enough to block the event
    loop and starve the SDK's own connection health-check.
    """
    data = depth_img.data
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    arr = np.frombuffer(data[24:], dtype=">u2")
    return arr.reshape(height, width)


def detect_red_knobs(bgr_img, img_h):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 800:
            continue
        aspect = w / h
        if not (0.6 <= aspect <= 1.6):
            continue
        if y < img_h * 0.45:
            continue
        candidates.append((x, y, w, h, area))
    candidates.sort(key=lambda c: c[4], reverse=True)
    candidates = candidates[:3]
    candidates.sort(key=lambda c: c[0])  # left to right
    return candidates


async def capture_with_retry(cam, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        try:
            imgs, _meta = await cam.get_images(timeout=30)
            return imgs
        except Exception as e:
            print(f"    get_images attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(2)


def knob_camera_frame_point(imgs, knob_index):
    color_named, depth_named = imgs[0], imgs[1]
    color_arr = np.frombuffer(color_named.data, dtype=np.uint8)
    bgr = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
    depth_arr = fast_depth_array(depth_named)

    knobs = detect_red_knobs(bgr, bgr.shape[0])
    if len(knobs) <= knob_index:
        raise RuntimeError(f"expected knob index {knob_index}, only found {len(knobs)} knobs")

    x, y, w, h, _area = knobs[knob_index]
    u = x + w / 2
    v = y + h / 2
    patch = depth_arr[y : y + h, x : x + w]
    valid = patch[(patch > 0) & (patch < 2000)]
    if valid.size == 0:
        raise RuntimeError("no valid depth in knob bbox")
    z_mm = float(np.median(valid))
    x_mm = (u - CX) * z_mm / FX
    y_mm = (v - CY) * z_mm / FY
    return np.array([x_mm, y_mm, z_mm])


def kabsch_fit(src_points, dst_points):
    """Best-fit rotation R (and a translation, discarded by the caller)
    s.t. dst ~= R @ src + t, via SVD."""
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_centered = src - src_c
    dst_centered = dst - dst_c

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = dst_c - R @ src_c
    return R, t


async def main():
    robot = await connect()
    arm = Arm.from_robot(robot, "arm-1")
    cam = Camera.from_robot(robot, "cam-1")
    motion = MotionClient.from_robot(robot, "builtin")

    home_pose = await arm.get_end_position()
    print(
        f"home pose: x={home_pose.x:.1f} y={home_pose.y:.1f} z={home_pose.z:.1f} "
        f"o_x={home_pose.o_x:.4f} o_y={home_pose.o_y:.4f} o_z={home_pose.o_z:.4f} "
        f"theta={home_pose.theta:.2f}"
    )

    arm_points = []
    cam_points = []

    try:
        for i, (dx, dy, dz) in enumerate(WAYPOINT_OFFSETS):
            target = Pose(
                x=home_pose.x + dx,
                y=home_pose.y + dy,
                z=home_pose.z + dz,
                o_x=home_pose.o_x,
                o_y=home_pose.o_y,
                o_z=home_pose.o_z,
                theta=home_pose.theta,
            )
            print(f"\nwaypoint {i}: offset=({dx},{dy},{dz})")
            ok = await motion.move(
                component_name="arm-1",
                destination=PoseInFrame(reference_frame="world", pose=target),
            )
            print(f"  move success={ok}")

            actual = await arm.get_end_position()
            print(f"  actual arm pos: ({actual.x:.1f},{actual.y:.1f},{actual.z:.1f})")

            imgs = await capture_with_retry(cam)
            cam_pt = knob_camera_frame_point(imgs, REFERENCE_KNOB_INDEX)
            print(f"  reference knob cam-frame point: {cam_pt}")

            arm_points.append([actual.x, actual.y, actual.z])
            cam_points.append(cam_pt)

    finally:
        print("\nReturning to home pose...")
        await motion.move(
            component_name="arm-1",
            destination=PoseInFrame(reference_frame="world", pose=home_pose),
        )

    R, t = kabsch_fit(cam_points, arm_points)
    print("\n=== Calibration result ===")
    print("Rotation R (camera-frame vector -> world/arm-frame vector):")
    print(R)

    arm_points_np = np.array(arm_points)
    cam_points_np = np.array(cam_points)
    predicted = (R @ cam_points_np.T).T + t
    errors = np.linalg.norm(predicted - arm_points_np, axis=1)
    print(f"\nfit residual errors (mm) per waypoint: {errors}")
    print(f"mean residual: {errors.mean():.2f}mm, max: {errors.max():.2f}mm")
    print("(low residual = R is self-consistent across all waypoints; the")
    print(" translation t is NOT saved/used, only R.)")

    out_path = os.path.join(os.path.dirname(__file__), "cam_to_arm_rotation.npy")
    np.save(out_path, R)
    print(f"\nSaved rotation matrix to {out_path}")

    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
