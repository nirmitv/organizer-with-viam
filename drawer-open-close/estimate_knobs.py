"""Read-only: detect the 3 drawer knobs from cam-1 alone (no vision
services), backproject to 3D using depth + intrinsics, and transform
into arm-1 frame. No motion commands.
"""
import asyncio
import os

import numpy as np
import cv2

from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials
from viam.components.camera import Camera
from viam.gen.common.v1.common_pb2 import PoseInFrame, Pose

ROBOT_ADDRESS = os.environ.get("ROBOT_ADDRESS", "robot8-main.ag9khwy6jn.viam.cloud")
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
API_KEY = os.environ["VIAM_API_KEY"]

# from cam-1 get_properties()
FX, FY, CX, CY = 910.4447021484375, 909.48077392578125, 653.95556640625, 375.75393676757812

# HSV red thresholds (red wraps around hue 0, so two ranges)
LOWER_RED_1 = np.array([0, 100, 80])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 80])
UPPER_RED_2 = np.array([180, 255, 255])


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


async def connect() -> RobotClient:
    opts = RobotClient.Options(
        dial_options=DialOptions(
            credentials=Credentials(type="api-key", payload=API_KEY),
            auth_entity=API_KEY_ID,
        ),
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


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
        if y < img_h * 0.45:  # knobs sit in the lower part of the frame
            continue
        candidates.append((x, y, w, h, area))

    # keep the 3 largest, most knob-like blobs
    candidates.sort(key=lambda c: c[4], reverse=True)
    candidates = candidates[:3]
    candidates.sort(key=lambda c: c[0])  # left to right
    return candidates


async def capture_with_retry(max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        robot = await connect()
        try:
            cam = Camera.from_robot(robot, "cam-1")
            imgs, _meta = await cam.get_images(timeout=30)
            return robot, imgs
        except Exception as e:
            print(f"get_images attempt {attempt} failed: {e}")
            await robot.close()
            if attempt == max_attempts:
                raise
            await asyncio.sleep(2)


async def main():
    robot, imgs = await capture_with_retry()
    color_named = imgs[0]
    depth_named = imgs[1]
    print(f"image[0] mime={color_named.mime_type}, image[1] mime={depth_named.mime_type}")

    color_arr = np.frombuffer(color_named.data, dtype=np.uint8)
    bgr = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
    depth_arr = fast_depth_array(depth_named)
    print("depth array shape:", depth_arr.shape, "dtype:", depth_arr.dtype)

    knobs = detect_red_knobs(bgr, bgr.shape[0])
    print(f"\nFound {len(knobs)} knob candidates (expect 3):")

    results = []
    for i, (x, y, w, h, area) in enumerate(knobs, start=1):
        u = int(x + w / 2)
        v = int(y + h / 2)
        # sample the whole bbox, not just the center, since glossy knobs
        # can produce a specular dropout (zero depth) right at the center
        patch = depth_arr[y : y + h, x : x + w]
        valid = patch[(patch > 0) & (patch < 2000)]
        if valid.size == 0:
            print(f"  drawer {i}: no valid depth at pixel ({u},{v}), skipping")
            continue
        z_mm = float(np.median(valid))
        x_mm = (u - CX) * z_mm / FX
        y_mm = (v - CY) * z_mm / FY

        query = PoseInFrame(
            reference_frame="cam-1",
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0, o_y=0, o_z=1, theta=0),
        )
        transformed = await robot.transform_pose(query, "arm-1")
        p = transformed.pose
        print(
            f"  drawer {i}: pixel=({u},{v}) bbox=({x},{y},{w},{h}) area={area:.0f} "
            f"depth={z_mm:.1f}mm cam_frame=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f}) "
            f"arm_frame=({p.x:.1f},{p.y:.1f},{p.z:.1f})"
        )
        results.append((i, p.x, p.y, p.z))
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(bgr, f"drawer {i}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out_path = os.path.join(os.path.dirname(__file__), "knobs_annotated.png")
    cv2.imwrite(out_path, bgr)
    print(f"\nSaved annotated frame to {out_path}")

    await robot.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())
