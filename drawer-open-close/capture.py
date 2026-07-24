"""Read-only: capture color + depth from cam-1 only, save the color
frame locally for visual inspection, and report on the depth image.
No motion commands, no vision services.
"""
import asyncio
import os

import numpy as np

from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image


def fast_depth_array(depth_img):
    """Vectorized replacement for NamedImage.bytes_to_depth_array().

    That SDK method decodes via a pure-Python nested loop over every
    pixel, which is slow enough to block the event loop and starve the
    SDK's own background connection health-check. This does the same
    parsing with numpy instead.
    """
    data = depth_img.data
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    arr = np.frombuffer(data[24:], dtype=">u2")  # big-endian uint16
    return arr.reshape(height, width)

ROBOT_ADDRESS = os.environ.get("ROBOT_ADDRESS", "robot8-main.ag9khwy6jn.viam.cloud")
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
API_KEY = os.environ["VIAM_API_KEY"]


async def connect() -> RobotClient:
    opts = RobotClient.Options(
        dial_options=DialOptions(
            credentials=Credentials(type="api-key", payload=API_KEY),
            auth_entity=API_KEY_ID,
        ),
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


async def main():
    robot = await connect()
    cam = Camera.from_robot(robot, "cam-1")

    imgs, _meta = await cam.get_images(timeout=30)
    print(f"got {len(imgs)} image(s)")

    color_img = imgs[0]
    print(f"image[0]: mime={color_img.mime_type} bytes={len(color_img.data)}")
    pil_img = viam_to_pil_image(color_img)
    out_path = os.path.join(os.path.dirname(__file__), "frame.png")
    pil_img.save(out_path)
    print(f"Saved color frame to {out_path}, size={pil_img.size}")

    if len(imgs) > 1:
        depth_img = imgs[1]
        print(f"image[1]: mime={depth_img.mime_type} bytes={len(depth_img.data)}")
        depth_arr = fast_depth_array(depth_img)
        print(f"depth array: shape={depth_arr.shape} dtype={depth_arr.dtype}")
        print(f"depth min={depth_arr.min()} max={depth_arr.max()} mean={depth_arr.mean():.1f}")
        nonzero = depth_arr[depth_arr > 0]
        print(f"nonzero pixels: {nonzero.size} / {depth_arr.size}")
    else:
        print("only 1 image returned, no depth present")

    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
