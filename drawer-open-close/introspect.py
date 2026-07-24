"""Read-only introspection of robot8: camera properties, vision service
configs, frame system, arm/gripper capabilities. No motion commands.
"""
import asyncio
import os

from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials
from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient

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

    print("=== Camera: cam-1 ===")
    cam = Camera.from_robot(robot, "cam-1")
    props = await cam.get_properties()
    print(props)

    print("\n=== Vision services ===")
    for svc_name in ["vision-1", "vision-green"]:
        try:
            vis = VisionClient.from_robot(robot, svc_name)
            vis_props = await vis.get_properties()
            print(f"{svc_name}: {vis_props}")
        except Exception as e:
            print(f"{svc_name}: error {e}")

    print("\n=== Arm: arm-1 ===")
    arm = Arm.from_robot(robot, "arm-1")
    kinematics = await arm.get_kinematics()
    print("kinematics frame:", kinematics[0])
    pos = await arm.get_end_position()
    print("current end position:", pos)
    joints = await arm.get_joint_positions()
    print("current joint positions:", joints)
    moving = await arm.is_moving()
    print("is_moving:", moving)

    print("\n=== Gripper: gripper-1 ===")
    grip = Gripper.from_robot(robot, "gripper-1")
    g_moving = await grip.is_moving()
    print("is_moving:", g_moving)

    print("\n=== Frame system config ===")
    frame_cfg = await robot.get_frame_system_config()
    for f in frame_cfg:
        print(f.frame)

    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
