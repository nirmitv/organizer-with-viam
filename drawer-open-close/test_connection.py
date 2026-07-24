"""Read-only connectivity check: connects to robot8 and lists resources.

Does not move the arm or actuate anything.
"""
import asyncio
import os

from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials

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
    print("Connected. Resources:")
    for name in robot.resource_names:
        print(f"  {name}")
    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
