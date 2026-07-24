"""Diagnostic: connect, then just wait/poll for 30s doing nothing but
cheap calls, to see if the connection drops on its own regardless of
any depth/image payload. No motion commands.
"""
import asyncio
import os
import time

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
    t0 = time.monotonic()
    robot = await connect()
    print(f"connected at t={time.monotonic() - t0:.1f}s")

    for i in range(30):
        await asyncio.sleep(1)
        try:
            names = robot.resource_names
            print(f"t={time.monotonic() - t0:.1f}s tick {i}: still ok, {len(names)} resources")
        except Exception as e:
            print(f"t={time.monotonic() - t0:.1f}s tick {i}: error {e}")

    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
