"""Standalone test for the manipulation block — no teammate code needed.

Run from the repo root:
  VIAM_ADDRESS=... VIAM_API_KEY=... VIAM_API_KEY_ID=... venv/bin/python pickplace/demo.py

Works against a fake arm in the machine config for plan-only testing;
switch to the real machine to tune poses and the approach offset.
"""

import asyncio
import os

from viam.components.gripper import Gripper
from viam.proto.common import Pose
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from pick_place import pick, place

GRIPPER_NAME = "gripper-1"
MOTION_NAME = "builtin"

# Arbitrary proof-of-concept points near the arm's idle end position
# (x=20, y=300, z=398, measured via setup.py). Grasp z=300 caps the descent at
# ~100mm below idle so the arm never dips more than 10cm toward the table.
# For OBJECT_POSE, pick() uses x/y/z and theta (block's long-axis angle in degrees,
# so the jaws close across the short side); o_x/o_y/o_z are ignored (always top-down).
OBJECT_POSE = Pose(x=20, y=250, z=300, o_x=0, o_y=0, o_z=-1, theta=0)
# Overhead drop point: the object free-falls from here, so keep z just above the
# drawer rim (a big drop bounces objects out of small drawers).
DROP_POSE = Pose(x=150, y=250, z=380, o_x=0, o_y=0, o_z=-1, theta=0)


async def main() -> None:
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    machine = await RobotClient.at_address(os.environ["VIAM_ADDRESS"], opts)
    try:
        motion = MotionClient.from_robot(machine, MOTION_NAME)
        gripper = Gripper.from_robot(machine, GRIPPER_NAME)

        grabbed = await pick(motion, gripper, OBJECT_POSE)
        print(f"pick: {'grabbed' if grabbed else 'missed'}")
        # Proof-of-concept: run the drop-off motion even with empty jaws.
        # Once real objects are in play, gate this on `grabbed`.
        await place(motion, gripper, DROP_POSE)
        print("place: done")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
