"""
Knee Actuator Single-Oscillation Control

Allows manual control of a Dephy knee actuator to perform exactly one
oscillation (0° → 45° → 0°) per command.

Features:
- Homing before starting.
- Position control using cosine trajectory.
- User input to run one oscillation ('R') or stop ('S').
- Displays desired vs actual position in real time.
"""

import numpy as np

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.utilities import SoftRealtimeLoop, units

FREQUENCY = 200

knee = DephyActuator(
    tag="knee",
    firmware_version="7.2.0",
    port="/dev/ttyACM0",
    gear_ratio=9 * 83 / 18,
    frequency=FREQUENCY,
)

actuators = [knee]

clock = SoftRealtimeLoop(dt=1 / FREQUENCY)

def make_periodic_trajectory(period, minimum, maximum):
    amplitude = (maximum - minimum) / 2
    mean = amplitude + minimum
    return lambda t: amplitude * np.cos(t * 2 * np.pi / period) + mean


# One oscillation trajectory (5 second period)
knee_traj = make_periodic_trajectory(5, 0, 45)


def run_one_oscillation():
    """Runs exactly one oscillation (0→90→0)."""
    period = 5
    t_end = period

    for t in clock:
        knee.update()

        knee_setpoint = units.convert_to_default(knee_traj(t), units.Position.deg)
        knee.set_output_position(knee_setpoint)

        print(
            f"Knee Desired {knee_setpoint:+.2f} rad, Actual {knee.output_position:+.2f} rad",
            end="\r",
        )

        if t >= t_end:
            break


with knee:

    input("Homing complete: Press Enter to start")

    knee.set_control_mode(CONTROL_MODES.POSITION)
    knee.set_position_gains(kp=3, kd=0.1)

    while True:

        user_input = input("\nPress 'R' for running one oscillation or 'S' to stop: ")

        if user_input.lower() == "s":
            print("Stopping motor...")
            knee.set_output_position(knee.output_position)
            break

        elif user_input == "r":
            run_one_oscillation()

        else:
            print("Invalid key. Press 'R' to move or 'S' to stop.")


print("\nProgram ended.")

