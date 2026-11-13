
import time
import matplotlib.pyplot as plt
import numpy as np

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.robots.osl import OpenSourceLeg


# ---------------- PARAMETERS ---------------- #

PORT = "/dev/ttyACM0"
GEAR_RATIO = 9 * (83 / 18)
FREQUENCY = 200
DT = 1.0 / FREQUENCY

# Start with a very small output torque
TEST_TORQUE = 0.25       # Nm at knee output
PULSE_DURATION = 0.20   # seconds
REST_DURATION = 1.0     # seconds

MAX_ANGLE_CHANGE = 3.0  # degrees; safety cutoff

CSV_FILE = "knee_torque_sign_test.csv"


# ---------------- INITIALIZATION ---------------- #

actuators = {
    "knee": DephyActuator(
        tag="knee",
        port=PORT,
        gear_ratio=GEAR_RATIO,
        frequency=FREQUENCY,
        debug_level=0,
        dephy_log=False,
    )
}

osl = OpenSourceLeg(
    tag="osl",
    actuators=actuators,
    sensors={},
)

results = []
start_time = None


# ---------------- TEST FUNCTIONS ---------------- #

def record_sample(phase, commanded_torque, baseline_angle):

    global start_time

    osl.update()

    position = osl.knee.output_position
    actual_torque = osl.knee.output_torque
    motor_current = osl.knee.motor_current

    position_deg = np.rad2deg(position)
    baseline_deg = np.rad2deg(baseline_angle)

    angle_change = position_deg - baseline_deg

    print(
        f"{time.monotonic() - start_time:7.3f}s | "
        f"{phase:12s} | "
        f"Command: {commanded_torque:+.3f} Nm | "
        f"Actual: {actual_torque:+.3f} Nm | "
        f"Angle: {position_deg:+.2f} deg"
    )

    results.append([
        time.monotonic() - start_time,
        phase,
        commanded_torque,
        actual_torque,
        position_deg,
        motor_current,
    ])

    # Stop if the knee moves more than the allowed amount
    if abs(angle_change) > MAX_ANGLE_CHANGE:
        osl.knee.set_output_torque(0.0)
        raise RuntimeError(
            "SAFETY STOP: Knee movement exceeded 3 degrees!"
        )


def run_phase(phase, commanded_torque, duration, baseline_angle):

    print(f"\nStarting {phase}: {commanded_torque:+.3f} Nm")

    end_time = time.monotonic() + duration

    while time.monotonic() < end_time:

        osl.knee.set_output_torque(commanded_torque)

        time.sleep(DT)

        record_sample(
            phase,
            commanded_torque,
            baseline_angle,
        )


def run_rest(duration, baseline_angle):

    run_phase(
        phase="REST",
        commanded_torque=0.0,
        duration=duration,
        baseline_angle=baseline_angle,
    )


# ---------------- MAIN TEST ---------------- #

if __name__ == "__main__":

    try:

        with osl:

            print("\nStarting OSL knee torque test...")

            osl.update()

            input(
                "\nSecure the leg in the test fixture, "
                "clear the movement area and press ENTER to continue..."
            )

            # Switch to current control
            osl.knee.set_control_mode(
                mode=CONTROL_MODES.CURRENT
            )

            osl.knee.set_current_gains()

            # Begin with zero torque
            osl.knee.set_output_torque(0.0)

            start_time = time.monotonic()

            # Allow measurements to stabilize
            print("\nInitial zero-torque period...")

            for _ in range(int(FREQUENCY * REST_DURATION)):
                osl.update()
                time.sleep(DT)

            baseline_angle = osl.knee.output_position

            print(
                f"\nBaseline knee angle: "
                f"{np.rad2deg(baseline_angle):.2f} degrees"
            )

            # 1. Positive torque
            run_phase(
                phase="POSITIVE",
                commanded_torque=+TEST_TORQUE,
                duration=PULSE_DURATION,
                baseline_angle=baseline_angle,
            )

            # 2. Return to zero
            run_rest(REST_DURATION, baseline_angle)

            # 3. Negative torque
            run_phase(
                phase="NEGATIVE",
                commanded_torque=-TEST_TORQUE,
                duration=PULSE_DURATION,
                baseline_angle=baseline_angle,
            )

            # 4. Final zero torque
            run_rest(REST_DURATION, baseline_angle)

            print("\nTest completed successfully.")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    except Exception as error:
        print(f"\nTEST STOPPED: {error}")

    finally:

        # Always request zero output torque before exiting
        try:
            osl.knee.set_output_torque(0.0)
            print("\nZero torque command sent.")
        except Exception:
            pass

        # Save measurements
        # ---------------- PLOT RESULTS ---------------- #
        if results:

            data = np.array(results, dtype=object)

            time_data = data[:, 0].astype(float)
            commanded = data[:, 2].astype(float)
            actual = data[:, 3].astype(float)
            angle = data[:, 4].astype(float)

            error = actual - commanded

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            # 1. Commanded vs actual torque
            axes[0].plot(time_data, commanded, label="Commanded Torque")
            axes[0].plot(time_data, actual, label="Actual Torque")

            axes[0].set_title("Actual vs Commanded Knee Torque")
            axes[0].set_ylabel("Torque (Nm)")
            axes[0].legend()
            axes[0].grid(True)

            # 2. Torque error
            axes[1].plot(time_data, error, label="Torque Error (Actual - Commanded)")
            axes[1].axhline(0, color="black", linestyle="--")

            axes[1].set_title("Knee Torque Error")
            axes[1].set_ylabel("Error (Nm)")
            axes[1].legend()
            axes[1].grid(True)

            # 3. Knee angle
            axes[2].plot(time_data, angle, label="Knee Angle")

            axes[2].set_title("Knee Position During Torque Test")
            axes[2].set_ylabel("Angle (degrees)")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()

            plt.savefig("knee_torque_sign_test.png", dpi=150)

            plt.show()


