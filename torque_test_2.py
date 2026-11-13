
"""
Impedance Control Test - Open Source Leg

Test sequence:
1. Early stance (3 s)
2. Rest (3 s)
3. Late stance (3 s)
4. Rest (3 s)
5. Swing (3 s)
6. Rest (3 s)

Compares theoretical impedance torque with actual Dephy torque.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.robots.osl import OpenSourceLeg
from opensourceleg.sensors.loadcell import DephyLoadcellAmplifier
from opensourceleg.sensors.encoder import AS5048B


# ============================================================
# PARAMETERS
# ============================================================

GEAR_RATIO = 9 * (83 / 18)

FREQUENCY = 200
DT = 1 / FREQUENCY

PHASE_DURATION = 3.0
REST_DURATION = 3.0

# ============================================================
# LOADCELL CALIBRATION MATRIX
# ============================================================

LOADCELL_CALIBRATION_MATRIX = np.array([
    (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
    (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
    (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
    (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
    (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
    (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
])


# ============================================================
# TEST PHASES
# ============================================================

# name, stiffness, damping, reference angle (degrees), duration

phases = [
    ("Early Stance", 20, 1, 15, PHASE_DURATION),
    ("Rest 1", 0, 0, None, REST_DURATION),

    ("Late Stance", 50, 20, 5, PHASE_DURATION),
    ("Rest 2", 0, 0, None, REST_DURATION),

    ("Swing", 10, 0.5, 60, PHASE_DURATION),
    ("Rest 3", 0, 0, None, REST_DURATION),
]


# ============================================================
# INITIALIZE HARDWARE
# ============================================================

actuators = {
    "knee": DephyActuator(
        tag="knee",
        port="/dev/ttyACM0",
        gear_ratio=GEAR_RATIO,
        frequency=FREQUENCY,
        debug_level=0,
        dephy_log=False,
    ),
}

sensors = {
    "loadcell": DephyLoadcellAmplifier(
        calibration_matrix=LOADCELL_CALIBRATION_MATRIX,
        tag="loadcell",
        amp_gain=125,
        exc=5,
        bus=1,
        i2c_address=102,
    ),

    "encoder": AS5048B(
        tag="encoder1",
        bus="/dev/i2c-1",
        A1_adr_pin=True,
        A2_adr_pin=False,
        zero_position=0,
        enable_diagnostics=False,
    ),
}

osl = OpenSourceLeg(
    tag="osl",
    actuators=actuators,
    sensors=sensors,
)


# ============================================================
# DATA STORAGE
# ============================================================

time_log = []
phase_log = []

k_log = []
b_log = []

theta_actual_log = []
theta_ref_log = []

encoder_velocity_log = []

torque_theoretical_log = []
torque_actual_log = []
torque_error_log = []


# ============================================================
# MAIN TEST
# ============================================================

with osl:

    print("Starting Open Source Leg...")
    osl.update()

    print("Homing...")
    osl.home()

    input(
        "\nMake sure the leg is secured and the test area is clear."
        "\nPress ENTER when ready to start the test..."
    )

    # Set impedance control mode
    osl.knee.set_control_mode(
        mode=CONTROL_MODES.IMPEDANCE
    )

    osl.knee.set_impedance_cc_pidf_gains()
    osl.knee.set_output_impedance()

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # RUN EACH PHASE
        # ----------------------------------------------------

        for phase_name, k, b, ref_deg, duration in phases:

            print(f"\nStarting: {phase_name}")

            phase_start = time.perf_counter()

            # Convert reference angle to radians
            if ref_deg is not None:
                theta_ref_fixed = np.deg2rad(ref_deg)

            while time.perf_counter() - phase_start < duration:

                loop_start = time.perf_counter()

                # Update sensor and actuator measurements
                osl.update()

                # --------------------------------------------
                # READ MEASUREMENTS
                # --------------------------------------------

                theta_actual = osl.knee.output_position

                encoder_velocity = osl.sensors["encoder"].velocity

                # EXACTLY THE SAME VELOCITY CONVENTION
                # AS YOUR FINAL FSM
                angular_vel = encoder_velocity

                # --------------------------------------------
                # REFERENCE ANGLE
                # --------------------------------------------

                if ref_deg is not None:

                    theta_ref = theta_ref_fixed

                else:

                    # During rest, reference follows the
                    # current knee position
                    theta_ref = theta_actual

                # --------------------------------------------
                # THEORETICAL IMPEDANCE TORQUE
                # --------------------------------------------

                torque_theoretical = (
                    -k * (theta_actual - theta_ref)
                    -b * angular_vel
                )

                # --------------------------------------------
                # SEND IMPEDANCE COMMANDS TO DEPHY
                # --------------------------------------------

                osl.knee.set_output_impedance(
                    k=k,
                    b=b
                )

                osl.knee.set_motor_position(theta_ref)

                # --------------------------------------------
                # READ ACTUAL TORQUE
                # --------------------------------------------

                torque_actual = osl.knee.output_torque

                # Torque error
                torque_error = (
                    torque_actual - torque_theoretical
                )

                # --------------------------------------------
                # STORE DATA
                # --------------------------------------------

                elapsed_time = time.perf_counter() - start_time

                time_log.append(elapsed_time)
                phase_log.append(phase_name)

                k_log.append(k)
                b_log.append(b)

                theta_actual_log.append(theta_actual)
                theta_ref_log.append(theta_ref)

                encoder_velocity_log.append(encoder_velocity)

                torque_theoretical_log.append(torque_theoretical)
                torque_actual_log.append(torque_actual)
                torque_error_log.append(torque_error)

                # --------------------------------------------
                # MAINTAIN 200 Hz
                # --------------------------------------------

                loop_elapsed = time.perf_counter() - loop_start

                time.sleep(max(0, DT - loop_elapsed))

            print(f"Finished: {phase_name}")

    except KeyboardInterrupt:

        print("\nTest interrupted by user.")

    finally:

        # Set impedance gains to zero at the end
        osl.knee.set_output_impedance(k=0, b=0)

        print("\nTest finished.")


# ============================================================
# CONVERT DATA TO ARRAYS
# ============================================================

time_log = np.array(time_log)
phase_log = np.array(phase_log)

theta_actual_log = np.rad2deg(
    np.array(theta_actual_log)
)

theta_ref_log = np.rad2deg(
    np.array(theta_ref_log)
)

encoder_velocity_log = np.rad2deg(
    np.array(encoder_velocity_log)
)

torque_theoretical_log = np.array(torque_theoretical_log)
torque_actual_log = np.array(torque_actual_log)
torque_error_log = np.array(torque_error_log)


# ============================================================
# PLOT RESULTS
# ============================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 11),
    sharex=True
)


# ------------------------------------------------------------
# 1. THEORETICAL VS ACTUAL TORQUE
# ------------------------------------------------------------

axes[0].plot(
    time_log,
    torque_theoretical_log,
    label="Theoretical Impedance Torque",
    linewidth=1.5
)

axes[0].plot(
    time_log,
    torque_actual_log,
    label="Actual Dephy Torque",
    linewidth=1.5
)

axes[0].set_title("Theoretical vs Actual Knee Torque")
axes[0].set_ylabel("Torque (Nm)")
axes[0].legend()
axes[0].grid(True)


# ------------------------------------------------------------
# 2. TORQUE ERROR
# ------------------------------------------------------------

axes[1].plot(
    time_log,
    torque_error_log,
    label="Torque Error (Actual - Theoretical)"
)

axes[1].axhline(
    0,
    color="black",
    linestyle="--"
)

axes[1].set_title("Knee Torque Error")
axes[1].set_ylabel("Error (Nm)")
axes[1].legend()
axes[1].grid(True)


# ------------------------------------------------------------
# 3. KNEE ANGLE TRACKING
# ------------------------------------------------------------

axes[2].plot(
    time_log,
    theta_actual_log,
    label="Actual Knee Angle"
)

axes[2].plot(
    time_log,
    theta_ref_log,
    "--",
    label="Reference Knee Angle"
)

axes[2].set_title("Knee Angle Tracking")
axes[2].set_ylabel("Angle (degrees)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True)


# ------------------------------------------------------------
# MARK PHASE TRANSITIONS
# ------------------------------------------------------------

transition_times = np.cumsum([
    duration for _, _, _, _, duration in phases
])[:-1]

for ax in axes:

    for transition in transition_times:

        ax.axvline(
            transition,
            color="gray",
            linestyle=":",
            alpha=0.6
        )


plt.tight_layout()

plt.savefig(
    "impedance_control_test.png",
    dpi=150
)

plt.show()

print("\nPlot saved as impedance_control_test.png")

