"""
Finite state machine (FSM)-based knee control system for the OpenSourceLeg platform.
State transitions are based on FILTERED load cell feedback and encoder angle.
A minimum state duration / transition lockout avoids rapid switching due to noisy signals.
"""

import numpy as np
import matplotlib.pyplot as plt

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.control.fsm import State, StateMachine
from opensourceleg.logging.logger import Logger
from opensourceleg.robots.osl import OpenSourceLeg
from opensourceleg.sensors.loadcell import DephyLoadcellAmplifier
from opensourceleg.sensors.encoder import AS5048B
from opensourceleg.utilities import SoftRealtimeLoop


# ---------------- PARAMETERS ---------------- #

GEAR_RATIO = 9 * (83 / 18)
FREQUENCY = 200

LOADCELL_CALIBRATION_MATRIX = np.array([
    (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
    (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
    (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
    (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
    (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
    (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
])

BODY_WEIGHT = 30 * 9.8
MIN_STATE_TIME = 0.15  # seconds


# ---------------- FSM ---------------- #

LOAD_EARLY_STANCE = 0.05 * BODY_WEIGHT
LOAD_STANCE = 0.15 * BODY_WEIGHT
LOAD_SWING = 0.08 * BODY_WEIGHT

KNEE_ESTANCE = np.deg2rad(90)
KNEE_LSTANCE = np.deg2rad(98)


def create_knee_fsm(osl: OpenSourceLeg) -> StateMachine:

    # ---------- SWING ----------

    swing = State(
        name="swing",
        knee_theta=60,
        knee_stiffness=10,
        knee_damping=0.5,
    )

    # ---------- EARLY STANCE ----------

    early_stance = State(
        name="early_stance",
        knee_theta=8,
        knee_stiffness=20,
        knee_damping=1,
    )

    # ---------- LATE STANCE ----------

    late_stance = State(
        name="late_stance",
        knee_theta=1,
        knee_stiffness=100,
        knee_damping=20,
    )

    # Store last transition time
    osl.last_state_change_time = 0.0

    def transition_allowed(osl):
        return (
            osl.clock_time - osl.last_state_change_time
        ) >= MIN_STATE_TIME

    # Transition criteria
    def swing_to_early_stance(osl):
        if not transition_allowed(osl):
            return False

        theta = osl.sensors["encoder"].position

        if (
            osl.filtered_fz < -LOAD_EARLY_STANCE
            and theta > KNEE_ESTANCE
        ):
            osl.last_state_change_time = osl.clock_time
            return True

        return False

    def early_stance_to_late_stance(osl):
        if not transition_allowed(osl):
            return False

        theta = osl.sensors["encoder"].position

        if (
            osl.filtered_fz < -LOAD_STANCE
            and theta > KNEE_LSTANCE
        ):
            osl.last_state_change_time = osl.clock_time
            return True

        return False

    def late_stance_to_swing(osl):
        if not transition_allowed(osl):
            return False

        if osl.filtered_fz > -LOAD_SWING:
            osl.last_state_change_time = osl.clock_time
            return True

        return False

    # ---------- FSM ----------

    fsm = StateMachine(
        states=[
            swing,
            early_stance,
            late_stance,
        ],
        initial_state_name="early_stance",
    )

    fsm.add_transition(
        source=swing,
        destination=early_stance,
        event_name="heel_strike",
        criteria=swing_to_early_stance,
    )

    fsm.add_transition(
        source=early_stance,
        destination=late_stance,
        event_name="loading_response",
        criteria=early_stance_to_late_stance,
    )

    fsm.add_transition(
        source=late_stance,
        destination=swing,
        event_name="toe_off",
        criteria=late_stance_to_swing,
    )

    return fsm


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

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

    clock = SoftRealtimeLoop(dt=1 / FREQUENCY)

    fsm_logger = Logger(
        log_path="./logs",
        file_name="knee_fsm.log",
    )

    osl = OpenSourceLeg(
        tag="osl",
        actuators=actuators,
        sensors=sensors,
    )

    fsm = create_knee_fsm(osl)

    # ---------------- LOW PASS FILTER ---------------- #

    CUTOFF_HZ = 10
    DT = 1 / FREQUENCY
    alpha = DT / (1 / (2 * np.pi * CUTOFF_HZ) + DT)

    fz_filtered = 0.0

    # ---------------- LOGS ---------------- #

    time_log = []

    k_log = []
    b_log = []
    state_log = []

    theta_log = []
    theta_ref_log = []

    encoder_theta_log = []
    encoder_velocity_log = []

    loadcell_fz_log = []
    loadcell_fz_filt_log = []

    # Torque and angle error logs
    torque_actual_log = []
    torque_commanded_log = []
    torque_error_log = []

    angle_error_encoder_log = []
    angle_error_actuator_log = []

    # ---------------- LOADCELL AXES LOGS ---------------- #

    loadcell_fx_log = []
    loadcell_fy_log = []

    loadcell_mx_log = []
    loadcell_my_log = []
    loadcell_mz_log = []

    # ---------------- START ---------------- #

    with osl, fsm:

        print("Starting system...")
        osl.update()

        print("Homing...")
        osl.home()

        print("Calibrating loadcell...")
        osl.loadcell.reset()
        osl.loadcell.calibrate()

        input("Press Enter to start control...")

        osl.knee.set_control_mode(mode=CONTROL_MODES.IMPEDANCE)
        osl.knee.set_impedance_cc_pidf_gains()
        osl.knee.set_output_impedance()

        try:
            for t in clock:

                osl.update()
                osl.clock_time = t

                # ---------------- FILTER ---------------- #

                raw_fz = osl.loadcell.fz
                fz_filtered = alpha * raw_fz + (1 - alpha) * fz_filtered
                osl.filtered_fz = fz_filtered

                # ---------------- FSM ---------------- #

                fsm.update(osl=osl)

                # ---------------- CONTROL ---------------- #

                k = fsm.current_state.knee_stiffness
                b = fsm.current_state.knee_damping

                theta_ref = np.deg2rad(fsm.current_state.knee_theta)

                theta_actual = osl.knee.output_position

                encoder_theta = osl.sensors["encoder"].position
                encoder_velocity = osl.sensors["encoder"].velocity

                # Apply the impedance controller
                osl.knee.set_output_impedance(k=k, b=b)
                osl.knee.set_motor_position(theta_ref)

                # ---------------- TORQUE AND ANGLE ERRORS ---------------- #

                # The impedance equation requires radians and rad/s.
                # This uses encoder velocity as the angular velocity estimate.
                angular_vel = encoder_velocity

                # The requested commanded-torque equation uses actuator angle.
                torque_commanded = (
                    -k * (theta_actual - theta_ref)
                    - b * angular_vel
                )

                # Actual torque reading from the Dephy actuator.
                # Confirm that output_torque is supported by your installed OSL version.
                torque_actual = osl.knee.output_torque

                # Signed torque error: measured minus commanded (Nm)
                torque_error = torque_actual - torque_commanded

                # Signed angle errors (rad), converted to degrees for plotting
                angle_error_encoder = np.rad2deg(theta_ref - encoder_theta)
                angle_error_actuator = np.rad2deg(theta_ref - theta_actual)

                # ---------------- LOADCELL AXES ---------------- #

                fx = osl.loadcell.fx
                fy = osl.loadcell.fy

                mx = osl.loadcell.mx
                my = osl.loadcell.my
                mz = osl.loadcell.mz

                # ---------------- LOG ---------------- #

                fsm_logger.info(
                    f"T:{t:.3f}, "
                    f"State:{fsm.current_state.name}, "
                    f"Fz:{fz_filtered:.2f}, "
                    f"ThetaRef:{np.rad2deg(theta_ref):.2f}, "
                    f"ThetaActual:{np.rad2deg(theta_actual):.2f}, "
                    f"EncoderTheta:{np.rad2deg(encoder_theta):.2f}, "
                    f"EncoderVelocity:{np.rad2deg(encoder_velocity):.2f}, "
                    f"K:{k:.2f}, "
                    f"B:{b:.2f}, "
                    f"TorqueCommanded:{torque_commanded:.3f}, "
                    f"TorqueActual:{torque_actual:.3f}, "
                    f"TorqueError:{torque_error:.3f}, "
                    f"AngleErrorEncoder:{angle_error_encoder:.3f}, "
                    f"AngleErrorActuator:{angle_error_actuator:.3f}"
                )

                # ---------------- STORE ---------------- #

                time_log.append(t)

                k_log.append(k)
                b_log.append(b)
                state_log.append(fsm.current_state.name)

                theta_log.append(np.rad2deg(theta_actual))
                theta_ref_log.append(np.rad2deg(theta_ref))

                encoder_theta_log.append(np.rad2deg(encoder_theta))
                encoder_velocity_log.append(np.rad2deg(encoder_velocity))

                loadcell_fz_log.append(raw_fz)
                loadcell_fz_filt_log.append(fz_filtered)

                torque_actual_log.append(torque_actual)
                torque_commanded_log.append(torque_commanded)
                torque_error_log.append(torque_error)

                angle_error_encoder_log.append(angle_error_encoder)
                angle_error_actuator_log.append(angle_error_actuator)

                # ---------------- STORE LOADCELL AXES ---------------- #

                loadcell_fx_log.append(fx)
                loadcell_fy_log.append(fy)

                loadcell_mx_log.append(mx)
                loadcell_my_log.append(my)
                loadcell_mz_log.append(mz)

        except KeyboardInterrupt:
            print("\nStopped.")

    # ---------------- CONVERT ---------------- #

    time_log = np.array(time_log)

    k_log = np.array(k_log)
    b_log = np.array(b_log)
    state_log = np.array(state_log)

    theta_log = np.array(theta_log)
    theta_ref_log = np.array(theta_ref_log)

    encoder_theta_log = np.array(encoder_theta_log)
    encoder_velocity_log = np.array(encoder_velocity_log)

    loadcell_fz_log = np.array(loadcell_fz_log)
    loadcell_fz_filt_log = np.array(loadcell_fz_filt_log)

    torque_actual_log = np.array(torque_actual_log)
    torque_commanded_log = np.array(torque_commanded_log)
    torque_error_log = np.array(torque_error_log)

    angle_error_encoder_log = np.array(angle_error_encoder_log)
    angle_error_actuator_log = np.array(angle_error_actuator_log)

    # ---------------- CONVERT LOADCELL AXES ---------------- #

    loadcell_fx_log = np.array(loadcell_fx_log)
    loadcell_fy_log = np.array(loadcell_fy_log)

    loadcell_mx_log = np.array(loadcell_mx_log)
    loadcell_my_log = np.array(loadcell_my_log)
    loadcell_mz_log = np.array(loadcell_mz_log)

    # ---------------- PLOTTING HELPER ---------------- #

    def plot_by_state(y, title, ylabel, filename):

        plt.figure(figsize=(10, 5))

        state_colors = {
            "swing": "tab:blue",
            "early_stance": "tab:orange",
            "late_stance": "tab:green",
        }

        for i in range(1, len(time_log)):
            plt.plot(
                time_log[i - 1:i + 1],
                y[i - 1:i + 1],
                color=state_colors[state_log[i]],
                linewidth=2,
            )

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.grid(True)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------------- STIFFNESS ---------------- #

    plot_by_state(
        k_log,
        "Knee Stiffness Over Time",
        "Stiffness (Nm/rad)",
        "knee_stiffness_plot.png",
    )

    # ---------------- DAMPING ---------------- #

    plot_by_state(
        b_log,
        "Knee Damping Over Time",
        "Damping",
        "knee_damping_plot.png",
    )

    # ---------------- THETA ---------------- #

    plt.figure(figsize=(12, 6))
    plt.plot(time_log, theta_log, label="Actuator Knee Angle", linewidth=2)
    plt.plot(time_log, theta_ref_log, "--", label="Reference Knee Angle", linewidth=2)
    plt.title("Knee Angle Tracking Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (deg)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("knee_theta_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- TORQUE ERROR ---------------- #

    plt.figure(figsize=(12, 5))
    plt.plot(
        time_log,
        torque_error_log,
        label="Torque Error (Actual - Commanded)",
        linewidth=1.5,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Knee Torque Error Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque Error (Nm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("knee_torque_error_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- ANGLE ERRORS ---------------- #

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(
        time_log,
        angle_error_encoder_log,
        label="Reference - Encoder",
        linewidth=1.5,
    )
    axs[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0].set_title("Angle Error: Encoder")
    axs[0].set_ylabel("Angle Error (deg)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(
        time_log,
        angle_error_actuator_log,
        label="Reference - Dephy Actuator",
        linewidth=1.5,
    )
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[1].set_title("Angle Error: Dephy Motor Actuator")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angle Error (deg)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("knee_angle_error_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- ENCODER + LOADCELL ---------------- #

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time_log, encoder_theta_log)
    axs[0].set_title("Encoder Angle")
    axs[0].set_ylabel("Angle (deg)")
    axs[0].grid(True)

    axs[1].plot(time_log, encoder_velocity_log)
    axs[1].set_title("Encoder Velocity")
    axs[1].set_ylabel("Velocity (deg/s)")
    axs[1].grid(True)

    axs[2].plot(time_log, loadcell_fz_log, label="Raw")
    axs[2].plot(time_log, loadcell_fz_filt_log, label="Filtered")
    axs[2].set_title("Loadcell Force")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Fz (N)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("encoder_loadcell_outputs_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- 3x2 LOADCELL FORCES/MOMENTS ---------------- #

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    # ---------- FORCE FX ----------
    axs[0, 0].plot(time_log, loadcell_fx_log, label="Fx")
    axs[0, 0].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[0, 0].set_title("Force Fx")
    axs[0, 0].set_ylabel("Fx (N)")
    axs[0, 0].grid(True)

    # ---------- FORCE FY ----------
    axs[1, 0].plot(time_log, loadcell_fy_log, label="Fy")
    axs[1, 0].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[1, 0].set_title("Force Fy")
    axs[1, 0].set_ylabel("Fy (N)")
    axs[1, 0].grid(True)

    # ---------- FORCE FZ ----------
    axs[2, 0].plot(time_log, loadcell_fz_log, label="Fz")
    axs[2, 0].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[2, 0].set_title("Force Fz")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Fz (N)")
    axs[2, 0].grid(True)

    # ---------- MOMENT MX ----------
    axs[0, 1].plot(time_log, loadcell_mx_log, label="Mx")
    axs[0, 1].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[0, 1].set_title("Moment Mx")
    axs[0, 1].set_ylabel("Mx (Nm)")
    axs[0, 1].grid(True)

    # ---------- MOMENT MY ----------
    axs[1, 1].plot(time_log, loadcell_my_log, label="My")
    axs[1, 1].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[1, 1].set_title("Moment My")
    axs[1, 1].set_ylabel("My (Nm)")
    axs[1, 1].grid(True)

    # ---------- MOMENT MZ ----------
    axs[2, 1].plot(time_log, loadcell_mz_log, label="Mz")
    axs[2, 1].plot(
        time_log,
        encoder_theta_log,
        color="lightgrey",
        linewidth=1.5,
        label="Encoder Angle",
    )
    axs[2, 1].set_title("Moment Mz")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Mz (Nm)")
    axs[2, 1].grid(True)

    axs[0, 0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("loadcell_forces_moments_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved all plots:")
    print("- knee_stiffness_plot.png")
    print("- knee_damping_plot.png")
    print("- knee_theta_plot.png")
    print("- knee_torque_error_plot.png")
    print("- knee_angle_error_plot.png")
    print("- encoder_loadcell_outputs_plot.png")
    print("- loadcell_forces_moments_plot.png")
