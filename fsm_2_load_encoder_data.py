"""
Finite state machine (FSM)-based knee control system for the OpenSourceLeg platform.
State transitions are based on FILTERED load cell feedback only and applies
impedance control to the knee actuator in real time..

Encoder is logged but not used for control.
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

#LOADCELL_CALIBRATION_MATRIX = np.array([
    #(-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
    #(-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
    #(-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
    #(20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
    #(-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
    #(-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
#])

LOADCELL_CALIBRATION_MATRIX = np.array([ 
    (-12.59925, -1714.72670, 30.08768, 23.83767, -19.06937, 1591.12752), 
    (-20.61383, 973.84169, 32.63392, -1823.09994, 17.86314, 920.63255), 
    (-1009.60931, 11.02942, -968.67876, -8.28354, -982.19232, -4.29526), 
    (21.04895, 1.39746, 0.04007, -0.84455, -20.68956, -0.57042), 
    (-12.32609, 0.49963, 23.45320, 0.46924, -12.07187, -0.73778), 
    (-0.19264, -26.63914, 0.27437, -24.21785, 0.50465, -25.58366),
]) 

BODY_WEIGHT = 50 * 9.8

LOAD_STANCE = 0.15 * BODY_WEIGHT
LOAD_SWING = 0.10 * BODY_WEIGHT


# ---------------- FSM ---------------- #

def create_knee_fsm(osl: OpenSourceLeg) -> StateMachine:

    stance = State(
        name="stance",
        knee_theta=5,
        knee_stiffness=500,
        knee_damping=20,
    )

    swing = State(
        name="swing",
        knee_theta=60,
        knee_stiffness=10,
        knee_damping=0.5,
    )

    # USE FILTERED SIGNAL
    def stance_to_swing(osl):
        return osl.filtered_fz > -LOAD_SWING

    def swing_to_stance(osl):
        return osl.filtered_fz < -LOAD_STANCE

    fsm = StateMachine(
        states=[stance, swing],
        initial_state_name="stance",
    )

    fsm.add_transition(
        source=stance,
        destination=swing,
        event_name="toe_off",
        criteria=stance_to_swing,
    )

    fsm.add_transition(
        source=swing,
        destination=stance,
        event_name="heel_strike",
        criteria=swing_to_stance,
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

    # ---------------- LOADCELL AXES LOGS ---------------- #

    loadcell_fx_log = []
    loadcell_fy_log = []
    #loadcell_fz_full_log = []

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

                osl.knee.set_output_impedance(k=k, b=b)
                osl.knee.set_motor_position(theta_ref)

                # ---------------- LOADCELL AXES ---------------- #

                fx = osl.loadcell.fx
                fy = osl.loadcell.fy
                #fz = osl.loadcell.fz

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
                    f"B:{b:.2f}"
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

                # ---------------- STORE LOADCELL AXES ---------------- #

                loadcell_fx_log.append(fx)
                loadcell_fy_log.append(fy)
                #loadcell_fz_full_log.append(fz)

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

# ---------------- CONVERT LOADCELL AXES ---------------- #

loadcell_fx_log = np.array(loadcell_fx_log)
loadcell_fy_log = np.array(loadcell_fy_log)
#loadcell_fz_full_log = np.array(loadcell_fz_full_log)

loadcell_mx_log = np.array(loadcell_mx_log)
loadcell_my_log = np.array(loadcell_my_log)
loadcell_mz_log = np.array(loadcell_mz_log)


# ---------------- PLOT ---------------- #

def plot_by_state(y, title, ylabel, filename):

    plt.figure(figsize=(10, 5))

    for i in range(1, len(time_log)):

        style = "-" if state_log[i] == "stance" else "--"

        plt.plot(
            time_log[i - 1:i + 1],
            y[i - 1:i + 1],
            style,
            color="black"
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
    "knee_stiffness_plot.png"
)

# ---------------- DAMPING ---------------- #

plot_by_state(
    b_log,
    "Knee Damping Over Time",
    "Damping",
    "knee_damping_plot.png"
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

plt.savefig("knee_theta_plot.png", dpi=300, bbox_inches="tight")


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


# ---------------- 3x2 LOADCELL FORCES/MOMENTS ---------------- #

fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

# ---------- FORCE FX ----------
axs[0, 0].plot(time_log, loadcell_fx_log, label="Fx")
axs[0, 0].plot(
    time_log,
    encoder_theta_log,
    color="lightgrey",
    linewidth=1.5,
    label="Encoder Angle"
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
    label="Encoder Angle"
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
    label="Encoder Angle"
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
    label="Encoder Angle"
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
    label="Encoder Angle"
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
    label="Encoder Angle"
)
axs[2, 1].set_title("Moment Mz")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Mz (Nm)")
axs[2, 1].grid(True)

# Show legend only once
axs[0, 0].legend(loc="upper right")

plt.tight_layout()
plt.savefig("loadcell_forces_moments_plot.png", dpi=300, bbox_inches="tight")

plt.show()

# ---------------- Fz + My OVER TIME ---------------- #

fig, ax1 = plt.subplots(figsize=(12, 4))

# Primary axis: Fz
ax1.plot(time_log, loadcell_fz_log, label="Fz (N)", linewidth=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Fz (N)")
ax1.grid(True)

# Secondary axis: My
ax2 = ax1.twinx()
ax2.plot(time_log, loadcell_my_log, color="red", label="My (Nm)", linewidth=2)
ax2.set_ylabel("My (Nm)", color="red")

# Title
plt.title("Loadcell Fz and My Over Time")

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.tight_layout()
plt.savefig("fz_my_over_time.png", dpi=300, bbox_inches="tight")



print("Saved all plots:")
print("- knee_stiffness_plot.png")
print("- knee_damping_plot.png")
print("- knee_theta_plot.png")
print("- encoder_loadcell_outputs_plot.png")
print("- loadcell_forces_moments_plot.png")
