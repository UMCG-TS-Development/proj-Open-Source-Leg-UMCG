"""
Finite state machine (FSM)-based knee control system for the OpenSourceLeg platform.
It uses ONLY load cell feedback to switch between stance and swing phases and applies
impedance control to the knee actuator in real time.
"""

import numpy as np
import matplotlib.pyplot as plt

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.control.fsm import State, StateMachine
from opensourceleg.logging.logger import Logger
from opensourceleg.robots.osl import OpenSourceLeg
from opensourceleg.sensors.loadcell import DephyLoadcellAmplifier
from opensourceleg.utilities import SoftRealtimeLoop


# ---------------- PARAMETERS ---------------- #

GEAR_RATIO = 9 * (83 / 18)
FREQUENCY = 200

# LOADCELL_CALIBRATION_MATRIX = np.array([
#     (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
#     (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
#     (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
#     (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
#     (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
#     (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
# ])

LOADCELL_CALIBRATION_MATRIX = np.array([
    (-12.59925, -1714.72670, 30.08768, 23.83767, -19.06937, 1591.12752),
    (-20.61383, 973.84169, 32.63392, -1823.09994, 17.86314, 920.63255),
    (-1009.60931, 11.02942, -968.67876, -8.28354, -982.19232, -4.29526),
    (21.04895, 1.39746, 0.04007, -0.84455, -20.68956, -0.57042),
    (-12.32609, 0.49963, 23.45320, 0.46924, -12.07187, -0.73778),
    (-0.19264, -26.63914, 0.27437, -24.21785, 0.50465, -25.58366),
])

BODY_WEIGHT = 30 * 9.8

LOAD_STANCE = 0.25 * BODY_WEIGHT
LOAD_SWING = 0.15 * BODY_WEIGHT


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
        knee_theta=65,
        knee_stiffness=30,
        knee_damping=0.3,
    )

    def stance_to_swing(osl):
        return osl.loadcell.fz > -LOAD_SWING

    def swing_to_stance(osl):
        return osl.loadcell.fz < -LOAD_STANCE

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

    # ---------------- DATA LOGGING BUFFERS ---------------- #

    time_log = []
    k_log = []
    b_log = []
    state_log = []

    # Theta logging
    theta_log = []
    theta_ref_log = []

    # NEW: Load cell force logging
    fz_log = []

    # ---------------- SYSTEM STARTUP ---------------- #

    with osl, fsm:
        print("Initializing system...")

        osl.update()

        print("Homing system...")
        osl.home()

        print("Calibrating loadcell...")
        osl.loadcell.reset()
        osl.loadcell.calibrate()

        print("System ready.")
        input("Press Enter to start knee control...")

        osl.knee.set_control_mode(mode=CONTROL_MODES.IMPEDANCE)
        osl.knee.set_impedance_cc_pidf_gains()
        osl.knee.set_output_impedance()

        print("Control loop started.")

        # ---------------- REAL-TIME LOOP ---------------- #

        try:
            for t in clock:

                osl.update()
                fsm.update(osl=osl)

                k = fsm.current_state.knee_stiffness
                b = fsm.current_state.knee_damping

                # Reference angle
                theta_ref = np.deg2rad(fsm.current_state.knee_theta)

                # Actual knee angle
                theta_actual = osl.knee.output_position

                # Apply impedance control
                osl.knee.set_output_impedance(k=k, b=b)
                osl.knee.set_motor_position(theta_ref)

                # ---------------- FILE LOGGING ---------------- #

                fsm_logger.info(
                    f"T:{t:.3f}, "
                    f"State:{fsm.current_state.name}, "
                    f"Fz:{osl.loadcell.fz:.2f}, "
                    f"ThetaRef:{np.rad2deg(theta_ref):.2f}, "
                    f"ThetaActual:{np.rad2deg(theta_actual):.2f}, "
                    f"K:{k:.2f}, "
                    f"B:{b:.2f}"
                )

                # ---------------- PLOT LOGGING ---------------- #

                time_log.append(t)

                k_log.append(k)
                b_log.append(b)

                state_log.append(fsm.current_state.name)

                # Load cell force logging
                fz_log.append(osl.loadcell.fz)

                # Theta logging
                theta_log.append(np.rad2deg(theta_actual))
                theta_ref_log.append(np.rad2deg(theta_ref))

        except KeyboardInterrupt:
            print("\nControl loop stopped by user.")

    # ---------------- PLOTTING ---------------- #

    time_log = np.array(time_log)
    k_log = np.array(k_log)
    b_log = np.array(b_log)
    state_log = np.array(state_log)

    theta_log = np.array(theta_log)
    theta_ref_log = np.array(theta_ref_log)

    # Load cell force
    fz_log = np.array(fz_log)

    # ---------------- HELPER FUNCTION ---------------- #

    def plot_by_state(y, title, ylabel, filename):

        plt.figure(figsize=(10, 5))

        for i in range(1, len(time_log)):

            style = "-" if state_log[i] == "stance" else "--"

            plt.plot(
                time_log[i-1:i+1],
                y[i-1:i+1],
                style,
                color="black"
            )

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)

        plt.grid(True)

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ---------------- STIFFNESS PLOT ---------------- #

    plot_by_state(
        k_log,
        "Knee Stiffness Over Time",
        "Stiffness (Nm/rad)",
        "knee_stiffness_plot.png"
    )

    # ---------------- DAMPING PLOT ---------------- #

    plot_by_state(
        b_log,
        "Knee Damping Over Time",
        "Damping",
        "knee_damping_plot.png"
    )

    # ---------------- THETA PLOT ---------------- #

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_log,
        theta_log,
        label="Actual Knee Angle",
        linewidth=2
    )

    plt.plot(
        time_log,
        theta_ref_log,
        "--",
        label="Reference Knee Angle",
        linewidth=2
    )

    plt.title("Knee Angle (Theta) Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (degrees)")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        "knee_theta_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ---------------- LOAD CELL FORCE PLOT ---------------- #

    plt.figure(figsize=(12, 6))

    # Plot Fz with different line styles depending on FSM state
    for i in range(1, len(time_log)):

        style = "-" if state_log[i] == "stance" else "--"

        plt.plot(
            time_log[i-1:i+1],
            fz_log[i-1:i+1],
            style,
            color="black"
        )

    # Stance -> Swing threshold
    plt.axhline(
        -LOAD_SWING,
        linestyle=":",
        color="red",
        label=f"Stance → Swing threshold ({-LOAD_SWING:.1f} N)"
    )

    # Swing -> Stance threshold
    plt.axhline(
        -LOAD_STANCE,
        linestyle=":",
        color="blue",
        label=f"Swing → Stance threshold ({-LOAD_STANCE:.1f} N)"
    )

    plt.title("Load Cell Force and FSM State Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Fz (N)")

    plt.grid(True)

    # Legend for FSM states and thresholds
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="Stance"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Swing"
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle=":",
            label=f"Stance → Swing ({-LOAD_SWING:.1f} N)"
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            linestyle=":",
            label=f"Swing → Stance ({-LOAD_STANCE:.1f} N)"
        )
    ]

    plt.legend(handles=legend_elements)

    plt.savefig(
        "loadcell_force_fsm_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Plots saved:")
    print("- knee_stiffness_plot.png")
    print("- knee_damping_plot.png")
    print("- knee_theta_plot.png")
    print("- loadcell_force_fsm_plot.png")
