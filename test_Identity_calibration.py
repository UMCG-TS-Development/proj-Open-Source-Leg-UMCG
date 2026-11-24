
import numpy as np
import matplotlib.pyplot as plt

from opensourceleg.robots.osl import OpenSourceLeg
from opensourceleg.sensors.loadcell import DephyLoadcellAmplifier
from opensourceleg.utilities import SoftRealtimeLoop


# ---------------- PARAMETERS ----------------

FREQUENCY = 100

# False = Original calibration
# True  = Identity matrix
USE_IDENTITY_MATRIX = False

ORIGINAL_MATRIX = np.array([
    (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
    (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
    (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
    (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
    (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
    (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070)
])

CALIBRATION_MATRIX = (
    np.eye(6) if USE_IDENTITY_MATRIX
    else ORIGINAL_MATRIX
)


# ---------------- LOADCELL ----------------

sensors = {
    "loadcell": DephyLoadcellAmplifier(
        calibration_matrix=CALIBRATION_MATRIX,
        tag="loadcell",
        amp_gain=125,
        exc=5,
        bus=1,
        i2c_address=102,
    )
}

osl = OpenSourceLeg(
    tag="osl",
    actuators={},
    sensors=sensors
)

clock = SoftRealtimeLoop(dt=1/FREQUENCY)


# ---------------- PLOT SETUP ----------------

plt.ion()

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

fig.suptitle("Live Loadcell Measurements", fontsize=15)

# Forces
force_lines = []
for label in ["Fx", "Fy", "Fz"]:
    line, = axs[0].plot([], [], label=label)
    force_lines.append(line)

axs[0].set_title("Forces")
axs[0].set_ylabel("Force (N)")
axs[0].legend()
axs[0].grid(True)

# Moments
moment_lines = []
for label in ["Mx", "My", "Mz"]:
    line, = axs[1].plot([], [], label=label)
    moment_lines.append(line)

axs[1].set_title("Moments")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Moment (Nm)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show(block=False)


# ---------------- DATA STORAGE ----------------

time_log = []

fx_log = []
fy_log = []
fz_log = []

mx_log = []
my_log = []
mz_log = []


# ---------------- MEASUREMENT ----------------

with osl:

    print("\nInitializing loadcell...")

    osl.update()

    print("Resetting and calibrating loadcell...")
    osl.loadcell.reset()
    osl.loadcell.calibrate()

    input("\nPress ENTER to start measuring...")

    print("\nMeasuring loadcell. Press CTRL+C to stop.\n")

    try:
        for i, t in enumerate(clock):

            osl.update()

            # Read forces
            fx = osl.loadcell.fx
            fy = osl.loadcell.fy
            fz = osl.loadcell.fz

            # Read moments
            mx = osl.loadcell.mx
            my = osl.loadcell.my
            mz = osl.loadcell.mz

            # Store data
            time_log.append(t)

            fx_log.append(fx)
            fy_log.append(fy)
            fz_log.append(fz)

            mx_log.append(mx)
            my_log.append(my)
            mz_log.append(mz)

            # Print measurements
            print(
                f"Time: {t:6.2f} s | "
                f"Fx: {fx:8.2f} | "
                f"Fy: {fy:8.2f} | "
                f"Fz: {fz:8.2f} | "
                f"Mx: {mx:8.2f} | "
                f"My: {my:8.2f} | "
                f"Mz: {mz:8.2f}",
                end="\r"
            )

            # Update plots every 10 samples (10 Hz)
            if i % 10 == 0:

                force_data = [
                    fx_log, fy_log, fz_log
                ]

                moment_data = [
                    mx_log, my_log, mz_log
                ]

                for line, data in zip(force_lines, force_data):
                    line.set_data(time_log, data)

                for line, data in zip(moment_lines, moment_data):
                    line.set_data(time_log, data)

                # Show the last 10 seconds
                current_time = t
                start_time = max(0, current_time - 10)

                for ax in axs:
                    ax.set_xlim(start_time, max(10, current_time))
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n\nMeasurement stopped.")


# ---------------- FINAL PLOT ----------------

plt.ioff()

# Display all recorded measurements
for line, data in zip(
    force_lines,
    [fx_log, fy_log, fz_log]
):
    line.set_data(time_log, data)

for line, data in zip(
    moment_lines,
    [mx_log, my_log, mz_log]
):
    line.set_data(time_log, data)

for ax in axs:
    ax.set_xlim(
        min(time_log) if time_log else 0,
        max(time_log) if time_log else 10
    )
    ax.relim()
    ax.autoscale_view()

plt.tight_layout()

plt.savefig(
    "loadcell_test_results.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nPlot saved as loadcell_test_results.png")

plt.show()
