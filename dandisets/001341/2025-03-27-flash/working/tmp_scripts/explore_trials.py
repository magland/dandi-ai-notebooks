import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/5738ae8a-dd82-425b-8966-bbcfd35649a7/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Get the trials data
    trials = nwb.intervals["trials"]
    start_time = trials["start_time"].data[:]
    stop_time = trials["stop_time"].data[:]

    # Create a plot of start_time vs stop_time
    plt.figure(figsize=(10, 6))
    plt.plot(start_time, stop_time, marker='o', linestyle='-', markersize=2)
    plt.xlabel("Start Time (s)")
    plt.ylabel("Stop Time (s)")
    plt.title("Trial Start Time vs Stop Time")
    plt.grid(True)

    # Save the plot to a PNG file
    plt.savefig("tmp_scripts/trial_start_stop.png")
    print("Plot saved to tmp_scripts/trial_start_stop.png")

except Exception as e:
    print(f"Error: {e}")