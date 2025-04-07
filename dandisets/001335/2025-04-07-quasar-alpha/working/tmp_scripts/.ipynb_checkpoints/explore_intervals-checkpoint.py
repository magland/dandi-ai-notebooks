# This script loads event intervals from the NWB file:
# It extracts counts and timing for Blocks 1-3 and Odors A-F to inform subsequent analysis.
import lindi
import pynwb
import numpy as np

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

lines = []

blocks = ["Block 1", "Block 2", "Block 3"]
for b in blocks:
    tbl = nwb.intervals[b]
    starts = tbl["start_time"][:]
    stops = tbl["stop_time"][:]
    duration = stops - starts
    lines.append(f"{b}: #epochs={len(starts)}; start={starts}; stop={stops}; duration={duration}")

odors = ["Odor A ON", "Odor B ON", "Odor C ON", "Odor D ON", "Odor E ON", "Odor F ON"]
for o in odors:
    tbl = nwb.intervals[o]
    starts = tbl["start_time"][:]
    stops = tbl["stop_time"][:]
    duration = stops - starts
    lines.append(f"{o}: #epochs={len(starts)}; mean_duration={np.mean(duration):.3f}s; first_start={starts[0]:.3f}s")

with open("tmp_scripts/intervals_summary.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")