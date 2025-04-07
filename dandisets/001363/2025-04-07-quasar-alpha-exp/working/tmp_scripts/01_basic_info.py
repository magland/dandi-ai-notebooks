# This script loads the NWB file and prints basic metadata, dimensions, and structure info.

import pynwb
import lindi

url = "https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Start time: {nwb.session_start_time}")

subject = nwb.subject
print(f"Subject ID: {subject.subject_id}")
print(f"Species: {subject.species}")
print(f"Sex: {subject.sex}")
print(f"Age: {subject.age}")
print(f"Description: {subject.description}")

acq_keys = list(nwb.acquisition.keys())
intervals_keys = list(nwb.intervals.keys())
print(f"Acquisition keys: {acq_keys}")
print(f"Intervals keys: {intervals_keys}")

electrical_series = nwb.acquisition.get("ElectricalSeries", None)
if electrical_series is not None:
    data = electrical_series.data
    print(f"ElectricalSeries data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Sampling rate: {electrical_series.rate} Hz")
else:
    print("No ElectricalSeries found.")

if "trials" in nwb.intervals:
    trials = nwb.intervals["trials"]
    print(f"Number of trials: {len(trials['id'])}")
else:
    print("No trials found.")