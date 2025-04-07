# This script loads the NWB file and extracts basic session and subject metadata.
# The goal is to generate information for notebook documentation and verify file access.
import lindi
import pynwb

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

lines = []

lines.append(f"Session description: {nwb.session_description}")
lines.append(f"Identifier: {nwb.identifier}")
lines.append(f"Session start time: {nwb.session_start_time}")
lines.append(f"Experimenter(s): {nwb.experimenter}")
lines.append(f"Experiment description: {nwb.experiment_description}")
lines.append(f"Institution: {nwb.institution}")
lines.append(f"Lab: {nwb.lab}")
lines.append(f"Keywords: {nwb.keywords}")
lines.append("")
lines.append("Subject info:")
subj = nwb.subject
lines.append(f"  subject_id: {subj.subject_id}")
lines.append(f"  species: {subj.species}")
lines.append(f"  sex: {subj.sex}")
lines.append(f"  age: {subj.age}")
lines.append(f"  description: {subj.description}")

with open("tmp_scripts/metadata_summary.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")