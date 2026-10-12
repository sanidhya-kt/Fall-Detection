# Local dataset paths (Mac)

These are the datasets found on your machine and wired as defaults in `config.py`.

## SisFall
`/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/sisfall`

- Subjects: `SA01`… / `SE01`…
- Files: `Dxx_SAxx_Rxx.txt` (ADL) / `Fxx_SAxx_Rxx.txt` (Fall)
- ~4505 trials

## KFall (preferred full release)
`/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/kfall_download/extracted/KFall Dataset/KFall Dataset`

- `sensor_data/SAxx/*.csv`
- `label_data/SAxx_label.xlsx` (impact frames)
- 5075 trials = **2729 ADL + 2346 Fall** (matches paper)

## KFall fallback (flat sensors)
`/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/kfall`

Labels fallback:
`.../KFall Dataset/label_data`

## Out of scope
SlowFall — private; not used.
