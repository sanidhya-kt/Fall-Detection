"""
data/sisfall_dataset.py
========================
Parser for the public SisFall dataset (Sucerquia et al., 2017), matching
Table 1 / Section 5.1 description used in the paper:
    "2 accelerometers and 1 gyroscope", 200 Hz, 38 volunteers,
    19 ADL types, 15 Fall types, 4505 samples total.

Expected directory layout after downloading & extracting the official
SisFall release (http://sistemic.udea.edu.co/en/research/projects/english-sisfall/):

    <root>/
        SA01/
            D01_SA01_R01.txt
            D01_SA01_R02.txt
            ...
            F01_SA01_R01.txt
            ...
        SA02/
        ...
        SE01/   (elderly subjects, same naming convention)
        ...

File naming convention:  <ActivityCode>_<SubjectCode>_R<trial>.txt
    ActivityCode starts with 'D' for an ADL ("Daily-living") and 'F' for a Fall.
    The two digits after the letter (e.g. D01, F03) give the specific
    activity/fall TYPE -- this is what the paper calls the fall "type bias"
    used by the counterfactual-intervention experiments (Table 7 / Section 4.3.2).

Row format inside each file: 9 comma-separated integer ADC counts per line,
terminated with a semicolon, sampled at 200 Hz:
    ADXL345_X, ADXL345_Y, ADXL345_Z,
    ITG3200_X, ITG3200_Y, ITG3200_Z,
    MMA8451Q_X, MMA8451Q_Y, MMA8451Q_Z

We convert raw ADC counts to physical units using the official SisFall
resolution constants, then use accelerometer channels 0-2 (ADXL345) as the
reference tri-axial accelerometer for SMV / impact-time detection.
"""
import os
import glob
import re
import numpy as np

from data.preprocessing import build_sample

SISFALL_FS = 200.0  # Hz, both accelerometers and gyroscope

# Official SisFall conversion constants (see dataset README):
#   ADXL345: range +-16g,  resolution 13 bits
#   ITG3200: range +-2000 dps, resolution 16 bits
#   MMA8451Q: range +-8g, resolution 14 bits
_ADXL_SCALE = (2 * 16.0) / (2 ** 13)     # g / count
_ITG_SCALE = (2 * 2000.0) / (2 ** 16)    # dps / count
_MMA_SCALE = (2 * 8.0) / (2 ** 14)       # g / count
_SCALES = np.array([_ADXL_SCALE] * 3 + [_ITG_SCALE] * 3 + [_MMA_SCALE] * 3)

_FNAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subj>S[AE]\d{2})_R(?P<trial>\d{2})\.txt$",
                       re.IGNORECASE)


def _parse_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        text = f.read()
    text = text.replace(";", "")
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [v for v in line.split(",") if v.strip() != ""]
        if len(vals) < 9:
            continue
        rows.append([float(v) for v in vals[:9]])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = arr * _SCALES[None, :]
    return arr


def scan_sisfall(root: str):
    """
    Returns a list of dicts: {'path', 'label' (0/1), 'event_type', 'subject'}
    label = 1 for Fall (code starts with 'F'), 0 for ADL ('D').
    event_type = the raw activity code, e.g. 'F03' -- used by the
    counterfactual-intervention "type bias" experiments.
    """
    items = []
    for path in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        fname = os.path.basename(path)
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        code = m.group("code").upper()
        label = 1 if code.startswith("F") else 0
        items.append({
            "path": path,
            "label": label,
            "event_type": code,
            "subject": m.group("subj").upper(),
        })
    if not items:
        raise FileNotFoundError(
            f"No SisFall .txt trial files found under '{root}'. "
            "Check that you pointed --sisfall_root at the extracted dataset "
            "(it should contain SA01/, SA02/, ..., SE01/, ... subfolders).")
    return items


def load_sisfall_samples(root: str, cfg):
    """
    Walk the whole dataset, clean + resample + segment every trial, and
    return a list of sample dicts ready to be handed to a torch Dataset:
        {'X': [C, delta] float32, 'Y_impact': [C, delta] float32,
         'label': int, 'event_type': str, 'subject': str}
    """
    items = scan_sisfall(root)
    samples = []
    for it in items:
        raw = _parse_file(it["path"])
        if raw.shape[0] < 10:
            continue
        built = build_sample(raw, SISFALL_FS, cfg)
        if built is None:
            continue
        samples.append({
            "X": built["X"],
            "Y_impact": built["Y_impact"],
            "label": it["label"],
            "event_type": it["event_type"],
            "subject": it["subject"],
        })
    print(f"[SisFall] kept {len(samples)}/{len(items)} trials with full falling windows")
    return samples
