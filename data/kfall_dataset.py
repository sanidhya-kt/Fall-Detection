"""
data/kfall_dataset.py
=======================
Parser for the public KFall dataset (Yu, Jang & Xiong, 2021).

Expected directory layout:

    <root>/
        sensor_data/
            SA06/
                S06T01R01.csv
                ...
        label_data/
            SA06_label.xlsx
            ...

If `sensor_data/` is missing, `<root>` itself is scanned for CSVs (agentic_fall
flat layout). Label dir can be passed separately via `label_dir=` or is taken
as `<root>/label_data`.
"""
import os
import glob
import re
import numpy as np
import pandas as pd

from data.preprocessing import (
    build_sample,
    clean_outliers,
    resample_signal,
    extract_falling_window,
    extract_impact_window,
)

KFALL_FS = 100.0  # Hz

_ACC_ALIASES = [["accx", "acc_x", "ax"], ["accy", "acc_y", "ay"], ["accz", "acc_z", "az"]]
_GYR_ALIASES = [["gyrx", "gyr_x", "gx"], ["gyry", "gyr_y", "gy"], ["gyrz", "gyr_z", "gz"]]
_EUL_ALIASES = [["eulerx", "euler_x", "roll"], ["eulery", "euler_y", "pitch"],
                ["eulerz", "euler_z", "yaw"]]

_FNAME_RE = re.compile(r"S(?P<subj>\d+)T(?P<task>\d+)R(?P<trial>\d+)\.csv$", re.IGNORECASE)
# Label sheet cells look like "F01 (20)" / "D01 (1)" / bare ints.
_TASK_ID_RE = re.compile(r"\((\d+)\)")
_TASK_BARE_RE = re.compile(r"^\s*(\d+)\s*$")


def _norm_col(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
        .replace("_", "")
        .replace("-", "")
    )


def _pick_col(columns, aliases):
    """Match columns by normalized name; also allow alias-as-substring."""
    lower = {_norm_col(c): c for c in columns}
    for alias_group in aliases:
        for a in alias_group:
            key = _norm_col(a)
            if key in lower:
                return lower[key]
            # e.g. alias "taskcode" matches normalized "taskcodetaskid"
            for nk, original in lower.items():
                if key and key in nk:
                    return original
    return None


def _parse_task_id(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    m = _TASK_ID_RE.search(s)
    if m:
        return int(m.group(1))
    m = _TASK_BARE_RE.match(s)
    if m:
        return int(m.group(1))
    # Fallback: trailing digits, e.g. "T20"
    digits = re.findall(r"\d+", s)
    if digits:
        return int(digits[-1])
    return None


def _load_trial_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = list(df.columns)
    chosen = []
    for group in (_ACC_ALIASES, _GYR_ALIASES, _EUL_ALIASES):
        for aliases in group:
            c = _pick_col(cols, [aliases])
            if c is None:
                raise ValueError(
                    f"Could not find a column matching {aliases} in {path}. "
                    f"Available columns: {cols}"
                )
            chosen.append(c)
    return df[chosen].to_numpy(dtype=np.float64)


def _load_label_table(label_dir: str, subj_id: str):
    """
    Returns dict keyed by (task_id, trial_id) ->
        {'task': int, 'is_fall': bool, 'onset': int|None, 'impact': int|None}

    KFall label sheets forward-fill Task Code across trials of the same fall
    type (first row has "F01 (20)", next rows have NaN for the same fall).
    Fall tasks are IDs 20..34 (F01-F15); ADLs are 1..19 (+ any <20).
    """
    candidates = (
        glob.glob(os.path.join(label_dir, f"*{subj_id}*.xlsx"))
        + glob.glob(os.path.join(label_dir, f"*{subj_id}*.xls"))
        + glob.glob(os.path.join(label_dir, f"*{subj_id}*.csv"))
    )
    if not candidates:
        return {}
    path = candidates[0]
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    cols = list(df.columns)

    task_col = _pick_col(cols, [["taskcode", "taskid", "task_id", "task"]])
    trial_col = _pick_col(cols, [["trialid", "trial_id", "trial"]])
    onset_col = _pick_col(cols, [["fall_onset_frame", "fallonsetframe", "onsetframe"]])
    impact_col = _pick_col(cols, [["fall_impact_frame", "fallimpactframe", "impactframe"]])

    if task_col is not None:
        # Forward-fill "F01 (20)" across NaN rows belonging to the same fall type.
        df[task_col] = df[task_col].ffill()

    table = {}
    for _, row in df.iterrows():
        task = _parse_task_id(row[task_col]) if task_col else None
        try:
            trial = int(row[trial_col]) if trial_col is not None and pd.notna(row[trial_col]) else None
        except (TypeError, ValueError):
            trial = None
        onset = impact = None
        if onset_col is not None and pd.notna(row.get(onset_col, np.nan)):
            onset = int(row[onset_col])
        if impact_col is not None and pd.notna(row.get(impact_col, np.nan)):
            impact = int(row[impact_col])
        if task is None or trial is None:
            continue
        # Fall tasks are F01-F15 == Task IDs 20..34. T35/T36 are extra ADLs.
        is_fall = 20 <= task <= 34
        table[(task, trial)] = {
            "task": task,
            "is_fall": is_fall,
            "onset": onset,
            "impact": impact,
        }
    return table


def scan_kfall(root: str, label_dir: str = None):
    sensor_dir = os.path.join(root, "sensor_data")
    if label_dir is None:
        label_dir = os.path.join(root, "label_data")
    if not os.path.isdir(sensor_dir):
        sensor_dir = root
    items = []
    for path in glob.glob(os.path.join(sensor_dir, "**", "*.csv"), recursive=True):
        fname = os.path.basename(path)
        m = _FNAME_RE.search(fname)
        if not m:
            continue
        subj = f"SA{int(m.group('subj')):02d}"
        task = int(m.group("task"))
        trial = int(m.group("trial"))
        items.append({"path": path, "subject": subj, "task": task, "trial": trial})
    if not items:
        raise FileNotFoundError(
            f"No KFall trial CSVs found under '{root}'. "
            "Expected 'sensor_data/SAxx/SxxTyyRzz.csv' or flat SAxx folders."
        )
    return items, label_dir


def load_kfall_samples(root: str, cfg, label_dir: str = None):
    items, label_dir = scan_kfall(root, label_dir=label_dir)
    label_cache = {}
    samples = []
    n_label_hits = 0
    for it in items:
        subj = it["subject"]
        if subj not in label_cache:
            label_cache[subj] = _load_label_table(label_dir, subj)
        labels = label_cache[subj]

        task = it["task"]
        trial = it["trial"]
        # KFall: T01-T19 + T35-T36 = 21 ADLs; T20-T34 = 15 falls (paper Sec 5.1).
        is_fall = 20 <= task <= 34
        onset = impact = None
        entry = labels.get((task, trial))
        if entry is not None:
            n_label_hits += 1
            is_fall = entry["is_fall"]
            onset, impact = entry["onset"], entry["impact"]

        raw = _load_trial_csv(it["path"])
        if raw.shape[0] < 10:
            continue

        if is_fall and impact is not None and 0 <= impact < raw.shape[0]:
            sig = clean_outliers(raw, cfg.outlier_clip_value, cfg.median_filter_kernel)
            method = getattr(cfg, "resample_method", "decimate")
            sig = resample_signal(sig, KFALL_FS, cfg.fs, method=method)
            scale = cfg.fs / KFALL_FS
            t_impact = int(round(impact * scale))
            allow_pad = bool(getattr(cfg, "allow_window_pad", False))
            x_window = extract_falling_window(
                sig, t_impact, cfg.theta, cfg.delta, allow_pad=allow_pad
            )
            if x_window is None:
                continue
            y_window = extract_impact_window(sig, t_impact, cfg.delta)
            built = {
                "X": x_window.astype(np.float32).T,
                "Y_impact": y_window.astype(np.float32).T,
            }
        else:
            built = build_sample(raw, KFALL_FS, cfg)
            if built is None:
                continue

        samples.append({
            "X": built["X"],
            "Y_impact": built["Y_impact"],
            "label": int(is_fall),
            "event_type": f"T{task:02d}",
            "subject": subj,
        })
    print(
        f"[KFall] label-sheet matches: {n_label_hits}/{len(items)} "
        f"(falls={sum(s['label'] for s in samples)}, "
        f"adls={len(samples) - sum(s['label'] for s in samples)})"
    )
    return samples
