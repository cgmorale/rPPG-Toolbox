#!/usr/bin/env python3
import argparse
import os
import glob
import csv
from typing import Any, Dict, Tuple, List, Optional

# --- Robust .mat loader (scipy first, h5py fallback) ---
def load_mat(path: str) -> Dict[str, Any]:
    try:
        import scipy.io as sio
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        # Likely a v7.3 MAT file; use h5py
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def _read_obj(obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    # Convert bytes/char arrays to str
                    if isinstance(data, (bytes, bytearray)):
                        return data.decode("utf-8", errors="ignore")
                    # MATLAB char arrays often come as uint16/uint8
                    if hasattr(data, "dtype") and str(data.dtype).startswith("uint"):
                        try:
                            return "".join(map(chr, data.flatten()))
                        except Exception:
                            pass
                    # Squeeze 1-d arrays
                    try:
                        import numpy as np
                        data = data.squeeze()
                        if data.shape == ():
                            data = data.item()
                    except Exception:
                        pass
                    return data
                elif isinstance(obj, h5py.Group):
                    return {k: _read_obj(obj[k]) for k in obj.keys()}
                else:
                    return obj
            for k in f.keys():
                out[k] = _read_obj(f[k])
        return out

# --- Utilities to coerce MATLAB-ish values into Python strings/bools ---
def to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    # MATLAB may give arrays of chars or nested arrays
    try:
        import numpy as np
        if hasattr(x, "dtype") and str(x.dtype).startswith("uint"):
            try:
                return "".join(map(chr, x.flatten()))
            except Exception:
                pass
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return to_str(x.item())
            # Sometimes nested char arrays like array(['LED-low'], dtype='<U7')
            if x.dtype.kind in ("U", "S"):
                return str(x.tolist()[0])
    except Exception:
        pass
    # Fallback to str()
    return str(x)

def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = to_str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    # If it's numeric-like
    try:
        return float(s) != 0.0
    except Exception:
        return False

def to_int(x: Any) -> int:
    # skin_color is already numeric in MMPD (3,4,5,6), but be robust
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.squeeze()
            if x.shape == ():
                x = x.item()
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        # sometimes comes as string like '4'
        try:
            return int(float(to_str(x)))
        except Exception:
            raise ValueError(f"Cannot convert value to int: {x!r}")

# --- Your exact mapping from get_information() ---
def encode_fields(info: Tuple[str, str, bool, int, str, bool, bool, bool]) -> Dict[str, int]:
    light_s, motion_s, exercise_b, skin_int, gender_s, glasser_b, hair_b, makeup_b = info

    # Light
    light_map = {
        "led-low": 1,
        "led-high": 2,
        "incandescent": 3,
        "nature": 4,
    }
    L = light_map.get(light_s.lower(), 0)

    # Motion
    if motion_s in ("Stationary", "Stationary (after exercise)"):
        MO = 1
    elif motion_s == "Rotation":
        MO = 2
    elif motion_s == "Talking":
        MO = 3
    elif motion_s == "Walking":
        MO = 4
    else:
        MO = 0

    E = 1 if exercise_b else 2
    S = int(skin_int)  # already numeric (3–6)

    GE = 1 if gender_s.lower() == "male" else (2 if gender_s.lower() == "female" else 0)
    GL = 1 if glasser_b else 2
    H  = 1 if hair_b else 2
    MA = 1 if makeup_b else 2

    return dict(L=L, MO=MO, E=E, S=S, GE=GE, GL=GL, H=H, MA=MA)

# --- Extract canonical fields from a loaded .mat dict ---
def extract_fields(mat: Dict[str, Any]) -> Tuple[str, str, bool, int, str, bool, bool, bool]:
    # Keys per your dataset description:
    # 'light','motion','exercise','skin_color','gender','glasser','hair_cover','makeup'
    light = to_str(mat.get("light", ""))
    motion = to_str(mat.get("motion", ""))
    exercise = to_bool(mat.get("exercise", False))
    skin_color = to_int(mat.get("skin_color", 0))
    gender = to_str(mat.get("gender", ""))
    glasser = to_bool(mat.get("glasser", False))
    hair_cover = to_bool(mat.get("hair_cover", False))
    makeup = to_bool(mat.get("makeup", False))
    return (light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup)

# --- Compose the processed filename pattern you use ---
def processed_glob(processed_root: str, subject_id: int, codes: Dict[str, int]) -> List[str]:
    # Example: subject17_L2_MO4_E2_S4_GE1_GL2_H2_MA2_input0.npy
    pattern = (
        f"subject{subject_id}_"
        f"L{codes['L']}_MO{codes['MO']}_E{codes['E']}_S{codes['S']}_"
        f"GE{codes['GE']}_GL{codes['GL']}_H{codes['H']}_MA{codes['MA']}_*.npy"
    )
    # Files live somewhere under a long config folder; search recursively
    return glob.glob(os.path.join(processed_root, "**", pattern), recursive=True)

# --- Parse px_y.mat pieces from path ---
def parse_mat_identity(mat_path: str) -> Tuple[int, int]:
    # mat_path like .../subject17/p17_12.mat  (or p1_19.mat under subject1)
    base = os.path.basename(mat_path)
    # Formats seen: p<subject>_<exp>.mat  (e.g., p17_12.mat)
    # but safer: derive subject from directory name "subject<id>"
    try:
        subject_dir = os.path.basename(os.path.dirname(mat_path))
        assert subject_dir.startswith("subject")
        sid = int(subject_dir.replace("subject", ""))
    except Exception:
        sid = -1

    name_no_ext = os.path.splitext(base)[0]
    # exp index is after underscore
    try:
        exp_idx = int(name_no_ext.split("_")[1])
    except Exception:
        exp_idx = -1

    return sid, exp_idx

def main():
    ap = argparse.ArgumentParser(description="Map MMPD .mat files to processed .npy paths by metadata codes.")
    ap.add_argument("--raw-root", required=True,
                    help="Root folder containing MMPD_Dataset/subject*/pX_Y.mat")
    ap.add_argument("--processed-root", required=True,
                    help="Root folder containing processed_dataset/.../subject*_L*_MO*_..._*.npy")
    ap.add_argument("--out-csv", default="mmpd_mat_to_npy_mapping.csv",
                    help="Where to write the mapping CSV.")
    args = ap.parse_args()

    mat_paths = sorted(glob.glob(os.path.join(args.raw_root, "subject*", "p*.mat")))
    if not mat_paths:
        print("No .mat files found. Check --raw-root.")
        return

    out_rows = []
    for mp in mat_paths:
        try:
            mat = load_mat(mp)
            fields = extract_fields(mat)
            codes = encode_fields(fields)
            sid, exp_idx = parse_mat_identity(mp)
            npys = processed_glob(args.processed_root, sid, codes)

            out_rows.append({
                "mat_path": mp,
                "subject_id": sid,
                "exp_idx": exp_idx,
                "light": fields[0],
                "motion": fields[1],
                "exercise": fields[2],
                "skin_color": fields[3],
                "gender": fields[4],
                "glasser": fields[5],
                "hair_cover": fields[6],
                "makeup": fields[7],
                "L": codes["L"],
                "MO": codes["MO"],
                "E": codes["E"],
                "S": codes["S"],
                "GE": codes["GE"],
                "GL": codes["GL"],
                "H": codes["H"],
                "MA": codes["MA"],
                "matched_npy_count": len(npys),
                "matched_npys": " | ".join(npys),
            })
        except Exception as e:
            out_rows.append({
                "mat_path": mp,
                "error": repr(e),
            })

    # Write CSV
    fieldnames = [
        "mat_path","subject_id","exp_idx",
        "light","motion","exercise","skin_color","gender","glasser","hair_cover","makeup",
        "L","MO","E","S","GE","GL","H","MA",
        "matched_npy_count","matched_npys","error"
    ]
    # Ensure all keys exist in rows
    for r in out_rows:
        for k in fieldnames:
            r.setdefault(k, "")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Quick summary
    total = len(out_rows)
    matched = sum(1 for r in out_rows if str(r.get("matched_npy_count","0")) not in ("","0"))
    print(f"Done. {matched}/{total} .mat files matched at least one processed .npy.")
    print(f"CSV written to: {args.out_csv}")

if __name__ == "__main__":
    main()
