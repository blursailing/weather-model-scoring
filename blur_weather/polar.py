"""Parse and interpolate Expedition polar files.

Used for performance analysis — comparing actual BSP/VMG against target.
Not directly needed for weather scoring, but part of the full system.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PolarPoint:
    tws: float
    twa: float
    bsp: float


def parse_expedition_polar(filepath: str) -> list[dict]:
    """Parse an Expedition polar file.
    
    Format: tab-separated, first column is TWS, then TWA/BSP pairs.
    First line is "!Expedition polar"
    
    Returns:
        List of dicts: [{"tws": 6, "points": [(42, 5.4), (52, 5.92), ...]}]
    """
    filepath = Path(filepath)
    polar_curves = []
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("!") or not line:
            continue
        
        parts = line.split("\t")
        tws = float(parts[0])
        
        points = []
        i = 1
        while i < len(parts) - 1:
            twa = float(parts[i])
            bsp = float(parts[i + 1])
            if twa > 0:
                points.append((twa, bsp))
            i += 2
        
        polar_curves.append({"tws": tws, "points": points})
    
    return polar_curves


def interpolate_polar(polar_curves: list[dict], tws: float, twa: float) -> float:
    """Interpolate target BSP for given TWS and TWA.
    
    Uses bilinear interpolation between the two nearest TWS curves
    and TWA points on each curve.
    """
    twa = abs(twa)  # Polars are symmetric
    
    # Find bracketing TWS curves
    tws_values = [c["tws"] for c in polar_curves]
    
    if tws <= tws_values[0]:
        curve_lo = curve_hi = polar_curves[0]
        tws_frac = 0
    elif tws >= tws_values[-1]:
        curve_lo = curve_hi = polar_curves[-1]
        tws_frac = 0
    else:
        for i in range(len(tws_values) - 1):
            if tws_values[i] <= tws <= tws_values[i + 1]:
                curve_lo = polar_curves[i]
                curve_hi = polar_curves[i + 1]
                tws_frac = (tws - tws_values[i]) / (tws_values[i + 1] - tws_values[i])
                break
    
    # Interpolate on each curve
    def interp_curve(curve, twa_target):
        points = curve["points"]
        twas = [p[0] for p in points]
        bsps = [p[1] for p in points]
        
        if twa_target <= twas[0]:
            return bsps[0]
        if twa_target >= twas[-1]:
            return bsps[-1]
        
        for i in range(len(twas) - 1):
            if twas[i] <= twa_target <= twas[i + 1]:
                frac = (twa_target - twas[i]) / (twas[i + 1] - twas[i])
                return bsps[i] + frac * (bsps[i + 1] - bsps[i])
        
        return bsps[-1]
    
    bsp_lo = interp_curve(curve_lo, twa)
    bsp_hi = interp_curve(curve_hi, twa)
    
    return bsp_lo + tws_frac * (bsp_hi - bsp_lo)
