"""
debug_lane4.py - TRAFFIQ AI diagnostic (READ-ONLY: it changes no project files)

Run from the project root folder:
    python debug_lane4.py

It answers three questions:
  1. When did each lane have traffic demand (as the scheduler sees it)?
  2. At every green decision, what did the scheduler see for each lane?
  3. Do the lane polygons overlap each other or the crossing zone?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ANALYTICS = Path("runs/traffiq/lane_detection/lane_analytics.csv")
ADAPTIVE = Path("runs/traffiq/scheduler/adaptive_scheduler_log.csv")
ROIS = Path("config/lane_rois.json")

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]
GDC_THRESHOLD = 45  # must match scheduler.py


def longest_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def flips(flags):
    flags = list(flags)
    return sum(1 for a, b in zip(flags, flags[1:]) if a != b)


def score(queue, waiting, vehicles):
    # Same formula as calculate_score() in scheduler.py
    return queue * 2 + min(waiting, 30) * 0.5 + vehicles * 0.5


# ------------------------------------------------------------
# PART 1: demand per lane, using the scheduler's own snapshot rule
# (latest frame of each whole second)
# ------------------------------------------------------------
print("=" * 70)
print("PART 1: LANE DEMAND AS THE SCHEDULER SEES IT")
print("=" * 70)

if not ANALYTICS.exists():
    print(f"Missing file: {ANALYTICS}")
else:
    an = pd.read_csv(ANALYTICS)
    an["second"] = an["video_time_seconds"].astype(int)
    an["demand"] = (an["queue_length"] > 0) | (an["current_vehicles"] > 0)

    snap = an.sort_values("frame").groupby(["second", "lane"]).tail(1)
    demand = snap.pivot(index="second", columns="lane", values="demand")

    print(f"Seconds in analytics: {len(demand)}  "
          f"(from {demand.index.min()} to {demand.index.max()})\n")

    print(f"{'lane':8} {'sec with demand':>16} {'% of run':>9} "
          f"{'first':>6} {'last':>6} {'longest run':>12} {'on/off flips':>13}")

    longest_any = 0
    for lane in LANES:
        if lane not in demand.columns:
            print(f"{lane:8} NOT FOUND IN CSV")
            continue
        flags = demand[lane].fillna(False).astype(bool)
        secs = flags[flags].index
        first = secs.min() if len(secs) else "-"
        last = secs.max() if len(secs) else "-"
        run = longest_run(flags)
        longest_any = max(longest_any, run)
        print(f"{lane:8} {int(flags.sum()):>16} "
              f"{100 * flags.mean():>8.1f}% {str(first):>6} {str(last):>6} "
              f"{run:>12} {flips(flags):>13}")

    print(f"\nA lane needs {GDC_THRESHOLD} CONTINUOUS seconds of demand "
          f"for starvation prevention (GDC) to trigger.")
    print(f"Longest continuous demand run of any lane: {longest_any} s")

# ------------------------------------------------------------
# PART 2: what the scheduler saw at every green decision
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: EVERY GREEN DECISION")
print("=" * 70)

if not ADAPTIVE.exists():
    print(f"Missing file: {ADAPTIVE}")
else:
    log = pd.read_csv(ADAPTIVE)

    for i in range(len(log)):
        row = log.iloc[i]
        starts_green = row["phase"] == "GREEN" and (
            i == 0 or log.iloc[i - 1]["phase"] != "GREEN"
        )
        if not starts_green:
            continue

        print(f"\nt={int(row['time_seconds'])}s  chose {row['active_lane']}  "
              f"({row['selection_reason']}), green {int(row['green_duration'])}s")

        for lane in LANES:
            queue = row[f"{lane}_queue"]
            vehicles = row[f"{lane}_vehicles"]
            waiting = row[f"{lane}_waiting"]
            # The GDC the selection used = value after the PREVIOUS second
            gdc = 0 if i == 0 else log.iloc[i - 1][f"{lane}_gdc"]
            has_demand = (queue > 0) or (vehicles > 0)
            sc = score(queue, waiting, vehicles) if has_demand else None
            sc_text = f"{sc:5.1f}" if sc is not None else "  n/a"
            mark = "  <== chosen" if lane == row["active_lane"] else ""
            print(f"   {lane}: demand={str(has_demand):5} queue={int(queue)} "
                  f"vehicles={int(vehicles)} wait={waiting:4.1f}s "
                  f"score={sc_text} gdc={int(gdc):2}{mark}")

    print("\nGreen seconds per lane in the adaptive log:")
    greens = log[log["phase"] == "GREEN"].groupby("active_lane").size()
    for lane in LANES:
        print(f"   {lane}: {int(greens.get(lane, 0))} s")

    print("\nMaximum GDC value each lane ever reached:")
    for lane in LANES:
        print(f"   {lane}: {int(log[f'{lane}_gdc'].max())}  "
              f"(threshold {GDC_THRESHOLD})")

# ------------------------------------------------------------
# PART 3: do the lane polygons overlap?
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 3: LANE REGION OVERLAP CHECK")
print("=" * 70)

try:
    import cv2
except ImportError:
    cv2 = None
    print("OpenCV (cv2) not installed - skipping this part.")

if cv2 is not None:
    if not ROIS.exists():
        print(f"Missing file: {ROIS}")
    else:
        roi = json.loads(ROIS.read_text())
        names = LANES + ["crossing_zone"]
        width = int(max(p[0] for n in names for p in roi[n])) + 1
        height = int(max(p[1] for n in names for p in roi[n])) + 1

        masks = {}
        for name in names:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.array(roi[name], np.int32)], 1)
            masks[name] = mask

        print("Region areas (pixels):")
        for name in names:
            print(f"   {name}: {int(masks[name].sum())}")

        print("\nOverlaps (as % of the first region's area):")
        found = False
        for a in names:
            for b in names:
                if a >= b:
                    continue
                overlap = int((masks[a] & masks[b]).sum())
                if overlap > 0:
                    found = True
                    pct = 100 * overlap / max(1, int(masks[a].sum()))
                    pct_b = 100 * overlap / max(1, int(masks[b].sum()))
                    print(f"   {a} & {b}: {overlap} px "
                          f"({pct:.1f}% of {a}, {pct_b:.1f}% of {b})")
        if not found:
            print("   No overlaps found.")

print("\nDone. Copy everything above and paste it back to me.")