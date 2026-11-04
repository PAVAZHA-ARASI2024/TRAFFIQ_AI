
import os
import pandas as pd

# ---------------- CONFIGURATION ----------------

INPUT_CSV = "runs/traffiq/lane_detection/lane_analytics.csv"
OUTPUT_DIR = "runs/traffiq/scheduler"

MIN_GREEN = 10
MAX_GREEN = 25
YELLOW_TIME = 3
ALL_RED_TIME = 1

GDC_THRESHOLD = 20

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- LOAD ANALYTICS ----------------

df = pd.read_csv(INPUT_CSV)

df["second"] = df["video_time_seconds"].astype(int)

# Keep the latest available measurement for each lane in each second
snapshots = {}

for second, group in df.groupby("second"):
    lane_data = group.sort_values("frame").groupby("lane").tail(1)
    snapshots[second] = lane_data.set_index("lane").to_dict("index")


# ---------------- SCHEDULER VARIABLES ----------------

gdc = {lane: 0 for lane in LANES}

current_lane = None
phase = "ALL_RED"
remaining_time = 0
green_duration = 0

scheduler_log = []


# ---------------- PRIORITY CALCULATION ----------------

def calculate_score(metrics):

    queue = metrics["queue_length"]
    vehicles = metrics["current_vehicles"]
    waiting = metrics["average_waiting_seconds"]

    score = (
        queue * 2
        + min(waiting, 30) * 0.5
        + vehicles * 0.5
    )

    return score


def has_demand(metrics):
    return (
        metrics["queue_length"] > 0
        or metrics["current_vehicles"] > 0
    )


def select_next_lane(lane_metrics):

    # Consider only lanes with actual traffic demand
    waiting_lanes = [
        lane for lane in LANES
        if has_demand(lane_metrics[lane])
    ]

    if not waiting_lanes:
        return None, "No traffic demand"

    # Prevent starvation only for lanes that have traffic
    starved_lanes = [
        lane for lane in waiting_lanes
        if gdc[lane] >= GDC_THRESHOLD
    ]

    if starved_lanes:
        selected = max(starved_lanes, key=lambda lane: gdc[lane])
        reason = "GDC starvation prevention"
    else:
        scores = {
            lane: calculate_score(lane_metrics[lane])
            for lane in waiting_lanes
        }

        selected = max(scores, key=scores.get)
        reason = "Highest traffic priority"

    return selected, reason

# ---------------- ADAPTIVE GREEN TIME ----------------

def calculate_green_time(metrics):

    queue = metrics["queue_length"]
    waiting = metrics["average_waiting_seconds"]

    duration = 10 + (queue * 2) + (waiting * 0.3)

    return int(max(MIN_GREEN, min(MAX_GREEN, duration)))



# ---------------- SIMULATION ----------------

start_second = min(snapshots.keys())
end_second = max(snapshots.keys())

selection_reason = "No traffic demand"

for second in range(start_second, end_second + 1):

    # Get latest available traffic snapshot
    available_seconds = [s for s in snapshots if s <= second]

    if not available_seconds:
        continue

    snapshot_second = max(available_seconds)
    lane_metrics = snapshots[snapshot_second]

    if not all(lane in lane_metrics for lane in LANES):
        continue

    # Start next signal phase when timer reaches zero
    if remaining_time == 0:

        if phase == "GREEN":
            phase = "YELLOW"
            remaining_time = YELLOW_TIME

        elif phase == "YELLOW":
            phase = "ALL_RED"
            remaining_time = ALL_RED_TIME

        else:
            current_lane, selection_reason = select_next_lane(
                lane_metrics
            )

            if current_lane is not None:

                green_duration = calculate_green_time(
                    lane_metrics[current_lane]
                )

                phase = "GREEN"
                remaining_time = green_duration

            else:
                # No vehicles waiting: keep all signals red
                phase = "ALL_RED"
                remaining_time = 1
                green_duration = 0

    # Update GDC counters
    for lane in LANES:

        if not has_demand(lane_metrics[lane]):
            gdc[lane] = 0

        elif phase == "GREEN" and lane == current_lane:
            gdc[lane] = 0

        else:
            gdc[lane] += 1

    # Record current signal state
    row = {
        "time_seconds": second,
        "phase": phase,
        "active_lane": (
            current_lane if phase in ["GREEN", "YELLOW"] else "NONE"
            ),
        "remaining_time": remaining_time,
        "green_duration": green_duration,
        "selection_reason": (
            selection_reason if phase == "GREEN" else ""
        )
    }

    for lane in LANES:

        metrics = lane_metrics[lane]

        row[f"{lane}_queue"] = metrics["queue_length"]
        row[f"{lane}_vehicles"] = metrics["current_vehicles"]
        row[f"{lane}_waiting"] = metrics["average_waiting_seconds"]
        row[f"{lane}_gdc"] = gdc[lane]

    scheduler_log.append(row)

    remaining_time -= 1

# ---------------- SAVE RESULTS ----------------

result = pd.DataFrame(scheduler_log)

output_file = os.path.join(
    OUTPUT_DIR,
    "adaptive_scheduler_log.csv"
)

result.to_csv(output_file, index=False)

print("\nHYBRID ADAPTIVE SCHEDULER COMPLETED")
print("-----------------------------------")
print("Total simulated seconds:", len(result))
print("Signal phases:", result["phase"].value_counts().to_dict())
print("Output:", output_file)

print("\nFirst 15 signal decisions:")
print(
    result[
        ["time_seconds", "phase", "active_lane",
         "remaining_time", "selection_reason"]
    ].head(15).to_string(index=False)
)
