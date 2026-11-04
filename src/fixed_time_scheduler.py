
import pandas as pd
import math
from pathlib import Path

# --------------------------------
# 1. CONFIGURATION
# --------------------------------

INPUT_FILE = Path("runs/traffiq/lane_detection/lane_analytics.csv")

OUTPUT_DIR = Path("runs/traffiq/scheduler")
OUTPUT_FILE = OUTPUT_DIR / "fixed_time_scheduler_log.csv"

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]

GREEN_TIME = 15
YELLOW_TIME = 3
ALL_RED_TIME = 1

# --------------------------------
# 2. READ TRAFFIC ANALYTICS
# --------------------------------

print("Starting Fixed-Time Traffic Scheduler...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Analytics CSV not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = ["video_time_seconds", "lane"]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing from analytics CSV.")

total_seconds = max(1, math.ceil(df["video_time_seconds"].max()))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 3. FIXED-TIME SCHEDULER
# --------------------------------

cycle_time = GREEN_TIME + YELLOW_TIME + ALL_RED_TIME
full_cycle_time = cycle_time * len(LANES)

results = []

for second in range(total_seconds):

    # Find the current position in the repeating cycle
    cycle_position = second % full_cycle_time

    # Identify the lane being served
    lane_index = cycle_position // cycle_time
    lane_position = cycle_position % cycle_time

    current_lane = LANES[lane_index]

    # Determine signal phase
    if lane_position < GREEN_TIME:

        phase = "GREEN"
        active_lane = current_lane
        remaining_time = GREEN_TIME - lane_position

    elif lane_position < GREEN_TIME + YELLOW_TIME:

        phase = "YELLOW"
        active_lane = current_lane
        remaining_time = (
            GREEN_TIME + YELLOW_TIME - lane_position
        )

    else:

        phase = "ALL_RED"
        active_lane = "NONE"
        remaining_time = ALL_RED_TIME

    results.append({
        "time_seconds": second,
        "phase": phase,
        "active_lane": active_lane,
        "remaining_time": remaining_time,
        "selection_reason": "Fixed round-robin schedule"
    })

# --------------------------------
# 4. SAVE SCHEDULER RESULTS
# --------------------------------

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------
# 5. DISPLAY RESULTS
# --------------------------------

print("\n========== FIXED-TIME SCHEDULER RESULTS ==========")

print(f"Total simulated seconds: {total_seconds}")

print("\nSignal phase counts:")
print(result_df["phase"].value_counts())

print("\nGreen time allocated to each lane:")

green_summary = (
    result_df[result_df["phase"] == "GREEN"]
    .groupby("active_lane")
    .size()
)

for lane in LANES:
    print(f"{lane}: {green_summary.get(lane, 0)} seconds")

print("\nFirst 20 signal decisions:")
print(result_df.head(20).to_string(index=False))

print(f"\nOutput saved to: {OUTPUT_FILE}")

print("\nFIXED-TIME SCHEDULER COMPLETED")
