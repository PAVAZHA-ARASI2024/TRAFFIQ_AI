
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------
# 1. FILE PATHS
# --------------------------------

BASE_DIR = Path("runs/traffiq/scheduler")

ADAPTIVE_FILE = BASE_DIR / "adaptive_scheduler_log.csv"
FIXED_FILE = BASE_DIR / "fixed_time_scheduler_log.csv"

OUTPUT_DIR = BASE_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]

# --------------------------------
# 2. LOAD SCHEDULER DATA
# --------------------------------

print("Starting scheduler comparison...")

adaptive = pd.read_csv(ADAPTIVE_FILE)
fixed = pd.read_csv(FIXED_FILE)

# Check simulation duration
if len(adaptive) != len(fixed):
    raise ValueError("Both scheduler logs must have equal duration.")

# --------------------------------
# 3. CALCULATE GREEN TIME
# --------------------------------

adaptive_green = (
    adaptive[adaptive["phase"] == "GREEN"]
    .groupby("active_lane").size()
)

fixed_green = (
    fixed[fixed["phase"] == "GREEN"]
    .groupby("active_lane").size()
)

green_comparison = pd.DataFrame({
    "Lane": LANES,
    "Adaptive_Green_Seconds": [
        adaptive_green.get(lane, 0) for lane in LANES
    ],
    "Fixed_Green_Seconds": [
        fixed_green.get(lane, 0) for lane in LANES
    ]
})

green_comparison["Difference_Seconds"] = (
    green_comparison["Adaptive_Green_Seconds"]
    - green_comparison["Fixed_Green_Seconds"]
)

# --------------------------------
# 4. COMPARE SIGNAL PHASES
# --------------------------------

phases = ["GREEN", "YELLOW", "ALL_RED"]

phase_comparison = pd.DataFrame({
    "Phase": phases,
    "Adaptive_Seconds": [
        (adaptive["phase"] == phase).sum()
        for phase in phases
    ],
    "Fixed_Seconds": [
        (fixed["phase"] == phase).sum()
        for phase in phases
    ]
})

# --------------------------------
# 5. COUNT GREEN SERVICE PERIODS
# --------------------------------

def count_green_periods(df):
    green = df["phase"].eq("GREEN")
    previous = df["active_lane"].shift(1)

    starts = (
        green
        & (
            ~green.shift(1, fill_value=False)
            | df["active_lane"].ne(previous)
        )
    )

    return df.loc[starts, "active_lane"].value_counts()


adaptive_services = count_green_periods(adaptive)
fixed_services = count_green_periods(fixed)

service_comparison = pd.DataFrame({
    "Lane": LANES,
    "Adaptive_Service_Count": [
        adaptive_services.get(lane, 0) for lane in LANES
    ],
    "Fixed_Service_Count": [
        fixed_services.get(lane, 0) for lane in LANES
    ]
})

# --------------------------------
# 6. SAVE COMPARISON REPORTS
# --------------------------------

green_comparison.to_csv(
    OUTPUT_DIR / "green_time_comparison.csv",
    index=False
)

phase_comparison.to_csv(
    OUTPUT_DIR / "phase_comparison.csv",
    index=False
)

service_comparison.to_csv(
    OUTPUT_DIR / "service_comparison.csv",
    index=False
)

# --------------------------------
# 7. GENERATE GREEN TIME CHART
# --------------------------------

x = range(len(LANES))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    green_comparison["Adaptive_Green_Seconds"],
    width,
    label="Adaptive Scheduler"
)

plt.bar(
    [i + width / 2 for i in x],
    green_comparison["Fixed_Green_Seconds"],
    width,
    label="Fixed-Time Scheduler"
)

plt.xlabel("Traffic Lanes")
plt.ylabel("Green Time (Seconds)")
plt.title("Green Time Allocation Comparison")
plt.xticks(x, LANES)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "green_time_comparison.png"
)

plt.close()

# --------------------------------
# 8. GENERATE PHASE CHART
# --------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    phases,
    phase_comparison["Adaptive_Seconds"],
    label="Adaptive Scheduler"
)

plt.bar(
    phases,
    phase_comparison["Fixed_Seconds"],
    alpha=0.7,
    label="Fixed-Time Scheduler"
)

plt.xlabel("Signal Phase")
plt.ylabel("Duration (Seconds)")
plt.title("Signal Phase Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "phase_comparison.png"
)

plt.close()

# --------------------------------
# 9. DISPLAY RESULTS
# --------------------------------

print("\n========== GREEN TIME COMPARISON ==========")
print(green_comparison.to_string(index=False))

print("\n========== PHASE COMPARISON ==========")
print(phase_comparison.to_string(index=False))

print("\n========== SERVICE COUNT COMPARISON ==========")
print(service_comparison.to_string(index=False))

print("\nComparison completed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")
