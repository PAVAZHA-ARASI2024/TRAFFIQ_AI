
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TRAFFIQ AI | Smart Traffic",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).parent
CSS_FILE = BASE / "assets" / "CSS" / "style.css"

def load_css():
    if CSS_FILE.exists():
        with open(CSS_FILE, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )

load_css()
VIDEO_FILE = BASE / "runs" / "traffiq" / "lane_detection" / "lane_analytics_h264.mp4"
ANALYTICS_FILE = BASE / "runs/traffiq/lane_detection/lane_analytics.csv"

ADAPTIVE_FILE = BASE / "runs/traffiq/scheduler/adaptive_scheduler_log.csv"
FIXED_FILE = BASE / "runs/traffiq/scheduler/fixed_time_scheduler_log.csv"

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]

# ==========================================
# 2. CUSTOM LIGHT THEME
# ==========================================


# ==========================================
# 3. LOAD PROJECT DATA
# ==========================================

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


analytics = load_csv(ANALYTICS_FILE)
adaptive = load_csv(ADAPTIVE_FILE)
fixed = load_csv(FIXED_FILE)


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def create_signal_card(lane, phase, active_lane):

    if phase == "GREEN" and lane == active_lane:
        status = "GREEN"
        light = "signal-green"

    elif phase == "YELLOW" and lane == active_lane:
        status = "YELLOW"
        light = "signal-yellow"

    else:
        status = "STOP"
        light = "signal-red"

    st.markdown(f"""
    <div class="signal-card">
        <div class="small-label">{lane.upper()}</div>
        <div style="margin:14px 0;">
            <span class="signal-light {'signal-red' if light != 'signal-red' else 'signal-red'}"></span>
            <span class="signal-light {'signal-yellow' if status == 'YELLOW' else 'signal-red'}"></span>
            <span class="signal-light {'signal-green' if status == 'GREEN' else 'signal-red'}"></span>
        </div>
        <h3 style="margin:5px;color:#1a2639;">{status}</h3>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. HEADER AND NAVIGATION
# ==========================================

header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:34px;">🚦</span>
        <div>
            <h2 style="margin:0;color:#1a2639;">TRAFFIQ AI</h2>
            <p style="margin:0;color:#68758a;font-size:12px;">
                AI-POWERED SMART TRAFFIC MANAGEMENT
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style="text-align:right;padding-top:10px;">
        <span style="background:#e2f0e8;color:#1a7a4a;
        padding:8px 15px;border-radius:30px;font-size:12px;
        font-weight:700;">
        ● ANALYTICS READY
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

page = st.radio(
    "Main Navigation",
    [
        "Overview",
        "Traffic Analytics",
        "Signal Simulation",
        "Scheduler Comparison"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")


# ==========================================
# 6. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.markdown("""
    <div style="padding:25px 5px 20px 5px;">
        <div style="display:inline-block;background:#e6effd;
        color:#2a7de1;padding:8px 18px;border-radius:30px;
        font-size:12px;font-weight:700;">
        SMART CITY TRAFFIC SOLUTION
        </div>
        <h1 style="font-size:38px;margin-top:18px;">
        Intelligent Traffic <span style="color:#2a7de1;">
        Monitoring & Optimisation
        </span>
        </h1>
        <p style="color:#3e4a5e;font-size:16px;max-width:800px;">
        An AI-powered traffic monitoring system that detects
        vehicles, analyses lane-wise traffic conditions and
        dynamically allocates traffic signal green time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Traffic Approaches", "4")
    c2.metric("Scheduling Methods", "2")
    c3.metric("Detection Model", "YOLOv8n")
    c4.metric("Tracking Method", "ByteTrack")

    st.markdown("---")

    left, right = st.columns([1.5, 1])

    with left:
        section_title(
            "Traffic Video Monitoring",
            "Processed Four Corners Downtown traffic footage"
        )

        if VIDEO_FILE.exists():
            st.video(str(VIDEO_FILE))
        else:
            st.warning("Processed traffic video not found.")

    with right:
        section_title(
            "System Architecture",
            "TRAFFIQ AI processing pipeline"
        )

        st.markdown("""
        <div style="background:white;padding:22px;
        border-radius:18px;border:1px solid #e9edf4;">

        <div style="background:#e6effd;padding:13px;
        border-radius:12px;text-align:center;">
        <b>Traffic Video Input</b>
        </div>

        <div style="text-align:center;color:#2a7de1;font-size:20px;">↓</div>

        <div style="background:#e6effd;padding:13px;
        border-radius:12px;text-align:center;">
        <b>YOLO Vehicle Detection</b>
        </div>

        <div style="text-align:center;color:#2a7de1;font-size:20px;">↓</div>

        <div style="background:#e6effd;padding:13px;
        border-radius:12px;text-align:center;">
        <b>ByteTrack Tracking</b>
        </div>

        <div style="text-align:center;color:#2a7de1;font-size:20px;">↓</div>

        <div style="background:#e6effd;padding:13px;
        border-radius:12px;text-align:center;">
        <b>Lane-wise Analytics</b>
        </div>

        <div style="text-align:center;color:#2a7de1;font-size:20px;">↓</div>

        <div style="background:#e2f0e8;padding:13px;
        border-radius:12px;text-align:center;">
        <b>Adaptive Signal Scheduler</b>
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Demonstration mode: Recorded traffic footage is analysed "
        "and the resulting traffic measurements are used to simulate "
        "signal scheduling. Vehicles do not physically respond to "
        "the simulated signals."
    )


# ==========================================
# 7. TRAFFIC ANALYTICS PAGE
# ==========================================

elif page == "Traffic Analytics":

    section_title(
        "Traffic Analytics",
        "Lane-wise vehicle detection and traffic condition analysis"
    )

    if analytics is None:
        st.error("Traffic analytics CSV not found.")

    else:

        latest = (
            analytics.sort_values("video_time_seconds")
            .groupby("lane")
            .tail(1)
            .set_index("lane")
            .reindex(LANES)
        )

        total_vehicles = int(
            latest["current_vehicles"].fillna(0).sum()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Vehicles in Latest Snapshot", total_vehicles)

        c2.metric(
            "Maximum Queue Length",
            int(analytics["queue_length"].max())
        )

        c3.metric(
            "Maximum Estimated Waiting",
            f"{analytics['average_waiting_seconds'].max():.1f} sec"
        )

        st.markdown("---")

        section_title("Lane Conditions")

        display_data = latest[
            [
                "current_vehicles",
                "queue_length",
                "average_waiting_seconds",
                "traffic_intensity"
            ]
        ].copy()

        display_data.index = display_data.index.str.upper()

        display_data.columns = [
            "Vehicle Count",
            "Queue Length",
            "Estimated Waiting (sec)",
            "Traffic Intensity"
        ]

        st.dataframe(
            display_data,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            section_title("Vehicle Count by Lane")

            fig = px.bar(
                latest.reset_index(),
                x="lane",
                y="current_vehicles",
                color="current_vehicles",
                color_continuous_scale=["#c9dcfa", "#2a7de1"],
                labels={
                    "lane": "Traffic Lane",
                    "current_vehicles": "Vehicles"
                }
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1a2639"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            section_title("Traffic Intensity Distribution")

            intensity = (
                analytics.groupby(
                    ["lane", "traffic_intensity"]
                )
                .size()
                .reset_index(name="observations")
            )

            fig2 = px.bar(
                intensity,
                x="lane",
                y="observations",
                color="traffic_intensity",
                barmode="stack",
                color_discrete_map={
                    "LOW": "#16a34a",
                    "MEDIUM": "#eab308",
                    "HIGH": "#dc2626"
                }
            )

            fig2.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1a2639"
            )

            st.plotly_chart(fig2, use_container_width=True)

        section_title("Traffic Observation Timeline")

        timeline = analytics.groupby(
            "video_time_seconds"
        )["current_vehicles"].sum().reset_index()

        fig3 = px.line(
            timeline,
            x="video_time_seconds",
            y="current_vehicles",
            labels={
                "video_time_seconds": "Video Time (seconds)",
                "current_vehicles": "Total Vehicles"
            }
        )

        fig3.update_traces(
            line=dict(color="#2a7de1", width=3)
        )

        fig3.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 8. SIGNAL SIMULATION PAGE
# ==========================================

elif page == "Signal Simulation":

    section_title(
        "Traffic Signal Simulation",
        "Explore the saved signal decisions produced by each scheduler"
    )

    scheduler_choice = st.selectbox(
        "Select Scheduling Method",
        ["Adaptive Scheduler", "Fixed-Time Scheduler"]
    )

    selected_data = (
        adaptive if scheduler_choice == "Adaptive Scheduler"
        else fixed
    )

    if selected_data is None or selected_data.empty:

        st.error("Selected scheduler log not found.")

    else:

        selected_data = selected_data.reset_index(drop=True)

        max_second = len(selected_data) - 1

        second = st.slider(
            "Simulation Time (seconds)",
            min_value=0,
            max_value=max_second,
            value=0
        )

        current = selected_data.iloc[second]

        phase = current["phase"]
        lane = current["active_lane"]
        remaining = int(current["remaining_time"])

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric("Simulation Time", f"{second} sec")
        c2.metric("Signal Phase", phase)
        c3.metric("Remaining Phase Time", f"{remaining} sec")

        st.markdown("---")

        section_title(
            "Four-Lane Signal Control",
            "Signal indication for each monitored approach"
        )

        signal_cols = st.columns(4)

        for i, approach in enumerate(LANES):

            with signal_cols[i]:

                create_signal_card(
                    approach,
                    phase,
                    lane
                )

        if phase == "GREEN":

            st.success(
                f"GREEN SIGNAL: {lane.upper()} is currently being served."
            )

        elif phase == "YELLOW":

            st.warning(
                f"YELLOW SIGNAL: {lane.upper()} is completing its phase."
            )

        else:

            st.error(
                "ALL-RED: No vehicle approach is being given green."
            )

        st.markdown("---")

        section_title("Scheduling Decision")

        left, right = st.columns(2)

        with left:

            st.markdown(
                f"**Selected lane:** {lane.upper()}"
            )

            st.markdown(
                f"**Signal phase:** {phase}"
            )

            st.markdown(
                f"**Remaining time:** {remaining} seconds"
            )

        with right:

            if "selection_reason" in current.index:

                st.markdown("**Decision explanation**")

                st.info(str(current["selection_reason"]))

        section_title("Signal Timeline")

        start = max(0, second - 10)
        end = min(len(selected_data), second + 11)

        timeline = selected_data.iloc[start:end].copy()

        fig = px.scatter(
            timeline,
            x="time_seconds",
            y="active_lane",
            color="phase",
            color_discrete_map={
                "GREEN": "#16a34a",
                "YELLOW": "#eab308",
                "ALL_RED": "#dc2626"
            },
            hover_data=["remaining_time"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#1a2639"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is a replay of saved scheduler decisions, "
            "not a live traffic signal controller."
        )


# ==========================================
# 9. SCHEDULER COMPARISON PAGE
# ==========================================

elif page == "Scheduler Comparison":

    section_title(
        "Scheduler Performance Comparison",
        "Compare traffic signal allocation under fixed and adaptive scheduling"
    )

    if adaptive is None or fixed is None:

        st.error("One or both scheduler log files are missing.")

    else:

        adaptive_green = (
            adaptive[adaptive["phase"] == "GREEN"]
            .groupby("active_lane")
            .size()
        )

        fixed_green = (
            fixed[fixed["phase"] == "GREEN"]
            .groupby("active_lane")
            .size()
        )

        comparison = pd.DataFrame({
            "Lane": LANES,
            "Adaptive Scheduler": [
                adaptive_green.get(lane, 0)
                for lane in LANES
            ],
            "Fixed-Time Scheduler": [
                fixed_green.get(lane, 0)
                for lane in LANES
            ]
        })

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Adaptive Green Time",
            f"{(adaptive['phase'] == 'GREEN').sum()} sec"
        )

        c2.metric(
            "Fixed-Time Green Time",
            f"{(fixed['phase'] == 'GREEN').sum()} sec"
        )

        c3.metric(
            "Simulation Duration",
            f"{len(adaptive)} sec"
        )

        st.markdown("---")

        section_title("Green Time Allocation")

        fig = px.bar(
            comparison,
            x="Lane",
            y=[
                "Adaptive Scheduler",
                "Fixed-Time Scheduler"
            ],
            barmode="group",
            color_discrete_map={
                "Adaptive Scheduler": "#2a7de1",
                "Fixed-Time Scheduler": "#9bb9df"
            },
            labels={
                "value": "Green Time (Seconds)",
                "variable": "Scheduling Method"
            }
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#1a2639"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # SERVICE COUNT COMPARISON
        adaptive_services = adaptive[
            (adaptive["phase"] == "GREEN") &
            (
                (adaptive["phase"].shift() != "GREEN") |
                (adaptive["active_lane"] != adaptive["active_lane"].shift())
            )
        ].groupby("active_lane").size()

        fixed_services = fixed[
            (fixed["phase"] == "GREEN") &
            (
                (fixed["phase"].shift() != "GREEN") |
                (fixed["active_lane"] != fixed["active_lane"].shift())
            )
        ].groupby("active_lane").size()

        service_comparison = pd.DataFrame({
            "Lane": LANES,
            "Adaptive Service Count": [
                adaptive_services.get(lane, 0)
                for lane in LANES
            ],
            "Fixed-Time Service Count": [
                fixed_services.get(lane, 0)
                for lane in LANES
            ]
        })

        section_title("Signal Service Count Comparison")

        st.caption(
            "Number of separate green-signal services allocated to each lane."
        )

        st.dataframe(
            service_comparison,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")



        section_title("Signal Phase Distribution")

        phases = ["GREEN", "YELLOW", "ALL_RED"]

        phase_data = pd.DataFrame({
            "Phase": phases,
            "Adaptive Scheduler": [
                (adaptive["phase"] == p).sum()
                for p in phases
            ],
            "Fixed-Time Scheduler": [
                (fixed["phase"] == p).sum()
                for p in phases
            ]
        })

        fig2 = px.bar(
            phase_data,
            x="Phase",
            y=[
                "Adaptive Scheduler",
                "Fixed-Time Scheduler"
            ],
            barmode="group",
            color_discrete_map={
                "Adaptive Scheduler": "#2a7de1",
                "Fixed-Time Scheduler": "#9bb9df"
            }
        )

        fig2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#1a2639"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Lane-wise Allocation Analysis")

        for _, row in comparison.iterrows():

            difference = (
                row["Adaptive Scheduler"]
                - row["Fixed-Time Scheduler"]
            )

            if difference > 0:
                st.write(
                    f"**{row['Lane'].upper()}** received "
                    f"{difference} additional green seconds "
                    "under adaptive scheduling."
                )

            elif difference < 0:
                st.write(
                    f"**{row['Lane'].upper()}** received "
                    f"{abs(difference)} fewer green seconds "
                    "under adaptive scheduling."
                )

            else:
                st.write(
                    f"**{row['Lane'].upper()}** received equal "
                    "green time under both methods."
                )

        st.info(
            "Interpretation: Adaptive scheduling changes green-time "
            "allocation according to its traffic priority logic. "
            "These results measure simulated signal allocation, "
            "not actual vehicle delay reduction, because recorded "
            "vehicles do not respond to simulated signals."
        )


# ==========================================
# 10. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    <b style="color:#2a7de1;">TRAFFIQ AI</b><br>
    AI-Powered Smart Traffic Signal Optimisation<br>
    <span>Intelligent Monitoring · Traffic Analytics · Adaptive Scheduling</span>
</div>
""", unsafe_allow_html=True)
