import os
import joblib
import pandas as pd
import gradio as gr


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_gradient_boosting_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "test_FD001.txt"
)


# =========================================================
# FINAL SENSOR LIST
# =========================================================

final_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

best_gb_model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_PATH)


# =========================================================
# LOAD TEST DATA
# =========================================================

TEST_COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    TEST_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=TEST_COLUMNS
)


# =========================================================
# HEALTH STATUS
# =========================================================

def get_health_status(rul):

    if rul <= 20:
        return "Critical"

    elif rul <= 50:
        return "Warning"

    elif rul <= 100:
        return "Normal"

    else:
        return "Healthy"


# =========================================================
# MAINTENANCE RECOMMENDATION
# =========================================================

def get_maintenance_recommendation(rul):

    if rul <= 20:
        return (
            "CRITICAL: Immediate inspection and "
            "maintenance required."
        )

    elif rul <= 50:
        return (
            "WARNING: Schedule maintenance soon and "
            "closely monitor the engine."
        )

    elif rul <= 100:
        return (
            "NORMAL: Continue monitoring engine condition."
        )

    else:
        return (
            "HEALTHY: No immediate maintenance required."
        )


# =========================================================
# RUL PREDICTION
# =========================================================

def predict_rul(engine_data):

    # Make a copy so the original data is not changed
    data = engine_data.copy()

    # Sort data by engine and cycle
    data = (
        data
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # 1. Rolling Mean - 5 cycles
    # -----------------------------------------------------

    for sensor in final_sensors:

        data[f"{sensor}_rolling_mean_5"] = (
            data
            .groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

    # -----------------------------------------------------
    # 2. Rolling Standard Deviation - 5 cycles
    # -----------------------------------------------------

    for sensor in final_sensors:

        data[f"{sensor}_rolling_std_5"] = (
            data
            .groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).std()
            )
        )

    # Replace NaN values in rolling standard deviations
    rolling_std_columns = [
        col
        for col in data.columns
        if "rolling_std_5" in col
    ]

    data[rolling_std_columns] = (
        data[rolling_std_columns].fillna(0)
    )

    # -----------------------------------------------------
    # 3. Trend - difference over 5 cycles
    # -----------------------------------------------------

    for sensor in final_sensors:

        data[f"{sensor}_trend_5"] = (
            data
            .groupby("unit")[sensor]
            .diff(5)
        )

    # Replace NaN values in trend features
    trend_columns = [
        col
        for col in data.columns
        if "trend_5" in col
    ]

    data[trend_columns] = (
        data[trend_columns].fillna(0)
    )

    # -----------------------------------------------------
    # 4. Select latest observation for each engine
    # -----------------------------------------------------

    latest_data = (
        data
        .sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # 5. Select exact training features
    # -----------------------------------------------------

    X = latest_data[feature_columns]

    # -----------------------------------------------------
    # 6. Predict RUL
    # -----------------------------------------------------

    predictions = best_gb_model.predict(X)

    # -----------------------------------------------------
    # 7. Create output
    # -----------------------------------------------------

    results = pd.DataFrame({
        "Engine": latest_data["unit"].values,
        "Last_Cycle": latest_data["cycle"].values,
        "Predicted_RUL": predictions
    })

    # Add health status
    results["Health_Status"] = (
        results["Predicted_RUL"]
        .apply(get_health_status)
    )

    return results


# =========================================================
# GENERATE FLEET PREDICTIONS
# =========================================================

prediction_results = predict_rul(test_df)


# =========================================================
# SINGLE ENGINE PREDICTION
# =========================================================

def predict_single_engine(engine_number):

    engine_number = int(engine_number)

    # Check whether engine exists
    if engine_number not in test_df["unit"].unique():
        return None

    # Select requested engine
    engine_data = test_df[
        test_df["unit"] == engine_number
    ].copy()

    # Generate prediction
    result = predict_rul(engine_data)

    return result


# =========================================================
# ENGINE MONITOR
# =========================================================

def monitor_engine(engine_number):

    try:
        engine_number = int(engine_number)
    except (TypeError, ValueError):

        return (
            "Invalid engine number",
            "—",
            "—",
            "Please enter an engine number between 1 and 100."
        )

    result = predict_single_engine(engine_number)

    if result is None:

        return (
            "Engine not found",
            "—",
            "—",
            "Please enter an engine number between 1 and 100."
        )

    rul = result["Predicted_RUL"].iloc[0]

    status = result["Health_Status"].iloc[0]

    cycle = result["Last_Cycle"].iloc[0]

    recommendation = (
        get_maintenance_recommendation(rul)
    )

    return (
        f"{rul:.2f} cycles",
        status,
        str(cycle),
        recommendation
    )


# =========================================================
# FLEET KPIs
# =========================================================

def get_fleet_kpis():

    total = len(prediction_results)

    healthy = (
        prediction_results["Health_Status"] == "Healthy"
    ).sum()

    normal = (
        prediction_results["Health_Status"] == "Normal"
    ).sum()

    warning = (
        prediction_results["Health_Status"] == "Warning"
    ).sum()

    critical = (
        prediction_results["Health_Status"] == "Critical"
    ).sum()

    return (
        str(total),
        str(healthy),
        str(normal),
        str(warning),
        str(critical)
    )


# =========================================================
# GRADIO DASHBOARD
# =========================================================

with gr.Blocks(
    title="AI RUL Monitoring Dashboard"
) as demo:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    gr.Markdown(
        """
        # ✈️ AI-Based RUL Monitoring Dashboard

        ### Predictive Maintenance System

        Monitor engine health and Remaining Useful Life
        using Machine Learning.
        """
    )

    # -----------------------------------------------------
    # FLEET OVERVIEW
    # -----------------------------------------------------

    gr.Markdown("## 📊 Fleet Overview")

    with gr.Row():

        total_output = gr.Textbox(
            label="Total Engines",
            value=str(len(prediction_results)),
            interactive=False
        )

        healthy_output = gr.Textbox(
            label="🟢 Healthy",
            value=str(
                (
                    prediction_results["Health_Status"]
                    == "Healthy"
                ).sum()
            ),
            interactive=False
        )

        normal_output = gr.Textbox(
            label="🟡 Normal",
            value=str(
                (
                    prediction_results["Health_Status"]
                    == "Normal"
                ).sum()
            ),
            interactive=False
        )

        warning_output = gr.Textbox(
            label="🟠 Warning",
            value=str(
                (
                    prediction_results["Health_Status"]
                    == "Warning"
                ).sum()
            ),
            interactive=False
        )

        critical_output = gr.Textbox(
            label="🔴 Critical",
            value=str(
                (
                    prediction_results["Health_Status"]
                    == "Critical"
                ).sum()
            ),
            interactive=False
        )

    # -----------------------------------------------------
    # ENGINE MONITORING
    # -----------------------------------------------------

    gr.Markdown("## 🔍 Engine Monitoring")

    with gr.Row():

        engine_input = gr.Number(
            label="Engine Number",
            value=1,
            minimum=1,
            maximum=100,
            precision=0
        )

        monitor_button = gr.Button(
            "🔎 Monitor Engine",
            variant="primary"
        )

    # -----------------------------------------------------
    # ENGINE STATUS
    # -----------------------------------------------------

    gr.Markdown("## ⚙️ Engine Status")

    with gr.Row():

        rul_output = gr.Textbox(
            label="Predicted RUL"
        )

        health_output = gr.Textbox(
            label="Health Status"
        )

        cycle_output = gr.Textbox(
            label="Last Recorded Cycle"
        )

    recommendation_output = gr.Textbox(
        label="🔧 Maintenance Recommendation",
        lines=3
    )

    # -----------------------------------------------------
    # BUTTON ACTION
    # -----------------------------------------------------

    monitor_button.click(
        fn=monitor_engine,
        inputs=engine_input,
        outputs=[
            rul_output,
            health_output,
            cycle_output,
            recommendation_output
        ]
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 10000)
        )
    )
