import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fitbit ML Analytics",
    page_icon="🏃",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/calorie_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.title("🏃 Fitbit ML Analytics Dashboard")

st.markdown(
    """
    ### Calorie Burn Prediction & Workout Pattern Analysis

    This application uses Machine Learning to predict calories burned
    during a workout and analyze workout behavior patterns.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "🔥 Calorie Prediction",
        "📊 Model Performance",
        "👥 Workout Clusters",
        "ℹ️ About Project"
    ]
)


# =========================================================
# PAGE 1 - CALORIE PREDICTION
# =========================================================

if page == "🔥 Calorie Prediction":

    st.header("🔥 Calorie Burn Prediction")

    st.write(
        "Enter the user's physical and workout information "
        "to estimate calories burned."
    )

    # -----------------------------------------------------
    # INPUT COLUMNS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=30
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0
        )

        height = st.number_input(
            "Height (m)",
            min_value=1.0,
            max_value=2.5,
            value=1.70
        )

        fat_percentage = st.number_input(
            "Fat Percentage",
            min_value=1.0,
            max_value=60.0,
            value=20.0
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:

        max_bpm = st.number_input(
            "Maximum BPM",
            min_value=80,
            max_value=220,
            value=180
        )

        avg_bpm = st.number_input(
            "Average BPM",
            min_value=50,
            max_value=220,
            value=120
        )

        resting_bpm = st.number_input(
            "Resting BPM",
            min_value=30,
            max_value=120,
            value=70
        )

        session_duration = st.number_input(
            "Session Duration (hours)",
            min_value=0.1,
            max_value=5.0,
            value=1.0
        )

        water_intake = st.number_input(
            "Water Intake (liters)",
            min_value=0.0,
            max_value=10.0,
            value=2.0
        )

    with col3:

        workout_frequency = st.number_input(
            "Workout Frequency (days/week)",
            min_value=0,
            max_value=7,
            value=3
        )

        experience_level = st.selectbox(
            "Experience Level",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Beginner",
                2: "Intermediate",
                3: "Advanced"
            }[x]
        )

        workout_type = st.selectbox(
            "Workout Type",
            ["Cardio", "HIIT", "Mixed", "Strength", "Yoga"]
        )

    # -----------------------------------------------------
    # DERIVED FEATURES
    # -----------------------------------------------------

    bmi = weight / (height ** 2)

    hr_reserve_ratio = (
        (avg_bpm - resting_bpm) /
        (max_bpm - resting_bpm)
        if max_bpm != resting_bpm
        else 0
    )

    st.info(
        f"Calculated BMI: **{bmi:.2f}**  |  "
        f"HR Reserve Ratio: **{hr_reserve_ratio:.2f}**"
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🔥 Predict Calories",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Age": [age],
            "Weight (kg)": [weight],
            "Height (m)": [height],
            "BMI": [bmi],
            "Fat_Percentage": [fat_percentage],
            "Max_BPM": [max_bpm],
            "Avg_BPM": [avg_bpm],
            "Resting_BPM": [resting_bpm],
            "Session_Duration (hours)": [session_duration],
            "Water_Intake (liters)": [water_intake],
            "Workout_Frequency (days/week)": [workout_frequency],
            "Experience_Level": [experience_level],
            "HR_Reserve_Ratio": [hr_reserve_ratio],
            "Gender": [gender],
            "Workout_Type": [workout_type]
        })

        try:

            prediction = model.predict(input_data)[0]

            st.success(
                f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**"
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)


# =========================================================
# PAGE 2 - MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.header("📊 Regression Model Performance")

    results_path = "reports/regression_model_results.csv"

    if os.path.exists(results_path):

        results = pd.read_csv(results_path)

        st.subheader("Model Comparison")

        st.dataframe(
            results,
            use_container_width=True
        )

        # -----------------------------------------------
        # BEST MODEL
        # -----------------------------------------------

        best_model = results.loc[
            results["R2"].idxmax()
        ]

        st.success(
            f"""
            🏆 Best Model: **{best_model['Model']}**

            R² Score: **{best_model['R2']:.4f}**

            MAE: **{best_model['MAE']:.2f}**

            RMSE: **{best_model['RMSE']:.2f}**
            """
        )

        # -----------------------------------------------
        # R2 CHART
        # -----------------------------------------------

        st.subheader("R² Score Comparison")

        r2_chart = results.set_index("Model")["R2"]

        st.bar_chart(r2_chart)

        # -----------------------------------------------
        # MAE CHART
        # -----------------------------------------------

        st.subheader("MAE Comparison")

        mae_chart = results.set_index("Model")["MAE"]

        st.bar_chart(mae_chart)

        # -----------------------------------------------
        # RMSE CHART
        # -----------------------------------------------

        st.subheader("RMSE Comparison")

        rmse_chart = results.set_index("Model")["RMSE"]

        st.bar_chart(rmse_chart)

    else:

        st.warning(
            "Regression results file not found."
        )


# =========================================================
# PAGE 3 - CLUSTERING
# =========================================================

elif page == "👥 Workout Clusters":

    st.header("👥 Workout Pattern Clustering")

    st.write(
        """
        KMeans clustering was used to identify hidden workout
        behavior patterns based on physiological and behavioral
        characteristics.
        """
    )

    # -----------------------------------------------
    # CLUSTER FEATURE MEANS
    # -----------------------------------------------

    cluster_path = "reports/cluster_feature_means.csv"

    if os.path.exists(cluster_path):

        clusters = pd.read_csv(cluster_path)

        st.subheader("Cluster Profiles")

        st.dataframe(
            clusters,
            use_container_width=True
        )

    else:

        st.warning(
            "Cluster feature means file not found."
        )


metrics_path = "reports/clustering_metrics.csv"

if os.path.exists(metrics_path):

    metrics = pd.read_csv(metrics_path)

    silhouette = metrics.loc[
        metrics["Metric"] == "Silhouette Score",
        "Value"
    ].iloc[0]

    threshold = metrics.loc[
        metrics["Metric"] == "Acceptance Threshold",
        "Value"
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Silhouette Score",
            f"{silhouette:.4f}"
        )

    with col2:
        st.metric(
            "Acceptance Threshold",
            f"{threshold:.2f}"
        )

    if silhouette >= threshold:
        st.success(
            "✅ Clustering meets the acceptance criterion"
        )
    else:
        st.warning(
            "⚠️ Clustering is below the acceptance criterion"
        )

    # PCA
    pca_path = "visuals/pca_clusters.png"

    if os.path.exists(pca_path):
        st.subheader("📈 PCA Cluster Visualization")
        st.image(pca_path, use_container_width=True)
    else:
        st.warning("PCA visualization not found.")


    # Silhouette
    silhouette_path = "visuals/silhouette_plot.png"

    if os.path.exists(silhouette_path):
        st.subheader("📊 Silhouette Analysis")
        st.image(silhouette_path, use_container_width=True)
    else:
        st.warning("Silhouette plot not found.")


# =========================================================
# PAGE 4 - ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        ## Fitbit: Calorie Burn Prediction & Workout Pattern Clustering

        ### Domain

        Fitness Analytics / Health Tech / Machine Learning

        ### Supervised Learning

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - KNN Regressor
        - Decision Tree
        - Random Forest
        - SVR
        - XGBoost

        ### Unsupervised Learning

        - PCA
        - KMeans Clustering

        ### Regression Metrics

        - MAE
        - RMSE
        - R² Score

        ### Clustering Metric

        - Silhouette Score

        ### Business Applications

        - Real-time calorie estimation
        - Personalized fitness coaching
        - Workout recommendations
        - User segmentation
        - Nutrition planning
        """
    )