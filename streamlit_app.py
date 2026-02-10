import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("productivity_rf_model.pkl")

## Streamlit app
st.title("Productivity of garment store worker")
st.write("Enter production details to predict the actual productivity of the garment worker.")


## User inputs
quarter_selected = st.selectbox("Select Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4"])
department_selected = st.selectbox("Select Department", ["finishing", "sweing"])
day_selected = st.slider("Select Day", 
                          min_value=1, 
                          max_value=31,
                          step=1)
month_selected = st.slider("Select Month", 
                          min_value=1, 
                          max_value=12,
                          step=1)
team_selected = st.slider("Select Team Number", 
                          min_value=1, 
                          max_value=12,
                          step=1)
targeted_productivity_selected = st.slider("Select Targeted Productivity", 
                                min_value=0.00, 
                                max_value=1.00,
                                step=0.01)
smv_selected = st.number_input("Select SMV", 
                          min_value=0.00, 
                          max_value=60.00)
wip_selected = st.number_input("Select WIP", 
                          min_value=0.00, 
                          max_value=2000.00)
overtime_selected = st.number_input("Select Overtime",
                              min_value=0, 
                              max_value=25000)
incentive_selected = st.number_input("Select Incentive", 
                          min_value=0.00, 
                          max_value=3600.00)
idle_men_selected = st.number_input("Select Number of Idle Men",
                                    min_value=0,
                                    max_value=45)
no_of_style_change_selected = st.slider("Select Number of Style Changes", 
                                            min_value=0, 
                                            max_value=2,
                                            step=1)
no_of_workers_selected = st.number_input("Select Number of Workers",
                                    min_value=2,
                                    max_value=89,
                                    step=1)



if st.button("Predict Actual Productivity"):

    df_input = pd.DataFrame({
        "quarter": [quarter_selected],
        "department": [department_selected],
        "day": [day_selected],
        "month": [month_selected],
        "team": [team_selected],
        "targeted_productivity": [targeted_productivity_selected],
        "smv": [smv_selected],
        "wip": [wip_selected],
        "overtime": [overtime_selected],
        "incentive": [incentive_selected],
        "idle_men": [idle_men_selected],
        "no_of_style_change": [no_of_style_change_selected],
        "no_of_workers": [no_of_workers_selected],
    })

    # One-hot encoding
    df_input = pd.get_dummies(
        df_input,
        columns=["quarter", "department"]
    )

    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(df_input)[0]

    st.success(f"Predicted Productivity: **{prediction:.2f}**")

## Page design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)