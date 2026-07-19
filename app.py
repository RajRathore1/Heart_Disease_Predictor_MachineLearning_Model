"""CardioCheck entry point. Routes to the views/ pages via st.navigation so
the sidebar shows proper page names/icons (the default pages/ folder nav
would label the home page "app" after the filename). Shared startup - global
styles, database init, the automatic-retraining check, and the branded
sidebar - happens once here rather than in every view.
"""

import streamlit as st

from core import database, retrain, ui

st.set_page_config(page_title="CardioCheck — Heart Disease Risk", page_icon="🩺", layout="wide")
ui.apply_global_styles()

database.init_db()

if retrain.should_retrain():
    with st.spinner("New confirmed patient outcomes are available — retraining the model..."):
        retrain.run_retraining()

pages = [
    st.Page("views/risk_assessment.py", title="Risk Assessment", icon="🩺", default=True),
    st.Page("views/batch_prediction.py", title="Batch Prediction", icon="📊"),
    st.Page("views/insights.py", title="Clinic Insights", icon="📈"),
    st.Page("views/prediction_history.py", title="Prediction History", icon="📁"),
]
pg = st.navigation(pages, position="hidden")

ui.sidebar_nav(pages)

pg.run()
