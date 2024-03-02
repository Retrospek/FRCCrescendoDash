import streamlit as st
import Calc_Web as cw


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout='wide'
)

st.title("Scouting Application")

st.sidebar.success("Select a Function Above")

st.markdown("""This app's main function is to provide statistics that i've assembeled from FRC Crescendo 2024 data. 
            Additionally, I've done feature engineering to create more insightful data for any user coming here""")
