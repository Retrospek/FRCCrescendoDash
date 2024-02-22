import streamlit as st
import statbotics as sb

sb = sb.Statbotics()
year = 2024

events = sb.get_events(year=year, state="TX", )


events = sb.get_events

event = st.multiselect