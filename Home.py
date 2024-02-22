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

st.markdown("# Set-Up")
st.write("""I wanted to create a scouting prediction app that would take the likelyhood of
          any given 3 teams beating any other 3 teams. To accomplish this, I researched how to 
          compare two comparable averages and standard deviations. Additionally, I had experience in neural networks and because this was 
          a predictive task, I thought that perhaps some level of ML or even deep learning could be used. However, I ran into the issue of a lack of data, and I quickly
          realized that in order for me to create a predictive model that took at least 4 attributes that it would need approximately 40 data inputs, according to the 10x rule
          . With this knowledge, I instead went towards a more statistics based approach. The method of using a normal cumumalitive distribution function required normalized data
          . Knowing this I delved into past matches, and proved that as a team progresses throughout a tournament it will have a normally distributed graph of data points that represent points scored.
          Knowing this I generated a cumulative average and standard deviation for each of the alliances and applied the norm.cdf function to calculate the possibility of the Blue team winning. """)