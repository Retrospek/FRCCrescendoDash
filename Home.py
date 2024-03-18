import streamlit as st
import Calc_Web as cw

def main():
    
    # Google Authentication
    

    st.set_page_config(
        page_title="Hello",
        page_icon="chart_with_upwards_trend",
        layout='wide'
    )

    st.title("Scouting Application")

#    if 'data' not in st.session_state:
#        st.session_state['data'] = st.file_uploader("Tournament Data")


    st.sidebar.success("Select a Function Above")

    st.markdown("""This app's main function is to provide statistics that i've assembeled from FRC Crescendo 2024 data. 
                Additionally, I've done feature engineering to create more insightful data for any user coming here""")

    st.write("By: Arjun Mahableshwarkar")

    st.header("Gradient Boosted Output", divider='rainbow')
    st.markdown("All Tournament Accuracy: :green[89.34%]")
    st.markdown("FIT Waco Tournament Accuracy: :green[91.03%]")
    st.markdown("FIT Fort Worth Tournament Accuracy: :green[88.64%]")
    
    st.header("Set-Up Before WACO", divider='rainbow')
    st.write("""I wanted to create a scouting prediction app that would take the likelyhood of
            any given 3 teams beating any other 3 teams. To accomplish this, I researched how to 
            compare two comparable averages and standard deviations. Additionally, 
            I had experience in neural networks and because this was 
            a predictive task, I thought that perhaps some level of ML or even deep
            learning could be used. However, I ran into the issue of a lack of data, and I quickly
            realized that in order for me to create a predictive model that took at least 4 attributes 
            that it would need approximately 40 data inputs, according to the 10x rule
            . With this knowledge, I instead went towards a more statistics based approach.
            The method of using a normal cumumalitive distribution function required normalized data
            . Knowing this I delved into past matches, and proved that as a team progresses throughout
            a tournament it will have a normally distributed graph of data points that represent points scored.
            Knowing this I generated a cumulative average and standard deviation for each of the alliances and 
            applied the norm.cdf function to calculate the possibility of the Blue team winning. """)
    #st.header("Set-Up After Waco", divider='rainbow')
    st.write("""After Waco, there was more data for me to use, and after requesting for a new field to be added to the dataset, who won or lost, I was able to finally approach an ml method of solving match prediction problems.
                With this better data, I found out how to create a dynamic pattern between each team by taking their standardized scores between 0 and 1 to solve field-heavy ml tasks.
                In the first stages, I approached ensemble learning and found 2 different methods that work the best: logistic regression, gradient boosting, and while it isn't a model, hyperparameter tuning.
                After developing a clean and stable model that doesn't overfit the WACO event tournament I developed the ml model, and it gave back amazing results that I listed below.""")
                

    #st.header("Statistics Results", divider='rainbow')
    st.write("""My match predictions for the FRC District Event at Waco accurately predicted the outcomes of 85% of the first 54 matches.
              This accuracy rate is 11% higher than that of Statbotics, which predicted matches with 74% accuracy.""")
              
    st.write("""My match predictions for the FRC District Event at Waco predicted 86.66% of the match outcomes from the Playoffs.""")

    #st.header("ML Results", divider='rainbow')
    st.write("""My match predictions for the FRC District Event at Waco predicted 91% of the qualifications results. This accuracy beat both my statistics approach by 6 percent 
                and the statbotics approach by 17%""")
    st.write("""However, all ml models must face the frustrating obstacle of having a lack of data, so be cautious when consulting the ml predictions if you don't have an abundance of field records.
                To combat this, I would recommend looking at the statistic based approach alongside the ml prediction.""")
    st.write(""" The issue is that if there is too little data, then the team-based statistics may not represent the real strength of a robot""")
if __name__ == "__main__":
    main()
