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

    with st.expander("What is a Gradient Boosted Machine Learning Model?"):
        st.write(""" 
        A gradient boosted machine learning model is an ensemble learning method that iteratively combines the predictions of multiple individual models, typically decision trees, to improve overall performance. It starts by building a single decision tree to make predictions on the target variable. It then calculates the errors and creates a new decision tree to predict these errors, repeating this process iteratively. This approach allows the model to capture complex relationships in the data and make more accurate predictions. In the context of categorical classification, gradient boosting is particularly useful because it can handle categorical features directly, automatically handle missing data and outliers, and is robust to overfitting. These characteristics make it a powerful tool for effectively modeling and predicting categorical outcomes.
        """)
                 
    
    st.markdown("All Tournament Accuracy: :green[94.26%]")
    st.markdown("FIT Waco Tournament Accuracy: :green[92.31%]")
    st.markdown("FIT Fort Worth Tournament Accuracy: :green[97.72%]")
    
    st.header("Set-Up Before WACO", divider='rainbow')
    st.write("""I wanted to create a scouting prediction app that would take the likelihood of any given 3 teams beating any other 3 teams. To accomplish this, I researched how to compare two comparable averages and standard deviations. Additionally, I had experience in neural networks and because this was a predictive task, I thought that perhaps some level of ML or even deep learning could be used. However, I ran into the issue of a lack of data, and I quickly realized that for me to create a predictive model that took at least 4 attributes it would need approximately 40 data inputs, according to the 10x rule. With this knowledge, I instead went towards a more statistics-based approach. The method of using a normal cumulative distribution function required normalized data. Knowing this I delved into past matches and proved that as a team progresses throughout a tournament it will have a normally distributed graph of data points that represent points scored. Knowing this I generated a cumulative average and standard deviation for each of the alliances and applied the norm.cdf function to calculate the possibility of the Blue team winning. """)
    st.header("Set-Up After Waco", divider='rainbow')
    st.write("""After Waco, there was more data for me to use, and after requesting for a new field to be added to the dataset, who won or lost, I was able to finally approach an ML method of solving match prediction problems. With this better data, I found out how to create a dynamic feature between each team by taking their standardized scores between 0 and 1 and creating differing features that would categorize each alliance's outcome as a loss or win. In the first stages, I approached ensemble learning and found 4 different methods that work the best: logistic regression, gradient boosting, hyperparameter tuning, and feature engineering. After developing a stable model that didn't overfit the WACO event tournament I trained the ML model on more generalized matches, and it gave back amazing results that I listed below.""")
                

    st.header("Statistics Results", divider='rainbow')
    st.write("""My match predictions for the FRC District Event at Waco predicted the outcomes of 74.4% of all of the qualification matches.
              This accuracy rate is 1.2% lower than that of Statbotics, which predicted matches with 75.6% accuracy. With this knowledge I looked into the idea of standardizing features so that there is less of a skew, so that numbers can be more accurate for a robot""")
              

    st.header("ML Results", divider='rainbow')
    st.write("""Before I explain the ml's accuracy of the matches, I want to ensure that everyone realizes that robot's are "affected" by each other. A key flaw in the statistics model is that it doesn't account for other robot's behaviours: bumping, defending, etc. This is why a machine learing model is so powerful as it can display the relationship between variables and how they effect each other""")
    st.write("""My match predictions for the FRC District Event at Waco predicted 92.3% of the qualifications results. This accuracy beat both my statistics approach, by 17.9% percent, 
                and the statbotics approach, with a 75.6% accuracy rate, by 16.7%. This is huge as it destroyed previous predictive model by more than 15%!""")
    st.write("""The Machine Learning match predictor set an even higher bar at the Fort Worth tournament by beating statbotics's approach, which achieved an accuracy rate of 79%, by achieving a accuracy rate of 97.7%. This is a crazy statistic as the model only missed 1 match out of the 44 that blue alliance posted""")
    st.write("""However, all ml models must face the frustrating obstacle of having a lack of data, so be cautious when consulting the ml predictions if you don't have an abundance of field records.
                To combat this, I would recommend looking at the statistic based approach alongside the ml prediction.""")
    st.write(""" The issue is that if there is too little data, then the team-based statistics may not represent the real strength of a robot""")
if __name__ == "__main__":
    main()
