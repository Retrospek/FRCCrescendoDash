# FRCCrescendoDash

This program is hosted on the Streamlit Cloud. If the cloud says "waking up" that just mean that the server has a timed out session. Instead just take the code, copy it, and run it through the terminal. You would do this by grabbing the relative path of Home.py and then running "streamlit run (relative path)" - exclude the parenthesis.


I aimed to create an app that could predict the likelihood of any given three teams defeating any other three teams. This required me to research how to compare two comparable averages and standard deviations. I considered using my experience with neural networks for this predictive task, thinking that some level of machine learning or even deep learning could be beneficial. However, I faced a data shortage. I realized that to create a predictive model that considers at least four attributes, I would need approximately 40 data inputs, based on the 10x rule. Given this, I pivoted towards a more statistics-based approach. Using the normal cumulative distribution function required normalized data. I examined past matches and found that as a team advances through a tournament, it will have a normally distributed graph of data points representing points scored. With this insight, I generated a cumulative average and standard deviation for each alliance and applied the norm.cdf function to calculate the possibility of the Blue team winning.

My match predictions for the FRC District Event at Waco accurately predicted the outcomes of 88% of the first 54 matches. This accuracy rate is 14% higher than that of Statbotics, which predicted matches with 74% accuracy.

My match predictions for the FRC District Event at Waco predicted 86.66% of the match outcomes from the Playoffs.
