**College Basketball March Madness Predictor**/n
Brandon Poblette, Geoff Spilker
CPSC 322, Fall 2022
Introduction
Link to Dataset: https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset
Format: CSV
The Goal for this project is to create a predictor which based off several attributes can predict if a team will make March Madness. While we originally wanted to create a championship predictor, because of the small sample size of champions, we decided to pivot and work on a March Madness predictor. Every year, the AP come together to decide which teams will make the tournament. They base their decisions based off advanced analytics, strength of schedule, and if they won their conference tournament. We want to try to predict which teams will be invited to the tournament based off statistics alone.

Dataset Analysis
The dataset we decided to use is a college basketball dataset from the 2013 season to the 2021 season. This dataset has 3524 instances with no incomplete data. This allows this dataset to be really useful for our analysis. There are 24 attributes in this dataset. There are only a couple of columns which we will take into account. The attributes we chose were determined by our correlation matrix, which shows which attributes correlate well with other attributes in the dataset.

Attributes explored in final_report.ipynb:
ADJOE-Adjusted Offensive Efficiency
ADJDE-Adjusted Defensive Efficiency
BARTHAG-Power Rating (Chance of beating an average Division I team)
EFG_O-Effective Field Goal Percentage Shot
3P_O-Three-Point Shooting Percentage
ORB-Offensive Rebound Rate
DRB-Offensive Rebound Rate Allowed

Attributes explored in data_analysis.ipynb:
Wins - Amount of wins
Postseason - How Far a team got in March Madness
BARTHAG-Power Rating (Chance of beating an average Division I team)
ADJOE-Adjusted Offensive Efficiency
ADJDE-Adjusted Defensive Efficiency
EFG_O-Effective Field Goal Percentage Shot

Attributes explored in datamining.ipynb
ADJOE-Adjusted Offensive Efficiency
ADJDE-Adjusted Defensive Efficiency
BARTHAG-Power Rating (Chance of beating an average Division I team)
EFG_O-Effective Field Goal Percentage Shot
3P_O-Three-Point Shooting Percentage
ORB-Offensive Rebound Rate
DRB-Offensive Rebound Rate Allowed







