"""
Replication of the Methodology from Convex Clustering [1]

[1] D.F. Sun, K.C. Toh, and Y.C. Yuan, Convex clustering: model, theoretical guarantee and efficient algorithm, Journal of Machine Learning Research, 22(9):1?32, 2021.
"""

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 
