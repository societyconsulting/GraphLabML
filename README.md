GraphLab Code for Machine Learning and Predictive Services
==========================================================

This contains code using GraphLab create to generate some classification
algorithms. It also sets up a predictive service for one of the algorithms.
More information can be found on our blog post located here
http://www.societyconsulting.com/insights/detail/graphlab-ml-predictive-services

Contents
--------

analyze_algorithms.py : This runs all of our classification algorithms and
compares their accuracy.

predictive_service.py : This runs the predictive service.

visualize_data.py : This will graph the data using ggplot. You must make sure
ggplot is installed before you can run this code. 

process_data.py : This processes the raw data and turns it into features that
can be used by our classification algorithms.

data/features.csv : This contains the processed data (our features) that our
classification algorithms will use to learn.

data/labeled_data.csv : This contains the unprocessed data that's used to
generate our features.
