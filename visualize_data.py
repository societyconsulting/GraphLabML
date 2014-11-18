import pandas as pd

from ggplot import *


def graph_data():
    """
    Graphs the features with event_counts on the x axis and std on the y axis
    using a ggplot2 extension for Python. You can find out more at 
    ggplot.yhathq.com
    """

    data = pd.read_table('data/features.csv', sep=',')
    
    print ggplot(data, aes(x='event_counts', y='std', color='bot')) + \
            geom_point()


if __name__ == '__main__':
    graph_data()
