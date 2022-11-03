import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

def draw_line(x,name):
    sns.set(style='darkgrid')
    for feature in x:
        print(len(feature))
        data ={'timepoint': range(len(feature)),'signal':feature}
        df = pd.DataFrame(data)
        sns.lineplot(x='timepoint', y='signal', data=df)

    plt.savefig(name)