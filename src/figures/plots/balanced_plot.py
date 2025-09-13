import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import os

from .save_image import *

def balanced_plot(data: (pd.DataFrame | pl.DataFrame), name = str):
    fq_label = data\
                .select(pl.col('Activity Label'))\
                .group_by('Activity Label')\
                .agg(pl.count().alias("Count"))

    sns.barplot(fq_label, x = 'Activity Label', y = 'Count')
    save_plot(os.path.join(ROOT_IMAGES, f'{name}.png'), plt)



