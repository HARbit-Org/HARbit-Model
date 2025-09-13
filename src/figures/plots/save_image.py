import matplotlib.pyplot as plt
import seaborn as sns

ROOT_IMAGES = r"F:\UPC\Tesis\HARbit-Model\src\figures\images"

def save_plot(name : str, plot):
    plot.savefig(name, dpi = 300, bbox_inches = 'tight')
    plot.close()