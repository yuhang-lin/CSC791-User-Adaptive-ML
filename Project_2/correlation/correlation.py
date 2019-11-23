import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def find_correlation(file, corr_op):
    df = pd.read_csv(file)
    best_columns = df.iloc[:, [18, 42, 85, 114]]
    # print(best_columns)

    # calculate the correlation matrix
    corr = best_columns.corr()
    corr.to_csv(corr_op)

    # plot the heatmap
    plt.suptitle('correlation between different features', fontsize=16)
    plt.autoscale()
    ax = sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=False, annot_kws={"fontsize":11, "weight": "bold"}, cmap=ListedColormap(['gray']))

    plt.xticks(rotation=20)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)


    plt.savefig('corr_plot_4b_4f.png')
    plt.show()

def main():
    find_correlation(file = '../binned_4_reorder.csv', corr_op = 'corr_output_4b_4f.csv')


if __name__ == '__main__':
    main()
