import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def find_correlation(file, corr_op):
    df = pd.read_csv(file)
    best_columns = df.iloc[:, [76, 77, 88, 102, 110, 111, 121, 128]]
    print(best_columns)

    # calculate the correlation matrix
    corr = best_columns.corr()
    corr.to_csv(corr_op)

    # plot the heatmap
    plt.suptitle('correlation between different features')
    plt.autoscale()
    ax = sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('corr_plot_4.png')
    plt.show()

def main():
    find_correlation(file = '../binned_4_reorder.csv', corr_op = 'corr_output_4.csv')


if __name__ == '__main__':
    main()
