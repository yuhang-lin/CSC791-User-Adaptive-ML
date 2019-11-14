
def find_correlation(file, corr_op):
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    df = pd.read_csv(file)

    # calculate the correlation matrix
    corr = df.corr()
    corr.to_csv(corr_op)

    # plot the heatmap
    plt.suptitle('correlation between different features')
    ax = sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('corr_plot.png')
    plt.show()

def main():
    find_correlation(file = '../sample_training_data.csv', corr_op = 'corr_output.csv')


if __name__ == '__main__':
    main()
