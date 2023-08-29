
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.5, 0.85)}
    )
    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)

    ax_box.set(xlabel="")
    plt.title(variable)
    plt.show()
