import matplotlib.pyplot as plt
import pandas as pd


def plot_data(data, condition_positive, number_of_images, label, marker):
    true_positive_rate_list = []
    false_positive_average_list = []
    for index, row in data.iterrows():
        true_positive = row['TP']
        false_positive = row['FP']
        false_negative = row['FN']

        true_positive_rate = true_positive / condition_positive
        false_positive_average = false_positive / number_of_images
        # plt.plot([0,1],[0,1])
        true_positive_rate_list.append(true_positive_rate)
        false_positive_average_list.append(false_positive_average)

    plt.scatter(false_positive_average_list, true_positive_rate_list, marker=marker)
    plt.plot(false_positive_average_list, true_positive_rate_list, label=label)
    return


n365 = pd.read_csv("n365vt.csv")
r = pd.read_csv("rvt.csv")
n20 = pd.read_csv("n20vt.csv")
n10 = pd.read_csv("n10vt.csv")

plt.figure(dpi=400)

plot_data(r, 26, 12, label='Random - 125 epochs', marker='x')
plot_data(n10, 27, 12, label='Difficult - 10 epochs', marker='.')
plot_data(n20, 27, 12, label='Difficult - 20 epochs', marker='.')
plot_data(n365, 27, 12, label='Difficult - 365 epochs', marker='.')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig('roc.png')
plt.show()