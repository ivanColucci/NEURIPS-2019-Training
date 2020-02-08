import os
import numpy as np
import pickle

def extract_data(filein, fileout):
    os.system('grep -Po "(?<=Best fitness: |Population\'s average fitness: |fitness: [\d.]{7} stdev: |Population\'s sparseness: |ness: [\d.]{7} stdev: |Archive\'s best: )[\d.]+" '+filein+' > '+fileout)

def get_stats(file):
    with open(file, 'r') as fin:
        content = fin.read()
        means = []
        stdevs = []
        bests = []
        diversity_means = []
        diversity_stdev = []
        i = 0
        for line in content.split("\n"):
            if line is "":
                continue
            n = float(str(line))
            if i == 0:
                means.append(n)
                i += 1
            elif i == 1:
                stdevs.append(n)
                i += 1
            elif i == 2:
                diversity_means.append(n)
                i += 1
            elif i == 3:
                diversity_stdev.append(n)
                i += 1
            elif i == 4:
                bests.append(n)
                i = 0
        return means, stdevs, diversity_means, diversity_stdev, bests

def plot_data(means, stdevs):
    import matplotlib.pyplot as plt

    plt.plot(np.add(means, stdevs), label="pop mean+std_dev")
    plt.plot(np.subtract(means, stdevs), label="pop mean-std_dev")
    plt.plot(means, label="pop mean")
    plt.fill_between([i for i in range(len(means))], np.add(means, stdevs), np.subtract(means, stdevs), alpha=0.4, color='green')
    plt.legend()
    plt.ylabel("fitness")
    plt.xlabel("#generations")
    plt.title("NEAT Evolution")
    plt.grid()
    plt.show()


def plot_comparison(means, labels, colors, title, ylabel, linewidth=1):
    import matplotlib.pyplot as plt

    for mean, label, color in zip(means, labels, colors):
        plt.plot(mean, label=label, color=color, linewidth=linewidth)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc=2, fontsize=20)
    plt.ylabel(ylabel, fontsize=24)
    plt.axis(fontsize=24)
    plt.xlabel("number of generations", fontsize=24)
    plt.title(title, fontsize=32)
    plt.grid()
    plt.show()

def plot_times():
    pass