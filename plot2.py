import pickle
from plot import plot_data, plot_comparison

folders = ["Trial_MO", "Trial_MO_Archive", "Trial_NS", "Trial_FO"] #, "TrialFO_NEAT_Classic", "TrialFO_OldConfig"]
labels = ['MO', 'MO_Archive', 'NS', 'FO']
colors = ['green', 'orange', 'red', 'blue']
means = []
stds = []
div_means = []
div_stds = []
b_means = []
b_var = []
for folder in folders:
    with open(folder+"/"+folder,"rb") as f:
        info = pickle.load(f)
        means.append(info['f_means'])
        stds.append(info['f_var'])
        div_means.append(info['d_means'])
        div_stds.append(info['d_var'])
        b_means.append(info['b_means'])
        b_var.append(info['b_var'])
        # plot_data(info['means'], info['std'], info['div_means'], info['div_std'], info['bests'])
# plot_comparison(means, folders, colors, 'Mean of Average Population\'s Fitness - 10 trials - Walker2D', 'fitness')
# plot_comparison(stds, folders, colors, 'Standard Deviation of Average Population\'s Fitness - 10 trials - Walker2D', 'fitness')
plot_comparison(div_means, labels, colors, 'Mean of Population\'s Diversity - 2 trials - Humanoid', 'diversity')
plot_comparison(div_stds, labels, colors, 'Standard Deviation of Population\'s Diversity - 2 trials - Humanoid', 'diversity')
plot_comparison(b_means, labels, colors, 'Mean of Population\'s Best - 2 trials - Humanoid', 'fitness', linewidth=2)
plot_comparison(b_var, labels, colors, 'Standard Deviation of Population\'s Best - 2 trials - Humanoid', 'fitness', linewidth=2)
