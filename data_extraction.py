from plot import extract_data, get_stats, plot_data
import numpy as np
import pickle

folders = ["Trial_MO", "Trial_MO_Archive", "Trial_NS", "Trial_FO"]
trials = 2
for folder in folders:
    f_means = []
    f_var = []
    d_means = []
    d_var = []
    bests = []
    info = {}
    with open(folder+"/"+folder, "wb") as f:
        for i in range(trials):
            filein = folder+"/output"+str(i)+".txt"
            fileout = "result"+str(i)+".txt"
            extract_data(filein, fileout)
            means, stdevs, diversity_means, diversity_stdev, best = get_stats(fileout)
            f_means.append(means)
            f_var.append(stdevs)
            d_means.append(diversity_means)
            d_var.append(diversity_stdev)
            bests.append(best)

        temp = np.array(f_means)
        f_means = np.mean(temp, axis=0)
        info['f_means'] = f_means
        temp = np.array(f_var)
        f_var = np.mean(temp, axis=0)
        info['f_var'] = f_var
        temp = np.array(d_means)
        d_means = np.mean(temp, axis=0)
        info['d_means'] = d_means
        temp = np.array(d_var)
        d_var = np.mean(temp, axis=0)
        info['d_var'] = d_var
        temp = np.array(bests)
        b_means = np.mean(temp, axis=0)
        info['b_means'] = b_means
        b_var = np.std(temp, axis=0)
        info['b_var'] = b_var
        pickle.dump(info, f)
