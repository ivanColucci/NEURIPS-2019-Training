import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 2
fo = [1.90685, 3.07336]
mo = [1.48151844, 0.90764657]
moa = [4.75139841, 1.64735842]
ns = [3.51136104, 5.02677294]

## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.15                       # the width of the bars

## the bars
rects1 = ax.bar(ind, mo, width, color='green')
rects2 = ax.bar(ind+width, moa, width, color='orange')
rects3 = ax.bar(ind+2*width, fo, width, color='blue')
rects4 = ax.bar(ind+3*width, ns, width, color='red')

# axes and labels
ax.set_xlim(-width, len(ind)+width)
ax.set_ylim(0, 6)
ax.set_ylabel('Fitness', fontsize=24)
ax.set_title('Comparison of Best Individual Found for each Trial', fontsize=32)
xTickMarks = ['Trial '+str(i) for i in range(1, N+1)]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
## add a legend
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('MO', 'MO_Archive', 'FO', 'NS'), fontsize=20)
ax.grid()
plt.show()
