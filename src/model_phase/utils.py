import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_chart(epochs, train_list, val_list, save_path, name=''):
    x = np.arange(epochs)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    lns1 = ax.plot(x, train_list, 'b', label='train {}'.format(name))
    lns2 = ax.plot(x, val_list, 'r', label='val {}'.format(name))
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, loc='upper right')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(name)

    plt.savefig(save_path)
    plt.close()