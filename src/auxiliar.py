#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#-----------------------------------------------------------------------------------------------------------------------------------------#

def CustomCmap(from_rgb,to_rgb):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

#-----------------------------------------------------------------------------------------------------------------------------------------#

def print_classificability(value):
    s = '\n'.join(['', '\t\t\t\t' + '#' * 68, f'\t\t\t\t\t\t\tClassificability: {value:.4f}', '\t\t\t\t' + '#' * 68, ''])
    print(s)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def print_numerical_classificability(value):
    s = '\n'.join(['', '\t\t\t\t' + '#' * 68, f'\t\t\t\t\t\t  Numerical Classificability: {value:.4f}', '\t\t\t\t' + '#' * 68, ''])
    print(s)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def print_sampling_classificability(mean, std):
    s = '\n'.join(['', '\t\t\t\t' + '#' * 68, f'\t\t\t\t\t      Estimated Classificability: {mean:.4f} ± {std:.4f}', '\t\t\t\t' + '#' * 68, ''])
    print(s)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def theoretical_plots(x, raws, ps, rhos, colors=None):

    fig, ax = plt.subplots(2,3,figsize=(12,8))

    num_classes = len(raws)
    for i in range(num_classes):
        ax[0,0].fill_between(x, 0, raws[i](x), color=colors[i] if colors is not None else None, alpha=0.1)
        ax[0,0].plot(x, raws[i](x), color=colors[i] if colors is not None else None)
    ax[0,0].set_title('Raw Probabilities')
    ax[0,0].set_xlabel('X')
    ax[0,0].set_ylabel('Unnormalized Probability Density')

    stack = []
    for i in range(num_classes):
        stack.append(raws[i](x))
        ax[1,0].fill_between(x, 0, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None, alpha=0.1)
        ax[1,0].plot(x, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None)
    ax[1,0].set_title('Stacked Raw Probabilities')
    ax[1,0].set_xlabel('X')
    ax[1,0].set_ylabel('Unnormalized Probability Density')

    norm = np.sum([p(x) for p in ps], axis=0)
    for i in range(num_classes):
        ax[0,1].fill_between(x, 0, ps[i](x)/norm, color=colors[i] if colors is not None else None, alpha=0.1, label=f'Class {i+1}')
        ax[0,1].plot(x, ps[i](x)/norm, color=colors[i] if colors is not None else None)
    ax[0,1].set_title('Relative Probabilities')
    ax[0,1].set_xlabel('X')
    ax[0,1].set_ylabel('Probability Density')

    stack = []
    for i in range(num_classes):
        stack.append(ps[i](x)/norm)
        ax[1,1].fill_between(x, 0, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None, alpha=0.1)
        ax[1,1].plot(x, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None)
    ax[1,1].set_title('Stacked Relative Probabilities')
    ax[1,1].set_xlabel('X')
    ax[1,1].set_ylabel('Probability Density')

    for i in range(num_classes):
        ax[0,2].fill_between(x, 0, rhos[i](x), color=colors[i] if colors is not None else None, alpha=0.1)
        ax[0,2].plot(x, rhos[i](x), color=colors[i] if colors is not None else None)
    ax[0,2].set_title('Density Function')
    ax[0,2].set_xlabel('X')
    ax[0,2].set_ylabel('Density')

    stack = []
    for i in range(num_classes):
        stack.append(rhos[i](x))
        ax[1,2].fill_between(x, 0, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None, alpha=0.1)
        ax[1,2].plot(x, np.sum(stack, axis=0) if len(stack) > 1 else stack[0], color=colors[i] if colors is not None else None)
    ax[1,2].set_title('Stacked Density Function')
    ax[1,2].set_xlabel('X')
    ax[1,2].set_ylabel('Density')

    handles, labels = ax[0,1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', ncol=len(ps), bbox_to_anchor=(0.5, 1.05))
    for handle in legend.legendHandles:
        handle.set_alpha(1.0)

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------#

def data_plots(x_data, y_data, optimal_threshold, ps, domain, raws=None, colors=None, bins=30):

    assert len(np.unique(y_data)) == len(ps), 'probability functions list does not match the number of classes'
    if raws is not None:
        assert len(np.unique(y_data)) == len(raws), 'raw probability functions list does not match the number of classes'

    x = np.linspace(domain[0], domain[1], 1000)
    max_y = np.max([np.max(fn(x)) for fn in raws]) if raws is not None else np.max([np.max(fn(x)) for fn in ps])
    x_bins = np.linspace(domain[0], domain[1], bins+1)
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    for i in range(len(ps)):
        ax[0].fill_between(x, 0, raws[i](x) if raws is not None else ps[i](x), color=colors[i] if colors is not None else None, alpha=0.1)
        ax[0].plot(x, raws[i](x) if raws is not None else ps[i](x), color=colors[i] if colors is not None else None)
    ax[0].plot([optimal_threshold, optimal_threshold], [0, max_y], 'k--', linewidth=2)
    ax[0].set_title('Raw Probabilities')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Unnormalized Probability Density')

    data_counts = []
    for i in range(len(ps)):
        counts, _ = np.histogram(x_data[y_data == i], bins=x_bins)
        data_counts.append(counts)
        ax[1].hist(x_bins[:-1], x_bins, weights=counts, color=colors[i] if colors is not None else None, alpha=0.2, label=f'Class {i+1}')
        ax[1].stairs(counts, x_bins, color=colors[i] if colors is not None else None)
    ax[1].plot([optimal_threshold, optimal_threshold], [0, np.max(np.sum(data_counts, axis=0))], 'k--', linewidth=2)
    ax[1].set_title('Data Distribution')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Counts')

    for i in range(len(ps)):
        ax[2].hist(x_bins[:-1], x_bins, weights=data_counts[i], bottom=data_counts[i-1] if i > 0 else 0, color=colors[i] if colors is not None else None, alpha=0.2)
        ax[2].stairs(data_counts[i] + data_counts[i-1] if i > 0 else data_counts[i], x_bins, color=colors[i] if colors is not None else None)
    ax[2].plot([optimal_threshold, optimal_threshold], [0, np.max(np.sum(data_counts, axis=0))], 'k--', linewidth=2)
    ax[2].set_title('Data Density')
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Counts')

    handles, labels = ax[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', ncol=len(ps), bbox_to_anchor=(0.5, 1.05))
    for handle in legend.legendHandles:
        handle.set_alpha(1.0)
    fig.subplots_adjust(top=0.95)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------#

def estimation_plots(classificabilities, analytical_classificability, numerical_classificability, c_scatter, probs, entropies, xs_data, ys_data, distance_thresholds, domain, colors=None, bins=30):

    assert len(classificabilities) > 1, ' Error 418: I\'m a teapot\' '

    xs_data, ys_data = np.array(xs_data), np.array(ys_data)
    x_bins = np.linspace(domain[0], domain[1], bins+1)
    num_classes = len(np.unique(ys_data))
    num_rows = len(classificabilities)

    fig, ax = plt.subplots(num_rows, 4,figsize=(16,4*num_rows))

    for r in range(num_rows):
        ax[r,0].plot(distance_thresholds, classificabilities[0], c='#00C882', label='Entropy')
        ax[r,0].plot([distance_thresholds[0], distance_thresholds[-1]], [analytical_classificability, analytical_classificability], c='#3C3C3C', linestyle='--', label='Analytical')
        ax[r,0].plot([distance_thresholds[0], distance_thresholds[-1]], [numerical_classificability, numerical_classificability], c='#2D9FFF', linestyle='--', label='Numerical')
        ax[r,0].scatter(c_scatter[r][0], c_scatter[r][1], color='k', s=30, label='Sampled Point')
        ax[r,0].legend()
        ax[r,0].set_title('Classificability Estimation')
        ax[r,0].set_xlabel('Distance Threshold')
        ax[r,0].set_ylabel('Classificability')
        

        data_counts = []
        for i in range(num_classes):
            counts, _ = np.histogram(xs_data[r][ys_data[r] == i], bins=x_bins)
            data_counts.append(counts)
            ax[r,1].hist(x_bins[:-1], x_bins, weights=counts, color=colors[i] if colors is not None else None, alpha=0.2, label=f'Class {i+1}')
            ax[r,1].stairs(counts, x_bins, color=colors[i] if colors is not None else None)
        ax[r,1].set_title('Data Distribution')
        ax[r,1].set_xlabel('X')
        ax[r,1].set_ylabel('Counts')

        order = np.argsort(xs_data[r])
        for i in range(num_classes):
            ax[r,2].plot(xs_data[r][order], probs[r][order][:,i], color=colors[i] if colors is not None else None)
            ax[r,3].plot(xs_data[r][order], entropies[r][order][:,i], color=colors[i] if colors is not None else None)

        ax[r,2].set_title('Estimated Relative Probability')
        ax[r,2].set_xlabel('X')
        ax[r,2].set_ylabel('Probability Density')

        ax[r,3].set_title('Estimated Entropies')
        ax[r,3].set_xlabel('X')
        ax[r,3].set_ylabel('Entropy')

    handles, labels = ax[r,1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', ncol=len(ps), bbox_to_anchor=(0.5, 1.05))
    for handle in legend.legendHandles:
        handle.set_alpha(1.0)

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#