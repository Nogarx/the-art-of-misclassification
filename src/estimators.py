#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------------#

def one_dim_two_class_threshold_estimator(x_data, y_data, domain, resolution=10000):
    optimal_threshold, max_accuracy = 0, 0
    for th in np.linspace(domain[0], domain[1], resolution):
        solutions = ((x_data > th).reshape(-1) == y_data)
        accuracy = np.sum(solutions) / len(x_data)
        accuracy = accuracy if accuracy > 1 - accuracy else 1 - accuracy
        if accuracy > max_accuracy:
            optimal_threshold = th
            max_accuracy = accuracy
    return optimal_threshold, max_accuracy

from sklearn.metrics import roc_curve, accuracy_score
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------------#

def one_dim_multiclass_class_estimator(x_data, y_data, domain=None, resolution=10000):
    """
    Estimates the optimal threshold for a one-dimensional two-class classification problem 
    using ROC curve analysis.

    Parameters:
        x_data (numpy.ndarray): Array of x-values (features).
        y_data (numpy.ndarray): Array of binary class labels (0 or 1).
        domain (tuple, optional): Range of thresholds to consider. If not provided, it 
            uses the full range of `x_data`.
        resolution (int, optional): Number of points to evaluate within the domain.

    Returns:
        tuple: Optimal threshold and the corresponding maximum accuracy.
    """
    # Sort thresholds within the domain if provided, else use x_data range
    domain = domain or (np.min(x_data), np.max(x_data))
    thresholds = np.linspace(domain[0], domain[1], resolution)

    # Compute ROC curve and optimal threshold
    fpr, tpr, roc_thresholds = roc_curve(y_data, x_data)
    accuracies = []
    
    for threshold in thresholds:
        # Classify based on threshold
        predictions = (x_data > threshold).astype(int)
        accuracies.append(accuracy_score(y_data, predictions))

    # Find threshold with maximum accuracy
    max_accuracy_index = np.argmax(accuracies)
    optimal_threshold = thresholds[max_accuracy_index]
    max_accuracy = accuracies[max_accuracy_index]

    return optimal_threshold, max_accuracy


#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#