#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
from scipy.interpolate import interp1d

#-----------------------------------------------------------------------------------------------------------------------------------------#

def generate_one_dim_dataset_from_functions(functions, domain, K, resolution=10000, force_balance=True):
    """
    Generates a labeled dataset by sampling from multiple probability density functions (PDFs).

    This function creates a dataset where x-values are sampled from the given 
    list of PDFs, and corresponding labels are assigned based on the index of 
    the PDF in the input list.

    Parameters:
        functions (list of callables): A list of probability density functions (PDFs), 
            where each function defines the distribution for sampling. Each callable 
            should take a single input (x) and be valid over the specified domain.
        domain (tuple): A tuple specifying the range `(start, end)` of the domain 
            over which the PDFs are defined, e.g., `(0, 1)`.
        K (int): The number of x-values to sample per PDF.
        resolution (int, optional): The number of points used to approximate 
            the cumulative density function (CDF) for each PDF. Higher values 
            improve accuracy but increase computation time. Default is 10,000.
        force_balance (bool, optional): If `True`, ensures exactly `K` samples 
            per PDF, even if the integral of the PDF is less than 1. If `False`, 
            the number of samples is proportional to the integral of each PDF. 
            Default is `True`.

    Returns:
        tuple: 
            - x_data (list): A list of x-values sampled from the PDFs.
            - y_data (list): A list of integer labels corresponding to the index 
              of the PDF each x-value was sampled from.

    Notes:
        - Each PDF in the `functions` list is assumed to be non-negative over the 
          specified domain.
        - Ensure `resolution` is high enough for accurate sampling, particularly 
          for complex PDFs or large domains.
        - The dataset size will depend on the number of functions in the list 
          and the parameter `K`.

    Example:
        >>> f1 = lambda x: 2*x  # A simple linear PDF
        >>> f2 = lambda x: 1 - x  # A descending linear PDF
        >>> functions = [f1, f2]
        >>> domain = (0, 1)
        >>> K = 100
        >>> x_data, y_data = generate_one_dim_dataset_from_functions(functions, domain, K)
        >>> print(len(x_data), len(y_data))  # Should be 200 samples total
    """

    x_data = []
    y_data = []
    for i, fn in enumerate(functions):
        samples = _one_dim_stochastic_sampler(fn, domain, K, resolution=resolution, force_balance=force_balance)
        x_data +=samples
        y_data += len(samples) * [i]
    return np.array(x_data), np.array(y_data)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def _one_dim_stochastic_sampler(fn, domain, K, resolution=10000, force_balance=True):
    """
    Stochastically samples x-values from a probability density function (PDF) 
    defined by a callable function.

    This function uses inverse transform sampling to generate random samples 
    from a given PDF. The PDF is numerically approximated over the specified 
    domain and normalized to ensure it integrates to 1.

    Parameters:
        fn (callable): The probability density function (PDF) defined as 
            a callable, e.g., `lambda x: 2*x`. The function should be valid 
            over the specified domain.
        domain (tuple): A tuple specifying the range `(start, end)` of the 
            domain over which the PDF is defined, e.g., `(0, 1)`.
        K (int): The desired number of x-values to sample from the PDF.
        resolution (int, optional): The number of points used to approximate 
            the cumulative density function (CDF). Higher values yield more 
            accurate results but increase computation time. Default is 10,000.
        force_balance (bool, optional): If `True`, the sampler enforces exactly 
            `K` samples, even if the total integral of the PDF is less than 1 
            (adjusts for incomplete normalization). If `False`, the number of 
            samples is proportional to the integral. Default is `True`.

    Returns:
        list: A list of `K` sampled x-values, distributed according to the PDF.

    Notes:
        - The provided function is assumed to be non-negative over the given 
          domain, as required for a valid PDF.
        - The domain should be chosen such that the PDF is significant within 
          the range; otherwise, results may be inaccurate due to numerical 
          approximation.
        - Ensure `resolution` is sufficiently high for accurate CDF inversion, 
          particularly for complex PDFs or large domains.
    """
    
    # Generate values for approximating the CDF
    x = np.linspace(domain[0], domain[1], resolution)
    y = fn(x)

    # Normalize the PDF to ensure it integrates to 1
    integral = np.trapz(y, x)
    pdf = y / integral

    # Compute the CDF
    cdf = np.cumsum(pdf) / np.sum(pdf)

    # Interpolation function for inverse transform sampling
    inverse_cdf = interp1d(cdf, x, bounds_error=False, fill_value=(domain[0], domain[1]))

    # Sample uniformly in [0, 1] and map to x-values using the inverse CDF
    uniform_samples = np.random.uniform(0, 1, K if force_balance else int(K * integral))
    x_samples = inverse_cdf(uniform_samples)

    return list(x_samples)

#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#