import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.title(f"Normal Distribution (n={n})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data

def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.title(f"Uniform Distribution (n={n})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data

def bernoulli_histogram(n):
    # Bernoulli is a Binomial with n=1 trial
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.title(f"Bernoulli Distribution (n={n})")
    plt.xlabel("Outcome")
    plt.ylabel("Frequency")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    return sum(data) / len(data)

def sample_variance(data):
    n = len(data)
    if n < 2:
        return 0
    mu = sample_mean(data)
    # Sum of squared differences
    sq_diff = [(x - mu)**2 for x in data]
    return sum(sq_diff) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        
    q1 = sorted_data[int(0.25 * (n - 1))]
    q3 = sorted_data[int(0.75 * (n - 1))]
    
    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(data1, data2):
    n = len(data1)
    mu1 = sample_mean(data1)
    mu2 = sample_mean(data2)
    
    covariance = sum((data1[i] - mu1) * (data2[i] - mu2) for i in range(n))
    return covariance / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(data1, data2):
    var_x = sample_variance(data1)
    var_y = sample_variance(data2)
    cov_xy = sample_covariance(data1, data2)
    
    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
