
import numpy as np
import antropy as ant
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy.random
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

#normal distribution

np.random.seed(seed=611)

#data with outliers
def generate(median=630, err=12, outlier_err=100, size=80, outlier_size=10):
    errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
    data = median + errs

    lower_errs = outlier_err * np.random.rand(outlier_size)
    lower_outliers = median - err - lower_errs

    upper_errs = outlier_err * np.random.rand(outlier_size)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)

    return data


def structual_entropy(x):
    _, _count = np.unique(x, return_counts=True)
    _countnorm = _count / _count.sum()
    _entropy = -(_countnorm*np.log2(_countnorm)).sum()
    return(_entropy)
    

def timothy_entropy(x):
    ## by timothy, length = 3000, we have 10 bins
    nbins = 10
    # x_min = np.min(x)
    # x_max = np.max(x)
    # bin_size = (x_max - xmin)/10
    # bin_list = np.arange(x_min, x_max, bin_size)
    hist, edges = np.histogram(x, bins=10)
    ent_sum = 0
    for i in hist:
        if(i != 0):
            p = i/len(x)
            ent_sum -= p * np.log(p)
    return(ent_sum)


def get_entropy_list(x):

    x = [i for i in x if str(i) != 'nan']
    ent_1 = ant.perm_entropy(x, normalize=True)
    ent_2 = ant.perm_entropy(x, normalize=False)
    ent_3 = structual_entropy(x)
    ent_4 = timothy_entropy(x)
    return( {'antropy_norm':ent_1, 'antropy_not_norm':ent_2, 'structual': ent_3, 'timothy':ent_4} )



x1 = ss.norm.rvs(loc=0, scale=1, size=3000)
x2 = [3] * 3000
x3 = np.arange(1,3000,1)
x4 = generate(size=(3000),  outlier_err=(900), outlier_size=(900))
# print("entropy of a data with more far placed outliers",ant.perm_entropy(x, normalize=norm))
x5 = generate(size=(3000))
# print("entropy of a data with less far placed outliers",ant.perm_entropy(x, normalize=norm))
#pareto distribution
a, m = 1*3., 2.  # shape and mode
x6 = (np.random.pareto(a, 3000) + 1) * m
# print("entropy of pareto (less clumped)",ant.perm_entropy(x, normalize=norm))

a, m = 100*3., 2.  # shape and mode
x7 = (np.random.pareto(a, 3000) + 1) * m
# print("entropy of pareto (more clumped)",ant.perm_entropy(x, normalize=norm))
#when the more frequent values become more spread out, entropy goes up
#with a, m = 100*3., 2.   more clumping and more entropy (0.9987363447865257)
#with a, m = 1*3., 2.  less clumping (frequent values more spread out) and more entropy (0.9993237804430154)


x_name = ['normal', '3*3000', '1-3000', 'far-placed-outlier', 'close-placed-outlier', 'small-clumped-pareto', 'large-clumped-pareto']
x_list = [x1,x2,x3,x4,x5,x6,x7]


res_list = []

for idx in list(range(0,7)):
    data_name = x_name[idx]
    temp_x = x_list[idx]
    entropy_dict = get_entropy_list(temp_x)
    entropy_dict['name'] = data_name
    entropy_dict['transform'] = 'None'
    res_list.append(entropy_dict)

    transformed_x = np.power(temp_x,(1/4))
    entropy_dict = get_entropy_list(transformed_x)
    entropy_dict['name'] = data_name
    entropy_dict['transform'] = '^0.25'
    res_list.append(entropy_dict)

    transformed_x = np.power(temp_x, 2)
    entropy_dict = get_entropy_list(transformed_x)
    entropy_dict['name'] = data_name
    entropy_dict['transform'] = '^2'
    res_list.append(entropy_dict)

    transformed_x = np.tanh(temp_x)
    entropy_dict = get_entropy_list(transformed_x)
    entropy_dict['name'] = data_name
    entropy_dict['transform'] = 'tanh'
    res_list.append(entropy_dict)

    transformed_x = np.log(temp_x)
    entropy_dict = get_entropy_list(transformed_x)
    entropy_dict['name'] = data_name
    entropy_dict['transform'] = 'log'
    res_list.append(entropy_dict)

df = pd.DataFrame(res_list)
print(df)
