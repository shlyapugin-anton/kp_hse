import samples
from manifold import manifold_data
import numpy as np
import time
from scipy.linalg import eigh
from unidip import UniDip
import seaborn as sns
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt 
from unidip import UniDip

# === ОБЩИЕ КОНСТАНТЫ ===
mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
seed = 0
n = 5000
dim = 2
k = 100
generate = "mobius"
algorithm = "odm"
# === КОНЕЦ СЕКЦИИ С ОБЩИМИ КОНСТАНТАМИ 

start_time = time.time()
### Генерация выборки на сфере
sample = samples.samples()
if generate == "sphere":
    sample = sample.generate_point_on_sphere(mean, cov, n, seed)
elif generate == "mobius":
    sample = sample.generate_point_on_mobius(n, seed)
### Конец генерации выборки


manifold = manifold_data(sample, k, dim)
manifold.initialize_manifold_data()

def find_maximum_pos(w):
    max_pos = 0
    max_val = w[0]
    for val_pos in range(len(w)):
        if w[val_pos] > max_val:
            max_val = w[val_pos]
            max_pos = val_pos
    return max_pos

if algorithm == "odm":
    manifold.initialize_o_ij()
    manifold.initialize_z()
    manifold.initialize_reversed_d()
    manifold.initialize_reflection_matrix()
    # manifold.reflection_operator - оператор "Z красивая" из статьи по ODM
    w, v = eigs(manifold.reflection_operator)
    max_pos = find_maximum_pos(w)
    v[max_pos] = v[max_pos].real

    intervals = UniDip(v[max_pos]).run()
    print(intervals)

    plt.hist(v[max_pos])
    plt.show()
    """
    svm = sns.distplot(v[max_pos].real)
    figure = svm.get_figure()
    figure.savefig('svm_2.png', dpi=4000)
    """
#    print(v[0])
else:                                               # сейчас только 2 алгоритма: ODM и из статьи Estimation of smooth vector fields
    is_orientable = manifold.set_orientation()
    print(is_orientable)
    end_time = time.time()
    print("рассчеты заняли времени: " + str(end_time - start_time))
### Ниже всякие тесты для отладки той или иной части функционала

### manifold.find_nearest
#sample_data = np.array([
#    [100, 100, 100],
#    [10, 10, 10],
#    [1, 1, 1],
#    [0, 0, 0],
#    [-1, -1, -1],
#    [-10, -10, -10],
#    [-100, -100, -100]
#])
#manifold = manifold_data(sample_data, 3)
#print(manifold.find_nearest_points(2))
