import samples
from manifold import manifold_data
import numpy as np
import time

# === ОБЩИЕ КОНСТАНТЫ ===
mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
seed = 0
n = 4000
dim = 2
k = 10
generate = "mobius"
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

### === ОТЛАДКА === ###
"""
Q_0 = manifold.oriented_frame_map[0]
Q_4 = manifold.oriented_frame_map[4]
Q_287 = manifold.oriented_frame_map[287]
product = np.matmul(Q_0.T, Q_0)
print(product)
"""
### === КОНЕЦ ОТЛАДКИ === ###

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
