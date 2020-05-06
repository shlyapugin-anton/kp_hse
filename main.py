from samples import points_on_sphere
from manifold import manifold_data
import numpy as np
import time

start_time = time.time()
### Генерация выборки
mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
seed = 0
n = 20000

sample_on_sphere = points_on_sphere(mean, cov, n, seed)
sample_on_sphere.normalize_points_set()
### Конец генерации выборки
k = 4
dim = 2

manifold = manifold_data(sample_on_sphere.points, k, dim)
manifold.initialize_manifold_data()

is_orientable = manifold.set_orientation()
print(is_orientable)
end_time = time.time()
print("рассчеты заняли времени: " + str(end_time - start_time))
### Ниже всякие тесты для отладки той или иной части функционала

print(manifold.points[0])
print(manifold.points[1])

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
