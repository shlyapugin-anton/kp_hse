import generate_point_on_sphere
import apply_pca
import apply_local_pca
import store_point_info
import store_manifold_info

# Что подправить:
# 1. Наименование find_transformation_operator, т.к. функция возвращает промежуточный оператор (с.в. которого образуют нужную ортогональную матрицу перехода)
# 2. На вход подается N - число ближайших соседей к каждой точке. Однако находим N соседей включая саму точку. Нужно тоже подправить логику

# ====
# Тестовые данные
# ====

mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
n = int(input())
k = int(input())
d = int(input())

point = generate_point_on_sphere.Points(mean, cov, n)
point.normalize_points_set()

mn_info = store_manifold_info.store_manifold_info(point.points, k, d)
mn_info.initialize_orthonormal_operators()

# ====
# Проверка корректности части кода
# ====

for key in mn_info.manifold_info.keys():
    print("====")
    print("key = " + str(key))
    print("point in value = " + str(mn_info.manifold_info[key].coordinates))
    print("operator in value = " + str(mn_info.manifold_info[key].operator))
    print("====")
