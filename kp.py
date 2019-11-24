import generate_point_on_sphere
import apply_pca
import apply_local_pca
import store_point_info
import store_manifold_info
from scipy.linalg import svd
import numpy as np
import transformation_operators

# Что подправить:
# 1. Наименование find_transformation_operator, т.к. функция возвращает промежуточный оператор (с.в. которого образуют нужную ортогональную матрицу перехода)
# 2. На вход подается N - число ближайших соседей к каждой точке. Однако находим N соседей включая саму точку. Нужно тоже подправить логику
# 3. В store_point_info нужно хранить набор ближайших точек (для корректной работы transformation_matrices...)
# 4. Не забыть, что transformation_matrices... сейчас строит только O_ii! (пока не особо понятно, как определять "близость" точек для построения O_ij)
# Это может быть реально близость точек, т.е. строим O_ij, если расстояние между i и j меньше фиксированного, но как подбирать это фиксированное расстояние?
# 5. Начинают появляться методы, которые можно применять в разных классах, возможно стоит задуматься о выносе методов в абстрактный класс
# Примером таких методов является:
# get_point_hash(point) - согласован с store_point_info, необходим для того, чтобы получить оператор по ключу


# Класс для хранения информации об отражениях (в статье обозначается как Z - n \times n оператор из определителей матриц перехода)
class reflections_operator:
    def __init__(self, tr_operators):
        self.tr_operators = tr_operators
        self.operator = [[0 for i in range(len(tr_operators.transformation_operators))] for j in range(len(tr_operators.transformation_operators))]

    def initialize_reflections_operator(self):
        for first_operator_pos in range(len(self.tr_operators.transformation_operators)):
            for second_operator_pos in range(len(self.tr_operators.transformation_operators)):
                self.operator[first_operator_pos][second_operator_pos] = np.linalg.det(self.tr_operators.transformation_operators[first_operator_pos][second_operator_pos])


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

transformation_operators = transformation_operators.transformation_matrices_between_near_point(mn_info)
transformation_operators.initialize_transformation_operators()

reflection_operator = reflections_operator(transformation_operators)
reflection_operator.initialize_reflections_operator()

print(reflection_operator.operator)

# for key in mn_info.manifold_info.keys():
    # print("====")
    # print("operator in value = " + str(mn_info.manifold_info[key].operator))
    # U, s, vt = svd(mn_info.manifold_info[key].operator)
    # print("U = " + str(U))
    # print("s = " + str(s))
    # print("vt = " + str(vt))
    # print("====")
