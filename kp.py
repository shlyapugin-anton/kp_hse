import generate_point_on_sphere
import apply_pca
import apply_local_pca
import store_point_info
import store_manifold_info
from scipy.linalg import svd
import numpy as np
import transformation_operators
import reflections_operator
import os
import seaborn as sns

# Что подправить:
# 1. Наименование find_transformation_operator, т.к. функция возвращает промежуточный оператор (с.в. которого образуют нужную ортогональную матрицу перехода)
# 2. На вход подается N - число ближайших соседей к каждой точке. Однако находим N соседей включая саму точку. Нужно тоже подправить логику
# 3. В store_point_info нужно хранить набор ближайших точек (для корректной работы transformation_matrices...)
# 4. Не забыть, что transformation_matrices... сейчас строит только O_ii! (пока не особо понятно, как определять "близость" точек для построения O_ij)
# Это может быть реально близость точек, т.е. строим O_ij, если расстояние между i и j меньше фиксированного, но как подбирать это фиксированное расстояние?
# 5. Начинают появляться методы, которые можно применять в разных классах, возможно стоит задуматься о выносе методов в абстрактный класс
# Примером таких методов является:
# get_point_hash(point) - согласован с store_point_info, необходим для того, чтобы получить оператор по ключу


# Оператор D из статьи
# in: square_operator - квадратный оператор
# out: диагональный оператор, на i-й диагонали которого число ненулевых элементов в i-й строке исходного оператора

# Методы нового класса Matrix
def calculate_diagonal_with_nonzeros_in_row(square_operator):
    diag = [len(np.nonzero(x)[0]) for x in square_operator] # Вычисляем значения диагональных элементов
    return np.diag(diag) # Возвращаем диагональный оператор, на i диагонали которого - число ненулевых элементов в i-й строке оператора square_operator

# Возможно тоже должен быть методом класса Matrix 
def calculate_normalize_reflections_operator(reflections_operator, diagonal_operator):
    return np.matmul(np.linalg.inv(diagonal_operator), reflections_operator)

# Метод Matrix
# Необходим для поиска макс с.з. и соотв. ему с.в. из алгоритма (2.3 Syncronization)
def find_max_eigenvalue_with_its_vector(operator):
    w, v = np.linalg.eig(operator)
    max_eigenvalue_pos = ind_of_max_element(w)
    return w[max_eigenvalue_pos], v[max_eigenvalue_pos]

def ind_of_max_element(tList):
    maxElem = tList[0]
    ind = 0
    for i in range(len(tList)):
        if (tList[i] > maxElem):
            maxElem = tList[i]
            ind = i
    return ind

# in: top eigenvector
# Метод для расчета z_i (их оценок, вообще говоря)
# В статье шаг 7
def calculate_reflections_array(tList):
    tList
    return np.sign(tList)

def symmetrization(operator):
    print(operator + np.transpose(operator))
    return 1

# ====
# Тестовые данные
# ====

mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# seed для генерации точек
seed = 0
print("Введите число точек")
n = int(input())
print("Введите число ближайших соседей, которые используем в алгоритме")
k = int(input())
print("Введите предполагаемую размерность многообразия, на котором лежат точки")
d = int(input())

point = generate_point_on_sphere.Points(mean, cov, n, seed)
point.normalize_points_set()

mn_info = store_manifold_info.store_manifold_info(point.points, k, d)
mn_info.initialize_orthonormal_operators()

# ====
# Проверка корректности части кода
# ====

print("Введите расстояние, которое считается _близким_ для двух точек")
close_distance = float(input()) # расстояние, которое мы считаем близким (для рассчета операторов перехода между близкими точками)
transformation_operators = transformation_operators.transformation_matrices_between_near_point(mn_info, close_distance)
transformation_operators.initialize_transformation_operators()

reflections_operator = reflections_operator.reflections_operator(transformation_operators)
reflections_operator.initialize_reflections_operator()

diagonal_operator = calculate_diagonal_with_nonzeros_in_row(reflections_operator.operator) # Инициализируем диагональный оператор
normalize_reflections_operator = calculate_normalize_reflections_operator(reflections_operator.operator, diagonal_operator)

# result_operator = symm

# eiv - eigen value
# eig - eigen vector
eiv, eig = find_max_eigenvalue_with_its_vector(normalize_reflections_operator)

svm = sns.distplot(eig)

figure = svm.get_figure()
figure.savefig('svm.png', dpi=400)

# f = open("result.txt", "w")
# f.write(str(eig))
# f.close()
