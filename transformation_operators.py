import generate_point_on_sphere
import apply_pca
import apply_local_pca
import store_point_info
import store_manifold_info
from scipy.linalg import svd
import numpy as np

# Класс для хранения информации об операторах перехода между ближними точками
# На вход:
# Количество ближайших точек, для которых строим O_ij - операторы перехода
# инстанс store_manifold_info - хранящий информацию о многообразии 
# Сейчас класс может построить только матрицы O_ii
# Впоследствии подправлю для находления 
class transformation_matrices_between_near_point:
    def __init__(self, store_mn_info):
        self.mn_info = store_mn_info
        # Инициализируем список с операторами перехода
        self.transformation_operators = [[0 for i in range(len(store_mn_info.set_of_points))] for j in range(len(store_mn_info.set_of_points))]
    
    def initialize_transformation_operators(self):
        # При доработке нужно будет добавить второй цикл по тем же точкам
        # Можно прямо сейчас добавить цикл и проверять расстояние между точками и уже строить операторы,
        # Но мне совершенно не нравится сложность вычислений, кажется можно сделать сильно эффективнее
        for point_pos in range(len(self.mn_info.set_of_points)):
            # Получить точку в этой позиции
            # Получить ее хеш
            # Получить оператор по хешу в этой точке (хранятся в mn_info.manifold_info[key].operator)
            # Помноить транспонированный оператор и обычный
            # Применить svd
            # Перемножить 2 оператора из svd (это и будет оператором перехода)
            for second_point_pos in range(len(self.mn_info.set_of_points)):
                # ===
                # Инициализация первого оператора
                if (True): # Переопределить, тут должно быть вычисление "близости" точек и в зависимости от этого определяем матрицы перехода
                    point = self.mn_info.set_of_points[point_pos]
                    point_for_hash = store_point_info.store_point_info(point, '')
                    point_for_hash.get_coordinates_hashed()

                    key = point_for_hash.coordinates_hashed

                    first_operator = self.mn_info.manifold_info[key].operator
                    # Конец инициализации первого оператора (такие блоки говорят о том, что store_manifold_info подготовлен плохо. Там нужен метод, позволяющий вытягивать оператор по списку точек)
                    # ===

                    # Инициализация второго оператора
                    # ===
                    point = self.mn_info.set_of_points[second_point_pos]
                    point_for_hash = store_point_info.store_point_info(point, '')
                    point_for_hash.get_coordinates_hashed()

                    key = point_for_hash.coordinates_hashed

                    second_operator = self.mn_info.manifold_info[key].operator
                    # Конец инициализации второго оператора
                    # ===

                    almost_O_ij = np.matmul(first_operator.T, second_operator)

                    U, s, vt = svd(almost_O_ij)

                    self.transformation_operators[point_pos][second_point_pos] = np.matmul(U, vt)
