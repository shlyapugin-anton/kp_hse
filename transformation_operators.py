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
# close_distance - параметр для определения близости точек
class transformation_matrices_between_near_point:
    def __init__(self, store_mn_info, close_distance):
        self.mn_info = store_mn_info # class store_manifold_info
        self.close_distance = close_distance # какое расстояние мы считаем "близким" для тчоек
        # Инициализируем список с операторами перехода
        self.transformation_operators = [[0 for i in range(len(store_mn_info.set_of_points))] for j in range(len(store_mn_info.set_of_points))] # матрица с операторами перехода
    
    def initialize_transformation_operators(self):
        for point_pos in range(len(self.mn_info.set_of_points)):
            # Получить точку в этой позиции
            # Получить ее хеш
            # Получить оператор по хешу в этой точке (хранятся в mn_info.manifold_info[key].operator)
            # Помноить транспонированный оператор и обычный
            # Применить svd
            # Перемножить 2 оператора из svd (это и будет оператором перехода)
            first_point = store_point_info.store_point_info(self.mn_info.set_of_points[point_pos], '') # преобразуем nd_array к экземпляру класса store_point_info
            first_point_hash = self.mn_info.pos_key_map[point_pos]
            for second_point_pos in range(len(self.mn_info.set_of_points)):
                second_point = store_point_info.store_point_info(self.mn_info.set_of_points[second_point_pos], '')
                # ===
                # Инициализация первого оператора
                if (first_point.distance_between_point(second_point) < self.close_distance): # НЕ ПРОТЕСТИРОВАНО
 #                   point = self.mn_info.set_of_points[point_pos]
 #                   point_for_hash = store_point_info.store_point_info(point, '')
  #                  point_for_hash.get_coordinates_hashed()

#                    key = point_for_hash.coordinates_hashed

                    first_operator = self.mn_info.manifold_info[first_point_hash].operator
                    # Конец инициализации первого оператора (такие блоки говорят о том, что store_manifold_info подготовлен плохо. Там нужен метод, позволяющий вытягивать оператор по списку точек)
                    # ===

                    # Инициализация второго оператора
                    # ===
                    second_point_hash = self.mn_info.pos_key_map[second_point_pos]
  #                  point = self.mn_info.set_of_points[second_point_pos]
   #                 point_for_hash = store_point_info.store_point_info(point, '')
    #                point_for_hash.get_coordinates_hashed()

 #                   key = point_for_hash.coordinates_hashed

                    second_operator = self.mn_info.manifold_info[second_point_hash].operator
                    # Конец инициализации второго оператора
                    # ===

                    almost_O_ij = np.matmul(first_operator, second_operator.T)

                    U, s, vt = svd(almost_O_ij)

                    self.transformation_operators[point_pos][second_point_pos] = np.matmul(U, vt)
