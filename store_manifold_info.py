import copy
import apply_local_pca
import store_point_info

# Храним информацию о нашем исходном многообразии, а именно:
# храним словарь, где ключ - хеш от строки координат точки (строка вида x0:данные,x1:данные,...xn:данные, - от нее берем хеш, это ключ)
# значение: объект store_point_info для этой точки
class store_manifold_info:
    def __init__(self, set_of_points, number_of_nearest_neighbours, dimension_of_operator):
        # Список со списком точек на многообразии
        self.set_of_points = copy.deepcopy(set_of_points)

        # Количество ближайших соседей, на которых строим ортогональный оператор
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # Размерность исходных данных (предполагаемая):
        self.dimension_of_operator = dimension_of_operator

        # Словарь с информацией по каждой точке многообразия:
        # Структура и смысл описан в комментариях к классу
        self.manifold_info = {}

    def initialize_orthonormal_operators(self):
        
        for point_pos in range(len(self.set_of_points)):
            local_pca = apply_local_pca.apply_local_pca(self.set_of_points, self.number_of_nearest_neighbours, self.dimension_of_operator, point_pos)
            local_pca.apply_pca_to_closest_points()

            point_info = store_point_info.store_point_info(self.set_of_points[point_pos], local_pca.pca.operator) # Структура данных для хранения информации о точке

            point_info.get_coordinates_hashed()

            self.manifold_info[point_info.coordinates_hashed] = point_info # Заполняем словарь с информацией о многобразии в каждой точке

    def return_point_by_its_coordinates(self, coordinates):
        point_info = store_point_info.store_point_info(coordinates, [])
        point_info.get_coordinates_hashed()
        return self.manifold_info[point_info.coordinates_hashed]
