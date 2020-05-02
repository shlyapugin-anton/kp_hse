import hashlib
import copy 
import sklearn.neighbors as sk
import numpy as np
from scipy.linalg import svd

class manifold_data:
    def __init__(self, points, k, dim):
        self.points = copy.deepcopy(points)                                 # на вход ожидается np.array, иначе find_nearest_points не будет работать
        self.k_ij = {}                                                      # словарь для функций k_ij. по позиции точки возвращает список ее ближайших соседей (их позиции)
        self.point_position_by_coordinates_hash_map = {}                    # словарь, по хэшу координат точки возвращает ее позицию, т.к. доступ к данным по точке по ее позиции предоставляется
        self.oriented_frame_map = {}                                        # словарь, по позиции точки возвращает Q_pca(X_i)
        self.k = k
        self.dim = dim

    def get_key(self, coordinates):                                         # Рассчитывает хэш по списку координат. Необходим для доступа к позиции точки по ее координатам
        strRepresentation = ''
        for axis in coordinates:
            strRepresentation = strRepresentation + "," + str(axis)
        hash_object = hashlib.sha512(bytes(strRepresentation, encoding='utf-8'))
        return hash_object.hexdigest()                                      # Возвращает хэш от координат
    
    def get_point_pos_by_coordinates(self, coordinates):
        key = self.get_key(coordinates)
        return self.point_position_by_coordinates_hash_map[key]
    
    def find_nearest_points(self, point_pos):                               # поиск ближайших точек к точке, позиция которой передается в параметрах
        tree = sk.KDTree(self.points, leaf_size=2)
        ind = tree.query([self.points[point_pos]], k=self.k + 1)[1][0]      # API: на вход подается k - число ближайших соседей, KDTree ищет с учетом самой точки, а ее нужно исключить
        return ind[1:]
    
    def initialize_manifold_data(self):
        for point_pos in range(len(self.points)):
            self.k_ij[point_pos] = self.find_nearest_points(point_pos)
            self.point_position_by_coordinates_hash_map[self.get_key(self.points[point_pos])] = point_pos       # эта штука выглядит очень страшно, но по сути
                                                                                                                # 1. мы берем координату точки:
                                                                                                                # self.points[point_pos]
                                                                                                                # 2. берем от координаты ключ - высчитываем хэш (self.get_key)
                                                                                                                # 3. именно этот хэш является ключом для получения позиции точки по ее координатам
                                                                                            
            closest_points_coordinates = []
            for point in self.k_ij[point_pos]:
                closest_points_coordinates.append(self.points[point])                                           # а тут по индексам ближайших точек находим их координаты для 
                                                                                                                # скармливания в pca
            pca_algorithm = pca(self.dim, closest_points_coordinates)
            self.oriented_frame_map[point_pos] = pca_algorithm.create_pca_frame()                               # а вот тут находим Q_pca(X_i) из статьи 

# класс для "больших" методов над многообразием
# сейчас сюда добавляется только pca
class manifold_methods:
    def __init__(self, manifold_data):
        self.points = copy.deepcopy(manifold_data.points)                                                       # ожидаю, что manifold_methods и его наследники НЕ изменяют само многообразие, 
                                                                                                                # а делают какие-то рассчеты с ним

class pca(manifold_methods):
    def __init__(self, dim, nearest_points):
#        super().__init__(manifold_data)                                                                        # сейчас этот код не нужен, т.к. для pca не нужны ВСЕ точки
        self.dim = dim
        self.nearest_points = nearest_points
        self.operator = ''

    def centralize_date(self):                                                                                  # нормализуем данные
        points_sum = sum(self.nearest_points)
        weighted_sum = points_sum / len(self.nearest_points)
        for point_ind in range(len(self.nearest_points)):
            self.nearest_points[point_ind] -= weighted_sum

    def find_x_i(self):
        first_point = np.array([self.nearest_points[0]])
        column_with_1_on_i_pos = np.array([0 for x in range(len(self.nearest_points))])
        column_with_1_on_i_pos[0] += 1
        column_with_1_on_i_pos = np.array([column_with_1_on_i_pos])
        self.operator = np.matmul(first_point.T, column_with_1_on_i_pos)
        for point in range(1, len(self.nearest_points)):
            column_with_1_on_i_pos = np.array([0 for x in range(len(self.nearest_points))])
            column_with_1_on_i_pos[point] += 1
            column_with_1_on_i_pos = np.array([column_with_1_on_i_pos])
            row = np.array([self.nearest_points[point]])
            self.operator += np.matmul(row.T, column_with_1_on_i_pos)   

    def find_transformation_operator(self):                                                                  # применяем SVD для поиска нужного оператора
        U, s, vt = svd(self.operator)
        self.operator = U[:self.dim] 
        self.operator = self.operator.T    

    def create_pca_frame(self):
        self.centralize_date()
        self.find_x_i()
        self.find_transformation_operator()
        return self.operator