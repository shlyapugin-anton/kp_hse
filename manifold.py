import hashlib
import copy 
import sklearn.neighbors as sk
import numpy as np
from scipy.linalg import svd

# формат self.oriented_frame_map[point_pos]:
# [[x_11, ... , x_1p]
#         ...
#  [x_q1, ..., x_qp]]
# где q = dim M - нашего многообразия
# [x_1i, ..., x_qi] - собственный вектор, полученный после pca. Он же - вектор базиса в T_{x_i}(M)

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

    # и это тоже вынести в manifold_methods
    def find_nearest_points(self, point_pos):                               # поиск ближайших точек к точке, позиция которой передается в параметрах
        tree = sk.KDTree(self.points, leaf_size=2)
        ind = tree.query([self.points[point_pos]], k=self.k + 1)[1][0]      # API: на вход подается k - число ближайших соседей, KDTree ищет с учетом самой точки, а ее нужно исключить
        return ind[1:]
    
    # отсюда все тоже вынести в manifold_methods
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

    # Возвращает False - если многообразие неориентируемо
    # Возвращает True - если ориентируемо (в обоих случаях меняет как-то ориентацию на фреймах
    # но в случае ориентированного многообразия - ориентация уже задается на всем многообразии)
    def set_orientation(self):                                                                                  # задаем ориентацию на многообразии на self.oriented_frame_map
                                                                                                                # если ориентацию задать нельзя (многообразие неорентируемо),
                                                                                                                # возвращаем "не ориентируемо"
        # после того, как синхронизировали знак det(Q_i.T, Q_j) - сохраняем об этом информацию в словарь
        # ключ - позиция точки. Значение - список позиций точек, с которыми синхронизировались уже
        syncronised_points_map = {} 
        # итерируемся по соседям точки (фиксируем точки и начинаем синхронизировать фреймы у соседей)
        # и смотрим на знак det(Q_pca(X_i)^T \times Q_pca(X_j))
        # где X_j - одна из "соседних точек", которые хранятся в self.k_ij[point_pos]
        # если определитель меньше нуля - меняем ориентацию первого вектора в Q_pca(X_j) на противоположную 
        # point_pos = i, self.oriented_frame_map(point_pos) = Q_pca(X_i)
        # важный момент: при смене ориентации Q_pca(X_j), так же проверяем, что ничего не сломалось 
        # среди всех det(X_j.T, X_k), где k - те точки, позиции которых уже синхронизированы с j (по сути просто проверяем непустоту словаря по ключу j - т.к. при смене 
        # знака первого вектора в Q_pca_j мы автоматом сломаем все предудыщие синхронизации)
        # (т.е. точки из syncronised_point_map[j])
        # если что-то сломалось - многообразие неориентируемо (т.к. банально не сможем задать ориентацию на фреймах точек (point_pos, j, k) ничего не сломав)
        # если ничего не сломалось - идем дальше
        # по выходу из цикла многообразие ориентируемо (с ориентацией на фреймах) (т.к. не было таких "плохих" троек)
        points_on_syncronised_path = [0]
        met_points_map = {}
        for point_pos in points_on_syncronised_path:
            if point_pos not in met_points_map:
                for j in self.k_ij[point_pos]:
                    if point_pos not in syncronised_points_map or j not in syncronised_points_map[point_pos]:       # синхронизируем только те пары, которые раньше не были синхронизированы
                        Q_pca_i = self.oriented_frame_map[point_pos]
                        Q_pca_j = self.oriented_frame_map[j]                                                        # для согласования с обозначениями в статье и для удобочитаемости
                        det = np.linalg.det(np.matmul(Q_pca_i.T, Q_pca_j))
                        if (det < 0):                                                                               # синхринизируем точки: меняем знак первого вектора в Q_pca_j, проверяем, что ничего не сломалось и добавляем инфу в синхр. мапу
                            self.change_frame_orientation(j)                                                        # меняем ориентацию Q_pca_j 
                            if j in syncronised_points_map:                                                         # значит, X_j уже синхронизирован с какой-то точкой. Значит, смена ориентации X_j разрушит прошлую синхронизацию
                                ### ОТЛАДКА ###
                                print(met_points_map)
                                print("point_pos = " + str(point_pos))
                                print("j = " + str(j))
                                print("syncronised[j] = " + str(syncronised_points_map[j]))
                                print("уже синхронизированы:")
                                for key in met_points_map.keys():
                                    print("key = " + str(key) + ", соседи = " + str(self.k_ij[key]))
                                print("в процессе:")
                                print("key = " + str(point_pos) + ", соседи = " + str(self.k_ij[point_pos]))
                                ### ОТЛАДКА ###
                                return False                                                                        # многообразие неориентируемо
                            syncronised_points_map[j] = [point_pos]                                                 # сюда доходим только если syncronised_points_map[j] был пустым, т.к. иначе - неориентируемо
                            if point_pos not in syncronised_points_map:
                                syncronised_points_map[point_pos] = []
                            syncronised_points_map[point_pos].append(j)
                    if j not in met_points_map:
                        points_on_syncronised_path.append(j)
                met_points_map[point_pos] = True
        return True

    def change_frame_orientation(self, frame_pos):
        for rows_pos in range(len(self.oriented_frame_map[frame_pos])):
            self.oriented_frame_map[frame_pos][rows_pos][0] *= -1

# класс для "больших" методов над многообразием
# сейчас сюда добавляется только pca
class manifold_methods:
    def __init__(self, manifold_data):
        self.manifold_data = manifold_data

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