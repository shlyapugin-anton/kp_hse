import numpy as np
import copy
from scipy.linalg import svd

class apply_pca:
    def __init__(self, d, set_of_points):
        self.dim = d
        # self.number_of_nearest_neighbours = k
        self.set_of_points = copy.deepcopy(set_of_points)
        self.operator = ''
        self.operator2 = '' # оператор, вычисленный по статье

    def centralize_data(self): # Нормализуем данные
        points_sum = sum(self.set_of_points)
        weighted_sum = points_sum / len(self.set_of_points)
        for point_ind in range(len(self.set_of_points)):
            self.set_of_points[point_ind] -= weighted_sum

    def find_x_i(self):
        first_point = np.array([self.set_of_points[0]])
        column_with_1_on_i_pos = np.array([0 for x in range(len(self.set_of_points))])
        column_with_1_on_i_pos[0] += 1
        column_with_1_on_i_pos = np.array([column_with_1_on_i_pos])
        self.operator2 = np.matmul(first_point.T, column_with_1_on_i_pos)
        for point in range(1, len(self.set_of_points)):
            column_with_1_on_i_pos = np.array([0 for x in range(len(self.set_of_points))])
            column_with_1_on_i_pos[point] += 1
            column_with_1_on_i_pos = np.array([column_with_1_on_i_pos])
            row = np.array([self.set_of_points[point]])
            self.operator2 += np.matmul(row.T, column_with_1_on_i_pos)

    def find_transformation_operator_2(self): # Оператор перехода, вычисленный по статье
        U, s, vt = svd(self.operator2)
#        test_2 = [[U[i], s[i]] for i in range(len(U))]
        self.operator2 = U[:self.dim] 
        self.operator = self.operator2.T 



#    def find_transformation_operator(self):
#        point = np.array([self.set_of_points[0]])
#        self.operator = np.matmul(point.T, point)
#        for point in self.set_of_points:
#            point = np.array([point])
#            point_transpose = point.T 
#            self.operator += np.matmul(point_transpose, point)
#        self.operator = self.operator / len(self.set_of_points)
        

#    def calculate_eigenvectors(self):
#        w, v = np.linalg.eig(self.operator)
#        self.operator = v[:self.dim]
#        self.operator = self.operator.T

    def create_orthogonal_matrix_on_nearest_points(self):
        self.centralize_data()
#        self.find_transformation_operator()
#        self.calculate_eigenvectors()
        self.find_x_i()
        self.find_transformation_operator_2()
