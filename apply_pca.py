import numpy as np
import copy

class apply_pca:
    def __init__(self, d, set_of_points):
        self.dim = d
        # self.number_of_nearest_neighbours = k
        self.set_of_points = copy.deepcopy(set_of_points)
        self.operator = ''

    def centralize_data(self): # Нормализуем данные
        for point_ind in range(len(self.set_of_points)):
            self.set_of_points[point_ind] -= self.set_of_points[0]

    def find_transformation_operator(self):
        point = np.array([self.set_of_points[0]])
        self.operator = np.matmul(point.T, point)
        for point in self.set_of_points:
            point = np.array([point])
            point_transpose = point.T 
            self.operator += np.matmul(point_transpose, point)
        self.operator = self.operator / len(self.set_of_points)
        

    def calculate_eigenvectors(self):
        w, v = np.linalg.eig(self.operator)
        self.operator = v[:self.dim]

    def create_orthogonal_matrix_on_nearest_points(self):
        self.centralize_data()
        self.find_transformation_operator()
        self.calculate_eigenvectors()
