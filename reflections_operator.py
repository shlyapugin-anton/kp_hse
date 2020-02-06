import generate_point_on_sphere
import apply_pca
import apply_local_pca
import store_point_info
import store_manifold_info
from scipy.linalg import svd
import numpy as np
import transformation_operators

# Класс для хранения информации об отражениях (в статье обозначается как Z - n \times n оператор из определителей матриц перехода)
class reflections_operator:
    def __init__(self, tr_operators):
        self.tr_operators = tr_operators
        self.operator = [[0 for i in range(len(tr_operators.transformation_operators))] for j in range(len(tr_operators.transformation_operators))]

    def initialize_reflections_operator(self):
        for first_operator_pos in range(len(self.tr_operators.transformation_operators)):
            for second_operator_pos in range(len(self.tr_operators.transformation_operators)):
                # если точки не близкие, то Z_ij = 0. Иначе Z_ij = +-1 (в зависимости от того, нужно ли отражение для перехода между двумя точками)
                try:
                    self.operator[first_operator_pos][second_operator_pos] = np.linalg.det(self.tr_operators.transformation_operators[first_operator_pos][second_operator_pos])
                except:
                    self.operator[first_operator_pos][second_operator_pos] = 0