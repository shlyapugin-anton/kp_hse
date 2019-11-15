import sklearn.neighbors as sk
import apply_pca
import copy

# Применяем Local PCA для одной (выделенной) точки многообразия (ind_of_main_point - индекс этой точки)
class apply_local_pca:
    def __init__(self, set_of_points, k, dim, ind_of_main_point):
        self.set_of_all_points = copy.deepcopy(set_of_points)
        self.k = k # Число ближайших соседей для точки, на которых строим ортогональные операторы
        self.dim = dim
        self.set_of_closest_points = ''
        self.ind_of_main_point = ind_of_main_point
        self.pca = ''

    
    def find_nearest_points(self):
        tree = sk.KDTree(self.set_of_all_points, leaf_size=2) # Почему два, кстати?
        ind = ""
        ind = tree.query([self.set_of_all_points[self.ind_of_main_point]], k=self.k)[1][0]
        self.set_of_closest_points = [self.set_of_all_points[pos] for pos in ind]

    def apply_pca_to_closest_points(self):
        self.find_nearest_points()
        self.pca = apply_pca.apply_pca(self.dim, self.set_of_closest_points)
        self.pca.create_orthogonal_matrix_on_nearest_points()
