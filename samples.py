import numpy as np 


class points_on_sphere:
    def __init__(self, mean, cov, n, seed):
        self.n = n
        self.mean = mean
        self.cov = cov
        np.random.seed(seed);
        self.points = np.random.multivariate_normal(self.mean, self.cov, self.n) # для нормального

    def find_length(self, point):
        length = 0
        for axis in point:
            length = length + axis ** 2
        return length ** 0.5

    def normalize_point(self, point):
        length = self.find_length(point)
        if (length != 0):
            return point / length
        else:
            return []
        
    def normalize_points_set(self):
        for point_pos in range(len(self.points)):
            point = self.normalize_point(self.points[point_pos])
            self.points[point_pos] = point
        return self.points

    def check_length_of_each_point(self):
        for point in self.points:
            length = 0
            for axis in point:
                length += axis ** 2
            print(length)
