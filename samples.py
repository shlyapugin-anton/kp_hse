import numpy as np 
import math

class samples:
    def generate_point_on_sphere(self, mean, cov, n, seed):
        sample_on_sphere = points_on_sphere(mean, cov, n, seed)
        return sample_on_sphere.normalize_points_set()
    
    def generate_point_on_mobius(self, n, seed):
        sample_on_mobius = points_on_mobius(n, seed)
        return sample_on_mobius.construct_mobius_strip()

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

class points_on_mobius:
    def __init__(self, n, seed):
        self.n = n
        np.random.seed(seed)
        self.theta = np.random.uniform(0, 2 * math.pi, n)
        self.w = np.random.uniform(-1, 1, n)
        self.points = list(zip(self.theta, self.w))
    
    def construct_mobius_strip(self):
        points = []
        for point in self.points:
            theta = point[0]
            w = point[1]
            phi = 0.5 * theta
            r = 1 + w * np.cos(phi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = w * np.sin(phi)
            point = [x, y, z]
            points.append(point)
        return np.array(points)