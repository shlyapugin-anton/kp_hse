import hashlib
import copy

class store_point_info:
    def __init__(self, coordinates, operator):
        self.coordinates = copy.deepcopy(coordinates)
        self.coordinates_string = '' # стринговое представление координат
        self.coordinates_hashed = '' # хэш от стринги - координат точки
        self.operator = copy.deepcopy(operator) # ортогональный оператор 

    def get_string_repr_of_coordinates(self):
        for axis in range(len(self.coordinates)):
            self.coordinates_string = self.coordinates_string + 'x' + str(axis) + ':' + str(self.coordinates[axis]) + ','
    
    def get_coordinates_hashed(self): 
        if (self.coordinates_hashed == ''):
            self.get_string_repr_of_coordinates()
        hash_object = hashlib.sha512(bytes(self.coordinates_string, encoding='utf-8'))
        self.coordinates_hashed = hash_object.hexdigest()
