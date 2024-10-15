import torch

class Matrix(object):
    def __init__(self, vectors):
        self.vectors = vectors
        self.num_samples = vectors.shape[0]
        self.dimension = vectors.shape[1]

    def __str__(self):
        return self.vectors

    def plus(self, v):
        return self.vectors + v.vectors

    def minus(self, v):
        return self.vectors - v.vectors

    def magnitude(self):
        return torch.sqrt(torch.sum(torch.pow(self.vectors, 2), dim=-1))


    def normalized(self):
        magnitude = self.magnitude()
        weight = (1.0 / magnitude).reshape(self.num_samples, 1)
        return self.vectors * weight

    def component_parallel_to(self, basis):
        u = basis.normalized()
        weight = torch.sum(self.vectors * u, dim=-1).reshape(self.num_samples, 1)
        return u * weight

    def component_orthogonal_to(self, basis):
        projection = self.component_parallel_to(basis)
        return self.vectors - projection


def Ortho_algorithm(primary_feature, auxiliary_feature):
    primary_feature = Matrix(primary_feature)
    auxiliary_feature = Matrix(auxiliary_feature)
    useful_feature = auxiliary_feature.component_orthogonal_to(primary_feature)
    useful_feature = Matrix(useful_feature)
    res = primary_feature.plus(useful_feature)
    return res

