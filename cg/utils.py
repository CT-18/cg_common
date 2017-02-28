from functools import reduce
import numpy as np


def gcd(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 1
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    return (a * b) // gcd(a, b)


def reduce_fraction(a: int, b: int) -> (int, int):
    d = gcd(a, b)
    return a // d, b // d


def int_det(A: np.ndarray) -> int:
    """
    целочисленный детерминант матрицы A

    :param A: квадратная матрица
    :return: определитель A
    """
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            d = gcd(A[i][i], A[j][i])
            A[j] = A[j] * (A[i][i] // d) - A[i] * (A[j][i] // d)
    return A.diagonal().prod()


def cmp_(x, y) -> int:
    if x == y:
        return 0
    if x > y:
        return 1
    return -1


class PointTestGenerator:
    dimensions = 10
    defaultTestsNumber = 500
    defaultSetSize = 500
    normalRandomSTD, normalRandomExpected = 1., 0.
    uniformRandomLower, uniformRandomUpper = -1., 1.
    intRandomLower, intRandomUpper = -1000, 1000
    cumulativeTestCount = 0

    @staticmethod
    def generate(generator, dimensions=None, size=None, **kwargs):
        """
        генератор множества точек

        :param generator: фугнция, генерирующая координаты
        :param dimensions: размерность пространства
        :param size: количество точек в множестве
        :param kwargs:
        :return: итератор по точкам в множестве
        """
        if 'testNumber' in kwargs:
            del kwargs['testNumber']
        pointSet = PointTestGenerator.generateSet(generator, dimensions=dimensions, size=size, testNumber=1, **kwargs)
        pointSet = next(pointSet)
        for point in pointSet:
            yield point

    @staticmethod
    def generateSet(generator, dimensions=None, size=None, testNumber=None, **kwargs):
        """
        общий генератор для множеств точек

        :param generator: фугнция, генерирующая координаты
        :param dimensions: размерность пространства
        :param size: количество точек в множестве
        :param testNumber: количество множеств
        :param kwargs:
        :return: итератор по точкам в множестве
        """
        if dimensions is None:
            dimensions = range(2, PointTestGenerator.dimensions + 1)
        elif isinstance(dimensions, int):
            dimensions = [dimensions]
        if size is None:
            size = PointTestGenerator.defaultSetSize
        if testNumber is None:
            testNumber = PointTestGenerator.defaultTestsNumber
        for dim in dimensions:
            points = generator(size=dim * size * testNumber, **kwargs).reshape((testNumber, size, dim))
            for i in range(points.shape[0]):
                yield points[i, ...]

    @staticmethod
    def generateNormal(dimensions=None, size=None, testNumber=None):
        """
        генератор с нормальным распределением.

        :param dimensions: размерность
        :param size: размер множества точек
        :param testNumber: количество множеств
        :return: если testNumber пропущен, возвращает итератор по точкам в множестве
                иначе по множествам
        """
        if testNumber is None:
            func = PointTestGenerator.generate
        else:
            func = PointTestGenerator.generateSet
        return func(np.random.normal, dimensions=dimensions, size=size,
                    loc=PointTestGenerator.normalRandomExpected,
                    scale=PointTestGenerator.normalRandomSTD,
                    testNumber=testNumber)

    @staticmethod
    def generateUniform(dimensions=None, size=None, testNumber=None):
        """
        генератор с равномерным распределением.

        :param dimensions: размерность
        :param size: размер множества точек
        :param testNumber: количество множеств
        :return: если testNumber пропущен, возвращает итератор по точкам в множестве
                иначе по множествам
        """
        if testNumber is None:
            func = PointTestGenerator.generate
        else:
            func = PointTestGenerator.generateSet
        return func(np.random.uniform, dimensions=dimensions, size=size,
                    low=PointTestGenerator.uniformRandomLower,
                    high=PointTestGenerator.uniformRandomUpper,
                    testNumber=testNumber)

    @staticmethod
    def generateInteger(dimensions=None, size=None, testNumber=None):
        """
        генератор точек с целыми координатами (равномерно распределенными).

        :param dimensions: размерность
        :param size: размер множества точек
        :param testNumber: количество множеств
        :return: если testNumber пропущен, возвращает итератор по точкам в множестве
                иначе по множествам
        """
        if testNumber is None:
            func = PointTestGenerator.generate
        else:
            func = PointTestGenerator.generateSet
        return func(np.random.randint, dimensions=dimensions, size=size,
                    low=PointTestGenerator.intRandomLower,
                    high=PointTestGenerator.intRandomUpper,
                    testNumber=testNumber)
