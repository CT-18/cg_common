import numpy as np
from functools import total_ordering, reduce
from numpy.linalg import det
from .utils import gcd, reduce_fraction
from fractions import Fraction
from operator import mul, attrgetter, methodcaller


@total_ordering
class Point:
    """
    Класс задает точку в афинном пространстве.
    Определены арифметические операции и лексикографическое сравнение.
    """

    def __init__(self, x, *args):
        """
        Конструктор n-мерной точки.
        Подразумевается две перегрузки:
            >>> Point(0., 0.) # конструктор от n аргументов (n > 0)
            (0.0; 0.0)
            >>> Point(np.array([0., 0.])) # конструктор от numpy.ndarray
            (0.0; 0.0)

        :param x: список координат в numpy.ndarray или первая координата float
        :param args: если первый аргумент float, то кортеж float - остальных координат
        """
        if isinstance(x, np.ndarray):
            self.coord = x
        else:
            self.coord = np.array([x] + list(args))
        self.coord = self.coord.astype(np.float64)

    def __add__(self, other):
        """
        покоординатная сумма двух точек.
        для ускорения, используем векторизацию.

        :param other: экземпляр Point
        :return: сумма точки self и other
        """
        return Point(self.coord + other.coord)

    def __sub__(self, other):
        """
        покоординатная разность двух точек.

        :param other: экземпляр Point
        :return: разность точки self и other
        """
        return Point(self.coord - other.coord)

    def __neg__(self):
        """
        арифметическое отрицание координат точки
        """
        return Point(-self.coord)

    def __mul__(self, other: float):
        """
        умножение точки на константу

        :return:
        """
        return Point(self.coord * other)

    def __rmul__(self, other):
        """
        умножение точки на константу (справа)

        :return:
        """
        return Point(self.coord * other)

    def __eq__(self, other):
        """
        сравнение двух точек на покоординатное равенство

        :param other: экземпляр Point
        """
        return np.all(self.coord == other.coord)

    def __lt__(self, other):
        """
        проверка на "<" лексикографически

        :param other: экземпляр Point
        """
        differs, = np.where(self.coord != other.coord)
        less = (self.coord < other.coord)[differs]
        return len(less) != 0 and less[0]

    def __str__(self):
        return '({0})'.format('; '.join(map(str, self.coord.tolist())))

    def __repr__(self):
        return str(self)


def vol(point: Point, *hyperplane):
    return det(np.array([pt.coord for pt in hyperplane]) - point.coord)


def turn(point: Point, *hyperplane):
    return np.sign(vol(point, *hyperplane))


@total_ordering
class HomogeneousPoint:
    """
    Класс определяет точку в однородных координатах.
    """

    def __init__(self, x, *args):
        """
        Конструктор n-мерной точки в однородных координатах.
        Подразумевается две перегрузки:
            >>> Point(0, 0) # конструктор от n аргументов (n > 1)
            (0; 0)
            >>> Point(np.array([0, 0, 1])) # конструктор от numpy.ndarray
            (0; 0; 1)

        :param x: список координат в numpy.ndarray или первая координата float
        :param args: если первый аргумент float, то кортеж float - остальных координат
        """
        if isinstance(x, np.ndarray):
            self.coord = x
        elif len(args) > 0:
            self.coord = np.array([x] + list(args))
        else:
            raise Exception('unable to create HomogeneousPoint')
        if self.coord.dtype != np.int64:
            fractions = list(map(Fraction, self.coord))
            denominators = list(map(attrgetter('denominator'), fractions))
            numerators = list(map(attrgetter('numerator'), fractions))
            last_num, last_denum = numerators[-1], denominators[-1]
            del denominators[-1]
            lcm = last_denum * (reduce(mul, denominators) // reduce(gcd, denominators))
            numerators = [n * lcm // d for n, d in zip(numerators, denominators)]
            numerators.append(last_num * lcm // last_denum)
            self.coord = np.array(numerators, dtype=np.int64)

    def same_level(self, other):
        """
        проецирует точки на одну и ту же гиперплоскость (не обязательно на единичной высоте)

        :param other: экземпляр HomogeneousPoint
        :return: пара из координат точек на одной гиперплоскости
        """
        m = gcd(self.coord[-1], other.coord[-1])
        self_height = other.coord[-1] / m
        other_height = self.coord[-1] / m
        return self.coord * self_height, other.coord * other_height

    def __add__(self, other):
        """
        сумма точек.

        :param other: экщемпляр HomogeneousPoint
        :return: сумма точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        res = HomogeneousPoint(new_self + new_other)
        res.coord[-1] //= 2
        return res

    def __sub__(self, other):
        """
        разность точек.

        :param other: экщемпляр HomogeneousPoint
        :return: разность точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        h = new_self[-1]
        res = HomogeneousPoint(new_self - new_other.coord)
        res.coord[-1] = h
        return res

    def __neg__(self):
        """
        арифметическое отрицание точки.

        :return: арифметическое отрицание точки в однородных координатах
        """
        res = HomogeneousPoint(-self.coord)
        res.coord[-1] *= -1
        return res

    def __mul__(self, other: float):
        """
        умножение точки на константу
        """
        value = Fraction(other)
        num, denom = value.numerator(), value.denominator()
        res = HomogeneousPoint(self.coord * num)
        res.coord[-1] //= num
        res.coord[-1] *= denom
        return res

    def __rmul__(self, other):
        return other * self

    def __eq__(self, other):
        """
        проверяет две точки в однородных координатах на равенство

        :param other: экземпляр HomogeneousPoint
        :return:
        """
        return np.all(self.coord * other.coord[-1] == other.coord * self.coord[-1])

    def __lt__(self, other):
        """

        :param other:
        :return:
        """
        return self.to_affine() < other.to_affine()

    def is_finite(self):
        return self.coord[-1] != 0

    def to_affine(self):
        if not self.is_finite():
            raise Exception('No representation in affine space')
        return Point(self.coord[:-1] / self.coord[-1])


if __name__ == '__main__':
    p = HomogeneousPoint(1.25, 1.75, 1)
    q = Point(0, 2.)
    print(p)
