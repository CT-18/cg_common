import numpy as np
from functools import total_ordering, reduce
from numpy.linalg import det
from cg.utils import gcd


@total_ordering
class Point:
    """
    Класс определяет точку в однородных координатах.
    """

    def __init__(self, x: 'int or np.ndarray', *args, homogeneous=False):
        """
        Конструктор n-мерной точки в однородных координатах.
        Подразумевается две перегрузки:
            >>> Point(0, 0) # конструктор от n аргументов (n > 1)
            (0; 0; 1)
            >>> Point(np.array([0, 0, 1]), homogeneous=True) # конструктор от numpy.ndarray
            (0; 0; 1)

        :param x: список координат в numpy.ndarray или первая координата int
        :param args: если первый аргумент int, то кортеж int - остальных координат
        """
        if isinstance(x, np.ndarray) and (x.dtype == np.int32 or x.dtype == np.int64):
            if not homogeneous:
                self.coord = np.append(x, [1])
            else:
                self.coord = x
        elif len(args) > 0 and all(isinstance(y, int) for y in args) and isinstance(x, int):
            self.coord = [x] + list(args)
            if not homogeneous:
                self.coord += [1]
            self.coord = np.array(self.coord)
        else:
            raise Exception('unable to create HomogeneousPoint')

    def same_level(self, other):
        """
        проецирует точки на одну и ту же гиперплоскость (не обязательно на единичной высоте)

        :param other: экземпляр Point
        :return: пара из координат точек на одной гиперплоскости
        """
        m = gcd(self.coord[-1], other.coord[-1])
        self_height = other.coord[-1] // m
        other_height = self.coord[-1] // m
        return self.coord * self_height, other.coord * other_height

    def __add__(self, other):
        """
        сумма точек.

        :param other: экземпляр Point
        :return: сумма точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        res = Point(new_self + new_other, homogeneous=True)
        res.coord[-1] //= 2
        return res

    def __sub__(self, other):
        """
        разность точек.

        :param other: экземпляр Point
        :return: разность точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        h = new_self[-1]
        res = Point(new_self - new_other, homogeneous=True)
        res.coord[-1] = h
        return res

    def __neg__(self):
        """
        арифметическое отрицание точки.

        :return: арифметическое отрицание точки в однородных координатах
        """
        res = Point(-self.coord, homogeneous=True)
        res.coord[-1] *= -1
        return res

    def __mul__(self, other: int):
        """
        умножение точки на константу
        """
        res = Point(self.coord * other, homogeneous=True)
        res.coord[-1] //= other
        return res

    def __rmul__(self, other: int):
        return other * self

    def __eq__(self, other):
        """
        проверяет две точки в однородных координатах на равенство

        :param other: экземпляр Point
        :return:
        """
        return np.all(self.coord * other.coord[-1] == other.coord * self.coord[-1])

    def __lt__(self, other):
        """
        Лексикографическое "меньше" для однородных точек

        :param other: экземпляр Point
        :return:
        """
        slf, otr = self.same_level(other)
        differs, = np.where(slf != otr)
        less = (slf < otr)[differs]
        return len(less) != 0 and less[0]

    def __getitem__(self, item):
        """
        getter для координат точек

        :param item: индекс
        :return: координата по оси item
        """
        return self.coord[item]

    def __str__(self):
        return '({0})'.format('; '.join(map(str, self.coord)))

    def __repr__(self):
        suffix = ''
        coord = self.coord[:-1].tolist()
        if self.coord[-1] != 1:
            suffix = ', homogeneous=True'
            coord.append(self.coord[-1])
        return 'Point({0}{1})'.format(', '.join(map(str, coord)), suffix)

    def is_finite(self):
        return self.coord[-1] != 0


def vol(point: Point, *hyperplane):
    return det(np.array([pt.coord for pt in hyperplane] + [point.coord], dtype=np.int32))


def turn(point: Point, *hyperplane):
    return np.sign(vol(point, *hyperplane))
