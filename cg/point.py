import numpy as np
from functools import total_ordering, reduce, cmp_to_key
from numpy.linalg import det
from cg.utils import gcd, cmp_ as compare
from typing import Callable


@total_ordering
class Point:
    """
    Класс определяет точку в однородных координатах.
    """

    def __init__(self, x: 'int or np.ndarray', *args, homogeneous=False):
        """
        Конструктор n-мерной точки в однородных координатах.
        Возможные способы задания точки:
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

    def same_level(self, other) -> (np.ndarray, np.ndarray):
        """
        проецирует точки на одну и ту же гиперплоскость (не обязательно на единичной высоте)

        :param other: экземпляр Point
        :return: (np.ndarray, np.ndarray) пара из координат точек на одной гиперплоскости
        """
        m = gcd(self.coord[-1], other.coord[-1])
        self_height = other.coord[-1] // m
        other_height = self.coord[-1] // m
        return self.coord * self_height, other.coord * other_height

    def __add__(self, other) -> 'Point':
        """
        сумма точек.

        :param other: экземпляр Point
        :return: Point сумма точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        res = Point(new_self + new_other, homogeneous=True)
        res.coord[-1] //= 2
        return res

    def __sub__(self, other) -> 'Point':
        """
        разность точек.

        :param other: экземпляр Point
        :return: Point разность точек в однородных координатах
        """
        new_self, new_other = self.same_level(other)
        h = new_self[-1]
        res = Point(new_self - new_other, homogeneous=True)
        res.coord[-1] = h
        return res

    def __neg__(self) -> 'Point':
        """
        арифметическое отрицание точки.

        :return: Point арифметическое отрицание точки в однородных координатах
        """
        res = Point(-self.coord, homogeneous=True)
        res.coord[-1] *= -1
        return res

    def __mul__(self, other: int) -> 'Point':
        """
        умножение точки на константу

        :return: Point
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
        :return: bool
        """
        return np.all(self.coord * other.coord[-1] == other.coord * self.coord[-1])

    def __lt__(self, other):
        """
        Лексикографическое "меньше" для однородных точек

        :param other: экземпляр Point
        :return: bool
        """
        slf, otr = self.same_level(other)
        differs, = np.where(slf != otr)
        less = (slf < otr)[differs]
        return len(less) != 0 and less[0]

    def __getitem__(self, item: int) -> int:
        """
        getter для координат точек

        :param item: индекс
        :return: int координата по оси item
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
        """
        предикат который возвращает True тогда и только тогда когда
        точка является бесконечно удаленной

        :return: bool
        """
        return self.coord[-1] != 0

    def dim(self):
        """
        размерность пространства которому принадлежит точка

        :return: int
        """
        return len(self.coord) - 1


def vol(point: Point, *hyperplane):
    return det(np.array([pt.coord for pt in hyperplane] + [point.coord], dtype=np.int32))


def turn(point: Point, *hyperplane):
    return np.sign(vol(point, *hyperplane))


class PointSet:
    """
    Класс описывающий множество точек (в однородных координатах),
    все координаты хранятся в одном массиве
    для эффективности
    """
    def __init__(self, initial=None):
        """
        Конструктор множества точек
        возможные способы задания множеств:
        >>> PointSet(2) # конструктор от размерности пространства
        PointSet(2)
        >>> PointSet(np.array([[1, 2], [3, 4]])) # конструктор от np.ndarray
        PointSet(Point(1, 2), Point(3, 4))
        >>> PointSet([Point(1, 2), Point(3, 4)]) # конструктор от списка точек
        PointSet(Point(1, 2), Point(3, 4))
        :param initial: одно из трех: число (int), np.ndarray, список Point
        """
        if isinstance(initial, int):
            self.points = np.array([], dtype=np.int32).reshape((-1, initial + 1))
            self.size = 0
        elif isinstance(initial, np.ndarray):
            assert len(initial.shape) == 2 and (initial.dtype == np.int32 or initial.dtype == np.int64)
            self.points = np.array(initial, dtype=np.int32)
            self.size = len(initial)
        elif isinstance(initial, list):
            if all(map(lambda x: isinstance(x, Point) and x.dim() == initial[0].dim(), initial)):
                self.points = np.array([point.coord for point in initial], dtype=np.int32)
            elif all(map(lambda x: isinstance(x, np.ndarray) and x.shape == initial[0].shape, initial)):
                self.points = np.array(initial, dtype=np.int32)
            else:
                raise Exception('wrong argument')
            self.size = len(self.points)
        else:
            raise Exception('wrong argument')

    def __iter__(self):
        """
        итератор по точкам множества
        зачительно более мендленный чем .map или .map_coord

        :return:
        """
        for __i in range(len(self)):
            if __i == self.size:
                raise StopIteration
            yield Point(self.points[__i], homogeneous=True)

    def __apply(self, func, axis):
        return np.apply_along_axis(func, axis, self.points[:self.size])

    def __apply_all(self, func, axis):
        return np.apply_along_axis(func, axis, self.points)

    def append(self, point: Point):
        """
        добавляет точку в конец массива содержащего множество

        :param point: точка
        :return: None
        """
        if point.dim() != self.dim():
            raise Exception('incorrect dimension')
        if self.size == 0:
            self.points = np.append(self.points, point.coord.reshape((1, -1)), axis=0)
        elif self.size == len(self.points):
            new = np.zeros((len(self.points) * 2, self.dim() + 1), dtype=self.points.dtype)
            new[:len(self.points)] = self.points
            self.points = new
        self.points[self.size] = point.coord
        self.size += 1

    def pop(self) -> Point:
        """
        удаляет последний элемент множества

        :return: Point последний элемент
        """
        if self.size == 0:
            raise Exception('Set is empty')
        last = Point(self.points[self.size - 1], homogeneous=True)
        self.size -= 1
        if self.size == 0:
            self.points = np.array([], dtype=np.int32).reshape((-1, self.dim() + 1))
        else:
            self.points[self.size] = np.zeros(self.dim() + 1, dtype=np.int32)
        if self.size * 2 == len(self.points):
            self.points = self.points[:self.size]
        return last

    def insert(self, index: int, point: Point):
        """
        вставляет точку на позицию index

        :param index: позиция для вставки (int)
        :param point: точка (Point)
        :return: None
        """
        self.append(point)
        self.points[index + 1:self.size] = self.points[index:self.size - 1]
        self.points[index] = point.coord
        pass

    def delete(self, index) -> Point:
        """
        удаляет точку по индексу

        :param index: int позиция удаления
        :return: Point удаленная точка
        """
        deleted = self.points[index]
        self.points[index:self.size - 1] = self.points[index + 1:self.size]
        self.pop()
        return deleted

    def __getitem__(self, item: int) -> Point:
        """
        геттер для точек в множестве.
        вызов PointSet(...).__getitem__(item) эквивалентен PointSet(...)[item]

        :param item: int - индекс
        :return: Point точка по номеру
        """
        if item >= self.size:
            raise IndexError
        return Point(self.points[item], homogeneous=True)

    def __setitem__(self, item: int, point: Point):
        """
        сеттер для точек в множестве.
        вызов PointSet(...).__setitem__(item, point) эквивалентен PointSet(...)[item] = point

        :param item: int номер точки
        :param point: Point
        :return: None
        """
        if point.dim() != self.dim() and point.coord.dtype == np.int32:
            raise Exception('incorrect point')
        self.points[item] = point.coord

    def dim(self) -> int:
        """
        размерность простравства, которому принадлежат точки

        :return: int
        """
        return self.points.shape[-1] - 1

    def __len__(self):
        """
        количество точек в множестве

        :return: int
        """
        return self.size

    def map(self, func: Callable[..., int]):
        """
        применяет функцию к координатам каждой точки

        :param func: функция от массива координат (np.ndarray). для эффективности,
            она принимает не Point а массив координат для него. Массив удовлетворяет тем же инвариантам,
            как если бы он был внутри класса Point
        :return: int результат поэлементного применения функции func
        """
        return self.__apply(func, axis=1)

    def map_coords(self, func: Callable[..., int]):
        """
        применяет функцию последовательно, к каждой координате всех точек сразу

        :param func: функция от массива координат (np.ndarray)
        :return: int результат поэлементного применения функции func
        """
        return self.__apply(func, axis=0)

    def filter(self, pred: Callable[..., bool]) -> 'PointSet':
        """
        фильтрует множество, оставляя только точки удовлетворяющие предикату

        :param pred: предикат
        :return: PointSet новое множество
        """
        select = np.logical_and(self.__apply_all(pred, axis=1), np.arange(len(self.points)) < len(self))
        return PointSet(self.points[select])

    def maximize_arg(self, func: Callable[..., int]) -> int:
        """
        индекс точки на которой достигается максимум функции func.
        Если таковых несколько, возвращается первый максимум.

        :param func:
        :return: int индеск максимума
        """
        return np.argmax(self.map(func))

    def maximize(self, func: Callable[..., int]) -> Point:
        """
        Точка на которой достигается максимум функции func.
        Если таковых несколько, возвращается первый максимум.

        :param func:
        :return: Point максимум
        """
        return Point(self.points[self.maximize_arg(func)], homogeneous=True)

    def max(self, cmp: Callable[[Point, Point], int]=None) -> Point:
        """
        Находит максимальный элемент в множестве точек, сравнивая по компаратору.
        если компаратор не задан, используется стандартный

        :param cmp: Callable[(Point, Point) -> int] компаратор
        :return: Point максимальный элемент множества
        """
        return self[self.argmax(cmp)]

    def argmax(self, cmp: Callable[[Point, Point], int]=None) -> int:
        """
        Находит позицию максимального элемента в множестве точек, сравнивая по компаратору.
        если компаратор не задан, используется стандартный

        :param cmp: Callable[(Point, Point) -> int] компаратор
        :return: int номер максимума
        """
        max_ = self[0]
        argmax = 0
        cmp_ = compare
        if cmp is not None:
            cmp_ = cmp
        for i, point in enumerate(self):
            if cmp_(point, max_) > 0:
                max_ = point
                argmax = i
        return argmax

    def sort(self, cmp: Callable[[Point, Point], int]=None, inplace=True) -> Point:
        """
        сортирует точки множества по указанному компаратору.
        Если компаратор не задан используется лексикографическое '<'.

        :param inplace: bool если True то сохраняет отсортированный список
            точек в том же множестве
        :param cmp: Callable компаратор
        :return: None
        """
        cmp_ = compare
        if cmp is not None:
            cmp_ = cmp
        points = self.map(lambda x: Point(x, homogeneous=True)).tolist()
        points = [point.coord for point in sorted(points, key=cmp_to_key(cmp_))]
        if inplace:
            self.points = np.array(points, dtype=np.int32)
        else:
            return PointSet(points)
