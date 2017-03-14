import numpy as np
from cg import Point, turn
from typing import List, Iterator
from operator import methodcaller


class Vertex:
    def __init__(self, coords: Point, face: 'Face'):
        self.geometry = coords
        self.face = face

    def __eq__(self, other):
        return self.geometry == other.geometry


class Face:
    def __init__(self, vertices: List[Vertex], neighbours: List['Face']):
        self.vertices = vertices
        self.neighbours = neighbours

    def rest(self, point: Vertex):
        for i in range(3):
            if self.vertices[i] == point:
                return self.vertices[Face.cw(i)], self.vertices[Face.ccw(i)]

    @staticmethod
    def cw(i):
        """
        clock-wise
        :param i:
        :return:
        """
        return [2, 0, 1][i]

    @staticmethod
    def ccw(i):
        """
        counter clock-wise
        :param i:
        :return:
        """
        return [1, 2, 0][i]

    def step(self, point: Vertex, dir):
        i = 0
        while point != self.neighbours[i]:
            i += 1
        return self.neighbours[dir(i)]

    def nxt(self, point: Vertex):
        return self.step(point, Face.ccw)

    def prv(self, point: Vertex):
        return self.step(point, Face.cw)

    def __getitem__(self, item: Vertex) -> int:
        i = 0
        while item != self.vertices[i]:
            i += 1
        return i


def walk_around(point: Vertex) -> Iterator[Face]:
    yield point.face
    nxt_ = point.face.nxt(point)
    while nxt_ != point.face:
        yield nxt_
        nxt_ = nxt_.nxt(point)


def walk_along(p: Vertex, q: Vertex) -> Iterator:
    is_last_point = True
    intersected = p
    while True:
        yield intersected
        if is_last_point:
            for face in walk_around(intersected):
                yield face
                a, b = face.rest(intersected)
                fst = turn(intersected, q, a)
                snd = turn(intersected, q, b)
                if fst == 0 or snd == 0:
                    is_last_point = True
                    if fst:
                        intersected = a
                    else:
                        intersected = b
                elif fst != snd:
                    is_last_point = False
                    intersected = face.neighbours[face[intersected]]
        else:
            if not all(map(methodcaller('is_finite'), intersected.vertices)):
                break
            turns = [turn(p, q, vertex) for vertex in intersected.vertices]
            for i in range(3):
                if turns[i] == 0:
                    intersected = intersected.vertices[i]
                    is_last_point = True
                    break
            else:
                continue
            for i in range(3):
                if turns[i] != turns[Face.ccw(i)]:
                    intersected = intersected.neighbours[Face.cw(i)]
                    break
