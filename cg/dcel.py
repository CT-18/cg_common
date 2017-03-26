import numpy as np
from cg import Point, turn
from cg.utils import look_back
from typing import List, Iterator
from operator import methodcaller


class Vertex:
    def __init__(self, coords: Point, face: 'Face'):
        self.geometry = coords
        self.face = face

    def __eq__(self, other):
        return self.geometry == other.geometry

    def __str__(self):
        return str(self.geometry)

    def is_finite(self):
        return self.geometry.is_finite()


class Face:
    def __init__(self, vertices: List[Vertex], neighbours: List['Face']):
        self.vertices = vertices
        self.neighbours = neighbours

    def rest(self, point: Vertex):
        for i in range(3):
            if self.vertices[i] == point:
                return self.vertices[Face.ccw(i)], self.vertices[Face.cw(i)]

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
        while point != self.vertices[i]:
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


def walk_along(p: Vertex, q: Point) -> Iterator:
    is_last_point = True
    intersected = p
    while True:
        if is_last_point:
            prev = -2
            for prev_face, curr_face in look_back(walk_around(intersected)):
                a, _ = curr_face.rest(intersected)
                curr = turn(intersected.geometry, q, a.geometry)
                if curr != prev:
                    if curr >= 0:
                        is_last_point = curr == 0
                        intersected = prev_face
                        break
                prev = curr
        else:
            yield intersected
            if not all(map(methodcaller('is_finite'), intersected.vertices)):
                break
            turns = [turn(p.geometry, q, vertex.geometry) for vertex in intersected.vertices]
            for i in range(3):
                if turns[i] == 0 and p != intersected.vertices[i]:
                    intersected = intersected.vertices[i]
                    is_last_point = True
                    i -= 1
                    break
            if i < 2:
                continue
            for i in range(3):
                if p == intersected.vertices[i]:
                    continue
                if turns[i] != turns[Face.ccw(i)] and turns[Face.ccw(i)] == 1:
                    intersected = intersected.neighbours[Face.cw(i)]
                    break
