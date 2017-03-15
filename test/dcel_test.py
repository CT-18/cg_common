from unittest import TestCase
from cg import Point, Vertex, Face, walk_along, walk_around


class TriangulationTest(TestCase):
    def setUp(self):
        points = [
            (0, 0),
            (-1, 3),
            (3, 0),
            (2, 3),
            (0, 6),
            (4, 6),
            (7, 4)
        ]
        points = [Point(*args) for args in points]
        points.append(Point.infinity(2))
        points = [Vertex(p, None) for p in points]
        faces = [
            [0, 2, 1],
            [1, 2, 3],
            [1, 3, 4],
            [3, 5, 4],
            [3, 6, 5],
            [3, 2, 6],
            [0, 1, 7],
            [1, 4, 7],
            [4, 5, 7],
            [5, 6, 7],
            [6, 2, 7],
            [2, 0, 7]
        ]
        connections = [0, 0, 0, 1, 2, 3, 4, 6]
        faces = [Face([points[i], points[j], points[k]], None) for i, j, k in faces]
        for i, v in enumerate(points):
            v.face = faces[connections[i]]
        neighbours = [
            [1, 6, 11],
            [5, 2, 0],
            [3, 7, 1],
            [8, 2, 4],
            [9, 3, 5],
            [10, 4, 1],
            [7, 11, 0],
            [8, 6, 2],
            [9, 7, 3],
            [10, 8, 4],
            [11, 9, 5],
            [6, 10, 0]
        ]
        for f, (i, j, k) in zip(faces, neighbours):
            f.neighbours = [faces[i], faces[j], faces[k]]
        self.vertices = points
        self.faces = faces

    def testIterator(self):
        right_order = [0, 1, 5, 4, 9]
        for i, ver in enumerate(walk_along(self.vertices[0], Point(7, 7))):
            self.assertTrue(ver is self.faces[right_order[i]])