import unittest
from ComplexVectorSpaceLibrary import *
import numpy as np


class TestComplexVectorSpaceLibrary(unittest.TestCase):

    def test_vectorAdd(self):
        a = np.array([2 + 4j, 6 + 5j])
        b = np.array([3 + 2j, 1j])
        self.assertTrue((vectorAdd(a, b) == np.array([5 + 6j, 6 + 6j])).all())

    def test_vectorInverse(self):
        a = np.array([4 + 5j, 7 - 9j, -1j])
        self.assertTrue((addVectorInverse(a) == np.array([-4 - 5j, -7 + 9j, 1j])).all())

    def test_scalarVectorMultiplication(self):
        c = 6 + 7j
        vector = np.array([6 + 4j, 9 - 3j, 1 - 5j])
        self.assertTrue((scalarVectorMultiplication(c, vector) == np.array([8 + 66j, 75 + 45j, 41 - 23j])).all())

    def test_matrixAdd(self):
        a = np.array([[1 + 2j, 2 - 5j, 3 + 2j], [4 - 3j, 5 + 1j, 6 - 9j]])
        b = np.array([[4 - 6j, 3 - 5j, 2 + 7j], [7 + 0j, 6 - 2j, 5 - 1j]])
        self.assertTrue((matrixAdd(a, b) == np.array([[5 - 4j, 5 - 10j, 5 + 9j], [11 - 3j, 11 - 1j, 11 - 10j]])).all())

    def test_addMatrixInverse(self):
        matrix = np.array([[1, 6 + 7j], [9 + 34j, 6 - 5j]])
        self.assertTrue((addMatrixInverse(matrix) == np.array([[-1, -6 - 7j], [-9 - 34j, -6 + 5j]])).all())

    def test_scalarMatrixMultiplication(self):
        c = 6 + 7j
        matrix = np.array([[5 + 7j, 3 - 4j], [2 + 3j, 5 - 7j]])
        self.assertTrue((scalarMatrixMultiplication(c, matrix) == np.array([[-19 + 77j, 46 - 3j], [-9 + 32j, 79 - 7j]]))
                        .all())

    def test_transpose(self):
        matrix = np.array([[2 - 6j, 7 + 2j], [1 - 1j, 3 - 8j]])
        self.assertTrue((transpose(matrix) == np.array([[2 - 6j, 1 - 1j], [7 + 2j, 3 - 8j]])).all())

    def test_conjugate(self):
        vector = np.array([2 - 5j, 7 + 45j, 1 - 6j])
        self.assertTrue((conjugate(vector) == np.array([2 + 5j, 7 - 45j, 1 + 6j])).all())

    def test_adjoint(self):
        matrix = np.array([[3 - 5j, 7 + 23j], [7 - 1j, -1j]])
        self.assertTrue((adjoint(matrix) == np.array([[3 + 5j, 7 + 1j], [7 - 23j, 1j]])).all())

    # Accion de un vector

    def test_matrixProduct(self):
        a = np.array([[3 + 6j, 5 - 9j], [1 + 1j, 6j]])
        b = np.array([[5 - 4j, 9 - 1j], [2 + 9j, 4 - 1j]])
        self.assertTrue((matrixProduct(a, b) == np.array([[130 + 45j, 44 + 10j], [-45 + 13j, 16 + 32j]])).all())

    def test_dotProduct(self):
        a = np.array([3 - 6j, 4 + 2j, 5 - 1j])
        b = np.array([6 - 2j, 1j, -1j])
        self.assertTrue((dotProduct(a, b) == 33 + 29j).all())

    def test_norm(self):
        vector = np.array([[3 - 4j, 9 - 1j], [-1j, 0]])
        self.assertTrue(norm(vector) == np.sqrt(108))

    def test_vectorDistance(self):
        a = np.array([0, -1j])
        b = np.array([1j, 8 - 5j])
        self.assertTrue((vectorDistance(a, b) == np.sqrt(81)).all())

    def test_isHermitian(self):
        matrix = np.array([[1, 4 + 2j], [4 - 2j, 5]])
        self.assertTrue(isHermitian(matrix))

    def test_isUnitary(self):
        matrix = np.array([[1j, 0, 0], [0, 1j, 0], [0, 0, 1j]])
        self.assertTrue(isUnitary(matrix))

    def test_tensorProduct(self):
        matrix1 = np.array([[1 + 5j, 2 - 8j], [3 + 1j, 4 + 0j]])
        matrix2 = np.array([[1 - 5j, 2 + 9j, 3 - 7j], [3 - 9j, 4 + 1j, 3 - 1j]])
        self.assertTrue(np.array_equal(tensorProduct(matrix1, matrix2), np.array([[26 + 0j, -43 + 19j, 38 + 8j, -38 -
        18j, 76 + 2j, -50 - 38j], [48 + 6j, -1 + 21j, 8 + 14j, -66 - 42j, 16 - 30j, -2 - 26j], [8 - 14j, -3 + 29j, 16 -
        18j, 4 - 20j, 8 + 36j, 12 - 28j], [18 - 24j, 11 + 7j, 10 + 0j, 12 - 36j, 16 + 4j, 12 - 4j]])))


if __name__ == '__main__':
    unittest.main()
