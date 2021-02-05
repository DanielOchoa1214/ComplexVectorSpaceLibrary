import numpy as np


def vectorAdd(vector1, vector2):
    return np.array([i + j for i, j in zip(vector1, vector2)])


def addVectorInverse(vector):
    return np.array(-vector)


def scalarVectorMultiplication(c, vector):
    return np.array([c * i for i in vector])


def matrixAdd(matrix1, matrix2):
    answer = np.array([np.array([0 + 0j for i in range(len(matrix1[0]))]) for j in range(len(matrix1))])
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            answer[i][j] = matrix1[i][j] + matrix2[i][j]
    return answer


def addMatrixInverse(matrix):
    return np.array([-i for i in matrix])


def scalarMatrixMultiplication(c, matrix):
    return np.array([[c * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))])


def transpose(any_vector):
    if any_vector.ndim == 1:
        transposed_vector = np.array([[0 + 0j] for i in range(len(any_vector))])
        for i in range(len(transposed_vector)):
            transposed_vector[i] = any_vector[i]
        return transposed_vector
    if any_vector.ndim == 2:
        transposed_matrix = np.array([[0 + 0j for k in range(len(any_vector))] for l in range(len(any_vector[0]))])
        for i in range(len(any_vector)):
            for j in range(len(any_vector[0])):
                transposed_matrix[j][i] = any_vector[i][j]
        return transposed_matrix


def conjugate(any_vector):
    if any_vector.ndim == 1:
        for i in range(len(any_vector)):
            any_vector[i] = np.conj(any_vector[i])
        return np.array(any_vector)
    if any_vector.ndim == 2:
        for i in range(len(any_vector)):
            for j in range(len(any_vector[i])):
                any_vector[i][j] = np.conj(any_vector[i][j])
        return np.array(any_vector)


def adjoint(any_vector):
    return conjugate(transpose(any_vector))


def actionOnVector(matrix, vector):
    return matrixProduct(matrix, transpose(vector))


def matrixProduct(matrix1, matrix2):
    answer = np.array([[0 + 0j for i in range(len(matrix2[0]))] for x in range(len(matrix1))])
    for i in range(len(answer)):
        for j in range(len(answer[0])):
            for k in range(len(matrix2)):
                answer[i][j] += matrix1[i][k] * matrix2[k][j]
    return answer


def dotProduct(vector1, vector2):
    answer = 0
    if vector1.ndim == 1:
        vector1 = conjugate(vector1)
        for i in range(len(vector1)):
            answer += vector1[i] * vector2[i]
        return answer
    if vector1.ndim == 2:
        vector1 = adjoint(vector1)
        multiplied_matrices = matrixProduct(vector1, vector2)
        for i in range(len(multiplied_matrices)):
            answer += multiplied_matrices[i][i]
        return answer


def norm(vector):
    return np.sqrt(dotProduct(vector, vector.copy()))


def vectorDistance(vector1, vector2):
    if vector1.ndim == 1:
        return norm(vectorAdd(vector1, addVectorInverse(vector2)))
    if vector1.ndim == 2:
        return norm(matrixAdd(vector1, addMatrixInverse(vector2)))


def isHermitian(vector):
    if vector.ndim == 2:
        if np.array_equal(vector, adjoint(vector)):
            return True
        return False
    if vector.ndim == 1:
        if list(vector) == list(adjoint(vector)):
            return True
        return False


def isUnitary(matrix):
    identity = [[1 + 0j if i == j else 0 + 0j for i in range(len(matrix[0]))] for j in range(len(matrix))]
    if np.array_equal(matrixProduct(matrix, adjoint(matrix)), identity):
        return True
    return False


def tensorProduct(matrix1, matrix2):
    if matrix1.ndim == 1 and matrix2.ndim == 1:
        answer = np.array([])
        for i in matrix1:
            answer = np.append(answer, scalarVectorMultiplication(i, matrix2))
        return answer
    if matrix1.ndim == 2 and matrix2.ndim == 2:
        final_list, sub_list, count = [], [], len(matrix2)
        for row1 in matrix1:
            counter, check = 0, 0
            while check < count:
                for num1 in row1:
                    for num2 in matrix2[counter]:
                        sub_list.append(num1 * num2)
                counter += 1
                final_list.append(sub_list)
                sub_list = []
                check += 1
        return np.array(final_list)
