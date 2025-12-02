import numpy as np
import time


random_image = np.random.randint(0, 4096, (10000, 10000, 3), dtype=np.uint16)


matrix = np.array([
    [200, 450, 500],
    [100, 150, 350],
    [400, 900, 1100],
    [300, 500, 1050]
])

vector = np.array([1000, 2000, 3000])
new_vector = np.array([2500, 4500, 9500])
result_matrix = np.zeros((10000, 10000, 4))
percentage_mat = (matrix / vector)
# print(new_vector * percentage_mat)
# print(np.sum(new_vector * percentage_mat, axis=1))
start_loop = time.time()
for i in range(5000):
    for j in range(5000):
        # print(np.sum(random_image[i, j] * percentage_mat, axis=1).shape)
        # result_matrix[i, j] = np.sum(random_image[i, j] * percentage_mat, axis=1)
        result_matrix[i, j] = np.sum(random_image[i, j] * percentage_mat, axis=1)
print('Time taken for loop:', time.time() - start_loop)
start_vectorized = time.time()
roi = random_image[:5000, :5000, :]  # shape (10, 10, 3)

# Apply percentage_mat to every pixel
# roi[:, :, None, :] → shape (10, 10, 1, 3)
# percentage_mat → shape (4, 3)
# Broadcasting result → (10, 10, 4, 3)
result = roi[:, :, None, :] * percentage_mat  # shape (10, 10, 4, 3)

# Sum across the last axis (channels)
result_matrix2 = np.sum(result, axis=-1)  # shape (10, 10, 4)
print('Time taken for vectorized loop:', time.time() - start_vectorized)

# Print the result
# print('result_matrix', result_matrix)
# print('result_matrix2: ', result_matrix2)
print('result_matrix', result_matrix.shape)
print('result_matrix2: ', result_matrix2.shape)

if np.array_equal(result_matrix2, result_matrix):
    print("They are exactly equal.")
else:
    print("They do not match.")