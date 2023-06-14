from munkres import Munkres, print_matrix, make_cost_matrix

matrix = [[5, 1, 9],
          [10, 3, 2],
          [8, 7, 4]]
cost_matrix = make_cost_matrix(matrix)
m = Munkres()
indexes = m.compute(cost_matrix)
print_matrix(matrix, msg='Highest profit through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total profit: {total}')
