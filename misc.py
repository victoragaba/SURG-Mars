import numpy as np
import collections

x = np.array([3,2,1])
print(len(x))

print(not {1})

arr = [1,2,3,4]
arr2 = [n for n in arr]
arr2[3] = 5
print(arr)
print(arr2)

# str1 = "boy"
# str2 = str1
# str2[2] = 'x'
# print(str1)
# print(str2)

q = collections.deque([1,2,3])
print(q)
print(len(q))

print(np.dot([0,0,1], [1,1,0]))

print(max(3,9,6))

A = [[1, 1],
     [0, 1]]
B = [[1, 0],
     [1, 1]]

print(np.matmul([1,1],B))

# twoD = [[2*j for j in range(3)] for i in range(4)]
# print(twoD)
# print(twoD[:,0])

print(2*np.arccos(0))

print(1-1/3)

x = np.array([1,2,3])
y = np.array([.5,.5,.5])
print(x*y)

print(1/2*3)

print(np.array((1,2,3)))