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

print(max([3,9,6]))