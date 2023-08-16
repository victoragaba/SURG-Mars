import numpy as np
import collections
import matplotlib.pyplot as plt

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

print(np.linspace(0,1,5))

# fig = plt.figure()
# plt.show()

nested_list = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]

flattened_list = [item for sublist in nested_list for item in sublist]
print(nested_list)
print(flattened_list)

print(np.rad2deg(np.arctan2(1,1)))

from matplotlib import colors
orange_rgb = colors.hex2color(colors.cnames['orange'])
print(orange_rgb)
print(colors.rgb2hex(orange_rgb))

array = [np.sin(((np.pi/2)/50)*(i+1)) for i in range(50)]
print(30000*sum(array))

import matplotlib.pyplot as plt
import numpy as np

# Sample data for three categories
data_category1 = np.random.normal(0, 1, 1000)
data_category2 = np.random.normal(2, 1, 1000)
data_category3 = np.random.normal(-2, 1, 1000)

# Create histograms for each category
plt.hist(data_category1, bins=20, alpha=0.5, label='Category 1')
plt.hist(data_category2, bins=20, alpha=0.5, label='Category 2')
plt.hist(data_category3, bins=20, alpha=0.5, label='Category 3')

# Add labels, title, legend, etc.
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram Example')
plt.legend()

plt.show()
