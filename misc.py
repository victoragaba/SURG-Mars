import numpy as np
import collections
import matplotlib.pyplot as plt
import function_repo as fr
import scipy.stats as stats
import importlib
importlib.reload(fr)

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

# import matplotlib.pyplot as plt
# import numpy as np

# # Sample data for three categories
# data_category1 = np.random.normal(0, 1, 1000)
# data_category2 = np.random.normal(2, 1, 1000)
# data_category3 = np.random.normal(-2, 1, 1000)

# # Create histograms for each category
# plt.hist(data_category1, bins=20, alpha=0.5, label='Category 1')
# plt.hist(data_category2, bins=20, alpha=0.5, label='Category 2')
# plt.hist(data_category3, bins=20, alpha=0.5, label='Category 3')

# # Add labels, title, legend, etc.
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Stacked Histogram Example')
# plt.legend()

# plt.show()



# import matplotlib.pyplot as plt

# # Create some sample data
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 4, 8]

# # Create a subplot
# fig, ax = plt.subplots()

# # Plot your data
# ax.plot(x, y)

# # Define the vertical lines and labels
# vertical_lines = [2, 3, 4]
# line_labels = ['Line 1', 'Line 2', 'Line 3']

# # Add vertical lines with labels
# for line_x, label in zip(vertical_lines, line_labels):
#     ax.axvline(x=line_x, color='r', linestyle='--', label=label)

# # Set labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('Plot with Multiple Vertical Lines')

# # Add legend
# ax.legend()

# # Show the plot
# plt.show()

print(5.5/2)

print(np.linspace(0,1,6)[:-1])
print(fr.space_evenly(5, [0,1]))
print(np.arange(0,1,1/5))
print([][:-1])
print(np.rad2deg(1/40))

# count = 0
# n = 10000
# for i in range(n):
#      vec = stats.norm.rvs(size=3)
#      # print(f"Perp: {np.dot(fr.perp(vec), vec) < fr.eps}")
#      # print(np.linalg.norm(fr.perp(vec)))
#      # print(f"Start: {np.dot(fr.starting_direc(vec, fr.k_hat), vec) < fr.eps}")
#      # print(np.linalg.norm(fr.starting_direc(vec, fr.k_hat)))
#      new = np.cross(fr.starting_direc(vec, fr.k_hat), fr.perp(vec))
#      if not abs(np.dot(new, vec)/(np.linalg.norm(new)*np.linalg.norm(vec))) > 1-fr.eps:
#           count += 1
# print(count/n*100)

a, b = np.array([1,2])
print(a,b)

test_cases = [
    (1, 0, 0),      # Along positive x-axis, azimuth = 0, elevation = 0
    (0, 1, 0),      # Along positive y-axis, azimuth = pi/2, elevation = 0
    (0, 0, 1),      # Along positive z-axis, azimuth = 0, elevation = pi/2
    (1, 1, 0),      # In the xy-plane, azimuth = pi/4, elevation = 0
    (1, 0, 1),      # Along x-axis and z-axis, azimuth = 0, elevation = pi/4
    (0, 1, 1),      # Along y-axis and z-axis, azimuth = pi/2, elevation = pi/4
    (-1, -1, 0),    # In the xy-plane, azimuth = -3*pi/4, elevation = 0
    (1, -1, 0),     # In the xy-plane, azimuth = pi/4, elevation = 0
    (-1, 1, 0),     # In the xy-plane, azimuth = -3*pi/4, elevation = 0
    (1, 1, -1),      # General direction, azimuth and elevation angles will vary
]
test_cases = [np.array(t) for t in test_cases]
for t in test_cases:
     new_t = np.vectorize(np.rad2deg)(fr.rect2pol(np.array(t)))
     print(f"Test case: {t}\nAzimuth: {new_t[1]}\nDecline: {new_t[2]}\n")