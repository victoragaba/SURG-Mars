# print(rect2pol([0,-1,-1]))
# print(np.rad2deg(np.arctan(np.tan(np.deg2rad(120)))))

# point = [1,-1,np.sqrt(2)]
# point /= np.linalg.norm(point)
# print(point)
# sd = sphere_to_sd(point)
# print([np.rad2deg(rad) for rad in sd])
# print(f"Point: {point}")
# point2 = sd_to_sphere(sd)
# print(f"Point2: {point2}")

# test_sd = [np.deg2rad(45)]*2
# print(test_sd)
# print(sd_to_sphere(sphere_to_sd(k_hat)))
# point = [1,-1,1]
# point /= np.linalg.norm(point)
# print(point)

# pol = rect2pol([-1,-1])
# print(np.rad2deg(pol[1]))

# rect = pol2rect([np.sqrt(2), np.deg2rad(315)])
# print(rect)
# print(np.rad2deg(angle2bearing(np.deg2rad(50))))
# print(np.rad2deg(bearing2angle(np.deg2rad(-1))))

# axis = [1,1,0]
# axis /= np.linalg.norm(axis)
# rotated = rotate(k_hat, axis, np.deg2rad(-90))
# print(rotated)

# normal = [1,1,np.sqrt(2)]
# normal /= np.linalg.norm(normal)
# start = starting_direc(normal)
# print(start)

# print(uniform_samples(15,[3,5]))

# ################### - needs fixing and testing

# t_orig1 = [0,-1,1]
# t_orig1 = t_orig1/np.linalg.norm(t_orig1)
# t_orig2 = [0,1,-1]
# t_orig2 = t_orig2/np.linalg.norm(t_orig2)
# t1 = coord_switch(t_orig1)
# t2 = coord_switch(t_orig2)
# # print(coord_switch(t2))

# p1 = coord_switch(starting_direc(t_orig1, k_hat))
# p2 = coord_switch(starting_direc(t_orig1, -k_hat))
# # print(coord_switch(p2))

# orig, img = tp2sdr(t1, p2)

# orig = map(np.rad2deg, orig); img = map(np.rad2deg, img)
# print(tuple(orig)); print(tuple(img))

################### - more on sdr

# t_orig = [1,0,1]
# t_orig = t_orig/np.linalg.norm(t_orig)
# t = coord_switch(t_orig)
# # print(coord_switch(t))

# p = coord_switch(starting_direc(t_orig, k_hat))
# # print(coord_switch(p))

# orig, img = tp2sdr(t, -p)

# orig = map(np.rad2deg, orig); img = map(np.rad2deg, img)
# print(tuple(orig)); print(tuple(img))

# [np.rad2deg(i) for i in sdr1_emt]

"""
[[array([-0.83195995, -0.49620208,  0.24824613]),
  array([ 0.55434415, -0.72455212,  0.40954461])],
 
 [array([-0.83195995, -0.49620208,  0.24824613]),
  array([ 0.33316757, -0.08900409,  0.93865736])],
 
 [array([-0.83195995, -0.49620208,  0.24824613]),
  array([ 0.47728907, -0.41190982,  0.77622512])],
 
 [array([-0.83195995, -0.49620208,  0.24824613]),
  array([-0.21897079,  0.70474662,  0.67482145])],
 
 [array([-0.83195995, -0.49620208,  0.24824613]),
  array([ 0.41226866, -0.25342348,  0.87510633])]]
  """
  
# def other_sdr(sdr: tuple) -> tuple:
#     """
#     Get sdr of complementary nodal plane
    
#     Args:
#         sdr (tuple): (strike, dip, rake) of given nodal plane
        
#     Returns:
#         tuple: _(strike, dip, rake) of complementary nodal plane
#     """

# def modified_tp2sdr(t,p):
#     """
#     Fixing the dip distribution issue

#     Args:
#         t (list): _description_
#         p (list): _description_
#     """

# bins = dict()
    # bins["Strikes"] = np.linspace(0, 360, 51) # 7.2 degree bins
    # bins["Dips"] = np.linspace(0, 90, 51) # 1.8 degree bins
    # bins["Rakes"] = np.linspace(-180, 180, 51) # 7.2 degree bins