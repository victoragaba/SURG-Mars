import sys
import numpy as np
import function_repo as fr
import scipy.stats as stats

# print(fr.eps)
eps = 1e-5

"""
testing my_tp2sdr against tp2sdr
"""
# n = 1000
# for i in range(n):
#     t = stats.norm.rvs(size=3)
#     t = t/np.linalg.norm(t)
#     p_start = stats.norm.rvs(size=3)
#     p_start = p_start/np.linalg.norm(p_start)
#     p = fr.starting_direc(t, p_start)
#     if abs(np.dot(t, p)) > fr.eps:
#         print(f"t and p are not orthogonal at {i}")
#     sdr1 = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))[0]
#     my_sdr1 = fr.my_tp2sdr(t, p)[0]
#     for j in range(3):
#         if abs(sdr1[j] - my_sdr1[j]) > eps:
#             print(f"iteration {i}, component {j} is different")
#             print(f"sdr1: {sdr1}\nmy_sdr1: {my_sdr1}")
#     sdr2 = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))[1]
#     my_sdr2 = fr.my_tp2sdr(t, p)[1]
#     for j in range(3):
#         if abs(sdr2[j] - my_sdr2[j]) > eps:
#             print(f"iteration {i}, component {j} is different")
#             print(f"sdr2: {sdr2}\nmy_sdr2: {my_sdr2}")

# # special cases
# ts = [fr.i_hat, fr.j_hat, fr.k_hat, -fr.i_hat, -fr.j_hat, -fr.k_hat]
# ps = [fr.i_hat, fr.j_hat, fr.k_hat, -fr.i_hat, -fr.j_hat, -fr.k_hat]

# failed = set()
# count = 0
# for t in ts:
#     for p in ps:
#         if abs(np.dot(t, p)) > eps:
#             continue
#         count += 1
#         sdr1 = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))[0]
#         sdr1 = np.vectorize(np.rad2deg)(sdr1)
#         my_sdr1 = fr.my_tp2sdr(t, p, True)[0]
#         for j in range(3):
#             if abs(sdr1[j] - my_sdr1[j]) > eps:
#                 if abs(my_sdr1[j] - 180) < eps and abs(sdr1[j] + 180) < eps:
#                     continue
#                 failed.add(count)
#                 print(f"special case, t = {t}, p = {p}, component {j} is different")
#                 print(f"sdr1: {sdr1}\nmy_sdr1: {my_sdr1}")
# print(f"{len(failed)}/{count} mismatches for special cases") # 6

# # vertical dips
# for i in range(n):
#     t = stats.norm.rvs(size=2)
#     t = np.append(t, 0)
#     t = t/np.linalg.norm(t)
#     p = fr.starting_direc(t, fr.j_hat)
#     n = t + p
#     n = n/np.linalg.norm(n)
#     # angle = np.deg2rad(90*np.random.random())
#     # t = fr.rodrigues_rotate(t, n, angle) # change from strike-slip
#     # p = fr.rodrigues_rotate(p, n, angle) # change from strike-slip
#     if abs(np.dot(t, p)) > fr.eps:
#         print(f"t and p are not orthogonal at {i}")
#     sdr1 = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))[0]
#     sdr1 = np.vectorize(np.rad2deg)(sdr1)
#     my_sdr1 = fr.my_tp2sdr(t, p, True)[0]
#     for j in range(3):
#         if abs(sdr1[j] - my_sdr1[j]) > eps:
#             if abs(my_sdr1[j] - 180) < eps and abs(sdr1[j] + 180) < eps:
#                     continue
#             print(f"iteration {i}, component {j} is different")
#             print(f"sdr1: {sdr1}\nmy_sdr1: {my_sdr1}")
#     sdr2 = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))[1]
#     sdr2 = np.vectorize(np.rad2deg)(sdr2)
#     my_sdr2 = fr.my_tp2sdr(t, p, True)[1]
#     for j in range(3):
#         if abs(sdr2[j] - my_sdr2[j]) > eps:
#             if abs(my_sdr2[j] - 180) < eps and abs(sdr2[j] + 180) < eps:
#                     continue
#             print(f"iteration {i}, component {j} is different")
#             print(f"sdr2: {sdr2}\nmy_sdr2: {my_sdr2}")

# # just for fun
# vic, tor = fr.my_tp2sdr(fr.j_hat, fr.i_hat, True)
# su, zan = fr.tp2sdr(fr.coord_switch(fr.j_hat), fr.coord_switch(fr.i_hat))
# su = np.vectorize(np.rad2deg)(su)
# zan = np.vectorize(np.rad2deg)(zan)
# print(f"vic: {vic}\nsu: {su}\ntor: {tor}\nzan: {zan}")
            
# """
# my_tp2sdr is more precise than tp2sdr, but not by much
# strike-slip fault where p is anticlockwise from t (from above) mismatch on rake
# my_tp2sdr works for all special cases?
# need to confirm logic for normal fault
# """

"""
testing machine epsilon
"""
# # Using sys.float_info
# machine_epsilon_sys = sys.float_info.epsilon

# # Using numpy.finfo
# machine_epsilon_np = np.finfo(float).eps

# print("Machine Epsilon (sys.float_info):", machine_epsilon_sys)
# print("Machine Epsilon (numpy.finfo):", machine_epsilon_np)

# def estimate_machine_epsilon():
#     epsilon = 1.
#     while 1. + epsilon != 1.:
#         epsilon /= 2.
#     return epsilon

# machine_epsilon = estimate_machine_epsilon()
# print("Estimated Machine Epsilon:", machine_epsilon)

"""
testing Rpattern
"""
t = fr.i_hat
p = fr.k_hat
sdr1, sdr2 = fr.my_tp2sdr(t, p, True)
azimuth = 100 # subject to change
takeoff_angles = [20, 25] # subject to change
alpha_beta = [5.8000, 3.4600] # from Suzan's notes

# As1 = fr.Rpattern(sdr1, azimuth, takeoff_angles, alpha_beta)
# As2 = fr.Rpattern(sdr2, azimuth, takeoff_angles, alpha_beta)
# print(As1, As2)

strike, dip, rake = sdr1
rela = strike - azimuth
sinlam = np.sin(np.radians(rake))
coslam = np.cos(np.radians(rake))
sind = np.sin(np.radians(dip))
cosd = np.cos(np.radians(dip))
cos2d = np.cos(np.radians(2*dip))
sinrela = np.sin(np.radians(rela))
cosrela = np.cos(np.radians(rela))
sin2rela = np.sin(np.radians(2*rela))
cos2rela = np.cos(np.radians(2*rela))

sR = sinlam*sind*cosd
qR = sinlam*cos2d*sinrela + coslam*cosd*cosrela
pR = coslam*sind*sin2rela - sinlam*sind*cosd*cos2rela
pL = sinlam*sind*cosd*sin2rela + coslam*sind*cos2rela
qL = -coslam*cosd*sinrela + sinlam*cos2d*cosrela

iP = np.radians(takeoff_angles[0])
jS = np.radians(takeoff_angles[1])

AP = sR*(3*np.cos(iP)**2 - 1) - qR*np.sin(2*iP) - pR*np.sin(iP)**2
ASV = 1.5*sR*np.sin(2*jS) + qR*np.cos(2*jS) + 0.5*pR*np.sin(2*jS)
ASH = qL*np.cos(jS) + pL*np.sin(jS)

print(AP, ASV, ASH)