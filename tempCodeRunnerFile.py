N = 100000
    ramdom_samples = []

    for i in range(N):
        p = stats.norm.rvs(size=3)
        while np.linalg.norm(p) < 0.00001:
            p = stats.norm.rvs(size=3)
        p /= np.linalg.norm(p)
        p[2] = abs(p[2]) # only upper hemisphere
        random_samples.append(p)
        
    plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D([p[0] for p in random_samples],
                    [p[1] for p in random_samples],
                    [p[2] for p in random_samples], c='b', s=1/100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()