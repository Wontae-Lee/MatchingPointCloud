import torch


def create_3d_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = torch.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = torch.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = torch.stack([t1 + t2 * t3,
                     t7 - t9,
                     t11 + t12,
                     t7 + t9,
                     t1 + t2 * t15,
                     t19 - t20,
                     t11 - t12,
                     t19 + t20,
                     t1 + t2 * t24], dim=1)

    return torch.reshape(R, (-1, 3, 3))


def trans_rot_ptx(src, trans, rot_x, rot_y, rot_z):
    print("")
    print("*****************************************************************")
    print("************************ Traning End ****************************")
    print("*****************************************************************")
    print("")
    print("trans\n", trans)
    print("")
    print("rot x axis\n", rot_x)
    print("")
    print("rot y axis\n", rot_y)
    print("")
    print("rot z axis\n", rot_z)

    src = src + trans
    zero = torch.tensor(0.)

    # rotation x-axis
    s_x = torch.sin(rot_x)
    c_x = torch.cos(rot_x)
    rot_x = torch.stack([torch.tensor([1., 0., 0.]),
                         torch.stack([zero, c_x, -s_x]),
                         torch.stack([zero, s_x, c_x])])
    src = src @ rot_x.t()

    # rotation y-axis
    s_y = torch.sin(rot_y)
    c_y = torch.cos(rot_y)
    rot_y = torch.stack([torch.stack([c_y, zero, s_y]),
                         torch.tensor([0., 1., 0.]),
                         torch.stack([zero, s_y, c_y])])
    src = src @ rot_y.t()

    # rotation z-axis
    s_z = torch.sin(rot_z)
    c_z = torch.cos(rot_z)
    rot_z = torch.stack([torch.stack([c_z, - s_z, zero]),
                         torch.stack([s_z, c_z, zero]),
                         torch.tensor([0., 0., 1.])])
    src = src @ rot_z.t()
    return src


def get_pairwise_separations(ri, rj):
    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


def neighbor_search(x1, x2, radius):
    dx, dy, dz = get_pairwise_separations(x1, x2)

    # radius를 기준으로 neighbor만 있는 bool 매트릭스
    neighbor_matrix2d = torch.sqrt((dx ** 2 + dy ** 2 + dz ** 2)) < radius

    # Exclude itself from the list of neighbors.
    for idx, ind in enumerate(neighbor_matrix2d):
        neighbor_matrix2d[idx][idx] = 0

    return neighbor_matrix2d


def return_distance(x1, x2):
    dx, dy, dz = get_pairwise_separations(x1, x2)
    return torch.sqrt((dx ** 2 + dy ** 2 + dz ** 2))
