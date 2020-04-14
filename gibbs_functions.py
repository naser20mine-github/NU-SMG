import numpy as np
from tkinter import messagebox
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm



def parSGS(i, index, nreal, coord_new, coord, data_new, data, model, cc, nugget, krigtype, estimation, sigma, simul):

    if krigtype == "Multicollocated cokriging":
        coord_l = np.append(coord_new, coord[index[i]], axis=0)
        data_temp = np.append(data_new[:, 0], data_new[:, 1], axis=0)
        data_l = np.append(data_temp, data[index[i], 1], axis=0)
    elif krigtype == "Collocated cokriging":
        coord_l = np.append(coord_new, coord[index[i]], axis=0)
        data_l = np.append(data_new[:, 0], data[index[i], 1], axis=0)
    else:
        coord_l = np.append(coord_new, coord[index[i]], axis=0)
        data_l = np.append(data_new[:, 0], data_new[:, 1], axis=0)
    [lambda_T, sigma_1, sigma_2] = multi_co_weight(coord_l, model, cc, nugget, krigtype)
    estimation[i, :] = np.resize(data_l, [len(data_l), 1]).T @ lambda_T
    sigma[i, :] = np.hstack((sigma_1, sigma_2))
    simul[i, :] = (estimation[i, 0] + np.sqrt(sigma[i, 0]) * np.random.randn(nreal, 1)).T
    return  (estimation[i, 0] + np.sqrt(sigma[i, 0]) * np.random.randn(nreal, 1)).T

def parWeights(i,  index, coord, model, cc, nugget, krigtype):
    coord_l = np.append(coord, coord[index[i]], axis=0)
    coord_l = np.delete(coord_l, index[i], axis=0)
    [lambda_T, sigma_1, sigma_2] = multi_co_weight(coord_l, model, cc, nugget, krigtype)

    return lambda_T[:, 0], sigma_1

def first_SGS( cpuNumber, index, nreal, coord_new, coord, data_new, data, model, cc, nugget, krigtype):

    simul = np.empty([len(index), nreal], dtype=float)
    estimation = np.empty([len(index), 2], dtype=float)
    sigma = np.empty([len(index), 2], dtype=float)
    print('First iteration with SGS:')
    sim = Parallel(n_jobs=cpuNumber, prefer="threads")(delayed(parSGS)(i, index, nreal, coord_new, coord, data_new, data, model, cc, nugget, krigtype, estimation, sigma, simul) for i in tqdm(range(len(index)),position=0, leave=True))

    simul = np.asarray(sim).reshape(len(index),nreal)
    simul_1 = np.dot(np.diag(data[:, 0]), np.ones([len(data[:, 0]), nreal]))
    simul_1[index.T] = simul
    simul_2 = np.dot(np.diag(data[:, 1]), np.ones([len(data[:, 1]), nreal]))
    return simul, simul_1, simul_2


def calc_of_weights( cpuNumber, data, index, coord, model, cc, nugget, krigtype):

    print('Calculation of weights:')

    res = Parallel(n_jobs=cpuNumber, prefer="threads")(delayed(parWeights)(i,  index, coord, model, cc, nugget, krigtype) for i in tqdm(range(len(index)),position=0, leave=True))
    weights_final = [item[0] for item in res]
    prediction_var = [item[1] for item in res]


    return np.asarray(weights_final).T, np.asarray(prediction_var).T

def gibbs_sampler(self, index, simul_1, simul_2, data, niter, nreal, weights_final, prediction_var, krigtype):


    if krigtype == "Multicollocated cokriging":
        for j in range(niter):
            for i in range(len(index)):
                gaussian1 = np.delete(simul_1, index[i], axis=0)
                gaussian2 = np.delete(simul_2, index[i], axis=0)
                gaussian = np.vstack((gaussian1, gaussian2, data[index[i], 1] * np.ones([1, nreal])))
                Z = np.resize((gaussian.T @ weights_final[:, i]), [nreal, 1]) + np.sqrt(
                    prediction_var[0, i]) * np.random.randn(nreal, 1)
                simul_1[index[i], 0:nreal] = Z.T

            self.Gibbsprogbar['value'] = j
            self.Gibbsprogbar.update()
    elif krigtype == "Collocated cokriging":
        for j in range(niter):
            for i in range(len(index)):
                gaussian1 = np.delete(simul_1, index[i], axis=0)
                gaussian = np.vstack((gaussian1, data[index[i], 1] * np.ones([1, nreal])))
                Z = np.resize((gaussian.T @ weights_final[:, i]), [nreal, 1]) + np.sqrt(
                    prediction_var[0, i]) * np.random.randn(nreal, 1)
                simul_1[index[i], 0:nreal] = Z.T

            self.Gibbsprogbar['value'] = j
            self.Gibbsprogbar.update()
    elif krigtype == "Simple cokriging":
        for j in range(niter):
            for i in range(len(index)):
                gaussian1 = np.delete(simul_1, index[i], axis=0)
                gaussian2 = np.delete(simul_2, index[i], axis=0)
                gaussian = np.vstack((gaussian1, gaussian2))
                Z = np.resize((gaussian.T @ weights_final[:, i]), [nreal, 1]) + np.sqrt(
                    prediction_var[0, i]) * np.random.randn(nreal, 1)
                simul_1[index[i], 0:nreal] = Z.T

            self.Gibbsprogbar['value'] = j
            self.Gibbsprogbar.update()
    elif krigtype == "Simple kriging":
        for j in range(niter):
            for i in range(len(index)):
                gaussian1 = np.delete(simul_1, index[i], axis=0)
                gaussian2 = np.delete(simul_2, index[i], axis=0)
                gaussian = np.vstack((gaussian1, gaussian2))
                Z = np.resize((gaussian.T @ weights_final[:, i]), [nreal, 1]) + np.sqrt(
                    prediction_var[0, i]) * np.random.randn(nreal, 1)
                simul_1[index[i], 0:nreal] = Z.T

            self.Gibbsprogbar['value'] = j
            self.Gibbsprogbar.update()

    return simul_1

def multi_co_weight(coord, model, cc, nugget, krigtype):

    nst = len(model)
    model_rotationmatrix = np.zeros([3,3,nst],dtype=float)
    x = coord
    C = np.zeros([len(x), len(x),2], dtype=float)

    for i in range(nst):
        A = setrot(model, i)
        model_rotationmatrix[:,:,i] = A
        R = model_rotationmatrix[:,:,i]
        h = np.dot(x,R)
        h = np.dot(h,h.T)
        h = -2 * h + np.dot(np.diag(h).reshape(len(h),1),np.ones([1, len(x)])) + np.dot(np.ones([len(x),1]),np.diag(h).reshape(1,len(h)))
        h[h < 0] = 0
        h = np.sqrt(h)
        C[:,:,i] = cova(model[i,0], h)
    C[0,1,0] = 1
        # C(1: 1 + size(C, 1):end, i) = 1;


    ndata = len(coord)

    

    if krigtype == "Multicollocated cokriging":
        c11 = [np.arange(0, ndata - 1), np.arange(0, ndata - 1)]
        c12 = [np.arange(0, ndata - 1), np.arange(0, ndata)]
        c21 = [np.arange(0, ndata), np.arange(0, ndata - 1)]
        c22 = [np.arange(0, ndata), np.arange(0, ndata)]
        c101 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c102 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c201 = [np.arange(0, ndata), np.array([ndata - 1])]
        c202 = [np.arange(0, ndata), np.array([ndata - 1])]

    elif krigtype == "Collocated cokriging":
        c11 = [np.arange(0,ndata - 1), np.arange(0, ndata - 1)]
        c12 = [np.arange(0,ndata - 1), np.array([ndata - 1])]
        c21 = [np.array([ndata - 1]), np.arange(0,ndata - 1)]
        c22 = [np.array([ndata - 1]), np.array([ndata - 1])]
        c101 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c102 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c201 = [np.array([ndata - 1]), np.array([ndata - 1])]
        c202 = [np.array([ndata - 1]), np.array([ndata - 1])]

    elif krigtype == "Simple cokriging" or krigtype == "Simple kriging":
        c11 = [np.arange(0, ndata - 1), np.arange(0, ndata - 1)]
        c12 = [np.arange(0, ndata - 1), np.arange(0, ndata - 1)]
        c21 = [np.arange(0, ndata - 1), np.arange(0, ndata - 1)]
        c22 = [np.arange(0, ndata - 1), np.arange(0, ndata - 1)]
        c101 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c102 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c201 = [np.arange(0, ndata - 1), np.array([ndata - 1])]
        c202 = [np.arange(0, ndata - 1), np.array([ndata - 1])]


    cov_11 = np.zeros([len(c11[0]), len(c11[1]), 2], dtype=float)
    cov_12 = np.zeros([len(c12[0]), len(c12[1]), 2], dtype=float)
    cov_21 = np.zeros([len(c21[0]), len(c21[1]), 2], dtype=float)
    cov_22 = np.zeros([len(c22[0]), len(c22[1]), 2], dtype=float)
    cov_101 = np.zeros([len(c101[0]), len(c101[1]), 2], dtype=float)
    cov_102 = np.zeros([len(c102[0]), len(c102[1]), 2], dtype=float)
    cov_201 = np.zeros([len(c201[0]), len(c201[1]), 2], dtype=float)
    cov_202 = np.zeros([len(c202[0]), len(c202[1]), 2], dtype=float)


    for i in range(model.shape[0]):
        # Left hand
        cov_11[:, :, i] = cc[i, 0] * C[c11[0][:,None],c11[1],i]
        cov_12[:, :, i] = cc[i, 1] * C[c12[0][:,None],c12[1],i]
        cov_21[:, :, i] = cc[i, 2] * C[c21[0][:,None],c21[1],i]
        cov_22[:, :, i] = cc[i, 3] * C[c22[0][:,None],c22[1],i]
        # Right hand
        cov_101[:, :, i] = cc[i, 0] * C[c101[0][:,None], c101[1], i].reshape(len(c101[0]), 1)
        cov_102[:, :, i] = cc[i, 1] * C[c102[0][:,None], c102[1], i].reshape(len(c102[0]), 1)
        cov_201[:, :, i] = cc[i, 2] * C[c201[0][:,None], c201[1], i].reshape(len(c201[0]), 1)
        cov_202[:, :, i] = cc[i, 3] * C[c202[0][:,None], c202[1], i].reshape(len(c202[0]), 1)



    cov_11 = cov_11[:,:,0] + cov_11[:,:,1]
    cov_12 = cov_12[:,:,0] + cov_12[:,:,1]
    cov_21 = cov_21[:,:,0] + cov_21[:,:,1]
    cov_22 = cov_22[:,:,0] + cov_22[:,:,1]
    cov_101 = cov_101[:,:,0] + cov_101[:,:,1]
    cov_102 = cov_102[:,:,0] + cov_102[:,:,1]
    cov_201 = cov_201[:,:,0] + cov_201[:,:,1]
    cov_202 = cov_202[:,:,0] + cov_202[:,:,1]

    # Left and right covariance matrices
    left_up = np.concatenate((cov_11, cov_12), axis=1)
    left_down = np.concatenate((cov_21, cov_22), axis=1)
    left = np.concatenate((left_up, left_down), axis=0)
    right_up = np.hstack((cov_101, cov_102))
    right_down = np.hstack((cov_201, cov_202))
    right = np.vstack((right_up, right_down))

    # Adding Nugget effect
    left = left + nugget[0] * np.eye(left.shape[0], left.shape[1])
    right[right.shape[0] - 1, :] = right[right.shape[0] - 1, :] + nugget[1:2]

    lambda_T = np.linalg.solve(left, right)
    sigma_1 = cov_11[0, 0] - np.vstack((cov_101, cov_201)).T @ lambda_T[:, 0]
    sigma_2 = cov_22[0, 0] - np.vstack((cov_102, cov_202)).T @ lambda_T[:, 1]


    return lambda_T, sigma_1, sigma_2

def backtr(y, table, zmin, zmax, tail):
    p = table.shape[0]
    m = y.shape[0]
    n = y.shape[1]
    z = np.zeros([m * n, 1])
    yvec = np.reshape(y, [m * n, 1])


    zl = table[0, 0]
    yl = table[0, 1]
    Il = np.array(np.where(yvec < yl)).T

    if len(Il[0]) != 0:
        b0 = (zl - zmin) * np.exp(-tail[0] * yl)
        z[Il] = zmin + b0 * np.exp(tail[0] * yvec[Il])

    zp = table[p - 1, 0]
    yp = table[p - 1, 1]
    Ip = np.array(np.where(yvec > yp)).T

    if len(Ip[0]) != 0:
        bp = (zp - zmax) * np.exp(tail[1] * yp)
        z[Ip] = zmax + bp * np.exp(-tail[1] * yvec[Ip])

    I = np.vstack((Il[:], Ip[:]))
    I3 = np.arange(n * m)
    I3.shape = (m * n, 1)
    I3 = np.delete(I3, I, axis=0)

    fint = interp1d(table[:, 1] + 1e-12 * np.arange(table.shape[0]).T, table[:, 0])
    z[I3] = fint(yvec[I3])
    z = np.reshape(z, [m, n])

    return z

def setrot(model, it):
    deg2rad = np.pi/180
    ranges = model[it,1:4]
    angles = model[it,4:7]

    redmat = np.diag(1/(np.finfo(float).eps+ranges))

    a = (90 - angles[0]) * deg2rad
    b = -angles[1] * deg2rad
    c = angles[2] * deg2rad

    cosa = np.cos(a)
    sina = np.sin(a)
    cosb = np.cos(b)
    sinb = np.sin(b)
    cosc = np.cos(c)
    sinc = np.sin(c)

    rotmat = np.zeros([3, 3],dtype=float)
    rotmat[0, 0] = cosb * cosa
    rotmat[0, 1] = cosb * sina
    rotmat[0, 2] = -sinb
    rotmat[1, 0] = -cosc * sina + sinc * sinb * cosa
    rotmat[1, 1] = cosc * cosa + sinc * sinb * sina
    rotmat[1, 2] = sinc * cosb
    rotmat[2, 0] = sinc * sina + cosc * sinb * cosa
    rotmat[2, 1] = -sinc * cosa + cosc * sinb * sina
    rotmat[2, 2] = cosc * cosb

    rotred_matrix = (np.dot(redmat,rotmat)).T

    return rotred_matrix


def cova(it, h):

    eps = np.finfo(float).eps
    if (it<1):      # Nugget effect
        C = (h<eps) + 0
    elif (it<2):    # Spherical model
        C = 1 - 1.5*np.minimum(h, 1) + 0.5*np.power(np.minimum(h, 1), 3)
    elif (it<3):    # Exponential model
        C = np.exp(-3*h)
    elif (it<4):    # Cubic model
        C = 1 - 7*np.power(np.minimum(h, 1), 2) + 35/4*np.power(np.minimum(h, 1), 3) - 7/2*np.power(np.minimum(h, 1), 5) + 3/4*np.power(np.minimum(h, 1), 7)
    elif (it<5):    # Gaussian model
        C = np.exp(-3*np.power(h,2))
    else:
        messagebox.showerror("Error", "Unavailable Covariance model!")

    return C
