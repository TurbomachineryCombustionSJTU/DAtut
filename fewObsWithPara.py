import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc,norm

savepath = 'E:\ing\EnKF\PartialPremixedFlame\codeplot\DA_parameter\\'
N = 50

'''构造观测值'''
x_obs = np.linspace(0,1,11)   # 稀疏观测值
x_plot = np.linspace(0,1,100) # 绘图点
y_plot_obs = 0.6*np.square(6*x_plot-2)*np.sin(12*x_plot-4)+2*(x_plot-0.5)-1
y_obs = 0.6*np.square(6*x_obs-2)*np.sin(12*x_obs-4)+2*(x_obs-0.5)-1

'''构造模型函数'''
def y_model(x,a,b,c,d):
    y = (a*x-b)**2 * np.sin(c*x-d)
    return y
a0 = 6
b0 = 2
c0 = 12
d0 = 5

'''确定抽样范围偏差'''
l_bound_mdl = (np.array([-a0,-b0,-c0,-d0])*0.05).tolist()
u_bound_mdl = (np.array([a0,b0,c0,d0])*0.05).tolist()
l_bound_obs = (-np.ones(x_obs.shape[0])*0.02).tolist()
u_bound_obs = (np.ones(x_obs.shape[0])*0.02).tolist()
theta = np.array((a0,b0,c0,d0),)
theta = np.array(theta, dtype=np.float64)
y_mdl = y_model(x_obs,a0,b0,c0,d0)
y_plot_mdl = y_model(x_plot,a0,b0,c0,d0)
plt.figure()
plt.scatter(x_obs,y_obs)
plt.scatter(x_obs,y_mdl)
plt.plot(x_plot, y_plot_obs, label='Observation')
plt.plot(x_plot, y_plot_mdl, label='Prediction')
plt.legend(frameon=False)
plt.savefig(savepath+'Initial_Distribution.pdf')
plt.close()

'''构造拉丁超立方抽样'''
def latin(matrix, dimensions, samples, l_bound, u_bound):
    sampler = qmc.LatinHypercube(d=dimensions)
    sample = sampler.random(n=samples)
    sample_scaled = qmc.scale(sample,l_bound,u_bound)
    matrix_ens = np.tile(matrix, (N,1))
    matrix_ens += sample_scaled
    return matrix_ens

theta_ens = latin(theta, theta.shape[0], N, l_bound_mdl, u_bound_mdl)
y_mdl_ens = np.zeros((N, x_obs.shape[0]))
for i in range(y_mdl.shape[0]):
    y_mdl_ens[i] = y_model(x_obs, theta_ens[i,0],theta_ens[i,1],theta_ens[i,2],theta_ens[i,3])
V_mdl_ens = np.concatenate((theta_ens, y_mdl_ens),axis=1)
V_obs_ens = latin(y_obs, y_obs.shape[0], N, l_bound_obs, u_bound_obs)
C = np.cov(V_mdl_ens.T)

'''构造EnKF'''
def EnKF(pk_f, pk_o, H):
    sizeh = H.shape[1]
    N_ens = pk_f.shape[0]
    C = np.cov(pk_f.T)
    R_e = np.cov(pk_o.T)
    S_jp1 = H @ C @ H.T + R_e
    K_gain = C @ H.T @ np.linalg.inv(S_jp1)
    pk_a = np.zeros_like(pk_f)
    for i in range(N_ens):
        pk_a[i] = pk_f[i] + K_gain @ (pk_o[i] - H @ pk_f[i])
    C_plus = (np.eye(sizeh) - K_gain @ H) @ C
    return pk_a, K_gain, C_plus

H = np.hstack((np.zeros((y_obs.shape[0],theta.shape[0])),np.eye(y_obs.shape[0])))
pk_a, _, C = EnKF(V_mdl_ens, V_obs_ens, H)
y_aly_ens = np.zeros((N,x_plot.shape[0]))
for i in range(y_aly_ens.shape[0]):
    y_aly_ens[i] = y_model(x_plot,pk_a[i,0],pk_a[i,1],pk_a[i,2],pk_a[i,3])
V_aly = np.mean(pk_a, axis=0)
y_plot_aly = y_model(x_plot,V_aly[0],V_aly[1],V_aly[2],V_aly[3])
'''验证结果'''
plt.figure()
for i in range(y_aly_ens.shape[0]):
    plt.plot(x_plot,y_aly_ens[i],color='lightgreen')
plt.scatter(x_obs,y_obs)
plt.scatter(x_obs,y_mdl)
plt.plot(x_plot, y_plot_obs, label='Observation')
plt.plot(x_plot, y_plot_mdl, label='Prediction')
plt.plot(x_plot, y_plot_aly, label = 'Analysis',color='r')
plt.legend(frameon=False)
plt.savefig(savepath+'After_assimilation.pdf')

