import numpy as np
import matplotlib.pyplot as plt

pred_rte_t = [0.1008, 0.0644, 0.0925, 0.1022, 0.0436, 0.0356, 0.0936]
pred_rte_t = np.array(pred_rte_t)
pred_rte_t_mean = np.mean(pred_rte_t)
pred_rte_t_average = np.full(pred_rte_t.size, pred_rte_t_mean)
pred_err_t = pred_rte_t_average - pred_rte_t

ekf_rte_t = [0.1260, 0.1377, 0.1377, 0.0992, 0.1747, 0.0885, 0.1169]
ekf_rte_t = np.array(ekf_rte_t)
ekf_rte_t_mean = np.mean(ekf_rte_t)
ekf_rte_t_average = np.full(ekf_rte_t.size,ekf_rte_t_mean)
ekf_err_t = ekf_rte_t_average - ekf_rte_t

cor_rte_t = [0.1345, 0.0971, 0.1234, 0.1468, 0.1794, 0.0480, 0.0561]
cor_rte_t = np.array(cor_rte_t)
cor_rte_t_mean = np.mean(cor_rte_t)
cor_rte_t_average = np.full(cor_rte_t.size,cor_rte_t_mean)
cor_err_t = cor_rte_t_average - cor_rte_t

y_zero = np.zeros(ekf_rte_t.size)

pred_rte_r = [1.043, 1.432, 1.272, 0.573, 0.665, 1.088, 0.602]
pred_rte_r = np.array(pred_rte_r)
pred_rte_r_mean = np.mean(pred_rte_r)
pred_rte_r_average = np.full(pred_rte_r.size, pred_rte_r_mean)
pred_err_r = pred_rte_r_average - pred_rte_r

ekf_rte_r = [1.432, 1.719, 1.593, 1.432, 1.954, 0.945, 0.751]
ekf_rte_r = np.array(ekf_rte_r)
ekf_rte_r_mean = np.mean(ekf_rte_r)
ekf_rte_r_average = np.full(ekf_rte_r.size,ekf_rte_r_mean)
ekf_err_r = ekf_rte_r_average - ekf_rte_r

cor_rte_r = [1.272, 1.461, 1.335, 1.089, 2.166, 0.693, 0.607]
cor_rte_r = np.array(cor_rte_r)
cor_rte_r_mean = np.mean(cor_rte_r)
cor_rte_r_average = np.full(cor_rte_r.size,cor_rte_r_mean)
cor_err_r = cor_rte_r_average - cor_rte_r

plt.rc('font',size=40)


fig1,ax1 = plt.subplots(figsize=(30,20))
x = ['Uneven17','Uneven18','Uneven19','Uneven20','Uneven21','Even05','Even06']
ax1.errorbar(x, ekf_rte_t_average, (ekf_err_t, y_zero), capsize=10, color='g', elinewidth=6, markeredgewidth=8, lw=4)
ax1.errorbar(x, cor_rte_t_average, (cor_err_t, y_zero), capsize=10, color='r', elinewidth=6, markeredgewidth=8, lw=4)
ax1.errorbar(x, pred_rte_t_average, (pred_err_t, y_zero), capsize=10, color='orange', elinewidth=6, markeredgewidth=8, lw=4)
ax1.grid()
ax1.legend(['EKF','LWOI[3]','Proposed'])
ax1.set(xlabel=r'test datasets', ylabel=r'error (m)', title="RTE translation")
plt.ylim([0,0.3])

print(ekf_rte_t_mean, ' ', pred_rte_t_mean, ' ', (ekf_rte_t_mean-pred_rte_t_mean)/ekf_rte_t_mean*100)
print(cor_rte_t_mean, ' ', pred_rte_t_mean, ' ', (cor_rte_t_mean-pred_rte_t_mean)/ekf_rte_t_mean*100)



fig2,ax2 = plt.subplots(figsize=(30,20))
x = ['Uneven17','Uneven18','Uneven19','Uneven20','Uneven21','Even05','Even06']
ax2.errorbar(x, ekf_rte_r_average, (ekf_err_r, y_zero), capsize=10, color='g', elinewidth=6, markeredgewidth=8, lw=4)
ax2.errorbar(x, cor_rte_r_average, (cor_err_r, y_zero), capsize=10, color='r', elinewidth=6, markeredgewidth=8, lw=4)
ax2.errorbar(x, pred_rte_r_average, (pred_err_r, y_zero), capsize=10, color='orange', elinewidth=6, markeredgewidth=8, lw=4)
ax2.grid()
ax2.legend(['EKF','LWOI[3]','Proposed'])
ax2.set(xlabel=r'test datasets', ylabel=r'error (deg)', title="RTE rotation")
plt.ylim([0,2.5])
print(ekf_rte_r_mean, " ",pred_rte_r_mean, " ",(ekf_rte_r_mean-pred_rte_r_mean)/ekf_rte_r_mean*100 )
print(cor_rte_r_mean, " ",pred_rte_r_mean, " ",(cor_rte_r_mean-pred_rte_r_mean)/cor_rte_r_mean*100 )


plt.show()