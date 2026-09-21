import numpy as np
import os

os.chdir(os.path.join("..",'code')) #On laptop

import ukf_voss 

class FNModel_with_parameter_tracking(ukf_voss.FNModel):
	def __init__(self, a=0.7, b=0.8, c=3., Q_par=0.015, Q_var=np.array((1.,)), R=1., track_a=False, track_b=False, track_c=False):
		super(FNModel_with_parameter_tracking, self).__init__(a, b, c, Q_par, Q_var, R)
		self.track_a = track_a
		self.track_b = track_b
		self.track_c = track_c

	def n_params(self):
		return 1 + int(self.track_a) + int(self.track_b) + int(self.track_c)

	def obs_g_model(self, x):
		return x[self.n_params(), :]

	def f_model(self, x, p):
		a, b, c = self.a, self.b, self.c

		n_p = 1; 
		if self.track_a:
			a = p[n_p, :]
			n_p += 1

		if self.track_b:
			b = p[n_p, :]
			n_p += 1

		if self.track_c:
			c = p[n_p, :]

		x = np.atleast_2d(x)
		# return np.array([c * (x[1,:] + x[0,:] - x[0,:]**3 / 3 + p), -(x[0,:] - a + b * x[1,:]) / c])
		rr = [np.atleast_2d(c * (x[1, :] + x[0, :] - x[0, :] ** 3 / 3 + p[0, :])),
			  np.atleast_2d(-(x[0, :] - a + b * x[1, :]) / c)]
		# print(rr)
		return np.vstack(rr)


def check_parameter(args):
    b0,c0,i,j,nature,chi2_mean_size = args
    q_I = 0.015
    q_a = 0.015

    a0 = .5
    
    Q_par = np.diag((q_I, q_a))
    Q_var0 = np.diag((nature.R, nature.R))
    #Define the model used for the Kalman filtering
    fn_model_ta = FNModel_with_parameter_tracking(Q_par=Q_par, Q_var=Q_var0, R=nature.R,track_a=True,b=b0,c=c0)
    
    #Send to the kalman filter the model we want
    uk_filter = ukf_voss.UKFVoss(model=fn_model_ta, ll=1600)
    
    init = np.array([0.,a0, nature.y[0,0], 0.]) #Initial conditions for the KF (p,x) so if you change the number of parameters you are tracking you need to change this (by default we allways track the current)
    #Run the Kalman filter simulation
    x_hat0, Pxx0, Ks0, errors0 = uk_filter.filter(nature.y, initial_condition=init)

    #Some code that returns a chi-squared error
    chi_sq = uk_filter.chi2s

    return i,j,np.mean(chi_sq[-chi2_mean_size:])
