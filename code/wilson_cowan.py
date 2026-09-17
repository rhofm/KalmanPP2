import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, block_diag
import ipywidgets as widgets
from tqdm import tqdm

class Wilson_Cowan():
    def __init__(self, 
                 q = 2, 
                 Threshold = .1, 
                 numsteps = 200, 
                 B = 1000, 
                 C = 15, 
                 g = 8, 
                 f = 8, 
                 K = 10, 
                 k = .05, 
                 dt = .001, 
                 tau = .01, 
                 #u0 = None, 
                 #a0 = None, 
                 NoiseFactor = 10, 
                 EnergyPlotYAxis = 0.00004, 
                 N = 500, 
                 nn = 1, 
                 MaxPlotEnergy = None, 
                 ControlTime = 40, 
                 EnergyThreshold = 4, 
                 ip = 0.0001, 
                 dq = 1
                ):
        
        self.q = q # any value up to 12
        self.Threshold = Threshold # firing threshold
        self.numsteps = numsteps # number of iterations
        self.B = B # beta in the equations above
        self.C = C # inverse time constant for u (\alpha in the equations above)
        self.g = g  # number of columns in the grid
        self.f = f  # number of rows in the grid
        self.K = K  # input gain for u (\phi in equations above) 
        self.k = k  # inverse spread of connectivity (\psi # beta in the equations abovein the equations abovce) 
        self.dt = dt  # integration interval 
        self.tau = tau  # time constant for a 
        self.NoiseFactor = NoiseFactor
        self.EnergyPlotYAxis = EnergyPlotYAxis
        self.N = N # number of data samples
        self.nn = nn
        self.dT = self.nn*self.dt # Step size dt
        if not MaxPlotEnergy: self.MaxPlotEnergy = np.finfo(float).eps
        self.ControlTime = ControlTime
        self.EnergyThreshold = EnergyThreshold
        self.ip = ip
        self.dq = dq
        self.dx = self.dq + 2 * self.f * self.g
        self.dy = self.f * self.g
        
        # Initial conditions for u and a
        self.u0 = np.zeros((self.g,self.f))
        self.a0 = np.zeros((self.g,self.f))
        
        #if u0 == None:
        #    self.u0 = np.array([
        #                    [-0.1637, -0.2244, -0.1982, -0.1410, -0.1029, -0.0811, -0.0656, -0.0396],
        #                    [-0.2593, -0.3580, -0.3021, -0.2008, -0.1321, -0.1028, -0.0840, -0.0516],
        #                    [-0.2444, -0.3386, -0.2746, -0.1726, -0.0950, -0.0742, -0.0637, -0.0417],
        #                    [-0.0055, -0.0383, 0.0012, -0.0657, -0.0395, -0.0388, -0.0387, -0.0270],
        #                    [0.2361, 0.3685, 0.2616, 0.2237, 0.0279, -0.0174, -0.0239, -0.0183],
        #                    [0.4698, 0.7002, 0.5627, 0.2726, 0.0939, -0.0071, -0.0171, -0.0135],
        #                    [0.3618, 0.5613, 0.4323, 0.2754, 0.0673, -0.0056, -0.0130, -0.0104],
        #                    [0.2442, 0.2669, 0.2571, 0.1067, 0.0152, -0.0049, -0.0077, -0.0061]
        #                ])
        #else:
        #    self.u0 = u0
        #if a0:
        #    self.a0 = a0
        #else:
        #    self.a0 = np.array([
        #                    [0.4104, 0.5339, 0.4345, 0.2786, 0.1904, 0.1454, 0.1157, 0.0694],
        #                    [0.7058, 0.9183, 0.6974, 0.4049, 0.2427, 0.1821, 0.1465, 0.0893],
        #                    [0.9263, 1.1846, 0.8515, 0.4104, 0.1757, 0.1283, 0.1085, 0.0707],
        #                    [1.0750, 1.4204, 1.0146, 0.4695, 0.0921, 0.0646, 0.0636, 0.0444],
        #                    [1.0367, 1.4612, 0.9507, 0.4993, 0.0532, 0.0309, 0.0380, 0.0293],
        #                    [0.7878, 1.1079, 0.7914, 0.3707, 0.0468, 0.0205, 0.0267, 0.0213],
        #                    [0.4156, 0.5573, 0.4218, 0.1886, 0.0255, 0.0150, 0.0202, 0.0162],
        #                    [0.1386, 0.1823, 0.1406, 0.0437, 0.0083, 0.0090, 0.0119, 0.0095]
        #                ])
        #self.n,self.p = self.u0.shape

        np.random.seed(3)

    #def kalmanwc_obs(self,x,dq):
    def kalmanwc_obs(self,x):
        # we only observe the u's, not the recovery variables, just like the FN case 
        r = x[self.dq:self.f*self.g+self.dq]
        return r

    def fc(self,x, p1):
        Rows, Columns = x.shape
        self.w = np.zeros((Rows, Columns))
    
        for NumberOfColumns in range(Columns):
            self.u = x[0:self.f*self.g, NumberOfColumns].reshape((self.f, self.g))
            self.a = x[self.f*self.g:2*self.f*self.g, NumberOfColumns].reshape((self.f, self.g))
            #q = 2
            # we are estimating only the threshold
            # vector p looks like this: p=[K,C,B,tau,z]
            #self.p = np.concatenate(([1.38, 3, 10, 4.85], p1[:, NumberOfColumns])) #HARD CODING THE CONSTANTS
            p = np.concatenate(([self.K,self.C,self.B,self.tau], p1[:, NumberOfColumns]))
    
            # ue = np.concatenate((np.zeros((g, q)), u, np.zeros((g, q))), axis=1)
            # ue = np.concatenate((np.zeros((q, f + 2 * q)), ue, np.zeros((q, f + 2 * q))), axis=0)
            self.ue = np.pad(self.u, ((self.q, self.q), (self.q, self.q)), mode='constant')
            self.integ = np.zeros((self.f, self.g))
    
            for i in range(-self.q, self.q + 1):
                for j in range(-self.q, self.q + 1):
                    self.integ += p[0] * np.exp(-self.k * (i**2 + j**2)) * (self.ue[i+self.q:i+self.f+self.q, j+self.q:j+self.g+self.q] > p[4])
    
            self.integ -= p[0] * (self.u > p[4])
            self.udot = (-p[1] * self.u - self.a + self.integ)
            self.adot = (p[2] * self.u - self.a) / p[3]
            self.w[:, NumberOfColumns] = np.concatenate([self.udot.flatten(), self.adot.flatten()])
    
        return self.w

    #def kalmanwc_fct(self,dq, x):
    def kalmanwc_fct(self, x):
      
        #self.dT = self.nn * self.dt
        p = x[:self.dq, :]  # strips out p from first row
        xn = x[self.dq:, :]
        #print(f'xn.shape = {xn.shape}')
    
        for n in range(1, self.nn + 1):
            k1 = self.dt * self.fc(xn, p)
            k2 = self.dt * self.fc(xn + k1 / 2, p)
            k3 = self.dt * self.fc(xn + k2 / 2, p)
            k4 = self.dt * self.fc(xn + k3, p)
            xn = xn + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    
        #print(f'xn.shape = {xn.shape}')
        r = np.vstack([p, xn])  # returns original p
        #print(f'r.shape = {r.shape}')
        return r

    def kalmanwc_int(self, x, z):
        
        u = np.reshape(x[:self.f * self.g], (self.f, self.g))
        a = np.reshape(x[self.f * self.g:2 * self.f * self.g], (self.f, self.g))
    
        p = [self.K, self.C, self.B, self.tau, z]
        ue = np.concatenate((np.zeros((self.g, self.q)), u, np.zeros((self.g, self.q))), axis=1)
        ue = np.concatenate((np.zeros((self.q, self.f + 2 * self.q)), ue, np.zeros((self.q, self.f + 2 * self.q))), axis=0)
    
        integ = np.zeros((self.f, self.g))
        for i in range(-self.q, self.q + 1):
            for j in range(-self.q, self.q + 1):
                integ += p[0] * np.exp(-self.k * (i**2 + j**2)) * (ue[i+self.q:i+self.f+self.q, j+self.q:j+self.g+self.q] > p[4])
    
        integ -= p[0] * (u > p[4])
        udot = (-p[1] * u - a + integ)
        adot = (p[2] * u - a) / p[3]
        r = np.concatenate((udot.flatten(), adot.flatten()))
    
        return r

    # Unscented transformation. Not specific to FitzHugh-Nagumo model
    #def kalmanwc_ut(self,xhat, Pxx, y, dq, R, fct, obsfct, dx=None, dy=None):#, fct = None, obsfct = None):
    def kalmanwc_ut(self,xhat, Pxx, y, R, fct, obsfct):#, fct = None, obsfct = None):
        #if not dx: dx = dq + 2 * self.f * self.g
        #if not dy: self.f * self.g
        #if not fct: fct = self.kalmanwc_fct
        #if not obsfct.all(): obsfct = self.kalmanwc_obs
        fct = self.kalmanwc_fct
        obsfct = self.kalmanwc_obs
        
        N = 2 * self.dx
        
        xsigma = sqrtm(self.dx * Pxx).T  # Pxx = root * root', but Pxx = chol' * chol
        Xa = xhat[:, np.newaxis] + np.hstack([xsigma, -xsigma])
        #print(f'xhat.shape = {xhat.shape}\n'+f'Xa.shape = {Xa.shape}\n'+f'xsigma.shape = {xsigma.shape}')
        #X = fct(self.dq, Xa)
        X = fct(Xa)
        #print(f'X.shape = {X.shape}')
    
        xtilde = np.mean(X, axis=1)  # same as xtilde = np.sum(X, axis=1) / N
    
        Pxx = np.zeros((self.dx, self.dx))
        for i in range(N):
            #print(f'X[:,i].shape = {X[:,i].shape}\n'+f'xtile.shape = {xtilde.shape}\n'+f'N = {N}\n'+f'Pxx.shape = {Pxx.shape}')
            Pxx += np.outer((X[:, i] - xtilde), (X[:, i] - xtilde)) / N
    
        Y = np.atleast_2d(obsfct(X))
        #Y = np.atleast_2d(obsfct(X, dq))
    
        ytilde = np.mean(Y, axis=1)
        #Pyy = self.R.copy()
        Pyy = R.copy()
        for i in range(N):
            Pyy += np.outer((Y[:, i] - ytilde), (Y[:, i] - ytilde)) / N
    
        Pxy = np.zeros((self.dx, self.dy))
        for i in range(N):
            Pxy += np.outer((X[:, i] - xtilde), (Y[:, i] - ytilde)) / N
    
        K = np.dot(Pxy, np.linalg.inv(Pyy))  # same as K = np.dot(Pxy, np.linalg.inv(Pyy))
        xhat = xtilde + np.dot(K, (y - ytilde))
        Pxx = Pxx - np.dot(K, Pxy.T)
    
        Pxx = (Pxx + Pxx.T) / 2. + self.ip * np.eye(Pxx.shape[0]) # covariance inflation
    
        return xhat, Pxx, K

    #def main(self, dq, G = -0.0006, dx = None, dy = None, Q = 0.0001, fct = None, obsfct = None):
    #def main(self, G = -0.0006, Q = 0.0001, fct = None, obsfct = None):
    def main(self, G = -0.0006, Q = 0.0001):
        #if not dx: dx = dq + 2*self.f*self.g
        #if not dy: dy = self.f*self.g
        #if not fct: fct = self.kalmanwc_fct
        #if not obsfct: obsfct = self.kalmanwc_obs
        fct = self.kalmanwc_fct
        obsfct = self.kalmanwc_obs
        
        for WithControl in (1,2):
            if WithControl == 1:
                Gain = 0  # get baseline # use with noise 10 - shows advantage of Kalman Observer
            else:
                Gain = G  # can set this + or -
        
            # Initial conditions for estimation
            self.MSEsum = 0  # initialize
            self.uEnergy_y_sum = 0
            self.uEnergy_yhat_sum = 0
        
            self.u = self.u0
            self.a = self.a0
            self.x0 = np.zeros((2*self.f*self.g,self.N))
            self.x = np.zeros((2*self.f*self.g+1,self.N))
        
            self.x0[:,0] = np.concatenate((self.u.flatten(), self.a.flatten()))
        
            # External input, estimated as parameter p(5) later on:
            self.z = np.ones((1,self.N)) * 0.24
            self.x[:,0] = np.concatenate((self.z[:,0], self.x0[:,0]))
        
            t = 0  # initialize
            self.xhat= np.zeros(self.x.shape)
            self.xhat[:,0] = self.x[:, 0]  # first guess of x_1 set to observation
            self.xhat[0, 0] = 0.55  # set first guess of first parameter arbitrarily
            self.yhat = np.zeros((self.dy,self.N))
            self.yhat[:,0] = self.xhat[self.dq:self.dq+self.f*self.g, 0]
        
            #Q = 0.0001  # process noise covariance matrix
            self.errors = np.zeros((self.dx, self.N))
            self.Energy = np.zeros(self.N)
            self.Energy_y = np.zeros(self.N)
            self.Energy_yhat = np.zeros(self.N)
            self.uEnergy_yhat = np.zeros((self.N,))
        
            tempx = self.x[:, :1] * np.ones((1, self.N))
            if t > 1:
                #self.R = 0.2**2 * np.cov(self.kalmanwc_obs(tempx[:, :self.N], dq))  # observation noise cov matrix
                R = 0.2**2 * np.cov(self.kalmanwc_obs(tempx[:, :self.N]))  # observation noise cov matrix
            else:
                #self.R = np.zeros((self.f*self.g, self.f*self.g))
                R = np.zeros((self.f*self.g, self.f*self.g))
            #self.R = self.R + np.finfo(float).eps * np.eye(self.R.shape[0])
            R = R + np.finfo(float).eps * np.eye(R.shape[0])
            #self.Pxx = block_diag(Q, self.R, self.R) 
            Pxx = block_diag(Q, R, R) 
            
            #np.random.seed(3)
            self.y = np.zeros((self.dy, self.N))
            #self.y[:,0] = obsfct(self.x[:, 0], dq) + (sqrtm(self.R) @ np.random.randn(dy, 1)).ravel()  # noisy data
            self.y[:,0] = obsfct(self.x[:, 0]) + (sqrtm(R) @ np.random.randn(self.dy, 1)).ravel()  # noisy data
            #np.random.seed(0)  # Not using this - set state above
            self.Energy_y[0] = np.sum(self.y[:, t] ** 2)  # y
            self.Energy_yhat[0] = np.sum(self.yhat[:, t] ** 2)  # y
            
            for t in tqdm(range(1, self.N)):
                xx = self.x0[:, t - 1]  # Pick column (at start, only 1 column of initial conditions)
                if t >= self.ControlTime:
                    self.uVector_yhat = Gain * self.yhat[:, t - 1]
                    self.uEnergy_yhat[t] = self.uVector_yhat.T @ self.uVector_yhat
                    self.ControlVector_yhat = np.concatenate([self.uVector_yhat, np.zeros((self.f * self.g,))])
                    xx = xx + self.ControlVector_yhat
                    self.xhat[:, t - 1] = self.xhat[:, t - 1] + np.concatenate([np.array((0,)), self.ControlVector_yhat])
                else:
                    self.uEnergy_yhat[t] = 0  # no control yet
        
                for i in range(self.nn):  # nn determines how many times to loop the RK - can use 1
                    k1 = self.dt * self.kalmanwc_int(xx, self.z[:, t - 1])  # feed a col of x and a value of z
                    k2 = self.dt * self.kalmanwc_int(xx + k1 / 2, self.z[:, t - 1])
                    k3 = self.dt * self.kalmanwc_int(xx + k2 / 2, self.z[:, t - 1])
                    k4 = self.dt * self.kalmanwc_int(xx + k3, self.z[:, t - 1])
                    xx = xx + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        
                self.x0[:, t] = xx
                self.x[:,t] = np.concatenate([self.z[:,t], self.x0[:,t]])  # augmented state vector
                if t > 1:
                    #self.R = 0.2**2 * np.cov(self.kalmanwc_obs(self.x[:, :t], dq))  # observation noise cov matrix
                    R = 0.2**2 * np.cov(self.kalmanwc_obs(self.x[:, :t]))  # observation noise cov matrix
                else:
                    #self.R = np.zeros((self.f*self.g,self.f*self.g))
                    R = np.zeros((self.f*self.g,self.f*self.g))
                #self.R = self.R + np.finfo(float).eps * np.eye(self.R.shape[0])
                R = R + np.finfo(float).eps * np.eye(R.shape[0])
                #self.y[:,t] = obsfct(self.x[:,t], dq) + self.NoiseFactor * (sqrtm(self.R) @ np.random.randn(dy, 1)).ravel()  # noisy data
                self.y[:,t] = obsfct(self.x[:,t]) + self.NoiseFactor * (sqrtm(R) @ np.random.randn(self.dy, 1)).ravel()  # noisy data
                #xhat[:, t], self.Pxx, self.kalman_gain = self.kalmanwc_ut(xhat[:, t - 1], self.Pxx, self.y[:, t], fct, obsfct, dq, dx, dy, self.R)
                #print(f'xhat.shape = {xhat.shape}')
                self.xhat[:, t], Pxx, self.kalman_gain = self.kalmanwc_ut(self.xhat[:, t - 1], Pxx, self.y[:, t], R, fct, obsfct)
                #self.Pxx[0, 0] = max(self.Pxx[0, 0], Q)  # keep the parameter cov from fading to 0
                Pxx[0, 0] = max(Pxx[0, 0], Q)  # keep the parameter cov from fading to 0
                self.yhat[:, t] = self.xhat[self.dq:self.dq + self.f * self.g, t]
                self.Energy_y[t] = np.sum(self.y[:, t] ** 2)  # the rms at time t
                self.Energy_yhat[t] = np.sum(self.yhat[:, t] ** 2)  # y
                #self.errors[:, t] = np.sqrt(np.diag(self.Pxx[:, :]))
                self.errors[:, t] = np.sqrt(np.diag(Pxx[:, :]))