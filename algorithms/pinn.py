import numpy as np
from typing import Dict, Any

class PINN:
    def __init__(self, input_dim=2, output_dim=1, hidden=[64,64], lr=1e-3):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.lr=lr
        self.layers=[input_dim]+hidden+[output_dim]
        self.weights=[]
        self.biases=[]
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(self.layers[i],self.layers[i+1])*np.sqrt(2.0/self.layers[i]))
            self.biases.append(np.zeros((1,self.layers[i+1])))
        self.history=[]
    def _relu(self,x): return np.maximum(0,x)
    def forward(self,X):
        a=X
        self.activations=[a]
        for i in range(len(self.weights)):
            z=a@self.weights[i]+self.biases[i]
            a=self._relu(z) if i<len(self.weights)-1 else z
            self.activations.append(a)
        return a
    def predict(self,X): return self.forward(X)
    def physics_loss(self,X):
        Xr=X.astype(np.float64);Xr[:,0]*=0.1
        u=self.forward(Xr)[:,0:1]
        dx=1e-5
        Xp=Xr.copy();Xp[:,0]+=dx;Xm=Xr.copy();Xm[:,0]-=dx
        d2u=(self.forward(Xp)-2*u+self.forward(Xm))[:,0:1]/(dx**2)
        dt=1e-5
        Xt=Xr.copy();Xt[:,1]+=dt;Xb=Xr.copy();Xb[:,1]-=dt
        du=(self.forward(Xt)-self.forward(Xb))[:,0:1]/(2*dt)
        return np.mean((du-0.14*d2u)**2)
    def fit(self,X,y,epochs=100,lam=10.0,verbose=False):
        hist=[]
        for ep in range(epochs):
            p=self.forward(X);dl=np.mean((p-y)**2);pl=self.physics_loss(X)
            tl=dl+lam*pl;hist.append(tl)
            self._grad_update(X,y,dl)
            if verbose and (ep%20==0 or ep==epochs-1):
                print("Epoch %d/%d loss=%.6f", ep, epochs, tl)
        self.history=hist
        return {"status": "success", "final_loss": hist[-1] if hist else None}
    def _grad_update(self,X,y,dl):
        eps=1e-4
        for i in range(len(self.weights)):
            for j in range(min(2,self.weights[i].shape[0])):
                for k in range(min(2,self.weights[i].shape[1])):
                    old=self.weights[i][j,k]
                    self.weights[i][j,k]=old+eps;pp=self.forward(X);dp=np.mean((pp-y)**2)
                    self.weights[i][j,k]=old-eps;pm=self.forward(X);dm=np.mean((pm-y)**2)
                    self.weights[i][j,k]=old-self.lr*(dp-dm)/(2*eps)
    def get_solution(self,X): return self.forward(X)