import numpy as np
import matplotlib.pyplot as plt


class particle_no_historic:
    def __init__(self,position,momentum):
        self.x=position[0]
        self.y=position[1]
        self.position=np.array([self.x,self.y])
        self.px=momentum[0]
        self.py=momentum[1]
        self.momentum=np.array([self.px,self.py])
        self.likelihood=None
        self.log_likelihood=None
        self.gradient_log_likelihood=None
        self.mass=None
    def _atribute_mass(self,mass_given):
        self.mass=mass_given
    def _atribute_position(self,position):
        self.x=position[0]
        self.y=position[1]
        self.position=np.array([self.x,self.y])
    def _atribute_momentum(self,momentum):
        self.px=momentum[0]
        self.py=momentum[1]
        self.momentum=np.array([self.px,self.py])
    def _atribute_likelihood(self,likelihood):
        self.likelihood=likelihood
    def _atribute_log_likelihood(self,log_likelihood):
        self.log_likelihood=log_likelihood

    def _atribute_gradient_log_likelihood(self,gradient_log_likelihood):
        self.gradient_log_likelihood=gradient_log_likelihood

    def move(self,time_step_size):
        if self.mass is None:
          Mass=np.identity(2)
          InvMass=np.identity(2)
        else:
          Mass=self.mass
          InvMass=np.linalg.inv(Mass)
        new_position=self.position+time_step_size*(np.dot(InvMass,self.momentum.T).T)
        self.position=new_position
        self.x=new_position[0]
        self.y=new_position[1]

    def half_accelerate(self,time_step_size,force):
        new_momentum=self.momentum+np.multiply(force,time_step_size/2)
        self.momentum=new_momentum
        self.px=new_momentum[0]
        self.py=new_momentum[1]