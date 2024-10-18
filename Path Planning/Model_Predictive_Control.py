# -*- coding: utf-8 -*-
"""
Unconstrained Model Predictive Control Implementation in Python
- This version is without an observer, that is, it assumes that the
  state vector is perfectly known
"""
import numpy as np
import matplotlib.pyplot as plt

class ModelPredictiveControl(object):
    def __init__(self, A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal):
        self.A = A
        self.B = B
        self.C = C
        self.f = f
        self.v = v
        self.W3 = W3
        self.W4 = W4
        self.desiredControlTrajectoryTotal = desiredControlTrajectoryTotal
        
        # dimensions of the matrices
        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]
        
        # this variable is used to track the current time step k of the controller
        self.currentTimeStep = 0
        
        # we store the state vectors of the controlled state trajectory
        self.states = []
        self.states.append(x0)
        
        # we store the computed inputs
        self.inputs = []
        
        # we store the output vectors of the controlled state trajectory
        self.outputs = []
        
        # form the lifted system matrices and vectors
        # the gain matrix is used to compute the solution
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()
        
    def formLiftedMatrices(self):
        f = self.f
        v = self.v
        r = self.r
        n = self.n
        m = self.m
        A = self.A
        B = self.B
        C = self.C
        
        # lifted matrix O
        O = np.zeros(shape=(f * r, n))
        for i in range(f):
            if i == 0:
                powA = A
            else:
                powA = np.matmul(powA, A)
            O[i * r : (i + 1) * r, :] = np.matmul(C, powA)
        
        # lifted matrix M
        M = np.zeros(shape=(f * r, v * m))
        for i in range(f):
            if i < v:
                for j in range(i + 1):
                    if j == 0:
                        powA = np.eye(n, n)
                    else:
                        powA = np.matmul(powA, A)
                    M[i * r : (i + 1) * r, (i - j) * m : (i - j + 1) * m] = np.matmul(C, np.matmul(powA, B))
            else:
                for j in range(v):
                    if j == 0:
                        sumLast = np.zeros(shape=(n, n))
                        for s in range(i - v + 2):
                            if s == 0:
                                powA = np.eye(n, n)
                            else:
                                powA = np.matmul(powA, A)
                            sumLast = sumLast + powA
                        M[i * r : (i + 1) * r, (v - 1) * m : v * m] = np.matmul(C, np.matmul(sumLast, B))
                    else:
                        powA = np.matmul(powA, A)
                        M[i * r : (i + 1) * r, (v - 1 - j) * m : (v - j) * m] = np.matmul(C, np.matmul(powA, B))
        
        tmp1 = np.matmul(M.T, np.matmul(self.W4, M))
        tmp2 = np.linalg.inv(tmp1 + self.W3)
        gainMatrix = np.matmul(tmp2, np.matmul(M.T, self.W4))
        
        return O, M, gainMatrix

    def propagateDynamics(self, controlInput, state):
        xkp1 = np.matmul(self.A, state) + np.matmul(self.B, controlInput)
        yk = np.matmul(self.C, state)
        return xkp1, yk

    def computeControlInputs(self):
        # extract the segment of the desired control trajectory
        desiredControlTrajectory = self.desiredControlTrajectoryTotal[
            self.currentTimeStep : self.currentTimeStep + self.f, :
        ]
        # compute the vector s
        vectorS = desiredControlTrajectory - np.matmul(self.O, self.states[self.currentTimeStep])
        
        # compute the control sequence
        inputSequenceComputed = np.matmul(self.gainMatrix, vectorS)
        inputApplied = np.zeros(shape=(1, 1))
        inputApplied[0, 0] = inputSequenceComputed[0, 0]
        
        # compute the next state and output
        state_kp1, output_k = self.propagateDynamics(inputApplied, self.states[self.currentTimeStep])
        
        # append the lists
        self.states.append(state_kp1)
        self.outputs.append(output_k)
        self.inputs.append(inputApplied)
        self.currentTimeStep += 1

def generate_reference_trajectory(time_steps):
    return np.sin(time_steps)

# Example usage
if __name__ == "__main__":
    # Define system matrices
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    C = np.array([[1, 0]])
    
    # Define initial state
    x0 = np.array([[0], [0]])
    
    # Define horizons
    f = 10
    v = 5
    
    # Define weight matrices
    W3 = np.eye(v)
    W4 = np.eye(f)
    
    # Generate reference trajectory
    time_steps = np.linspace(0, 2 * np.pi, 100)
    desiredControlTrajectoryTotal = generate_reference_trajectory(time_steps).reshape(-1, 1)
    
    # Initialize the MPC
    mpc = ModelPredictiveControl(A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal)
    
    # Run the MPC
    for _ in range(len(time_steps) - f):
        mpc.computeControlInputs()
    
    # Extract states for plotting
    controlled_states = np.array([state[0, 0] for state in mpc.states])
    
    # Plot the reference trajectory and the actual path followed
    plt.plot(time_steps, generate_reference_trajectory(time_steps), label='Reference Trajectory (Sine Wave)')
    plt.plot(time_steps[:len(controlled_states)], controlled_states, label='Controlled Path')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.title('MPC Tracking of Sine Wave Trajectory')
    plt.show()
