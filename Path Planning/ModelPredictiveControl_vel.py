# -*- coding: utf-8 -*-
"""
Differential Drive Model Predictive Control Implementation in Python
- Enhanced tracking for a sine wave reference.
"""
import numpy as np
import matplotlib.pyplot as plt

class ModelPredictiveControl:
    def __init__(self, A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal, L, dt):
        self.A = A
        self.B = B
        self.C = C
        self.f = f
        self.v = v
        self.W3 = W3
        self.W4 = W4
        self.desiredControlTrajectoryTotal = desiredControlTrajectoryTotal
        self.L = L  # Distance between the wheels (wheelbase)
        self.dt = dt  # Time step for propagation
        
        # Dimensions of the matrices
        self.n = A.shape[0]  # State dimension
        self.r = C.shape[0]  # Output dimension
        self.m = B.shape[1]  # Control input dimension
        
        # Current time step
        self.currentTimeStep = 0
        
        # Store state vectors, control inputs, and output vectors
        self.states = [x0]
        self.inputs = []
        self.outputs = []

        # Form lifted system matrices and gain matrix
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

    def propagateDynamics(self, v_left, v_right, state):
        """
        Propagates the dynamics for a differential drive robot given wheel velocities.
        """
        x, y, theta = state.flatten()
        
        # Compute the linear and angular velocity
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / self.L

        # Update state based on the differential drive kinematics
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        
        # Return the next state and output
        return np.array([[x_next], [y_next], [theta_next]]), np.array([[x_next], [y_next]])

    def computeControlInputs(self):
        """
        Computes the optimal wheel velocities using the MPC strategy.
        """
        # Extract the segment of the desired control trajectory
        desiredControlTrajectory = self.desiredControlTrajectoryTotal[
            self.currentTimeStep: self.currentTimeStep + self.f, :
        ].reshape(-1, 1)  # Reshape to match (f * r, 1)

        # Compute vector s (tracking error)
        vectorS = desiredControlTrajectory - np.matmul(self.O, self.states[self.currentTimeStep])
        
        # Compute the control sequence (wheel velocities)
        inputSequenceComputed = np.matmul(self.gainMatrix, vectorS)
        v_left = np.clip(inputSequenceComputed[0, 0], -1.0, 1.0)  # Limit between -1 and 1
        v_right = np.clip(inputSequenceComputed[1, 0], -1.0, 1.0)  # Limit between -1 and 1

        # Propagate dynamics
        state_kp1, output_k = self.propagateDynamics(v_left, v_right, self.states[self.currentTimeStep])

        # Store the results
        self.states.append(state_kp1)
        self.outputs.append(output_k)
        self.inputs.append(np.array([[v_left], [v_right]]))
        self.currentTimeStep += 1

def generate_reference_trajectory(time_steps):
    # Generate a sine wave as the reference trajectory for x, with y following a sine wave pattern
    x_ref = time_steps  # x is time-based for smooth motion along x
    y_ref = np.sin(time_steps)  # Adjust frequency to enhance tracking
    return np.vstack((x_ref, y_ref)).T

# Example usage
if __name__ == "__main__":
    # Define the wheelbase and time step
    L = 0.7  # Distance between the wheels
    dt = 0.1  # Time step
    
    # Define system matrices for a simplified differential drive model
    A = np.eye(3)  # State transition matrix for (x, y, theta)
    B = np.array([[0.5 * dt, 0.5 * dt], [0, 0], [1 / L * dt, -1 / L * dt]])  # Control input matrix for (v_left, v_right)
    C = np.array([[1, 0, 0], [0, 1, 0]])  # Output matrix

    # Define initial state (x, y, theta)
    x0 = np.array([[0], [0], [0]])

    # Define horizons
    f = 10  # Prediction horizon (increased for better foresight)
    v = 10   # Control horizon

    # Define weight matrices (tuning these can improve tracking)
    W3 = np.eye(v * 2) * 0.1  # Increase control weight slightly for smoother control
    W4 = np.eye(f * 2) * 10   # Reduce the tracking weight for more balanced control

    # Generate reference trajectory for (x, y)
    time_steps = np.linspace(0, 2 * np.pi, 500)  # Longer time steps for a longer sine wave
    desiredControlTrajectoryTotal = generate_reference_trajectory(time_steps)

    # Initialize the MPC
    mpc = ModelPredictiveControl(A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal, L, dt)

    # Run the MPC
    for _ in range(len(time_steps) - f):
        mpc.computeControlInputs()

    # Extract states for plotting
    controlled_states = np.array([state[:2, 0] for state in mpc.states])
    controlled_x = controlled_states[:, 0]
    controlled_y = controlled_states[:, 1]

    # Plot the reference trajectory and the actual path followed
    plt.figure(figsize=(10, 6))
    plt.plot(desiredControlTrajectoryTotal[:, 0], desiredControlTrajectoryTotal[:, 1], label='Reference Trajectory (Sine Wave)', linestyle='--', color='green')
    plt.plot(controlled_x, controlled_y, label='Controlled Path', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title('MPC Tracking of Sine Wave Reference Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
