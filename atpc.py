import numpy as np
from numpy.linalg import inv, det, norm
from env import CartPoleWrapper
import matplotlib.pyplot as plt
import wandb

# Run Configs

configs = {
    'env_id': 'CartPole-v1',
    'tao': 0.1,
    'eta': 0.9,
    'inference_iterations': 10
}

env = CartPoleWrapper()

control_dimension       = env.control_dimension
observation_dimension   = env.observation_dimension

latent_dimension        = 4 # Arbitrary for now

# Linear Transformation Function

def f(x):
    return np.tanh(x)

def df(x):
    return 1 - f(x) ** 2

# Aux Functions to calculate common values

def calculate_energy(A, B, C, x, sigma_x, y, sigma_y, u, x_prev):

    t1 = (1 / 2) * np.transpose( y - C @ f(x) ) @ sigma_y @ ( y - C @ f(x) )

    t2 = (1 / 2) * np.transpose( x - A @ f(x_prev) - B @ u ) @ sigma_x @ ( x - A @ f(x_prev) - B @ u )

    return t1 + t2

def calculate_ex(A, B, x, sigma_x, u, x_prev):
    return inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u)

def calculate_ey(C, y, sigma_y, x):
    return inv(sigma_y) @ ( y - C @ f(x) )

def calculate_qx_grad(C, x, e_x, e_y):
    return - e_x + np.multiply(df(x), np.transpose(C) @ e_y)

def update_synaptic_weights(A, B, C, x, x_prev, sigma_x, y, sigma_y, u):

    # Hardcoded dimensions here I need to change

    vfes    = []

    for _ in range(inference_iterations):

        # A
        dA = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(f(x_prev), (1, latent_dimension))
        # B
        dB = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(u, (1, control_dimension))
        # C
        dC = eta * np.reshape(inv(sigma_y) @ ( y - C @ f(x)), (observation_dimension, 1)) @ np.reshape(f(x), (1, latent_dimension))

        A = A + tao * dA 
        B = B + tao * dB 
        C = C + tao * dC

        vfe = calculate_energy(A, B, C, x, sigma_x, y, sigma_y, u, x_prev)

        vfes.append(vfe)

    # plt.plot(vfes)
    # plt.show()

    return A, B, C

def infer_x(inference_iterations, A, B, C, x_prev, sigma_x, y, sigma_y, u):

    qx      = [np.zeros(latent_dimension)]
    vfes    = []

    for _ in range(inference_iterations):

        e_x = calculate_ex(A, B, qx[-1], sigma_x, u, x_prev)

        e_y = calculate_ey(C, y, sigma_y, qx[-1])

        qx_grad = calculate_qx_grad(C, qx[-1], e_x, e_y)

        qx.append(qx[-1] + tao * qx_grad)

        vfe = calculate_energy(A, B, C, qx[-1], sigma_x, y, sigma_y, u, x_prev)

        vfes.append(vfe)

    return qx[-1]

def predict_observation(A, B, C, x_prev, u):

    pred_next_x = A @ f(x_prev) + B @ u

    pred_next_y = C @ f(pred_next_x)

    return pred_next_y

inference_iterations    = configs['inference_iterations']
tao                     = configs['tao']
eta                     = configs['eta']

# Instantiating components of the model

A = np.random.rand(latent_dimension, latent_dimension)      # Dynamics Matrix
B = np.random.rand(latent_dimension, control_dimension)     # Control
C = np.random.rand(observation_dimension, latent_dimension) # Observation

# Static Covariance Matrices
sigma_x = np.eye(latent_dimension)
sigma_y = np.eye(observation_dimension)

# Biased Distribution over the observation space

biased_mu = np.array([0, 0, 0, 0, 1])

biased_cov = np.array([[1000, 0, 0, 0, 0], 
                      [0, 1000, 0, 0, 0],
                      [0, 0, 1000, 0, 0],
                      [0, 0, 0, 1000, 0],
                      [0, 0, 0, 0, 1 / 1000]])

# Training loop

wandb.init(project='active-temporal-predictive-coding', group='testing-learning', config=configs)

for episode in range(1000):

    x_prev  = np.zeros(latent_dimension)

    y, _    = env.reset(seed=2024)

    done            = False
    episode_steps   = 0

    episode_norms = []

    while not done:

        episode_steps += 1

        u = env.sample_action()

        pred_y = predict_observation(A, B, C, x_prev, u)

        y, terminated, truncated, info = env.step(u[0])

        episode_norms.append(norm(pred_y - y))

        # Relax the energy to obtain the estimate for x

        x_tilde = infer_x(inference_iterations, A, B, C, x_prev, sigma_x, y, sigma_y, u)

        A, B, C = update_synaptic_weights(A, B, C, x_tilde, x_prev, sigma_x, y, sigma_y, u)

        x_prev = x_tilde

        done = terminated or truncated
        
    wandb.log({
        'episode': episode,
        'averagePredictionError': np.average(episode_norms)
    })

wandb.finish()