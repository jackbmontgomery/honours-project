import numpy as np
from numpy.linalg import inv, det, norm, trace
from envs import CartPole, BayesianThermostat
import wandb

# Run Configs

configs = {
    'num_episodes': 250,
    'env_id': 'CartPole-v1',
    'tao': 0.2,
    'eta_start': 0.1,
    'eta_min': 0.1,
    'eta_decay': 0.99,
    'inference_iterations': 20,
    'activation': 'sigmoid',
    'numpy_seed': '',
    'latent_dimension': 8
}

env = CartPole()

num_episodes            = configs['num_episodes']

control_dimension       = env.control_dimension
observation_dimension   = env.observation_dimension
latent_dimension        = configs['latent_dimension']

inference_iterations    = configs['inference_iterations']

tao                     = configs['tao']

eta                     = configs['eta_start']
eta_min                 = configs['eta_min']
eta_decay               = configs['eta_decay']

# Linear Transformation Function

if configs['activation'] == 'sigmoid':
    from activations import sigmoid as f
    from activations import sigmoid_derivative as df
    
elif configs['activation'] == 'tanh':
    from activations import tanh as f
    from activations import tanh_derivative as df

elif configs['activation'] == 'linear':
    from activations import linear as f
    from activations import linear_derivative as df

else:
    raise Exception("Invalid Activation Function")

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

def update_synaptic_weights(A, B, C, x, x_prev, sigma_x, y, sigma_y, u, eta):
        
    # A
    dA = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(f(x_prev), (1, latent_dimension))
    # B
    dB = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(u, (1, control_dimension))
    # C
    dC = eta * np.reshape(inv(sigma_y) @ ( y - C @ f(x)), (observation_dimension, 1)) @ np.reshape(f(x), (1, latent_dimension))

    A += tao * dA 
    B += tao * dB 
    C += tao * dC

    return A, B, C

def infer_x(inference_iterations, A, B, C, x_prev, sigma_x, y, sigma_y, u):

    qx      = [x_prev]
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

def gaussian_entropy(mu, sigma):

    dim = len(mu)

    return (dim / 2) * ( 1 + np.log(2 * np.pi) ) + (1 / 2) * np.log(det(sigma))

def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    """
    D_{KL}[ p || q ]
    """
    dim = len(mu_p)

    return (1 / 2) * (np.log( det(sigma_q) / det(sigma_p) ) - dim + np.transpose((mu_p - mu_q)) @ inv(sigma_q) @ ( mu_p - mu_q ) + trace(inv(sigma_q) @ sigma_p))

def sample_action(A, B, C, x_prev, sigma_y, biased_mu, biased_cov):

    controls = np.array([-1, 1])

    efe = np.array([])

    for u in controls:

        u = np.reshape(u, (control_dimension,))

        pred_y = predict_observation(A, B, C, x_prev, u)

        predicted_divergence    = kl_divergence(pred_y, sigma_y, biased_mu, biased_cov)
        predicted_uncertainty   = gaussian_entropy(C @ f(x_prev), sigma_y)

        efe = np.append(efe, predicted_divergence + predicted_uncertainty)

    efe_dist = np.exp(- efe )/sum(np.exp(- efe))

    u = np.random.choice(controls, p=efe_dist)

    u = np.reshape(u, (control_dimension,))   

    return u, efe_dist


def init_weights(n, m):
    lower, upper = -(np.sqrt(6.0) / np.sqrt(n + m)), (np.sqrt(6.0) / np.sqrt(n + m))
    # generate random numbers
    numbers = np.random.rand(n*m)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)
    # summarize
    return np.reshape(scaled, (n, m))

# Instantiating components of the model

A = init_weights(latent_dimension, latent_dimension)
print('A:\n', A)
B = init_weights(latent_dimension, control_dimension)         # Control
print('B:\n', B)
C = init_weights(observation_dimension, latent_dimension)     # Observation
print('C:\n', C)

# A = np.zeros((latent_dimension, latent_dimension))
# B = np.zeros((latent_dimension, control_dimension))
# C = np.ones((observation_dimension, latent_dimension))

# Static Covariance Matrices
# sigma_x = np.eye(latent_dimension)
sigma_x = np.zeros((latent_dimension, latent_dimension))
np.fill_diagonal(sigma_x, 1)
# sigma_y = np.eye(observation_dimension)
sigma_y = np.zeros((observation_dimension, observation_dimension))
np.fill_diagonal(sigma_y, 1)


# Biased Distribution over the observation space

biased_mu = np.array([0, 0])
# biased_mu = np.array([0, 0, 0, 0])
# biased_mu = np.array([0, 0, 0, 0, 1])

biased_cov = np.array([[1, 0], [0, 1 / 10]])

# Training loop

wandb.init(project='active-temporal-predictive-coding-testing', config=configs)

try:

    for episode in range(num_episodes):

        x_prev  = np.zeros(latent_dimension)

        y, _    = env.reset()

        done            = False
        episode_steps   = 0

        episode_norms       = []
        episode_energies    = []

        while not done:

            episode_steps += 1

            u, efe_dist = sample_action(A, B, C, x_prev, sigma_y, biased_mu, biased_cov)

            pred_y = predict_observation(A, B, C, x_prev, u)

            y, terminated, truncated, info = env.step(u[0])
                
            episode_norms.append(norm(pred_y - y))

            # Relax the energy to obtain the estimate for x

            x_tilde = infer_x(inference_iterations, A, B, C, x_prev, sigma_x, y, sigma_y, u)

            A, B, C = update_synaptic_weights(A, B, C, x_tilde, x_prev, sigma_x, y, sigma_y, u, eta)

            episode_energies.append(calculate_energy(A, B, C, x_tilde, sigma_x, y, sigma_y, u, x_prev))

            x_prev = x_tilde

            done = terminated or truncated
            
        wandb.log({
            'averagePredictionError': np.average(episode_norms),
            'averageFreeEnergy': np.average(episode_energies),
            'steps': episode_steps,
            'eta': eta
        })

        print(f'Episode: {episode} - Steps: {episode_steps}')
        # Exponential Decap of eta and tao
        eta = max(eta * eta_decay, eta_min)

    # wandb.finish()

except KeyboardInterrupt:
    print("\nFinishing run")
    wandb.finish()

print('A:\n', A)
print('B:\n', B)
print('C:\n', C)