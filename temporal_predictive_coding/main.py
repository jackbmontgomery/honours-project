import numpy as np
from numpy.linalg import inv, det, norm, trace
from temporal_predictive_coding.env import CartPole
import wandb
import os
import randomname

# Run Configs

configs = {
    'num_runs': 3,
    'num_episodes': 1000,
    'tao': 0.2,
    'eta_start': 0.2,
    'eta_min': 0.025,
    'eta_decay': 0.999,
    'inference_iterations': 25,
    'activation': 'sigmoid',
    'numpy_seed': 2024,
    'latent_dimension': 6
}

# Training loop

num_runs                = configs['num_runs']
numpy_seed              = configs['numpy_seed']

np.random.seed(numpy_seed)

for run in range(num_runs):

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
        from temporal_predictive_coding.activations import sigmoid as f
        from temporal_predictive_coding.activations import sigmoid_derivative as df
        
    elif configs['activation'] == 'tanh':
        from temporal_predictive_coding.activations import tanh as f
        from temporal_predictive_coding.activations import tanh_derivative as df

    elif configs['activation'] == 'linear':
        from temporal_predictive_coding.activations import linear as f
        from temporal_predictive_coding.activations import linear_derivative as df

    else:
        raise Exception("Invalid Activation Function")

    # Aux Functions to calculate common values

    def calculate_energy(A, B, C, bias, x, sigma_x, y, sigma_y, u, x_prev):

        t1 = (1 / 2) * np.transpose( y - C @ f(x) - bias) @ sigma_y @ ( y - C @ f(x) - bias)

        t2 = (1 / 2) * np.transpose( x - A @ f(x_prev) - B @ u ) @ sigma_x @ ( x - A @ f(x_prev) - B @ u )

        return t1 + t2

    def calculate_ex(A, B, x, sigma_x, u, x_prev):
        return inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u)

    def calculate_ey(C, bias, y, sigma_y, x):
        return inv(sigma_y) @ ( y - C @ f(x) - bias)

    def calculate_qx_grad(C, x, e_x, e_y):

        return - e_x + np.multiply(df(x), np.transpose(C) @ e_y)

    def update_synaptic_weights(A, B, C, bias, x, x_prev, sigma_x, y, sigma_y, u, eta):
            
        # A
        dA = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(f(x_prev), (1, latent_dimension))
        # B
        dB = eta * np.reshape(inv(sigma_x) @ ( x - A @ f(x_prev) - B @ u), (latent_dimension, 1)) @ np.reshape(u, (1, control_dimension))
        # C
        dC = eta * np.reshape(inv(sigma_y) @ ( y - C @ f(x) - bias), (observation_dimension, 1)) @ np.reshape(f(x), (1, latent_dimension))
        # Bias
        dBias = eta * np.reshape(inv(sigma_y) @ ( y - C @ f(x) - bias), (observation_dimension, ))

        A += tao * dA 
        B += tao * dB 
        C += tao * dC

        bias += tao * dBias

        return A, B, C, bias

    def infer_x(inference_iterations, A, B, C, bias, x_prev, sigma_x, y, sigma_y, u):

        qx      = [x_prev]
        vfes    = []

        for _ in range(inference_iterations):

            e_x = calculate_ex(A, B, qx[-1], sigma_x, u, x_prev)

            e_y = calculate_ey(C, bias, y, sigma_y, qx[-1])

            qx_grad = calculate_qx_grad(C, qx[-1], e_x, e_y)

            qx.append(qx[-1] + tao * qx_grad)

            vfe = calculate_energy(A, B, C, bias, qx[-1], sigma_x, y, sigma_y, u, x_prev)

            vfes.append(vfe)

        return qx[-1]

    def predict_observation(A, B, C, bias, x_prev, u):

        pred_next_x = A @ f(x_prev) + B @ u

        pred_next_y = C @ f(pred_next_x) + bias

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

    def sample_action(A, B, C, bias, x_prev, sigma_y, biased_mu, biased_cov):

        controls = np.array([-1, 1])

        efe = np.array([])
        # divergence = np.array([]) 

        for u in controls:

            u = np.reshape(u, (control_dimension,))

            pred_y = predict_observation(A, B, C, bias, x_prev, u)

            predicted_divergence    = kl_divergence(pred_y, sigma_y, biased_mu, biased_cov)

            # divergence = np.append(divergence, predicted_divergence)

            predicted_uncertainty   = gaussian_entropy(C @ f(x_prev) + bias, sigma_y)

            efe = np.append(efe, predicted_divergence + predicted_uncertainty)

        efe_dist = np.exp(- efe )/sum(np.exp(- efe))

        u = np.random.choice(controls, p=efe_dist)
        u = np.reshape(u, (control_dimension,))

        # u = np.argmin(divergence)
        # u = np.reshape(u, (control_dimension,))

        return u, efe_dist


    def init_weights(n, m):
        lower, upper = -(np.sqrt(6.0) / np.sqrt(n + m)), (np.sqrt(6.0) / np.sqrt(n + m))
        # generate random numbers
        numbers = np.random.rand(n*m)
        # scale to the desired range
        scaled = lower + numbers * (upper - lower)
        # summarize
        return np.reshape(scaled, (n, m))

    def save_episode_data(run_name, episode_data: dict):

        data = np.array(list(episode_data.values()))

        file_path = f'./results/{run_name}.csv'
        if not os.path.isfile(file_path):
            # Write headers and initialize file
            header = 'episodeNum,averagePredictionError,averageFreeEnergy,steps,eta'
            np.savetxt(file_path, [data], delimiter=',', header=header, comments='')
        else:
            # Append new data
            with open(file_path, 'a') as f:
                np.savetxt(f, [data], delimiter=',')

    # Instantiating components of the model

    A = init_weights(latent_dimension, latent_dimension)
    print('A:\n', A)
    B = init_weights(latent_dimension, control_dimension)         # Control
    print('B:\n', B)
    C = init_weights(observation_dimension, latent_dimension)     # Observation
    print('C:\n', C)

    bias = np.zeros((observation_dimension, ))

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

    biased_cov = np.array([[1, 0], [0, 1 / 10]])

    # wandb.init(project='active-temporal-predictive-coding', config=configs)

    run_name = randomname.get_name() + "-" + str(np.random.randint(0, 100))

    # if not os.path.exists(f'./results/{run_name}'):
    #     os.makedirs(f'./results/{run_name}')

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

                u, efe_dist = sample_action(A, B, C, bias, x_prev, sigma_y, biased_mu, biased_cov)

                pred_y = predict_observation(A, B, C, bias, x_prev, u)

                y, terminated, truncated, info = env.step(u[0])
                    
                episode_norms.append(norm(pred_y - y))

                # Relax the energy to obtain the estimate for x

                x_tilde = infer_x(inference_iterations, A, B, C, bias, x_prev, sigma_x, y, sigma_y, u)

                A, B, C, bias = update_synaptic_weights(A, B, C, bias, x_tilde, x_prev, sigma_x, y, sigma_y, u, eta)

                episode_energies.append(calculate_energy(A, B, C, bias, x_tilde, sigma_x, y, sigma_y, u, x_prev))

                x_prev = x_tilde

                done = terminated or truncated


            episode_data = {
                'episodeNum': episode,
                'averagePredictionError': np.average(episode_norms),
                'averageFreeEnergy': np.average(episode_energies),
                'steps': episode_steps,
                'eta': eta
            }
                
            # wandb.log(episode_data)
            save_episode_data(run_name, episode_data)

            print(f'Episode: {episode} - Steps: {episode_steps}')
            # Exponential Decap of eta and tao
            eta = max(eta * eta_decay, eta_min)

            # Saving to local results file:

        # wandb.finish()

    except KeyboardInterrupt:
        print("\nFinishing run")
        # wandb.finish()
        exit()

    print('A:\n', A)
    print('B:\n', B)
    print('C:\n', C)
    print('Bias:\n', bias)