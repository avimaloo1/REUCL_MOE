import subprocess

lambda_values = [0.005] # upper bound of the random noise
eta_values = [0.3] # learning rate for gating model
sigma_0_values = [0.1] # variance of ground truths
alpha_values = [0.3]

def main():
    def update_config(_lambda, _eta, sigma_0, _alpha):
        # Edit ds.moe_param in datasets_config.py with new values
        config_path = 'main/utils/datasets_config.py'
        with open(config_path, 'r') as f:
            lines = f.readlines()
        with open(config_path, 'w') as f:
            for line in lines:
                if "'_lambda':" in line:
                    f.write(f"        '_lambda': {_lambda}, # upper bound of the random noise\n")
                elif "'_eta':" in line:
                    f.write(f"        '_eta': {_eta}, # learning rate for gating model\n")
                elif "'sigma_0':" in line:
                    f.write(f"        'sigma_0': {sigma_0}, # variance of ground truths\n")
                elif "'_alpha':" in line:
                    f.write(f"        '_alpha': {_alpha}, # scalar of the auxiliary loss function. originally 0.3\n")
                else:
                    f.write(line)

    for _lambda in lambda_values:
        for _eta in eta_values:
            for sigma_0 in sigma_0_values:
                for _alpha in alpha_values:
                    update_config(_lambda, _eta, sigma_0, _alpha)
                    subprocess.run(['make', 'run_cifar100'])


if __name__ == "__main__":
    main()