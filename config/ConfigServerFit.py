class ConfigServerFit:
    def __init__(self, ds_clients, min_fit_clients, rb_number,
                 # delay_requirement, energy_requirement,
                 error_rate_requirement, lmbda, fixed_parameters,
                 user_power, user_bw, user_cpu_freq, cm):
        self.ds_clients = ds_clients
        self.min_fit_clients = min_fit_clients
        self.rb_number = rb_number

        self.error_rate_requirement = error_rate_requirement

        self.lmbda = lmbda

        self.fixed_parameters = fixed_parameters

        self.user_power = user_power
        self.user_bw = user_bw
        self.user_cpu_freq = user_cpu_freq

        self.cm = cm
