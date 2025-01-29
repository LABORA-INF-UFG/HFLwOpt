import sys
import numpy as np


class RoundCosts:

    def __init__(self, n_server, n_server_rounds, comm_model, topology):

        self.n_server = n_server
        self.n_server_rounds = n_server_rounds
        self.comm_model = comm_model
        self.topology = topology

        self.round_costs_list = {
            'total_training': [],
            'total_uploads': [],
            'total_error_uploads': [],
            'energy_success': [],
            'energy_error': [],
            'total_energy': [],
            'user_energy_training': [],
            'round_sum_total_delay': [],
            'round_total_user_delay_success': [],
            'round_max_total_delay': [],
            'user_bandwidth': [],
            'user_power': [],
            'user_power_upload_success': [],
            'user_power_upload_unsuccess': [],
            'cpu_freq': [],
            'q': [],
            'user_data_rate_upload_success': []
        }

    def compute(self, server_list):

        total_server = dict()
        for _, key in enumerate(self.round_costs_list):
            total_server[key] = []

        for sid in range(self.n_server):

            if self.topology == "CloudEdgeServer":
                round_costs_list = server_list[sid].s_stf.round_costs_list
            else:
                round_costs_list = server_list[sid].ces_Fit.round_costs.round_costs_list

            for _, key in enumerate(self.round_costs_list):
                if key == "round_max_total_delay":
                    total_server[key].append(
                        sum(round_costs_list[key][
                            -1 * self.n_server_rounds:]) +
                        self.comm_model.total_delay[sid, self.comm_model.rb_allocation[sid]]
                    )
                else:
                    total_server[key].append(
                        sum(round_costs_list[key][-1 * self.n_server_rounds:]))

        for _, key in enumerate(self.round_costs_list):
            if key == "round_max_total_delay":
                self.round_costs_list[key].append(max(total_server[key]))
            else:
                self.round_costs_list[key].append(np.mean(total_server[key]))

    def print_result(self):
        for item, value in self.round_costs_list.items():
            print(item)
            print(value)
