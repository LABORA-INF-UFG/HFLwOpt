import sys
import numpy as np
from comm_model.DeviceToServerCommModel import DeviceToServerCommModel
from comm_model.ShowCommModel import ShowCommModel
from optmizer.MilpOptmizer import MilpOpt


class CommStrategyDeviceToServer:

    def __init__(self, total_params, sf, csf):

        self.sf = sf

        self.list_clients = self.sf.server_clients.get_cids()
        self.list_distances = self.sf.server_clients.list_distances
        self.clients_number_data_samples = self.sf.server_clients.clients_number_data_samples
        self.clients_loss_list = self.sf.server_clients.loss_list

        self.total_params = total_params

        self.csf = csf

        self.comm_model = None

        self.count_of_client_selected = dict(zip(self.list_clients, map(int, np.zeros(len(self.list_clients)))))
        self.count_of_client_uploads = dict(zip(self.list_clients, map(int, np.zeros(len(self.list_clients)))))

        self.ind_selected_clients = []
        self.selected_clients = []
        self.user_bandwidth_allocation = []
        self.rb_allocation = []
        self.user_power_allocation = []
        self.user_cpu_freq_allocation = []

        self.success_uploads = []
        self.error_uploads = []

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

    def random_user_selection(self, factor=1.0):
        self.selected_clients = np.random.permutation(self.list_clients)[:int(self.csf.min_fit_clients * factor)]
        self.ind_selected_clients = np.array(range(len(self.selected_clients)))

    def greater_data_user_selection(self, factor=1.0, k=0):
        if np.random.rand() >= 0:
            self.selected_clients = np.random.permutation(self.list_clients)[:int(self.csf.min_fit_clients * factor)]

            data_samples_list = [self.clients_number_data_samples[key] for key in self.selected_clients]
            pos_list = np.arange(len(data_samples_list))

            combined_data = list(zip(data_samples_list, pos_list))
            sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
            distance_list, pos_list = zip(*sorted_data)

            self.selected_clients = np.sort(
                self.selected_clients[np.array(pos_list)[:len(self.selected_clients) if k == 0 else int(k)]])

        else:
            least_selected_keys = sorted(self.count_of_client_selected, key=lambda x: self.count_of_client_selected[x])
            self.selected_clients = least_selected_keys[:int(self.csf.min_fit_clients)]

        self.ind_selected_clients = np.array(range(len(self.selected_clients)))

    def greater_loss_user_selection(self, factor=1.0, k=0):
        self.selected_clients = np.random.permutation(self.list_clients)[:int(self.csf.min_fit_clients * factor)]

        loss_samples_list = [self.clients_loss_list[key] for key in self.selected_clients]
        pos_list = np.arange(len(loss_samples_list))

        combined_data = list(zip(loss_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        loss_list, pos_list = zip(*sorted_data)

        self.selected_clients = np.sort(
            self.selected_clients[np.array(pos_list)[:len(self.selected_clients) if k == 0 else int(k)]])
        self.ind_selected_clients = range(len(self.selected_clients))

    def random_rb_allocation(self):
        self.rb_allocation = np.random.permutation(self.csf.rb_number)[:self.csf.min_fit_clients]

    def sinr_rb_allocation(self):
        self.fixed_allocation()

        distances = [self.list_distances[key] for key in self.selected_clients]

        pos_list = np.arange(len(distances))
        combined_data = list(zip(distances, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])

        _, self.rb_allocation = zip(*sorted_data)

        if True:
            selected_clients = []
            ind_selected_clients = []
            rb_allocation = []
            for i in self.ind_selected_clients:
                if self.comm_model.q[i][0][self.rb_allocation[i]][0] <= self.csf.error_rate_requirement:
                    selected_clients.append(self.selected_clients[i])
                    ind_selected_clients.append(i)
                    rb_allocation.append(self.rb_allocation[i])

            self.ind_selected_clients = np.array(ind_selected_clients)
            self.selected_clients = np.array(selected_clients)
            self.rb_allocation = np.array(rb_allocation)

    def fixed_user_power_allocation(self):
        self.user_power_allocation = np.zeros(len(self.selected_clients)).astype(int)

    def fixed_user_bandwidth_allocation(self):
        self.user_bandwidth_allocation = np.zeros(len(self.selected_clients)).astype(int)

    def fixed_user_cpu_freq_allocation(self):
        self.user_cpu_freq_allocation = np.zeros(len(self.selected_clients)).astype(int)

    def fixed_allocation(self):
        self.fixed_user_power_allocation()
        self.fixed_user_bandwidth_allocation()
        self.fixed_user_cpu_freq_allocation()

    def compute_comm_parameters(self):

        self.comm_model = DeviceToServerCommModel(csf=self.csf,
                                                  total_model_params=self.total_params,
                                                  list_clients=self.selected_clients,
                                                  user_distance=list(
                                                      map(self.list_distances.get, self.selected_clients))
                                                  )

    def upload_status(self):
        prob = np.random.rand(len(self.selected_clients))

        self.success_uploads = []
        self.error_uploads = []

        for i, ind in enumerate(self.ind_selected_clients):
            prob_w = self.comm_model.W[
                ind, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i],
                self.user_cpu_freq_allocation[i]]
            cid = self.selected_clients[i]
            print(
                f"[{(i + 1):2}] {cid:3} --> W: {prob_w:.6f} - P: {prob[i]:.6f} {'' if prob_w > 0 and prob_w >= prob[i] else ' - [X]'}")

            if prob_w > 0 and prob_w >= prob[i]:
                self.success_uploads.append(cid)
            else:
                self.error_uploads.append(cid)

    def compute_round_costs(self):
        total_training = len(self.selected_clients)
        total_uploads = len(self.success_uploads)

        round_energy_success = 0
        round_energy_error = 0
        user_energy_training = 0

        total_user_delay_success = 0
        round_total_delay = []

        user_bandwidth = 0
        user_power = 0
        user_power_upload_success = 0
        cpu_freq = 0

        q = 0
        user_data_rate_upload_success = 0

        for i, ind in enumerate(self.ind_selected_clients):
            cid = self.selected_clients[i]

            user_energy_training = user_energy_training + self.comm_model.user_energy_training[
                self.user_cpu_freq_allocation[i]]
            user_bandwidth = user_bandwidth + self.comm_model.user_bandwidth[self.user_bandwidth_allocation[i]]
            user_power = user_power + self.comm_model.user_power[self.user_power_allocation[i]]
            cpu_freq = cpu_freq + (self.comm_model.cpu_freq[self.user_cpu_freq_allocation[i]] / (10 ** 9))

            q = q + self.comm_model.q[
                ind, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]

            round_total_delay.append(
                self.comm_model.total_delay[
                    ind, 0, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i],
                    self.user_cpu_freq_allocation[i]])

            if cid in self.success_uploads:
                round_energy_success = round_energy_success + self.comm_model.total_energy[
                    ind, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i],
                    self.user_cpu_freq_allocation[i]]

                total_user_delay_success = total_user_delay_success + self.comm_model.total_user_delay[
                    ind, self.user_bandwidth_allocation[i], self.rb_allocation[i],
                    self.user_power_allocation[i], self.user_cpu_freq_allocation[i]]

                user_power_upload_success = user_power_upload_success + self.comm_model.user_power[
                    self.user_power_allocation[i]]

                user_data_rate_upload_success = user_data_rate_upload_success + self.comm_model.user_data_rate[
                    ind, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]

            else:
                round_energy_error = round_energy_error + self.comm_model.total_energy[
                    ind, self.user_bandwidth_allocation[i], self.rb_allocation[i], self.user_power_allocation[i],
                    self.user_cpu_freq_allocation[i]]

        self.round_costs_list['total_training'].append(total_training)
        self.round_costs_list['total_uploads'].append(total_uploads)
        self.round_costs_list['total_error_uploads'].append(total_training - total_uploads)

        self.round_costs_list['energy_success'].append(round_energy_success)
        self.round_costs_list['energy_error'].append(round_energy_error)
        self.round_costs_list['total_energy'].append(round_energy_success + round_energy_error)
        self.round_costs_list['user_energy_training'].append(user_energy_training)

        self.round_costs_list['round_sum_total_delay'].append(sum(round_total_delay))
        self.round_costs_list['round_total_user_delay_success'].append(total_user_delay_success)
        self.round_costs_list['round_max_total_delay'].append(
            max(round_total_delay) if len(round_total_delay) > 0 else 0)

        self.round_costs_list['user_bandwidth'].append(user_bandwidth)
        self.round_costs_list['user_power'].append(user_power)
        self.round_costs_list['user_power_upload_success'].append(user_power_upload_success)
        self.round_costs_list['user_power_upload_unsuccess'].append(user_power - user_power_upload_success)
        self.round_costs_list['cpu_freq'].append(cpu_freq)

        self.round_costs_list['q'].append(q)
        self.round_costs_list['user_data_rate_upload_success'].append(user_data_rate_upload_success)

    def optimization(self):
        (self.ind_selected_clients,
         self.selected_clients,
         self.user_bandwidth_allocation,
         self.rb_allocation,
         self.user_power_allocation,
         self.user_cpu_freq_allocation) = MilpOpt.opt(self)
