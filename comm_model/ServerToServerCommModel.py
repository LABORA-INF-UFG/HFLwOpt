import sys
import numpy as np


class ServerToServerCommModel:

    def __init__(self, total_model_params, server_number, lower_limit, upper_limit):

        self.data_size_model = 0
        self.total_model_params = total_model_params
        self.server_number = server_number
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.user_interference = np.array([])
        self.user_distance = np.array([])
        self.angles = np.array([])

        self.user_power = 1
        self.user_bandwidth = 20
        self.N = 10 ** -20
        self.q = np.array([])

        self.h = np.array([])
        self.user_sinr = np.array([])

        self.user_data_rate = np.array([])

        self.base_station_power = 1
        self.base_station_bandwidth = 20
        self.base_station_sinr = np.array([])
        self.base_station_data_rate = np.array([])

        self.user_delay = np.array([])
        self.base_station_delay = np.array([])

        self.total_delay = np.array([])
        self.rb_allocation = []

        self.init()

    def init(self):
        self.init_user_distance()
        self.init_user_interference()
        self.init_q()
        self.init_h()
        self.init_user_sinr()
        self.init_user_data_rate()
        self.init_base_station_sinr()
        self.init_base_station_data_rate()
        self.init_data_size_model()
        self.init_user_delay()
        self.init_base_station_delay()
        self.init_total_delay()
        self.init_rb_allocation()

    def init_user_distance(self):
        np.random.seed(1)
        a, b = self.lower_limit, self.upper_limit
        self.user_distance, self.angles = a + (b - a) * np.random.rand(self.server_number, 1), 2 * np.pi * np.random.rand(self.server_number)
        np.random.seed()

    def init_user_interference(self):
        i = np.array([0.05 + i * 0.01 for i in range(self.server_number)])
        self.user_interference = (i - 0.04) * 0.000001

    def init_q(self):
        self.q = 1 - np.exp(-1.08 * (self.user_interference + self.N * self.user_bandwidth) / (self.user_power * (self.user_distance ** -2)))

    def init_h(self):
        o = 1
        self.h = o * (self.user_distance ** (-2))

    def init_user_sinr(self):
        self.user_sinr = self.user_power * self.h / (self.user_interference + self.user_bandwidth * self.N)

    def init_user_data_rate(self):
        self.user_data_rate = self.user_bandwidth * np.log2(1 + self.user_sinr)

    def init_base_station_sinr(self):
        base_station_interference = 0.06 * 0.000003  # Interference over downlink
        self.base_station_sinr = (self.base_station_power * self.h /
                                  (base_station_interference + self.N * self.base_station_power))

    def init_base_station_data_rate(self):
        self.base_station_data_rate = self.base_station_bandwidth * np.log2(1 + self.base_station_sinr)

    def init_data_size_model(self):
        self.data_size_model = self.total_model_params * 4 / (1024 ** 2)

    def init_user_delay(self):
        self.user_delay = self.data_size_model / self.user_data_rate

    def init_base_station_delay(self):
        self.base_station_delay = self.data_size_model / self.base_station_data_rate

    def init_total_delay(self):
        self.total_delay = self.user_delay + self.base_station_delay

    def init_rb_allocation(self):
        distances = [item for sublist in self.user_distance for item in sublist]

        pos_list = np.arange(len(distances))
        combined_data = list(zip(distances, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        _, self.rb_allocation = zip(*sorted_data)
        self.rb_allocation = list(self.rb_allocation)
