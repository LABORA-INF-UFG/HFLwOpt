import numpy as np


class ShowCommModel:

    @staticmethod
    def show_q(comm_model, i=-1):
        print("-> q")
        if i == -1:
            print(comm_model.q)
        else:
            pass

    @staticmethod
    def show_w(comm_model, i=-1):
        print("-> W")
        if i == -1:
            print(comm_model.W)
        else:
            pass

    @staticmethod
    def show_total_user_delay(comm_model, i=-1):
        print("-> total_user_delay")
        if i == -1:
            print(comm_model.total_user_delay)
        else:
            pass

    @staticmethod
    def show_total_energy(comm_model, i=-1):
        print("-> total_energy")
        if i == -1:
            print(comm_model.total_energy)
        else:
            pass

    @staticmethod
    def show(comm_model):
        print("**************************")
        print("user_interference")
        print(comm_model.user_interference)
        print("user_power")
        print(comm_model.user_power)

        print("list_clients")
        print(comm_model.list_clients)
        print("user_distance")
        print(comm_model.user_distance)
        print("q")
        print(comm_model.q)
        print("h")
        print(comm_model.h)
        print("user_sinr")
        print(comm_model.user_sinr)
        print("user_data_rate")
        print(comm_model.user_data_rate)
        print("base_station_sinr")
        print(comm_model.base_station_sinr)
        print("base_station_data_rate")
        print(comm_model.base_station_data_rate)
        print("data_size_model")
        print(comm_model.data_size_model)
        print("user_delay")
        print(comm_model.user_delay)
        print("base_station_delay")
        print(comm_model.base_station_delay)
        print("total_delay")
        print(comm_model.total_delay)
        print("user_energy_training")
        print(comm_model.user_energy_training)
        print("user_upload_energy")
        print(comm_model.user_upload_energy)
        print("total_energy")
        print(comm_model.total_energy)
