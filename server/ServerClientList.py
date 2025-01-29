import numpy as np
from client.Client import Client
from ml_model.MLModel import Model


class ServerClientList:
    def __init__(self, list_clients, list_distances, cm):

        self.cm = cm
        self.list_clients = list_clients
        self.model = Model.create_model(self.cm)

        self.clients_model_dict = {}
        self.clients_number_data_samples = {}
        self.list_distances = {}
        self.loss_list = {}

        self.init_models()
        self.list_distances = dict(zip(list_clients, list_distances))

    def init_models(self):

        for i in self.list_clients:
            tmp_client = Client(i,
                                load_data_constructor=False,# Instantiate when necessary
                                cm=self.cm
                                )
            self.clients_number_data_samples[i] = tmp_client.number_data_samples()
            self.loss_list[i] = np.inf
            self.clients_model_dict[i] = None # Instantiate when necessary

        print("\nID Clients")
        print(self.list_clients)

    def instantiate(self, selected_clients):
        for i in selected_clients:
            self.clients_model_dict[i] = Client(i,
                                                load_data_constructor=False,
                                                cm=self.cm)

    def destroy(self, selected_clients):
        for i in selected_clients:
            self.clients_model_dict[i] = None


    def get_cids(self):
        return list(self.clients_model_dict.keys())









