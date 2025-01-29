import sys
from comm_model.ServerToServerCommModel import ServerToServerCommModel
from network_topologies.cloud_edge_server.RoundCosts import RoundCosts
from server.ServerAggregator import ServerAgregador
from network_topologies.cloud_server.ServerStrategyFit import ServerStrategyFit


class CloudEdgeServer:
    def __init__(self, n_es, n_rounds, n_edge_rounds, lower_limit, upper_limit, init_esi=0, csf=None):

        self.n_es = n_es
        self.n_rounds = n_rounds
        self.init_esi = init_esi
        self.n_edge_rounds = n_edge_rounds

        self.csf = csf

        self.edge_server = {}
        self.sa = ServerAgregador(self.csf.cm)
        self.init_edge_server()

        self.comm_model = ServerToServerCommModel(total_model_params=self.sa.total_params,
                                                  server_number=self.n_es,
                                                  lower_limit=lower_limit,
                                                  upper_limit=upper_limit)

        self.round_costs = RoundCosts(n_server=self.n_es,
                                      n_server_rounds=self.n_edge_rounds,
                                      comm_model=self.comm_model,
                                      topology="CloudEdgeServer")

    def init_edge_server(self):
        for esid in range(self.n_es):
            self.edge_server[esid] = EdgeServer(esid=(self.init_esi + esid),
                                                n_edge_rounds=self.n_edge_rounds,
                                                csf=self.csf)

    def fit(self, parameters=None, text="CloudRound"):

        if parameters is not None:
            self.sa.w_global = parameters

        sample_sizes_list = []
        for i in range(self.n_rounds):

            weight_list = []
            sample_sizes = []
            for esid in self.edge_server:
                weights, size, _ = self.edge_server[esid].fit(parameters=self.sa.w_global, text=f"{text}: {i+1}")
                weight_list.append(weights)
                sample_sizes.append(size)

            self.sa.aggregate_fit(weight_list, sample_sizes)
            self.sa.centralized_evaluation()
            sample_sizes_list.append(sum(sample_sizes))

            self.round_costs.compute(self.edge_server)
            print("\nAggregation -> Level: #2")
            self.sa.print_evaluate()

        return self.sa.w_global, sum(sample_sizes_list) / len(sample_sizes_list), {}


class EdgeServer:
    def __init__(self, esid, n_edge_rounds, csf):
        self.esid = esid
        self.n_edge_rounds = n_edge_rounds
        self.csf = csf
        self.round = 0
        self.s_stf = ServerStrategyFit(sid=self.esid, n_rounds=self.n_edge_rounds, csf=self.csf)

    def fit(self, parameters, text):
        self.s_stf.sa.w_global = parameters

        sample_sizes_list = []
        for i in range(self.n_edge_rounds):
            self.round = self.round + 1
            print("\n-----------------------------------------------------------------------------------")
            print(f"{text} | ESID: {self.esid + 1} | EdgeRound: {i + 1} | LocalRound: {self.round}")
            print("-----------------------------------------------------------------------------------\n")

            _, sample_size, _ = self.s_stf.fit()
            sample_sizes_list.append(sample_size)

        self.s_stf.sa.print_evaluate()
        return self.s_stf.sa.w_global, sum(sample_sizes_list) / len(sample_sizes_list), {}
