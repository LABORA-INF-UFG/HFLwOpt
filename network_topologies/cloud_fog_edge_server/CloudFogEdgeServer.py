import sys
from comm_model.ServerToServerCommModel import ServerToServerCommModel
from network_topologies.cloud_edge_server.RoundCosts import RoundCosts
from network_topologies.cloud_edge_server.CloudEdgeServer import CloudEdgeServer
from server.ServerAggregator import ServerAgregador
from network_topologies.cloud_server.ServerStrategyFit import ServerStrategyFit


class CloudFogEdgeServer:
    def __init__(self, n_fs, n_es,
                 n_cloud_rounds,
                 n_fog_rounds,
                 n_edge_rounds,

                 fog_lower_limit, fog_upper_limit,
                 edge_lower_limit, edge_upper_limit,

                 csf):

        self.n_fs = n_fs

        self.n_cloud_rounds = n_cloud_rounds
        self.n_fog_rounds = n_fog_rounds

        self.n_es = n_es
        self.n_edge_rounds = n_edge_rounds

        self.edge_lower_limit = edge_lower_limit
        self.edge_upper_limit = edge_upper_limit

        self.csf = csf

        self.fog_server = {}
        self.sa = ServerAgregador(self.csf.cm)
        self.init_fog_server()

        self.comm_model = ServerToServerCommModel(total_model_params=self.sa.total_params,
                                                  server_number=self.n_fs,
                                                  lower_limit=fog_lower_limit,
                                                  upper_limit=fog_upper_limit)

        self.round_costs = RoundCosts(n_server=self.n_fs,
                                      n_server_rounds=self.n_fog_rounds,
                                      comm_model=self.comm_model,
                                      topology="CloudFogEdgeServer")


    def init_fog_server(self):
        for fsid in range(self.n_fs):
            self.fog_server[fsid] = FogServer(fsid=fsid,
                                              n_fog_rounds=self.n_fog_rounds,

                                              n_es=self.n_es,
                                              n_edge_rounds=self.n_edge_rounds,
                                              edge_lower_limit=self.edge_lower_limit,
                                              edge_upper_limit=self.edge_upper_limit,
                                              csf=self.csf)

    def fit(self):

        for i in range(self.n_cloud_rounds):

            weight_list = []
            sample_sizes = []
            for fsid in self.fog_server:
                weights, size, _ = self.fog_server[fsid].fit(parameters=self.sa.w_global, text=f"CloudRound: {i + 1} | FSID: {fsid + 1}:")
                weight_list.append(weights)
                sample_sizes.append(size)

            self.sa.aggregate_fit(weight_list, sample_sizes)
            self.sa.centralized_evaluation()

            self.round_costs.compute(self.fog_server)

            print("\nAggregation -> Level: #3")
            self.sa.print_evaluate()


class FogServer:
    def __init__(self, fsid, n_fog_rounds, n_es, n_edge_rounds, edge_lower_limit, edge_upper_limit, csf):
        self.fsid = fsid
        self.n_fog_rounds = n_fog_rounds
        self.ces_Fit = CloudEdgeServer(n_es=n_es,
                                       init_esi=self.fsid * n_es,
                                       n_rounds=self.n_fog_rounds,
                                       n_edge_rounds=n_edge_rounds,
                                       lower_limit=edge_lower_limit,
                                       upper_limit=edge_upper_limit,
                                       csf=csf)

    def fit(self, parameters, text):
        return self.ces_Fit.fit(parameters, text=f"{text} | FogRound")
