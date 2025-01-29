import sys
import tensorflow as tf
from client.GenerateClientList import GenerateClientList
from config.ConfigModel import ConfigModel
from config.ConfigServerFit import ConfigServerFit
from network_topologies.cloud_edge_server.CloudEdgeServer import CloudEdgeServer

if __name__ == "__main__":
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")

    if device == '/CPU:0':
        sys.exit()

    for i in range(5):
        with tf.device(device):
            n_es = 3
            ds_clients = GenerateClientList(n_es=n_es, n_device=120, lower_limit=100, upper_limit=625)

            csf = ConfigServerFit(ds_clients=ds_clients,
                                  min_fit_clients=10,
                                  rb_number=15,

                                  error_rate_requirement=0.3,
                                  lmbda=2.6,
                                  fixed_parameters=True,

                                  user_power=0.01,
                                  user_bw=1,
                                  user_cpu_freq=1,

                                  cm=ConfigModel(model_type="CNN",
                                                 shape=(28, 28, 1),
                                                 path_clients="/home/Dev/Dev/Python/Datasets/2024-HFL/fmnist/0.9",
                                                 path_server="/home/Dev/Dev/Python/Datasets/2024-HFL/fmnist/fmnist")
                                  )

            print("Topology: CloudEdgeServer")
            ces_Fit = CloudEdgeServer(n_es=n_es,

                                      n_rounds=20,
                                      n_edge_rounds=5,

                                      lower_limit=1575,
                                      upper_limit=1625,

                                      csf=csf)

            ces_Fit.fit()
            ces_Fit.sa.print_evaluate(loss=True)
            ces_Fit.round_costs.print_result()
