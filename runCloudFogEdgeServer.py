import sys
import tensorflow as tf
from client.GenerateClientList import GenerateClientList
from config.ConfigModel import ConfigModel
from config.ConfigServerFit import ConfigServerFit
from network_topologies.cloud_fog_edge_server.CloudFogEdgeServer import CloudFogEdgeServer

if __name__ == "__main__":
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")

    if device == '/CPU:0':
        sys.exit()

    for i in range(1):
        with tf.device(device):
            n_fs = 3
            n_es = 3
            ds_clients = GenerateClientList(n_es=n_fs * n_es, n_device=120, lower_limit=50, upper_limit=250)

            csf = ConfigServerFit(ds_clients=ds_clients,
                                  min_fit_clients=10,
                                  rb_number=15,

                                  error_rate_requirement=0.3,
                                  lmbda=2.6,
                                  fixed_parameters=False,

                                  user_power=0.01,
                                  user_bw=1,
                                  user_cpu_freq=1,

                                  cm=ConfigModel(model_type="CNN",
                                                 shape=(28, 28, 1),
                                                 path_clients="/home/Dev/Python/Datasets/2024-HFL/fmnist/0.9",
                                                 path_server="/home/Dev/Python/Datasets/2024-HFL/fmnist/fmnist")
                                  )

            print("Topology: CloudFogEdgeServer")
            cfes_Fit = CloudFogEdgeServer(n_fs,
                                          n_es,

                                          n_cloud_rounds=10,
                                          n_fog_rounds=2,
                                          n_edge_rounds=5,

                                          fog_lower_limit=850,
                                          fog_upper_limit=900,

                                          edge_lower_limit=850,
                                          edge_upper_limit=900,

                                          csf=csf)

            cfes_Fit.fit()
            cfes_Fit.sa.print_evaluate(loss=True)
            cfes_Fit.round_costs.print_result()
