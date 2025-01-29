import sys
import tensorflow as tf
from client.GenerateClientList import GenerateClientList
from config.ConfigModel import ConfigModel
from config.ConfigServerFit import ConfigServerFit
from network_topologies.cloud_server.ServerStrategyFit import ServerStrategyFit

if __name__ == "__main__":
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")

    if device == '/CPU:0':
        sys.exit()

    with tf.device(device):
        ds_clients = GenerateClientList(n_es=1, n_device=120, lower_limit=250, upper_limit=2000) # 500 / 2000

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
                                             path_clients="/home/Dev/Python/Datasets/2024-HFL/fmnist/0.9",
                                             path_server="/home/Dev/Dev/Python/Datasets/2024-HFL/fmnist/fmnist")
                              )

        print("Topology: CloudServer")
        s_stf = ServerStrategyFit(sid=0, n_rounds=100, csf=csf)

        for i in range(s_stf.n_rounds):
            print(f"ServerRound: {i + 1}")
            s_stf.fit()

        s_stf.print_result()
