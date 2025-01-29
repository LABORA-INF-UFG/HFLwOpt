import sys
from comm_strategy.CommStrategyDeviceToServer import CommStrategyDeviceToServer
from server.ServerAggregator import ServerAgregador
from server.ServerFit import ServerFit


class ServerStrategyFit(CommStrategyDeviceToServer):
    def __init__(self, sid, n_rounds, csf):
        self.n_rounds = n_rounds
        self.sa = ServerAgregador(csf.cm)

        super().__init__(self.sa.total_params, ServerFit(list_clients=csf.ds_clients.group_list[sid],
                                                         list_distances=csf.ds_clients.user_distance[sid],
                                                         cm=csf.cm), csf)
        self.server_local_rounds = 0


    def configure_strategy_fit(self):

        # 1
        # self.random_user_selection(factor=1)
        # self.random_rb_allocation()
        # self.fixed_allocation()
        # self.compute_comm_parameters()

        # 2
        # self.greater_loss_user_selection(factor=1.5, k=self.csf.min_fit_clients)
        # self.random_rb_allocation()
        # self.fixed_allocation()
        # self.compute_comm_parameters()

        # 3
        # self.random_user_selection(factor=1)
        # self.compute_comm_parameters()
        # self.sinr_rb_allocation()

        # 4
        # self.greater_loss_user_selection(factor=1.5, k=self.csf.min_fit_clients)
        # self.compute_comm_parameters()
        # self.sinr_rb_allocation()

        # 5/6
        self.greater_data_user_selection(factor=2, k=self.csf.min_fit_clients*1.5)
        self.compute_comm_parameters()
        self.optimization()

        ################
        self.upload_status()
        self.compute_round_costs()
        self.selected_clients = self.success_uploads

    def fit(self):
        self.configure_strategy_fit()

        self.sf.configure_fit(self.selected_clients)
        print(f"success_uploads: {self.success_uploads} - error_uploads: {self.error_uploads}")
        print("-------------------------")

        for cid in self.selected_clients:
            self.count_of_client_selected[cid] = self.count_of_client_selected[cid] + 1

        sample_sizes_list = []
        if len(self.selected_clients) > 0:

            for cid in self.selected_clients:
                self.count_of_client_uploads[cid] = self.count_of_client_uploads[cid] + 1

            weight_list, sample_sizes, info = self.sf.fit(self.sa.w_global)
            self.sa.aggregate_fit(weight_list, sample_sizes)
            sample_sizes_list.append(sum(sample_sizes))

        print(f"Centralized Evaluation: R: {self.server_local_rounds + 1}")
        _, evaluate_accuracy = self.sa.centralized_evaluation()
        print(f"evaluate_accuracy: {evaluate_accuracy}")
        self.server_local_rounds = self.server_local_rounds + 1

        return self.sa.w_global, sum(sample_sizes_list) if len(self.selected_clients) > 0 else 1, {}

    def print_result(self):
        self.sa.print_evaluate(loss=True)

        for item, value in self.round_costs_list.items():
            print(item)
            print(value)

        print("\ncount_of_client_selected")
        print(self.count_of_client_selected)

        print("\ncount_of_client_uploads")
        print(self.count_of_client_uploads)
