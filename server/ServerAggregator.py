from load_data.LoadData import LoadData
from ml_model.MLModel import Model


class ServerAgregador:
    def __init__(self, cm, parameters=None):

        self.cm = cm
        self.load_data = LoadData(self.cm)

        self.model = Model.create_model(self.cm)
        self.total_params = self.model.count_params()

        if parameters is None:
            self.w_global = self.model.get_weights()
        else:
            self.w_global = parameters

        (_, _), (self.x_test, self.y_test) = self.load_data.data_server()

        self.evaluate_list = {"centralized": {"loss": [], "accuracy": []}}

    def aggregate_fit(self, parameters, sample_sizes):
        self.w_global = []
        for weights in zip(*parameters):
            weighted_sum = 0
            total_samples = sum(sample_sizes)
            for i in range(len(weights)):
                weighted_sum += weights[i] * sample_sizes[i]
            self.w_global.append(weighted_sum / total_samples)

    def centralized_evaluation(self):
        self.model.set_weights(self.w_global)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        self.evaluate_list["centralized"]["loss"].append(loss)
        self.evaluate_list["centralized"]["accuracy"].append(accuracy)
        return loss, accuracy

    def print_evaluate(self, loss=False):
        print(f"accuracy: \n{self.evaluate_list['centralized']['accuracy']}")
        if loss:
            print(f"loss: \n{self.evaluate_list['centralized']['loss']}")
