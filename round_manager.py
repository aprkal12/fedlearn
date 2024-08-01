
class RoundManager:
    def __init__(self, client_manager, aggregator):
        self.client_manager = client_manager
        self.aggregator = aggregator
        self.current_round = 0
        self.parameters = []

    def start_new_round(self):
        self.current_round += 1
        self.parameters = []
        # 클라이언트에게 새 라운드 시작 알림

    def end_round(self):
        aggregated_params = self.aggregator.aggregate_parameters(self.parameters)
        self.update_global_model(aggregated_params)
        # 새 라운드 시작 또는 대기

    def receive_parameters(self, client_id, params):
        if client_id in self.client_manager.clients:
            self.parameters.append(params)
            if len(self.parameters) == len(self.client_manager.clients):
                self.end_round()

    def update_global_model(self, aggregated_params):
        # 글로벌 모델 업데이트 로직
        pass
