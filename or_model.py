import numpy as np

class ORModel:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
        self.train()

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])
        for epoch in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                # 오차 계산
                error = outputs[i] - prediction
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

        self.test()

    def predict(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return 1 if total > 0 else 0
    
    def test(self):
        print("\nOR 연산 테스트:")
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                print(f"OR({x1}, {x2}) = {self.predict([x1, x2])}")

    def testJson(self):
        result = {}
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                predicted = self.predict([x1, x2])
                correct = 1 if x1 == 1 or x2 == 1 else 0  # OR 연산 정답
                result[f"OR({x1},{x2})"] = {"예측정답": predicted, "정답": correct}

        # JSON 형식으로 출력
        return result