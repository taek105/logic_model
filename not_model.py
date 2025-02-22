import numpy as np

class NOTModel:
    def __init__(self):
        self.weights = np.random.rand(1)
        self.bias = np.random.rand(1)
        self.train()

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([0, 1])
        outputs = np.array([1, 0])
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
        print("\nNOT 연산 테스트:")
        for x1 in [0, 1]:
            print(f"NOT({x1}) = {self.predict(x1)}")

    def testJson(self):
        result = {}
        for x1 in [0, 1]:
            result[f"NOT({x1})"] = self.predict(x1)

        return result
