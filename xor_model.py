import torch
import torch.nn as nn
import torch.optim as optim

inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
outputs = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 입력 2 -> 은닉 2
        self.fc2 = nn.Linear(2, 1)  # 은닉 2 -> 출력 1
        self.train()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 은닉층 (시그모이드)
        x = torch.sigmoid(self.fc2(x))  # 출력층 (시그모이드)
        return x

    def train(self, epochs=10000, attempt=1, max_attempts=5):
        criterion = nn.MSELoss()  # 평균제곱오차
        optimizer = optim.SGD(self.parameters(), lr=0.1) # 경사하강법

        print()
        for epoch in range(epochs):
            outputs_pred = self.forward(inputs)
            loss = criterion(outputs_pred, outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 2000 == 0:
                print(f"Epoch [{epoch+1}], Loss: {loss.item():.6f}")

        print("\nXOR 연산 테스트:")
        mismatch_found = self.test()
        
        if mismatch_found:
            if attempt < max_attempts:
                print("\n예측이 틀린 항목이 있습니다. 재학습을 시도합니다.")
                self.train(epochs=epochs, attempt=attempt+1, max_attempts=max_attempts)
            else:
                print("\n최대 재시도 횟수를 초과했습니다. 학습을 중단합니다.")
        else:
            print("\n모든 예측이 정답과 일치합니다. 학습을 종료합니다.")


    def predict(self, inputs):
        # 입력을 PyTorch 텐서로 변환 및 2차원(batch) 형태로 만들기
        x = torch.tensor(inputs, dtype=torch.float).unsqueeze(0)  # shape: (1, 2)
        
        with torch.no_grad():
            output = self.forward(x)
        
        return 1 if output.item() > 0.5 else 0
    
    def test(self):
        print("\nXOR 연산 테스트:")
        mismatch_found = False
        with torch.no_grad():
            for i in range(len(inputs)):
                input_data = inputs[i].unsqueeze(0)
                pred_val = self(input_data)
                pred_label = int(pred_val.item() >= 0.5)
                correct_label = int(outputs[i].item())

                print(f"XOR{inputs[i].tolist()}, "
                      f"예측: {pred_val.item():.4f}, "
                      f"예측정답: {pred_label}, "
                      f"정답: {correct_label}")

                if pred_label != correct_label:
                    mismatch_found = True

        return mismatch_found
    
    def testJson(self):
        result = {}
        with torch.no_grad():
            for i in range(len(inputs)):
                input_data = inputs[i].unsqueeze(0)
                pred_val = self(input_data)
                pred_label = int(pred_val.item() >= 0.5)
                correct_label = int(outputs[i].item())

                result[f"XOR{inputs[i].tolist()}"] = {
                    "예측정답": pred_label,
                    "정답": correct_label
                }

        return result