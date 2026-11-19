import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__() 
        
        # nn.Sequential связывает слои в единую цепочку (граф вычислений)
        self.net = nn.Sequential(
            # Слой 1 (Скрытый): Полносвязный (Linear). Принимает 13 признаков, выдает 64 нейрона
            nn.Linear(input_dim, 64),
            # Функция активации ReLU: обнуляет все отрицательные сигналы, внося нелинейность в сеть
            nn.ReLU(),
            
            # Слой 2 (Скрытый): Полносвязный. Принимает 64 нейрона, выдает 32 нейрона
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Слой 3 (Выходной): Полносвязный. Принимает 32 нейрона, выдает 1 выходной сигнал
            nn.Linear(32, 1),
            # Функция активации Sigmoid: сжимает число на выходе строго в диапазон от 0 до 1 (вероятность болезни)
            nn.Sigmoid()
        )
        
    def forward(self, x):
    
        return self.net(x)

# Создаем экземпляр нейросети
model = MLP(input_dim=13)
criterion = nn.BCELoss()

# Определяем оптимизатор Adam, который будет обновлять веса со скоростью обучения (Learning Rate) = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Считаем общее число обучаемых параметров в сети (весов и смещений) встроенными средствами PyTorch
pytorch_params = sum(p.numel() for p in model.parameters())
print(f"\nКоличество обучаемых параметров сети (вывод PyTorch): {pytorch_params} шт.")