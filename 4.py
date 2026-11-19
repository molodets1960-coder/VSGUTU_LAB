import copy
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка и подготовка данных (как в Шаге 1)
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'heart.csv'))

X = df.drop('condition', axis=1).values
y = df['condition'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).view(-1, 1)

# 2. Архитектура лучшей модели (64 -> 32) с возможностью добавить Dropout
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 3. Функция для прогона одного эксперимента регуляризации
def run_experiment(name, dropout=0.0, weight_decay=0.0, use_early_stopping=False):
    torch.manual_seed(42)  # фиксируем начальные веса для честного сравнения
    model = MLP(input_dim=13, dropout_rate=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    epochs = 50
    batch_size = 32
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        # Оценка на валидации
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val).item()

        # Логика раннего останова (Early Stopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if use_early_stopping and counter >= patience:
                print(f"[{name}] Early stopping сработал на эпохе {epoch + 1}")
                break

    if use_early_stopping:
        model.load_state_dict(best_weights)

    # Итоговые метрики на train и val
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        val_predictions = model(X_val)
        final_val_loss = criterion(val_predictions, y_val).item()
        final_val_acc = ((val_predictions > 0.5).float() == y_val).float().mean().item()

    overfit = "Да" if (final_val_loss - train_loss) > 0.08 else "Нет"
    return name, final_val_acc, final_val_loss, overfit, model

# 4. Запуск всех 5 вариантов из таблицы методички
experiments = [
    ("Базовая (без регуляризации)", 0.0, 0.0, False),
    ("+ L2 (weight decay)",          0.0, 1e-3, False),
    ("+ Dropout",                    0.3, 0.0, False),
    ("+ Early stopping",             0.0, 0.0, True),
    ("+ L2 + Dropout + Early stop",  0.3, 1e-3, True),
]

print("\n| Конфигурация | Val Accuracy | Val Loss | Переобучение (да/нет) |")
print("|---|---|---|---|")

best_model = None
for exp in experiments:
    name, acc, loss, overfit, trained_model = run_experiment(*exp)
    print(f"| {name} | {acc:.4f} | {loss:.4f} | {overfit} |")
    best_model = trained_model

# Сохраняем веса лучшей модели 
torch.save(best_model.state_dict(), os.path.join(base_dir, 'best_model.pt'))
print("\n[+] Веса лучшей модели сохранены в 'best_model.pt'")