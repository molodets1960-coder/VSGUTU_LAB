import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# Перенос данных из шага 1 и архитектуру из шага 2

# 1. Воспроизводим архитектуру модели из Шага 2
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Инициализируем модель, функцию потерь и оптимизатор Adam
model = MLP(input_dim=13)
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 
# Тензоры из 1 шага
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('heart.csv')
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



epochs = 50
#пустые списки для хранения истории:
train_losses, val_losses = [], []
train_accs, val_accs = [], []
print("--- Старт процесса обучения нейросети (100 эпох) ---")

for epoch in range(epochs):
    model.train() # Переводим модель в режим обучения
    epoch_loss = 0
    correct_train = 0
    
    # Обучение методом мини-батчей (размер батча = 32 пациента)
    batch_size = 32
    
    # Перебираем данные батчами
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # 1. Обнуляем градиенты оптимизатора, чтобы они не накапливались
        optimizer.zero_grad()
        
        # 2. Прямой проход (forward pass) — делаем предсказание
        predictions = model(X_batch)
        
        # 3. Расчет значения функции потерь (Loss)
        loss = criterion(predictions, y_batch)
        
        # 4. Обратный проход (backpropagation) — считаем производные (градиенты)
        loss.backward()
        
        # 5. Шаг оптимизации — обновляем веса сети
        optimizer.step()
        
        epoch_loss += loss.item() * len(X_batch)
        
        # Считаем точность (Accuracy) на обучении
        pred_classes = (predictions > 0.5).float()
        correct_train += (pred_classes == y_batch).float().sum().item()
        
    # Вычисляем средние метрики за всю эпоху
    total_train_loss = epoch_loss / len(X_train)
    total_train_acc = correct_train / len(X_train)
    
    # --- Проверка качества на валидационной выборке (каждую эпоху) ---
    model.eval() # Переводим модель в режим оценки
    with torch.no_grad(): # Отключаем расчет градиентов для экономии памяти
        val_predictions = model(X_val)
        total_val_loss = criterion(val_predictions, y_val).item()
        total_val_acc = ((val_predictions > 0.5).float() == y_val).float().mean().item()
        
        train_losses.append(total_train_loss)
    train_accs.append(total_train_acc)
    val_losses.append(total_val_loss)
    val_accs.append(total_val_acc)
    # Выводим лог в консоль каждые 10 эпох 
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Эпоха {epoch+1:02d}: Train Loss = {total_train_loss:.4f} | Val Loss = {total_val_loss:.4f} | Val Accuracy = {total_val_acc:.4f}")
# 4. Построение двух графиков рядом (Loss и Accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# График функции потерь (Loss)
ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss (Обучение)', color='royalblue', linewidth=2)
ax1.plot(range(1, epochs + 1), val_losses, label='Val Loss (Валидация)', color='crimson', linewidth=2)
ax1.set_title('Кривые функции потерь (Loss)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('BCELoss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# График точности (Accuracy)
ax2.plot(range(1, epochs + 1), train_accs, label='Train Accuracy (Обучение)', color='royalblue', linewidth=2)
ax2.plot(range(1, epochs + 1), val_accs, label='Val Accuracy (Валидация)', color='crimson', linewidth=2)
ax2.set_title('Кривые точности (Accuracy)')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
# Сохраняем картинку в папку проекта для вставки в Word
plt.savefig('learning_curves.png', dpi=150)
print("[+] График успешно сохранен в файл 'learning_curves.png'!")
plt.show()