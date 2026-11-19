import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# 1. Загрузка и подготовка данных (идентично Шагу 1, чтобы X_test был тем же самым)
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'heart.csv'))

X = df.drop('condition', axis=1).values
y = df['condition'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# 2. Восстанавливаем архитектуру лучшей модели и загружаем обученные веса
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
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

model = MLP(input_dim=13, dropout_rate=0.3)
model.load_state_dict(torch.load(os.path.join(base_dir, 'best_model.pt')))

# 3. Оценка качества на тестовой выборке (model.eval() отключает Dropout!)
model.eval()
with torch.no_grad():
    y_proba = model(X_test).numpy()
    y_pred = (y_proba > 0.5).astype(int)

y_true = y_test.numpy().astype(int)

print("=== ОТЧЁТ О КЛАССИФИКАЦИИ (TEST) ===")
print(classification_report(y_true, y_pred, target_names=['Здоров (0)', 'Болен (1)'], digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (Матрица ошибок):")
print(cm)

acc = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_proba)
print(f"\nИтоговая Accuracy: {acc:.4f}")
print(f"Итоговый ROC-AUC:  {roc_auc:.4f}")

# 4. Визуализация и сохранение тепловой карты матрицы ошибок
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Здоров (0)', 'Болен (1)'],
            yticklabels=['Здоров (0)', 'Болен (1)'])
plt.xlabel('Предсказание нейросети')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок на тестовой выборке')
plt.tight_layout()

cm_path = os.path.join(base_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
print(f"\n[+] Матрица ошибок сохранена в '{cm_path}'")
plt.show()