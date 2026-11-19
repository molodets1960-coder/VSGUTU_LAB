import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных и воспроизведение разбиения из Шага 1
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'heart.csv'))

feature_names = df.drop('condition', axis=1).columns
X = df.drop('condition', axis=1).values
y = df['condition'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test_raw)

# 2. Загрузка обученной нейросети из Этапа 4
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
model.eval()

# 3. База продукционных правил (формат из Практической работы №1)
def apply_production_rules(patient):
    # Правило 1 (критическое): Асимптоматическая боль (cp == 3) И поражены сосуды (ca >= 1) -> Болен (1)
    if patient['cp'] == 3 and patient['ca'] >= 1:
        return 1
    # Правило 2 (высокий приоритет): Депрессия сегмента ST > 1.5 И стенокардия напряжения (exang == 1) -> Болен (1)
    if patient['oldpeak'] > 1.5 and patient['exang'] == 1:
        return 1
    # Правило 3 (средний приоритет): Дефект талассемии (thal == 2) И низкий макс. пульс (< 145) -> Болен (1)
    if patient['thal'] == 2 and patient['thalach'] < 145:
        return 1
    # Правило 4: Возраст > 60 И холестерин > 260 И бессимптомная боль (cp == 3) -> Болен (1)
    if patient['age'] > 60 and patient['chol'] > 260 and patient['cp'] == 3:
        return 1
    # Иначе срабатывает правило по умолчанию -> Здоров (0)
    return 0

# 4. Сравнение на первых 5 примерах из test-выборки
df_test_orig = pd.DataFrame(X_test_raw[:5], columns=feature_names)
X_test_tensor = torch.FloatTensor(X_test_scaled[:5])

with torch.no_grad():
    mlp_probs = model(X_test_tensor).numpy().flatten()
    mlp_preds = (mlp_probs > 0.5).astype(int)

print("\n| Пример | Истинный класс | Предсказание правилами | Предсказание MLP | Совпадение правил | Совпадение MLP |")
print("|---|---|---|---|---|---|")

for i in range(5):
    true_cls = int(y_test[i])
    rule_cls = apply_production_rules(df_test_orig.iloc[i])
    mlp_cls = int(mlp_preds[i])

    rule_match = "Да" if rule_cls == true_cls else "Нет"
    mlp_match = "Да" if mlp_cls == true_cls else "Нет"

    print(f"| {i+1} | {true_cls} | {rule_cls} | {mlp_cls} (p={mlp_probs[i]:.2f}) | {rule_match} | {mlp_match} |")