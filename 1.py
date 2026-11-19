
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


df = pd.read_csv('heart.csv')

# Выделяем матрицу признаков и вектор ответов y (столбец condition)
X = df.drop('condition', axis=1).values
y = df['condition'].values


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Нормализация признаков при помощи StandardScaler (среднее 0, разброс 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Вычисляем параметры ТОЛЬКО на Train
X_val = scaler.transform(X_val)         # Применяем к Валидации
X_test = scaler.transform(X_test)       # Применяем к Тесту

# 4. Преобразование обработанных массивов NumPy в тензоры PyTorch
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1) # Разворачиваем в вертикальный столбец

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Вывод результатов для проверки в консоль
print(f"Размер обучающего тензора признаков: {X_train.shape} (Ровно 13 входов)")
print(f"Размер вектора ответов y_train: {y_train.shape}")