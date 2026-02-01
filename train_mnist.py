# train_mnist.py

# Импортируем необходимые библиотеки:
import os  # для работы с файловой системой
import torch  # основной фреймворк глубокого обучения
import torch.nn as nn  # модули нейронных сетей
import torch.optim as optim  # оптимизаторы
from torch.utils.data import DataLoader  # для загрузки данных батчами
from torchvision import datasets, transforms  # датасеты и преобразования изображений

# Создаем папку 'models' если она не существует
# exist_ok=True предотвращает ошибку если папка уже существует
os.makedirs("models", exist_ok=True)

# Определяем устройство для вычислений: GPU если доступен, иначе CPU
# torch.cuda.is_available() проверяет наличие поддерживаемой видеокарты NVIDIA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Определяем архитектуру сверточной нейронной сети
class SmallCNN(nn.Module):
    def __init__(self):
        # Вызываем конструктор родительского класса nn.Module
        super().__init__()

        # Sequential - контейнер, который последовательно применяет слои
        self.net = nn.Sequential(
            # Первый сверточный слой:
            # nn.Conv2d(1, 16, 3, padding=1) - свертка 2D
            #   1 входной канал (черно-белое изображение)
            #   16 выходных каналов (фильтров)
            #   ядро 3x3, padding=1 для сохранения размерности
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),  # Функция активации ReLU (Rectified Linear Unit)
            nn.MaxPool2d(2),  # Макс-пулинг с окном 2x2 (уменьшает размер в 2 раза)

            # Второй сверточный слой:
            nn.Conv2d(16, 32, 3, padding=1),  # 16 входных, 32 выходных канала
            nn.ReLU(),
            nn.MaxPool2d(2),  # Еще одно уменьшение размера в 2 раза

            # Преобразует многомерный тензор в одномерный вектор
            # После двух пулингов: 32 * 7 * 7 = 1568 элементов
            nn.Flatten(),

            # Полносвязные слои:
            nn.Linear(32 * 7 * 7, 128),  # 1568 входов -> 128 нейронов
            nn.ReLU(),
            nn.Linear(128, 10)  # 128 нейронов -> 10 выходов (по одному на каждую цифру)
        )

    def forward(self, x):
        """Прямой проход данных через сеть"""
        return self.net(x)


# Преобразования для данных:
# ToTensor() преобразует PIL Image или numpy array в тензор PyTorch
# и автоматически нормализует значения пикселей из [0,255] в [0,1]
transform = transforms.ToTensor()

# Загружаем тренировочный и тестовый датасеты MNIST:
# "data" - папка для хранения данных
# download=True - скачивает датасет если отсутствует локально
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

# Создаем DataLoader'ы для итерации по данным батчами:
# batch_size=128 - размер батча для тренировки (128 образцов за раз)
# shuffle=True - перемешивает данные каждый эпох для лучшего обучения
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

# batch_size=256 - больший батч для тестирования (быстрее)
# shuffle=False - не перемешивать тестовые данные
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

# Создаем модель, перемещаем ее на выбранное устройство (GPU/CPU)
model = SmallCNN().to(device)

# Создаем оптимизатор Adam для обновления весов модели:
# model.parameters() - все обучаемые параметры сети
# lr=1e-3 - learning rate (скорость обучения) = 0.001
opt = optim.Adam(model.parameters(), lr=1e-3)

# Функция потерь для многоклассовой классификации:
# CrossEntropyLoss объединяет Softmax и Negative Log Likelihood
crit = nn.CrossEntropyLoss()

# Цикл обучения на 3 эпохах:
for epoch in range(3):
    # Переводим модель в режим тренировки
    model.train()

    # Итерируемся по батчам тренировочных данных
    for x, y in train_dl:
        # Перемещаем данные на то же устройство, что и модель
        x, y = x.to(device), y.to(device)

        # Обнуляем градиенты от предыдущего шага
        # Если не сделать это, градиенты будут накапливаться
        opt.zero_grad()

        # Прямой проход: пропускаем данные через модель
        # model(x) возвращает тензор формы [batch_size, 10] - логиты для каждого класса
        # Вычисляем потери между предсказаниями и истинными метками
        loss = crit(model(x), y)

        # Обратный проход: вычисляем градиенты
        loss.backward()

        # Шаг оптимизации: обновляем веса модели используя вычисленные градиенты
        opt.step()

# После обучения переводим модель в режим оценки
model.eval()

# Переменные для подсчета точности
correct = 0  # количество правильных предсказаний
total = 0  # общее количество тестовых образцов

# torch.no_grad() отключает вычисление градиентов для экономии памяти и ускорения
with torch.no_grad():
    # Итерируемся по тестовым батчам
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)

        # Прямой проход: получаем предсказания модели
        # model(x) возвращает логиты [batch_size, 10]
        # argmax(1) находит индекс максимального значения по dimension 1 (классы)
        # Это дает предсказанную цифру для каждого образца в батче
        pred = model(x).argmax(1)

        # Считаем количество правильных предсказаний:
        # (pred == y) создает булев тензор, где True - правильное предсказание
        # sum() подсчитывает количество True, item() преобразует в Python число
        correct += (pred == y).sum().item()

        # numel() возвращает общее количество элементов в тензоре y
        total += y.numel()

# Вычисляем точность: правильные предсказания / общее количество
acc = correct / total

# Сохраняем обученную модель:
# state_dict() содержит все обучаемые параметры модели
# acc - точность на тестовом наборе для справки
torch.save({"state_dict": model.state_dict(), "acc": acc}, "models/mnist_cnn.pth")

# Выводим результат
print(f"✅ Saved models/mnist_cnn.pth | Test acc: {acc:.3f}")
