from keras.models import load_model
import numpy as np

# Загрузка сохраненной модели
model = load_model('btc_price_prediction_model.h5')

input_data = np.random.rand(20, 10, 4) # Генерация 20 чисел

predictions = model.predict(input_data)

# Вывод предсказанных значений
print("Сгенерированные числа:")
print(predictions.flatten()) # Выводи чисел
