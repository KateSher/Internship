# Задание к Занятию 17.

**Задание:**

1. Загрузить данные из файла dataset.csv в pandas-датафрейм.

2. Первые 2 столбца датафрейма (position и intention) принять в качестве входного вектора, третий столбец – в качестве целевых значений (можно воспользоваться методом drop и общими подходами к индексированию и выбору данных из датафрейма).

3. Используя функцию train_test_split из состава Sklearn (from sklearn.model_selection import train_test_split) разделить входные и целевые данные на тренировочные и тестовые; на обучение отвести 80% данных (установкой параметра train_size = 0.8).

4. Используя StandardScaler, преобразовать входные данные путем их шкалирования, например:

scaler = StandardScaler()

scaler.fit(X_train_unscaled)

X_train= scaler.transform(X_train_unscaled)

X_test = scaler.transform(X_test_unscaled)

Целевые данные не шкалировать.

5. Построить модель с такой же архитектурой, как в рассмотренном на занятии примере, но со следующими отличиями:

– размерность входного слоя – 2 (input_shape=(2,)).

– плотность выходного слоя – 2, а его активационная функция должна быть указана явно (activation='softmax');

6. Откомпилировать модель с таким же как в рассмотренном примере оптимизатором (Adam), но с другой loss-функцией loss = 'sparse_categorical_crossentropy'. Кроме того, задать параметр metrics = ['acc']. Это необходимо для логирования Accuracy.

7. Обучить модель классификатора. Количество эпох (50 – 200) и размер батча (5 – 10) подобрать самостоятельно, но так, чтобы не наблюдалось переобучения. Гиперпараметр verbose можно установить в значение 2 для вывода лога в процессе обучения.

8. Продемонстрировать результаты обучения в виде кривых loss-функций (тренировочной и валидационной), а также acc (тренировочного и валидационного). В истории (history) данные об изменении acc и val_acc доступны по соответствующим ключам, то есть:

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='val')




