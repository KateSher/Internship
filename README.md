# Задание к Занятию 20.

**Задание:**

Используя пример по фильтрации MNIST-датасета, предоставленный на занятии, обучить нейросетевые модели с различным фактором зашумления (noise_factor), выполнить следующее:

1. Установить noise_factor в значения 0.1; 0.2; 0.3; 0.4; 0.5; 0.6. Для каждого из этих значений подготовить датасет и обучить модели.

2. Протестировать все 6 моделей, сравнить конечные значения loss-функций.

3. Для любых 10 выбранных изображений из тестовой выборки произвести сравнение качества фильтрования изображений каждой из моделей. Для этого воспользоваться показателем MSE (Mean Squared Error). Можно воспользоваться следующей функцией для сравнения изображений:

```sh
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
```

4. Построить зависимость MSE от noise_factor.