# Задание к Занятию 8.

**Задание:**

Используя библиотеку Pandas, написать программу, которая выполняет следующее:

1. Загружает данные из dataset.csv в DataFrame. Заголовок csv-файла при этом должен быть проигнорирован. Датафрейм должен иметь следующий вид:

2. Выводит заданное количество идущих подряд строк из любой заданной позиции датафрейма.

3. Находит все уникальные жанры (Genre) в исходных данных и разделяет загруженный датафрейм по жанровым датафреймам (например, в отдельных датафреймах должны размещаться жанры Shooter, Racing, Fighting, Puzzle и т.д.).

4. Упорядочивает строки в каждом из образованных жанровых датафреймов по возрастанию значений в столбце Year (например, 1985, 1986, 1987 и т. д.).

5. В каждом из полученных жанровых датафреймов производит переиндексацию по возрастанию индекса (порядок строк внутри датафреймов не должен изменяться – соответствует пункту 4).

6. Сохраняет результаты в csv-файлы. Имя файла должно соответствовать названию жанра (например, action.csv).



**Решение:**

1. Загружаем данные из dataset.csv в DataFrame df, игнорируя заголовок.

2. Создаем функцию part_df для вывод заданного количества идущих подряд строк из любой заданной позиции датафрейма.

3. Выполняем поиск всех уникальных жанров и делим исходный датафрейм на жанровые датафреймы.

4. Упорядочиваем строки в каждом датафрейме по возрастанию в столбце Year.

5. Делаем переиндексацию каждого жанрового датафрейма по возрастанию индекса (без изменения порядка строк).

6. Сохраняем полученные результаты в csv-файлы.
