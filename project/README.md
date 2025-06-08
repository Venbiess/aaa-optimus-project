# aaa-optimus-project

## Названи команды
Оптимус

## Состав команды
* Тарасова Ульяна
* Алексеев Андрей 
* Нахамкин Константин

Капитан: Алексеев Андрей

Загружает ответы: Алексеев Андрей

## Тема проекта
Background Blur / Crop / Face Blur / Замена Фона

## Локальный запуск через Docker
Перейдя в директорию репозитория
```
docker-compose up
```
Затем перейти по адресу http://localhost:8080/lab/tree/car_segmentation.ipynb

## Запуск в Kaggle
Загрузить файл requirements.txt в Kaggle (правда могут быть конфликты, т.к. Kaggle не даёт изменять версию Python, в этом случае нужно закомментировать ошибки через #).

```
!pip install -r <Путь к requirements.txt>
```

## Установка новых библиотек
Стоит добавить их в environment.yml и для сохранения `requirements.txt`
```
!pip list --format=freeze > requirements.txt
```