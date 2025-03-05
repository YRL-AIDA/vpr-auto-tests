# vpr-auto-tests

Виртуальное окружение vpr-env
Включить окружение

`conda activate vpr-env`

## Установка и настройка виртуального окружения
1. Создать окружение
    `conda create --name vpr-env python=3.12`
2. Войти в окружение
    `conda activate vpr-env`
3. Установить зависимости
   `pip install -r requirements.txt`
5. Для использования в jupyther-notebook(в нужном окружении) выполнить:
    ```
   conda install ipykernel
    python -m ipykernel install --user --name vpr-env --display-name "Python3.12 (vpr-env)"
    ```
## Как добавить датасет

1. Создать дирректорию датасета в datasets и загрузить в нее данные.
2. В dataloaders/val/ создать python-файл и реализовать в нем класс датасета аналогично примерам из дирректории.

## Как добавить модель

Все модели добавляются в дирректорию models/
В данной дирректории нужно реализовать класс модели.

Для моделей с архитектурой backbone +  aggrigator:
Основной класс VPRmodel, представленный в test.ipynb

Для добавления новых аггрегаторов/бекбонов:

1. Добавить в соответствующую дирректорию (models/aggregators/, models/backbones) файл с классом соответствующей модели и импортировать его в `__init__.py`
2. Добавить необходимые параметры в get_aggregator и get_backbone соответствующие elif по названиям моделей. 