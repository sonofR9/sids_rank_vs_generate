# TransAct-like 

Проект реализует TransAct-like подход для предсказания реакции пользователя на рекомендованный трек в датасете Yambda

Для каждого candidate прослушивания модель должна предсказать:

- `like` — target `1`;
- `dislike` — target `0`.

Текущий эксперимент построен на версии Yambda `sequential/50m` и использует первые 2000 пользователей

## Постановка задачи

Каждый обучающий пример связан с событием `listen`

Последующий `like` или `dislike` связывается с последним предыдущим прослушиванием того же item. Используются только реакции, произошедшие не позднее чем через 6 часов после прослушивания

Момент предсказания — начало candidate-прослушивания:

```text
candidate listen
        ↓
предсказание модели
        ↓
последующий like или dislike
```

История пользователя содержит только события с timestamp строго меньше timestamp candidate

В историю не входят:

- текущее candidate-прослушивание;
- последующая реакция;
- любые события из будущего;
- `played_ratio` текущего candidate.

`played_ratio` candidate не используется, потому что становится известен уже после начала прослушивания и создаёт утечку target

## Подготовка данных

Для каждого валидного candidate-прослушивания строится история:

- последние 80 любых событий;
- до 20 дополнительных более старых explicit-событий.

К explicit-событиям относятся:

```text
like
dislike
unlike
undislike
```

Такой способ сохраняет недавний пользовательский контекст и более старые сигналы предпочтений

### Признаки candidate

```text
candidate item
is_organic
track length
флаг наличия track length
```

### Признаки события истории

```text
item
event type
is_organic
played ratio
флаг наличия played ratio
track length
флаг наличия track length
time delta до candidate
```

Длина трека и временная дистанция:

1. ограничиваются сверху;
2. преобразуются через `log1p`;
3. стандартизируются по статистикам train.

Mappings пользователей и items также строятся только по train. Новые пользователи и items в validation и test переводятся в специальные `UNK`-индексы

Данные делятся глобально по времени:

```text
первые 80% candidate-событий → train
следующие 10%                → validation
последние 10%                → test
```

Таким образом, validation и test находятся позже train по времени.

## Пример обучающего примера

Пусть у пользователя есть такая последовательность:

```text
t=100: listen item_A
t=120: like item_A
t=200: listen item_B
t=230: dislike item_B
t=300: listen item_C
t=340: like item_C
```

Для последнего взаимодействия:

```text
candidate item: item_C
candidate timestamp: 300
target: 1
```

В историю войдут только события строго до `t=300`:

```text
listen item_A
like item_A
listen item_B
dislike item_B
```

Концептуально обучающий пример выглядит так:

```python
{
    "user_id": "user_1",

    "candidate_item_id": "item_C",
    "candidate_is_organic": 0,
    "candidate_track_length": 210,

    "history_item_ids": [
        "item_A",
        "item_A",
        "item_B",
        "item_B",
    ],

    "history_event_types": [
        "listen",
        "like",
        "listen",
        "dislike",
    ],

    "history_played_ratio": [
        0.85,
        0.0,
        0.20,
        0.0,
    ],

    "history_has_played_ratio": [
        1,
        0,
        1,
        0,
    ],

    "history_time_delta_seconds": [
        1000,
        900,
        500,
        350,
    ],

    "target": 1,
}
```

Само `listen item_C` и последующий `like item_C` в историю не включаются

## Представление историй переменной длины

Истории пользователей имеют разную длину

Вместо padding в `Dataset` все события историй одного батча склеиваются в плоские тензоры:

```text
history_item_idxs_flat
history_event_type_idxs_flat
history_is_organic_flat
history_played_ratio_flat
history_track_length_z_flat
history_time_delta_z_flat
```

Отдельно хранится:

```text
history_lengths
```

Например:

```text
history 1: [a, b, c]
history 2: [d, e]
history 3: [f, g, h, i]
```

представляется как:

```text
history_flat = [a, b, c, d, e, f, g, h, i]
history_lengths = [3, 2, 4]
```

В текущем стандартном PyTorch Transformer padding временно восстанавливается только внутри encoder и сопровождается padding mask

## Модели

### CandidateOnlyMLP

Использует только:

```text
candidate item embedding
candidate is_organic
candidate track length
```

Модель показывает, сколько сигнала содержится только в candidate

### UserCandidateMLP

Использует:

```text
user embedding
candidate item embedding
candidate features
```

Это основной baseline без истории пользователя

### RichMeanHistoryMLP

Каждое событие истории кодируется через:

```text
item embedding
event-type embedding
числовые признаки события
```

Полученные rich history tokens усредняются в один history vector.

Финальный MLP получает:

```text
user embedding
candidate embedding
candidate features
mean-pooled history vector
```

Эта модель проверяет, полезна ли богатая история даже без sequence-aware attention

### CandidateTokenTransformer

Candidate представляется отдельным токеном и добавляется в начало последовательности:

```text
[candidate, history event 1, history event 2, ..., history event L]
```

Каждое history event представлено rich token, содержащим item, тип события и числовые признаки

Дополнительно используются:

- position embeddings для порядка событий;
- token-type embeddings для различения candidate и history

После Transformer берётся представление первого токена:

```python
candidate_history_vector = encoded_sequence[:, 0]
```

Это candidate token, обогащённый информацией из истории пользователя

Финальный ranking head получает:

```text
user embedding
candidate embedding
candidate features
candidate history representation
```


## Обучение

Все модели обучаются через общий `ExplicitFeedbackTrainer`.

Для каждого батча Trainer:

1. переносит тензоры на выбранное устройство;
2. передаёт в модель все поля, кроме `target`;
3. получает один logit для каждого примера;
4. считает `BCEWithLogitsLoss`;
5. запускает `backward`;
6. применяет gradient clipping;
7. обновляет параметры через `AdamW`.


Все модели следуют общему интерфейсу:

```text
input:  именованные тензоры батча
output: logits формы [batch_size]
```

Поэтому один Trainer используется для baseline, mean-history модели и Transformer.

Во время validation и test logits преобразуются в вероятности через sigmoid.

Считаются метрики:

```text
ROC-AUC
PR-AUC для like
PR-AUC для dislike
log loss
balanced accuracy
macro F1
```

Сохраняется состояние модели с лучшим validation ROC-AUC. Оно затем используется для оценки на test

## Структура проекта

```text
experiments/
└── TransAct-like.ipynb

src/
└── TransAct-like/
    ├── __init__.py
    ├── config.py
    ├── preprocessing.py
    ├── dataset.py
    ├── models.py
    └── training.py
```

Назначение файлов:

```text
config.py
    Константы, event mappings, PAD и UNK индексы.

preprocessing.py
    Чтение строк Yambda, связывание listen с реакциями,
    построение истории, temporal split, mappings,
    нормализация и кодирование выборок.

dataset.py
    PyTorch Dataset, varlen collate-функция и DataLoaders.

models.py
    Candidate-only baseline, user-candidate baseline,
    rich mean-history модель и candidate-token Transformer.

training.py
    Общий Trainer, optimizer, метрики и фиксация seed

experiments/
    Ноутбук с прогоном
```
