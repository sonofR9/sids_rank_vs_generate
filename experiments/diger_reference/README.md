# DIGER reference v0.1

Этот запуск использует официальную реализацию DIGER без копирования её исходников в
репозиторий. Launcher фиксирует upstream commit
`cd72dcdc25d28e0acfa5aea14788ad863b41ab4f`, готовит общий Yambda split и запускает
FrqUD, SDUD или совместный вариант в отдельном окружении.

По умолчанию используется протокол `three-way`: tokenizer обучается только на fit,
лучший checkpoint выбирается по validation, test остаётся финальным. Начальный
RQ-VAE и подготовленные данные сохраняются в общем work root и повторно используются
следующими DIGER-запусками.

## Окружение авторов

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n diger-ref python=3.12.11 pip -y
conda activate diger-ref
python -m pip install -r experiments/diger_reference/requirements.txt
```

Проект должен быть установлен в основном окружении с MLflow:

```bash
conda activate sids
python -m pip install -e ".[mlflow]"
```

## Полный FrqUD-запуск

```bash
export DIGER_ENV_BIN="$HOME/miniconda3/envs/diger-ref/bin"
export DIGER_WORK_ROOT="/mnt/storage-1/g.shevchenko/sids_rank_vs_generate/diger-v0.1"
export MLFLOW_TRACKING_URI="http://51.250.47.231:5000"

python -u experiments/diger_reference_v0_1.py \
  --work-root "$DIGER_WORK_ROOT" \
  --diger-env-bin "$DIGER_ENV_BIN" \
  --gpu 1 \
  --variant frqud \
  --mlflow \
  --tracking-uri "$MLFLOW_TRACKING_URI"
```

Для быстрого smoke-run можно ограничить данные и эпохи:

```bash
python -u experiments/diger_reference_v0_1.py \
  --work-root "$DIGER_WORK_ROOT/smoke" \
  --diger-env-bin "$DIGER_ENV_BIN" \
  --gpu 1 \
  --variant frqud \
  --latest-interactions 200000 \
  --max-train-examples 20000 \
  --rqvae-epochs 2 \
  --joint-epochs 1
```

Результаты одного запуска лежат в `runs/<run-name>/`:

- `diger.log` — полный upstream log
- `run_complete.json` — статус и Recall/NDCG авторов
- `sid_mapping/semantic_ids.parquet` — три raw SID-уровня
- `sid_mapping/semantic_ids_with_conflict.parquet` — SID с conflict token
- `sid_mapping/semantic_ids_original_items.parquet` — mapping на исходные item ID

Официальная реализация в этой версии считает Recall@1/5/10 и NDCG@5/10. После
первого сигнала raw SID можно прогнать через общий генератор проекта для Recall@100.

## Общий downstream Recall@100

Для сравнения с остальными SID используется исходный трёхуровневый mapping на
оригинальные item ID. Общий runner сам сопоставляет его с внутренним каталогом.

```bash
export DIGER_VARIANT="both"
export DIGER_SID_PATH="/path/to/semantic_ids_original_items.parquet"
export SID_REPORTS_DIR="/path/to/shared/results/diger-common-v0.1"

python -u run.py experiments/diger_poster_generation.py \
  --skip-ranking \
  --mlflow \
  --mlflow-tracking-uri "http://51.250.47.231:5000"
```

Для FrqUD нужно заменить `DIGER_VARIANT` на `frqud` и указать его SID-файл.
Конфиг фиксирует poster-каталог из 33 148 айтемов и тот же recipe генерации,
который использовался для общей серии качества SID.
