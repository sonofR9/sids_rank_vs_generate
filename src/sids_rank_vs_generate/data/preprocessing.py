from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    EVENT2IDX,
    EXPLICIT_EVENT_TYPES,
    TIME_DELTA_CLIP_SECONDS,
    TRACK_LENGTH_CLIP_SECONDS,
    UNK_EVENT_IDX,
    UNK_ITEM_IDX,
    UNK_USER_IDX,
    VALID_EVENT_TYPES,
)


def get_first_existing(row, *names):
    for name in names:
        if name in row:
            return row[name]

    raise KeyError(
        f"None of the fields {names} were found. "
        f"Available fields: {list(row.keys())}"
    )


def normalize_event_type(event):
    if isinstance(event, bytes):
        event = event.decode("utf-8")

    event = str(event).lower().strip()

    if event not in VALID_EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event!r}")

    return event


def unpack_user_row(row):
    items = np.asarray(
        get_first_existing(row, "item_id", "item_ids"),
        dtype=np.int64,
    )

    timestamps = np.asarray(
        get_first_existing(row, "timestamp", "timestamps"),
        dtype=np.int64,
    )

    event_types = np.asarray(
        get_first_existing(row, "event_type", "event_types"),
        dtype=object,
    )

    is_organic = np.asarray(
        get_first_existing(row, "is_organic"),
        dtype=np.int64,
    )

    played_ratio = np.asarray(
        get_first_existing(row, "played_ratio_pct"),
        dtype=object,
    )

    track_length = np.asarray(
        get_first_existing(row, "track_length_seconds"),
        dtype=object,
    )

    lengths = {
        len(items),
        len(timestamps),
        len(event_types),
        len(is_organic),
        len(played_ratio),
        len(track_length),
    }

    if len(lengths) != 1:
        raise ValueError(
            f"Sequence lengths do not match for uid={row['uid']}: "
            f"items={len(items)}, "
            f"timestamps={len(timestamps)}, "
            f"event_types={len(event_types)}, "
            f"is_organic={len(is_organic)}, "
            f"played_ratio={len(played_ratio)}, "
            f"track_length={len(track_length)}"
        )

    event_types = np.asarray(
        [normalize_event_type(x) for x in event_types],
        dtype=object,
    )

    return {
        "uid": int(row["uid"]),
        "items": items,
        "timestamps": timestamps,
        "event_types": event_types,
        "is_organic": is_organic,
        "played_ratio": played_ratio,
        "track_length": track_length,
    }


def safe_optional_float(value):
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_reaction_links(
    dataset,
    max_users=None,
):
    """
    Для каждого like/dislike ищет последнее предыдущее listen
    того же item-а
    """
    records = []
    stats = Counter()

    for user_number, row in enumerate(
        tqdm(dataset, total=max_users, desc="Linking reactions")
    ):
        if max_users is not None and user_number >= max_users:
            break

        data = unpack_user_row(row)

        uid = data["uid"]
        items = data["items"]
        timestamps = data["timestamps"]
        event_types = data["event_types"]
        is_organic = data["is_organic"]
        played_ratio = data["played_ratio"]
        track_length = data["track_length"]

        stats["users"] += 1
        stats["all_events"] += len(items)

        # Для каждого item храним индекс последнего listen
        last_listen_by_item = {}

        # Один listen теоретически может получить несколько reaction events
        # Здесь собираем метки по listen index
        labels_by_listen_idx = {}
        candidate_data_by_listen_idx = {}

        for event_idx in range(len(items)):
            event_type = event_types[event_idx]
            item_id = int(items[event_idx])
            event_timestamp = int(timestamps[event_idx])

            stats[f"event_{event_type}"] += 1

            if event_type == "listen":
                last_listen_by_item[item_id] = event_idx
                continue

            if event_type not in {"like", "dislike"}:
                continue

            target = 1 if event_type == "like" else 0

            listen_idx = last_listen_by_item.get(item_id)

            if listen_idx is None:
                stats["reaction_without_previous_listen"] += 1
                continue

            listen_timestamp = int(timestamps[listen_idx])
            delay_bins = event_timestamp - listen_timestamp

            if delay_bins < 0:
                stats["negative_delay"] += 1
                continue

            # В Yambda один timestamp-bin = 5 секунд
            delay_seconds = delay_bins * 5

            existing_target = labels_by_listen_idx.get(listen_idx)

            if existing_target is None:
                labels_by_listen_idx[listen_idx] = target

                candidate_data_by_listen_idx[listen_idx] = {
                    "uid": uid,
                    "candidate_item_id": item_id,

                    # Это индекс candidate listen внутри последовательности
                    # Он понадобится для построения history строго до candidate
                    "candidate_event_idx": int(listen_idx),

                    "candidate_timestamp": listen_timestamp,
                    "reaction_timestamp": event_timestamp,

                    "reaction_type": event_type,
                    "target": float(target),

                    "reaction_delay_bins": int(delay_bins),
                    "reaction_delay_seconds": int(delay_seconds),
                    "reaction_delay_hours": delay_seconds / 3600.0,

                    "candidate_is_organic": int(is_organic[listen_idx]),
                    "reaction_is_organic": int(is_organic[event_idx]),

                    "candidate_played_ratio_pct": safe_optional_float(
                        played_ratio[listen_idx]
                    ),

                    "candidate_track_length_seconds": safe_optional_float(
                        track_length[listen_idx]
                    ),
                }

            elif existing_target == target:
                # Например, два like для одного и того же listen
                stats["duplicate_same_reaction"] += 1

                # Оставляем первую реакцию — она ближе к моменту listen
                old_delay = candidate_data_by_listen_idx[
                    listen_idx
                ]["reaction_delay_seconds"]

                if delay_seconds < old_delay:
                    candidate_data_by_listen_idx[listen_idx].update({
                        "reaction_timestamp": event_timestamp,
                        "reaction_type": event_type,
                        "reaction_delay_bins": int(delay_bins),
                        "reaction_delay_seconds": int(delay_seconds),
                        "reaction_delay_hours": delay_seconds / 3600.0,
                        "reaction_is_organic": int(is_organic[event_idx]),
                    })

            else:
                # Один listen связан и с like и с dislike
                # Такой пример неоднозначен помечаем и позже выбрасываем
                stats["conflicting_reactions"] += 1
                labels_by_listen_idx[listen_idx] = -1

        for listen_idx, target in labels_by_listen_idx.items():
            if target not in {0, 1}:
                stats["dropped_ambiguous_examples"] += 1
                continue

            records.append(candidate_data_by_listen_idx[listen_idx])
            stats["linked_examples"] += 1
            stats[f"linked_target_{target}"] += 1

    links_df = pd.DataFrame(records)

    return links_df, stats


def print_reaction_report(df, stats):
    likes = int((df["target"] == 1).sum()) if len(df) else 0
    dislikes = int((df["target"] == 0).sum()) if len(df) else 0

    explicit_events = (
        stats.get("event_like", 0)
        + stats.get("event_dislike", 0)
    )

    linked = len(df)

    coverage = (
        linked / explicit_events
        if explicit_events > 0
        else np.nan
    )

    print("Users processed:", stats.get("users", 0))
    print("All events:", stats.get("all_events", 0))
    print()

    print("Like events:", stats.get("event_like", 0))
    print("Dislike events:", stats.get("event_dislike", 0))
    print("Explicit events total:", explicit_events)
    print()

    print(
        "Reactions without previous same-item listen:",
        stats.get("reaction_without_previous_listen", 0),
    )
    print(
        "Duplicate same reactions:",
        stats.get("duplicate_same_reaction", 0),
    )
    print(
        "Conflicting reactions:",
        stats.get("conflicting_reactions", 0),
    )
    print()

    print("Linked examples:", linked)
    print(f"Link coverage: {coverage:.3%}")
    print()

    print("Like targets:", likes)
    print("Dislike targets:", dislikes)

    if linked > 0:
        print(f"Like rate: {likes / linked:.3%}")
        print(f"Dislike rate: {dislikes / linked:.3%}")


def reaction_window_report(df, windows_hours):
    rows = []

    total_examples = len(df)

    for max_hours in windows_hours:
        filtered = df[
            df["reaction_delay_hours"] <= max_hours
        ]

        likes = int((filtered["target"] == 1).sum())
        dislikes = int((filtered["target"] == 0).sum())

        rows.append({
            "max_delay_hours": max_hours,
            "examples": len(filtered),
            "coverage_of_linked": (
                len(filtered) / total_examples
                if total_examples > 0
                else np.nan
            ),
            "users": filtered["uid"].nunique(),
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": (
                likes / len(filtered)
                if len(filtered) > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def process_played_ratio(value):
    """
    Возвращает:
        normalized_ratio
        has_ratio
    """
    value = safe_optional_float(value)

    if np.isnan(value):
        return 0.0, 0

    value = np.clip(value, 0.0, 100.0)

    return float(value / 100.0), 1


def process_track_length(value):
    """
    Сохраняем длину в секундах
    Нормализацию сделаем после split только по train
    """
    value = safe_optional_float(value)

    if np.isnan(value) or value < 0:
        return 0.0, 0

    return float(value), 1


def build_rich_examples(
    dataset,
    links_df,
    max_users=None,
    recent_history_size=80,
    explicit_history_size=20,
):
    """
    Для каждого связанного candidate listen строит историю строго
    до candidate_timestamp

    История:
        последние recent_history_size любых событий
        +
        последние explicit_history_size более старых explicit-событий
    """

    # uid -> candidate_event_idx -> информация о target
    links_by_user = {}

    for row in links_df.itertuples(index=False):
        uid = int(row.uid)
        candidate_idx = int(row.candidate_event_idx)

        links_by_user.setdefault(uid, {})[candidate_idx] = {
            "target": float(row.target),
            "reaction_type": row.reaction_type,
            "reaction_timestamp": int(row.reaction_timestamp),
            "reaction_delay_hours": float(row.reaction_delay_hours),
        }

    examples = []
    stats = Counter()

    for user_number, row in enumerate(
        tqdm(
            dataset,
            total=max_users,
            desc="Building rich examples",
        )
    ):
        if max_users is not None and user_number >= max_users:
            break

        uid = int(row["uid"])

        if uid not in links_by_user:
            continue

        data = unpack_user_row(row)

        items = data["items"]
        timestamps = data["timestamps"]
        event_types = data["event_types"]
        is_organic = data["is_organic"]
        played_ratio = data["played_ratio"]
        track_length = data["track_length"]

        user_links = links_by_user[uid]

        for candidate_idx, link in user_links.items():
            stats["candidate_links"] += 1

            if candidate_idx < 0 or candidate_idx >= len(items):
                stats["invalid_candidate_idx"] += 1
                continue

            if event_types[candidate_idx] != "listen":
                stats["candidate_is_not_listen"] += 1
                continue

            candidate_timestamp = int(timestamps[candidate_idx])
            candidate_item_id = int(items[candidate_idx])

            # Берём только события со строго меньшим timestamp
            prior_indices = np.flatnonzero(
                timestamps[:candidate_idx] < candidate_timestamp
            ).tolist()

            if len(prior_indices) == 0:
                stats["empty_history"] += 1
                continue

            # Последние N любых событий
            recent_indices = prior_indices[-recent_history_size:]
            selected_indices = set(recent_indices)

            # Добавляем более старые explicit events, которые не попали в recent history
            added_explicit = []

            for j in reversed(prior_indices):
                if j in selected_indices:
                    continue

                if event_types[j] in EXPLICIT_EVENT_TYPES:
                    added_explicit.append(j)

                    if len(added_explicit) >= explicit_history_size:
                        break

            selected_indices.update(added_explicit)

            # Возвращаем хронологический порядок
            history_indices = sorted(selected_indices)

            history_item_ids = []
            history_event_types = []
            history_is_organic = []

            history_played_ratio = []
            history_has_played_ratio = []

            history_track_length_seconds = []
            history_has_track_length = []

            history_time_delta_seconds = []

            for j in history_indices:
                ratio_value, has_ratio = process_played_ratio(
                    played_ratio[j]
                )

                length_value, has_length = process_track_length(
                    track_length[j]
                )

                # Один timestamp bin Yambda 5 секунд
                delta_seconds = (
                    candidate_timestamp - int(timestamps[j])
                ) * 5

                history_item_ids.append(int(items[j]))
                history_event_types.append(str(event_types[j]))
                history_is_organic.append(int(is_organic[j]))

                history_played_ratio.append(ratio_value)
                history_has_played_ratio.append(has_ratio)

                history_track_length_seconds.append(length_value)
                history_has_track_length.append(has_length)

                history_time_delta_seconds.append(
                    int(delta_seconds)
                )

            candidate_length, candidate_has_length = (
                process_track_length(
                    track_length[candidate_idx]
                )
            )

            examples.append({
                "user_id": uid,

                "candidate_item_id": candidate_item_id,
                "candidate_timestamp": candidate_timestamp,
                "candidate_is_organic": int(
                    is_organic[candidate_idx]
                ),

                "candidate_track_length_seconds": (
                    candidate_length
                ),
                "candidate_has_track_length": (
                    candidate_has_length
                ),

                "target": link["target"],
                "reaction_type": link["reaction_type"],
                "reaction_delay_hours": (
                    link["reaction_delay_hours"]
                ),

                "history_item_ids": history_item_ids,
                "history_event_types": history_event_types,
                "history_is_organic": history_is_organic,

                "history_played_ratio": (
                    history_played_ratio
                ),
                "history_has_played_ratio": (
                    history_has_played_ratio
                ),

                "history_track_length_seconds": (
                    history_track_length_seconds
                ),
                "history_has_track_length": (
                    history_has_track_length
                ),

                "history_time_delta_seconds": (
                    history_time_delta_seconds
                ),

                "history_length": len(history_indices),
            })

            stats["built_examples"] += 1

    examples_df = pd.DataFrame(examples)

    return examples_df, stats


HISTORY_COLUMNS = (
    "history_item_ids",
    "history_event_types",
    "history_is_organic",
    "history_played_ratio",
    "history_has_played_ratio",
    "history_track_length_seconds",
    "history_has_track_length",
    "history_time_delta_seconds",
)


def validate_history_row(row, history_columns=HISTORY_COLUMNS):
    expected_length = row["history_length"]
    return all(
        len(row[column]) == expected_length
        for column in history_columns
    )

def global_temporal_split(
    df,
    train_fraction=0.8,
    val_fraction=0.1,
):
    df = (
        df
        .sort_values(
            ["candidate_timestamp", "user_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(n * train_fraction)
    val_end = int(
        n * (train_fraction + val_fraction)
    )

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    split_info = {
        "train_end_position": train_end,
        "val_end_position": val_end,
        "train_max_timestamp": int(
            train_df["candidate_timestamp"].max()
        ),
        "val_min_timestamp": int(
            val_df["candidate_timestamp"].min()
        ),
        "val_max_timestamp": int(
            val_df["candidate_timestamp"].max()
        ),
        "test_min_timestamp": int(
            test_df["candidate_timestamp"].min()
        ),
    }

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_info,
    )


def fit_mappings(train_df):
    train_users = sorted(
        train_df["user_id"].unique().tolist()
    )

    user2idx = {
        user_id: idx
        for idx, user_id in enumerate(
            train_users,
            start=1,
        )
    }

    train_items = set(
        train_df["candidate_item_id"].tolist()
    )

    for history in train_df["history_item_ids"]:
        train_items.update(history)

    item2idx = {
        item_id: idx
        for idx, item_id in enumerate(
            sorted(train_items),
            start=2,
        )
    }

    return user2idx, item2idx


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, values):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.size == 0:
            return

        self.count += values.size
        self.sum += values.sum()
        self.sum_sq += np.square(values).sum()

    def finalize(self):
        mean = self.sum / self.count

        variance = (
            self.sum_sq / self.count
            - mean ** 2
        )

        std = np.sqrt(max(variance, 1e-12))

        return {
            "mean": float(mean),
            "std": float(std),
            "count": int(self.count),
        }


def fit_numeric_stats(train_df):
    track_moments = RunningMoments()
    time_moments = RunningMoments()

    candidate_mask = (
        train_df["candidate_has_track_length"]
        .astype(bool)
        .to_numpy()
    )

    candidate_lengths = (
        train_df.loc[
            candidate_mask,
            "candidate_track_length_seconds",
        ]
        .to_numpy(dtype=np.float64)
    )

    candidate_lengths = np.log1p(
        np.clip(
            candidate_lengths,
            0,
            TRACK_LENGTH_CLIP_SECONDS,
        )
    )

    track_moments.update(candidate_lengths)

    for lengths, has_lengths, time_deltas in zip(
        train_df["history_track_length_seconds"],
        train_df["history_has_track_length"],
        train_df["history_time_delta_seconds"],
    ):
        lengths = np.asarray(
            lengths,
            dtype=np.float64,
        )

        has_lengths = np.asarray(
            has_lengths,
            dtype=bool,
        )

        if has_lengths.any():
            transformed_lengths = np.log1p(
                np.clip(
                    lengths[has_lengths],
                    0,
                    TRACK_LENGTH_CLIP_SECONDS,
                )
            )

            track_moments.update(
                transformed_lengths
            )

        time_deltas = np.asarray(
            time_deltas,
            dtype=np.float64,
        )

        transformed_deltas = np.log1p(
            np.clip(
                time_deltas,
                0,
                TIME_DELTA_CLIP_SECONDS,
            )
        )

        time_moments.update(
            transformed_deltas
        )

    return {
        "track_length": track_moments.finalize(),
        "time_delta": time_moments.finalize(),
    }


def normalize_track_length(
    values,
    has_values,
    stats,
):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    has_values = np.asarray(
        has_values,
        dtype=bool,
    )

    result = np.zeros(
        len(values),
        dtype=np.float32,
    )

    if has_values.any():
        transformed = np.log1p(
            np.clip(
                values[has_values],
                0,
                TRACK_LENGTH_CLIP_SECONDS,
            )
        )

        result[has_values] = (
            transformed - stats["mean"]
        ) / stats["std"]

    return result.tolist()


def normalize_time_delta(values, stats):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    transformed = np.log1p(
        np.clip(
            values,
            0,
            TIME_DELTA_CLIP_SECONDS,
        )
    )

    normalized = (
        transformed - stats["mean"]
    ) / stats["std"]

    return normalized.astype(
        np.float32
    ).tolist()


def encode_split(
    df,
    user2idx,
    item2idx,
    numeric_stats,
):
    records = []

    track_stats = numeric_stats["track_length"]
    time_stats = numeric_stats["time_delta"]

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Encoding split",
    ):
        history_item_idxs = [
            item2idx.get(
                int(item_id),
                UNK_ITEM_IDX,
            )
            for item_id in row.history_item_ids
        ]

        history_event_type_idxs = [
            EVENT2IDX.get(
                str(event_type),
                UNK_EVENT_IDX,
            )
            for event_type in row.history_event_types
        ]

        history_track_length_z = (
            normalize_track_length(
                row.history_track_length_seconds,
                row.history_has_track_length,
                track_stats,
            )
        )

        history_time_delta_z = (
            normalize_time_delta(
                row.history_time_delta_seconds,
                time_stats,
            )
        )

        candidate_length_z = normalize_track_length(
            [row.candidate_track_length_seconds],
            [row.candidate_has_track_length],
            track_stats,
        )[0]

        records.append({
            "user_idx": user2idx.get(
                int(row.user_id),
                UNK_USER_IDX,
            ),

            "candidate_item_idx": item2idx.get(
                int(row.candidate_item_id),
                UNK_ITEM_IDX,
            ),

            "candidate_timestamp": int(
                row.candidate_timestamp
            ),

            "candidate_is_organic": float(
                row.candidate_is_organic
            ),

            "candidate_track_length_z": float(
                candidate_length_z
            ),

            "candidate_has_track_length": float(
                row.candidate_has_track_length
            ),

            "target": float(row.target),

            "history_item_idxs": (
                history_item_idxs
            ),

            "history_event_type_idxs": (
                history_event_type_idxs
            ),

            "history_is_organic": [
                float(x)
                for x in row.history_is_organic
            ],

            "history_played_ratio": [
                float(x)
                for x in row.history_played_ratio
            ],

            "history_has_played_ratio": [
                float(x)
                for x in row.history_has_played_ratio
            ],

            "history_track_length_z": (
                history_track_length_z
            ),

            "history_has_track_length": [
                float(x)
                for x in row.history_has_track_length
            ],

            "history_time_delta_z": (
                history_time_delta_z
            ),

            "history_length": int(
                row.history_length
            ),
        })

    return pd.DataFrame(records)


def unknown_report(df):
    total_history_tokens = sum(
        df["history_length"]
    )

    unknown_history_items = sum(
        item_idx == UNK_ITEM_IDX
        for history in df["history_item_idxs"]
        for item_idx in history
    )

    return {
        "examples": len(df),

        "unknown_user_rate": float(
            (df["user_idx"] == UNK_USER_IDX).mean()
        ),

        "unknown_candidate_item_rate": float(
            (
                df["candidate_item_idx"]
                == UNK_ITEM_IDX
            ).mean()
        ),

        "unknown_history_item_rate": (
            unknown_history_items
            / total_history_tokens
        ),
    }
