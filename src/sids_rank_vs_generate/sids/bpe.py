from __future__ import annotations

from collections import Counter
from typing import Hashable

import polars as pl

from ..tracking import ExperimentTracker, NullTracker


Token = Hashable
Sequence = list[Token]


class BPE:
    def __init__(
        self,
        num_merges: int = 1000,
        min_frequency: int = 2,
        codebook_size: int = 512,
    ):
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")

        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.codebook_size = codebook_size

        self.merges: dict[tuple[Token, Token], Token] = {}
        self.vocabulary: dict[Token, tuple[Token, Token]] = {}

        self._next_token_id = 0

        # Base SID:
        #   0 .. codebook_size - 1
        #
        # BPE merges:
        #   codebook_size .. codebook_size + num_merges - 1
        #
        # PAD:
        #   codebook_size + num_merges
        self.pad_id = self.codebook_size + self.num_merges
        self.vocab_size = self.pad_id + 1

        self._fitted = False

    def fit(
        self,
        sequences,
        *,
        tracker: ExperimentTracker | None = None,
    ) -> "BPE":
        tracker = tracker or NullTracker()
        sequences = [list(seq) for seq in sequences]

        if not sequences:
            raise ValueError("BPE: передан пустой корпус")

        sequences = [
            seq
            for seq in sequences
            if seq
        ]

        if not sequences:
            raise ValueError("BPE: все последовательности пустые")

        self._validate_base_tokens(sequences)

        self._next_token_id = 0
        self.merges = {}
        self.vocabulary = {}

        for _ in range(self.num_merges):
            pair_counts = self._count_pairs(sequences)

            if not pair_counts:
                break

            pair, frequency = pair_counts.most_common(1)[0]

            if frequency < self.min_frequency:
                break

            new_token = (
                "bpe",
                self._next_token_id,
            )
            self._next_token_id += 1

            self.merges[pair] = new_token
            self.vocabulary[new_token] = pair

            sequences = [
                self._merge_sequence(
                    sequence,
                    pair,
                    new_token,
                )
                for sequence in sequences
            ]
            tracker.log_metrics(
                {
                    "fit.bpe_pair_frequency": frequency,
                    "fit.bpe_merges": self._next_token_id,
                },
                step=self._next_token_id,
            )

        self._fitted = True

        return self

    def transform(self, sequences):
        if not self._fitted:
            raise RuntimeError("Сначала вызови BPE.fit()")

        sequences = [list(sequence) for sequence in sequences]
        self._validate_base_tokens(sequences)

        return [
            self._encode_sequence(sequence)
            for sequence in sequences
        ]

    def fit_transform(
        self,
        sequences,
        *,
        tracker: ExperimentTracker | None = None,
    ):
        # Materialize once so this also works when ``sequences`` is a generator.
        sequences = [list(sequence) for sequence in sequences]
        self.fit(sequences, tracker=tracker)
        return self.transform(sequences)

    def fit_polars(
        self,
        df: pl.DataFrame,
        sid_columns: list[str],
        *,
        tracker: ExperimentTracker | None = None,
    ) -> "BPE":
        sequences = self._df_to_sequences(
            df,
            sid_columns,
        )

        self.fit(sequences, tracker=tracker)

        return self

    def transform_polars(
        self,
        df: pl.DataFrame,
        sid_columns: list[str],
        pad_id: int,
    ) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("Сначала вызови BPE.fit()")

        if pad_id != self.pad_id:
            raise ValueError(
                f"Expected pad_id={self.pad_id}, "
                f"received {pad_id}"
            )

        sequences = self._df_to_sequences(
            df,
            sid_columns,
        )

        encoded = self.transform(sequences)

        max_length = len(sid_columns)

        encoded_ids = [
            self._sequence_to_padded_ids(
                sequence,
                max_length=max_length,
                pad_id=pad_id,
            )
            for sequence in encoded
        ]

        data = {
            "item_id": df["item_id"],
        }

        for level in range(max_length):
            data[f"sid_{level}"] = [
                sequence[level]
                for sequence in encoded_ids
            ]

        return pl.DataFrame(data)

    def _sequence_to_padded_ids(
        self,
        sequence: Sequence,
        *,
        max_length: int,
        pad_id: int,
    ) -> list[int]:
        """
        Возвращает SID фиксированной длины,
        сохраняя исходную RQ-позицию токена.

        Пример:

            [A0, B1, C2]
            merge A0+B1

        encoded:
            [AB, C2]

        output:
            [AB, PAD, C2]
        """

        result = [pad_id] * max_length

        for token in sequence:
            levels = self._token_levels(token)

            if not levels:
                raise ValueError(
                    "BPE token must cover at least one SID level"
                )

            expected_levels = list(
                range(
                    levels[0],
                    levels[-1] + 1,
                )
            )

            if levels != expected_levels:
                raise ValueError(
                    "BPE token spans non-contiguous "
                    f"SID levels: {levels}"
                )

            start_level = levels[0]

            if not 0 <= start_level < max_length:
                raise ValueError(
                    "BPE token starts at invalid "
                    f"SID level {start_level}"
                )

            if result[start_level] != pad_id:
                raise ValueError(
                    "Multiple BPE tokens start at "
                    f"SID level {start_level}"
                )

            result[start_level] = self._token_to_id(
                token
            )

        return result

    def _token_to_id(
        self,
        token: Token,
    ) -> int:
        """
        Наружная нумерация для трансформера.

        Base:
            ("L0", 17) -> 17
            ("L1", 17) -> 17
            ("L2", 17) -> 17

        BPE:
            ("bpe", 0) -> codebook_size
            ("bpe", 1) -> codebook_size + 1
        """

        if self._is_bpe_token(token):
            merge_index = int(token[1])

            if not 0 <= merge_index < self.num_merges:
                raise ValueError(
                    f"Invalid BPE merge index: {merge_index}"
                )

            token_id = (
                self.codebook_size
                + merge_index
            )

        else:
            token_id = int(token[1])

            if not 0 <= token_id < self.codebook_size:
                raise ValueError(
                    f"Base SID token {token_id} "
                    f"is outside "
                    f"[0, {self.codebook_size})"
                )

        if not 0 <= token_id < self.pad_id:
            raise ValueError(
                f"BPE produced token id {token_id}, "
                f"but valid non-padding range "
                f"is [0, {self.pad_id})"
            )

        return token_id

    def _token_levels(
        self,
        token: Token,
    ) -> list[int]:
        """
        Возвращает исходные RQ-levels,
        покрываемые токеном.

        ("L2", 7)
            -> [2]

        merge(("L0", 1), ("L1", 2))
            -> [0, 1]

        merge(merge(L0, L1), L2)
            -> [0, 1, 2]
        """

        if not self._is_bpe_token(token):
            return [
                self._base_token_level(token)
            ]

        if token not in self.vocabulary:
            raise ValueError(
                f"Unknown BPE token: {token!r}"
            )

        left, right = self.vocabulary[token]

        return (
            self._token_levels(left)
            + self._token_levels(right)
        )

    @staticmethod
    def _is_bpe_token(
        token: Token,
    ) -> bool:
        return (
            isinstance(token, tuple)
            and len(token) == 2
            and token[0] == "bpe"
        )

    @staticmethod
    def _base_token_level(
        token: Token,
    ) -> int:
        if (
            not isinstance(token, tuple)
            or len(token) != 2
            or not isinstance(token[0], str)
            or not token[0].startswith("L")
        ):
            raise ValueError(
                f"Invalid base SID token: {token!r}"
            )

        try:
            return int(token[0][1:])
        except ValueError as error:
            raise ValueError(
                f"Invalid base SID token: {token!r}"
            ) from error

    def _validate_base_tokens(
        self,
        sequences: list[Sequence],
    ) -> None:
        for sequence in sequences:
            for token in sequence:
                if self._is_bpe_token(token):
                    raise ValueError(
                        "Input sequences must contain "
                        "only base SID tokens"
                    )

                self._base_token_level(token)

                token_id = int(token[1])

                if not 0 <= token_id < self.codebook_size:
                    raise ValueError(
                        f"Base SID token {token_id} "
                        f"is outside "
                        f"[0, {self.codebook_size})"
                    )

    @staticmethod
    def _count_pairs(
        sequences: list[Sequence],
    ) -> Counter:
        counter = Counter()

        for sequence in sequences:
            for i in range(
                len(sequence) - 1
            ):
                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                counter[pair] += 1

        return counter

    @staticmethod
    def _merge_sequence(
        sequence: Sequence,
        pair: tuple[Token, Token],
        new_token: Token,
    ) -> Sequence:
        result = []
        i = 0

        while i < len(sequence):
            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(
                    sequence[i]
                )
                i += 1

        return result

    def _encode_sequence(
        self,
        sequence: Sequence,
    ) -> Sequence:
        for pair, new_token in self.merges.items():
            sequence = self._merge_sequence(
                sequence,
                pair,
                new_token,
            )

        return sequence

    @staticmethod
    def _df_to_sequences(
        df: pl.DataFrame,
        sid_columns: list[str],
    ) -> list[list[tuple[str, int]]]:
        sequences = []

        for row in df.select(
            sid_columns
        ).iter_rows():
            sequence = []

            for level, sid in enumerate(row):
                sequence.append(
                    (
                        f"L{level}",
                        int(sid),
                    )
                )

            sequences.append(sequence)

        return sequences
