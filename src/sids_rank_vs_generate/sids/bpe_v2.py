from collections import Counter
from typing import Hashable

import polars as pl


Token = Hashable
Sequence = list[Token]


class BPE:
    def __init__(
        self,
        num_merges: int = 1000,
        min_frequency: int = 2,
    ):
        self.num_merges = num_merges
        self.min_frequency = min_frequency

        self.merges: dict[tuple[Token, Token], Token] = {}
        self.vocabulary: dict[Token, tuple[Token, Token]] = {}

        self._base_token_ids: dict[Token, int] = {}
        self._base_token_count = 0
        self._next_token_id = 0
        self._fitted = False

    def fit(self, sequences) -> "BPE":
        sequences = [list(seq) for seq in sequences]

        if not sequences:
            raise ValueError("BPE: передан пустой корпус")

        sequences = [seq for seq in sequences if seq]

        if not sequences:
            raise ValueError("BPE: все последовательности пустые")

        unique_tokens = set()

        for sequence in sequences:
            unique_tokens.update(sequence)

        self._base_token_ids = {
            token: i
            for i, token in enumerate(sorted(unique_tokens))
        }

        self._base_token_count = len(self._base_token_ids)
        self.vocab_size = self._base_token_count + self.num_merges


        self._next_token_id = 0

        self.merges = {}
        self.vocabulary = {}

        max_merges = self.num_merges

        for _ in range(max_merges):
            pair_counts = self._count_pairs(sequences)

            if not pair_counts:
                break

            pair, frequency = pair_counts.most_common(1)[0]

            if frequency < self.min_frequency:
                break

            new_token = ("bpe", self._next_token_id)
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

        self._fitted = True

        return self

    def transform(self, sequences):
        if not self._fitted:
            raise RuntimeError("Сначала вызови BPE.fit()")

        return [
            self._encode_sequence(list(sequence))
            for sequence in sequences
        ]

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)

    def fit_polars(
        self,
        df: pl.DataFrame,
        sid_columns: list[str],
    ) -> "BPE":
        sequences = self._df_to_sequences(
            df,
            sid_columns,
        )

        self.fit(sequences)

        return self

    def transform_polars(
        self,
        df: pl.DataFrame,
        sid_columns: list[str],
    ) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("Сначала вызови BPE.fit()")

        sequences = self._df_to_sequences(
            df,
            sid_columns,
        )

        encoded = self.transform(sequences)

        encoded_ids = [
            [
                self._token_to_id(token)
                for token in sequence
            ]
            for sequence in encoded
        ]

        max_length = max(
            len(sequence)
            for sequence in encoded_ids
        )

        data = {
            "item_id": df["item_id"],
        }

        for i in range(max_length):
            data[f"sid_{i}"] = [
                sequence[i] if i < len(sequence) else None
                for sequence in encoded_ids
            ]

        return pl.DataFrame(data)

    def _token_to_id(self, token: Token) -> int:
        if token[0] == "bpe":
            token_id = self._base_token_count + token[1]
        else:
            token_id = self._base_token_ids[token]

        if not 0 <= token_id < self.vocab_size:
            raise ValueError(
                f"BPE produced token id {token_id}, "
                f"but valid range is [0, {self.vocab_size})"
            )

        return token_id

    @staticmethod
    def _count_pairs(
        sequences: list[Sequence],
    ) -> Counter:
        counter = Counter()

        for sequence in sequences:
            for i in range(len(sequence) - 1):
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
                result.append(sequence[i])
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

        for row in df.select(sid_columns).iter_rows():
            sequence = []

            for level, sid in enumerate(row):
                sequence.append(
                    (f"L{level}", int(sid))
                )

            sequences.append(sequence)

        return sequences