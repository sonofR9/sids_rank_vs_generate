def resolve_codebook_sizes(
    codebook_sizes: tuple[int, ...] | None = None,
    num_levels: int | None = None,
    codebook_size: int | None = None,
) -> tuple[int, ...]:
    if codebook_sizes is None:
        if num_levels is None or codebook_size is None:
            raise ValueError(
                "Необходимо передать codebook_sizes или оба параметра "
                "num_levels и codebook_size"
            )
        return (codebook_size,) * num_levels

    resolved_sizes = tuple(codebook_sizes)
    if num_levels is not None and num_levels != len(resolved_sizes):
        raise ValueError(
            "Количество уровней должно совпадать с количеством codebook_sizes"
        )
    return resolved_sizes
