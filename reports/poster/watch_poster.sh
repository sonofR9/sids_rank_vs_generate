#!/usr/bin/env bash

set -u

poster_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
poster_source="${poster_dir}/poster.tex"

signature() {
    stat --format='%Y:%s' -- "${poster_source}" 2>/dev/null
}

last_signature="$(signature)" || exit 1
echo "[$(date --iso-8601=seconds)] Watching ${poster_source}"

while true; do
    sleep 1
    current_signature="$(signature)" || continue

    if [[ "${current_signature}" == "${last_signature}" ]]; then
        continue
    fi

    # Debounce edits: compile only after the file has stayed unchanged
    # for a complete 20-second interval.
    while true; do
        last_signature="${current_signature}"
        echo "[$(date --iso-8601=seconds)] Change detected; waiting 20 seconds"
        sleep 20
        current_signature="$(signature)" || continue

        if [[ "${current_signature}" == "${last_signature}" ]]; then
            break
        fi
    done

    echo "[$(date --iso-8601=seconds)] Rebuilding poster.pdf"
    if (
        cd -- "${poster_dir}" &&
        xelatex -interaction=nonstopmode -halt-on-error poster.tex &&
        xelatex -interaction=nonstopmode -halt-on-error poster.tex
    ); then
        echo "[$(date --iso-8601=seconds)] Build completed"
    else
        echo "[$(date --iso-8601=seconds)] Build failed"
    fi
done
