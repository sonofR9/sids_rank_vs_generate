from __future__ import annotations

SEED = 12
SCALE = "50m"
MAX_USERS_FOR_ANALYSIS = 2000

MAX_REACTION_DELAY_HOURS = 6
RECENT_HISTORY_SIZE = 80
EXPLICIT_HISTORY_SIZE = 20

VALID_EVENT_TYPES = {
    "listen",
    "like",
    "dislike",
    "unlike",
    "undislike",
}

EXPLICIT_EVENT_TYPES = {
    "like",
    "dislike",
    "unlike",
    "undislike",
}

UNK_USER_IDX = 0

PAD_ITEM_IDX = 0
UNK_ITEM_IDX = 1

PAD_EVENT_IDX = 0
UNK_EVENT_IDX = 1

EVENT2IDX = {
    "listen": 2,
    "like": 3,
    "dislike": 4,
    "unlike": 5,
    "undislike": 6,
}

TRACK_LENGTH_CLIP_SECONDS = 30 * 60
TIME_DELTA_CLIP_SECONDS = 180 * 24 * 60 * 60

BATCH_SIZE = 256
NUM_WORKERS = 0
