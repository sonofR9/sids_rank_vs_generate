"""Single entry point for the full pipeline: SID training + evaluation + ranking training + candgen training, etc.

Performs the common pipeline parts: data preprocessing, calling abstract SID training,
common ranking training (user should be able to opt out of this stage), candgen training
(opt-out too), etc. Concrete SID training should be provided via a separate script.
Optionally supports Optuna.
"""
