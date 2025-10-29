"""Utilities that bridge :mod:`ZCPstatespy` with the external ``refquant`` package.

This module exposes a convenience wrapper that mirrors the manual steps that
users typically perform when preparing DIA-NN output for the ``refquant``
pipeline.  The wrapper first relies on
``ZCPstatespy.process_diann_and_fill_channels`` to create the intermediate
file and then forwards that file to ``refquant`` for the remaining processing
steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .process_diann_plex2 import process_diann_and_fill_channels


def _ensure_sequence(value: float | Sequence[float]) -> List[float]:
    """Normalize ``channel_q_cutoffs`` into a list of floats.

    Parameters
    ----------
    value:
        Either a single float cutoff or a sequence of floats.

    Returns
    -------
    list of float
        A list that contains all cutoff values.
    """

    if isinstance(value, (int, float)):
        normalized = [float(value)]
    else:
        normalized = [float(cutoff) for cutoff in value]

    if not normalized:
        raise ValueError("channel_q_cutoffs must contain at least one value")

    return normalized


def run_refquant_with_diann_processing(
    diann_input_file: str | Path,
    processed_output_path: str | Path,
    *,
    channel_q_cutoffs: Sequence[float] | float = (0.15,),
    run_prefix: Optional[str] = None,
    use_multiprocessing: bool = False,
    refquant_kwargs: Optional[dict] = None,
) -> Path:
    """Process a DIA-NN report and execute the ``refquant`` pipeline.

    This helper encapsulates the multi-step workflow that is demonstrated in
    the project README.  It first calls
    :func:`ZCPstatespy.process_diann_and_fill_channels` to post-process the
    ``report.tsv`` file.  Afterwards, it applies the provided Channel Q value
    cutoffs and triggers :func:`refquant.refquant_manager.run_refquant` on the
    filtered file.

    Parameters
    ----------
    diann_input_file:
        Path to the DIA-NN ``report.tsv`` file.
    processed_output_path:
        Destination path for the output of
        :func:`process_diann_and_fill_channels`.
    channel_q_cutoffs:
        Either a single cutoff or a sequence of cutoffs to be applied via
        ``refquant.utils.refquant_utils.write_shortened_diann_file_w_channel_lib_pg_cutoff``.
        The final file corresponding to the last cutoff will be fed to
        ``refquant``.
    run_prefix:
        Optional prefix forwarded to :func:`process_diann_and_fill_channels`.
    use_multiprocessing:
        Whether to enable multiprocessing in ``refquant``.
    refquant_kwargs:
        Extra keyword arguments propagated to
        :func:`refquant.refquant_manager.run_refquant`.

    Returns
    -------
    pathlib.Path
        The path to the final cutoff-filtered file that was provided to
        ``refquant``.

    Raises
    ------
    ImportError
        If the ``refquant`` package is not installed.
    """

    channel_q_cutoff_values = _ensure_sequence(channel_q_cutoffs)

    # Lazily import the refquant dependencies so the rest of the package
    # remains importable even when refquant is not installed.
    try:
        from refquant import refquant_manager
        from refquant.utils import refquant_utils
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "refquant is required for run_refquant_with_diann_processing. "
            "Install refquant before using this helper."
        ) from exc

    processed_output_path = Path(processed_output_path)
    diann_input_file = Path(diann_input_file)

    # Step 1: run the existing ZCPstatespy preprocessing helper.
    process_diann_and_fill_channels(
        input_path=str(diann_input_file),
        output_path=str(processed_output_path),
        channel_q_cutoff=channel_q_cutoff_values[0],
        run_prefix=run_prefix,
    )

    # Step 2: apply each requested Channel Q cutoff, keeping track of the
    # most recently produced file.
    filtered_file: Optional[Path] = None
    for cutoff in channel_q_cutoff_values:
        filtered_file = Path(
            refquant_utils.write_shortened_diann_file_w_channel_lib_pg_cutoff(
                str(processed_output_path), cutoff
            )
        )

    if filtered_file is None:
        # No cutoffs were provided; default to the processed file.
        filtered_file = processed_output_path

    # Step 3: trigger the refquant pipeline using the final filtered file.
    kwargs = dict(refquant_kwargs or {})
    kwargs.setdefault("use_multiprocessing", use_multiprocessing)
    if "diann_file_qvalfiltered" in kwargs:
        raise ValueError(
            "refquant_kwargs must not override diann_file_qvalfiltered; it is "
            "managed by run_refquant_with_diann_processing."
        )

    refquant_manager.run_refquant(
        diann_file_qvalfiltered=str(filtered_file),
        **kwargs,
    )

    return filtered_file


__all__ = ["run_refquant_with_diann_processing"]

