"""Dependency-free numeric kernels shared by every layer above."""

from min_gp.numeric.ranking import cross_section_rank
from min_gp.numeric.preprocessing import remove_outliers

__all__ = ["cross_section_rank", "remove_outliers"]
