"""Utility estimands; higher metric values are assumed to be preferable."""
from __future__ import annotations


def utility_estimands(m_ri: float, m_si: float, m_re: float, m_se: float) -> dict[str, float]:
    iul = m_ri - m_si
    eul = m_re - m_se
    return {"IUL": iul, "EUL": eul, "ITL": eul - iul}


def ap_skill(average_precision: float, prevalence: float) -> float:
    return (average_precision - prevalence) / (1.0 - prevalence)


def brier_skill(brier_score: float, prevalence: float) -> float:
    return 1.0 - brier_score / (prevalence * (1.0 - prevalence))


def log_loss_skill(log_loss: float, prevalence: float) -> float:
    import math
    null = -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence))
    return 1.0 - log_loss / null
