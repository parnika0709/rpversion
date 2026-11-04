"""Public API for the risk-mapping module."""

from braintumor.risk.mapping import RiskRegion, compute_risk_regions
from braintumor.risk.report import build_report_text

__all__ = ["RiskRegion", "compute_risk_regions", "build_report_text"]
