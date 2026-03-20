"""Directory Management logic for reports and alerts."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ...domain.enriched_entities import ValidationReport, ValidationBatch, EnrichedProvider
from ...infrastructure.models.grok_model import GrokModel
from ...infrastructure.logging import get_logger


logger = get_logger(__name__)

# This module doesn't strictly need to be a Graph Node if it acts on batches,
# but we can expose functions that can be called by a Supervisor if needed.
# For now, we update it to remove smolagents dependency.

class DirectoryManagementService:
    """Service for directory management, reporting, and alerts."""
    
    def __init__(self):
        # No model needed for deterministic logic
        pass
    
    async def generate_report(self, batch: ValidationBatch, providers: List[EnrichedProvider]) -> ValidationReport:
        """Generate validation report."""
        logger.info("generating_report", batch_id=batch.batch_id)
        
        # Calculate metrics
        validated_count = sum(1 for p in providers if p.validation_status.value == "validated")
        discrepancy_count = sum(1 for p in providers if p.discrepancies)
        review_count = sum(1 for p in providers if p.requires_manual_review)
        
        # Prioritize review list
        prioritized = sorted(
            [p for p in providers if p.requires_manual_review],
            key=lambda x: x.review_priority,
            reverse=True
        )[:50]  # Top 50
        
        # Generate recommendations
        recommendations = []
        if discrepancy_count > batch.total_providers * 0.2:
            recommendations.append("High discrepancy rate detected - review data sources")
        if review_count > batch.total_providers * 0.3:
            recommendations.append("Large number of providers require manual review - consider adjusting confidence thresholds")
        
        report = ValidationReport(
            report_id=str(uuid.uuid4()),
            batch_id=batch.batch_id,
            generated_at=datetime.utcnow(),
            summary={
                "total_providers": batch.total_providers,
                "validated": validated_count,
                "discrepancies": discrepancy_count,
                "requires_review": review_count,
                "average_confidence": sum(p.overall_confidence for p in providers) / len(providers) if providers else 0.0,
            },
            providers_validated=validated_count,
            providers_with_discrepancies=discrepancy_count,
            providers_requiring_review=review_count,
            prioritized_review_list=prioritized,
            recommendations=recommendations,
        )
        
        logger.info("report_generated", report_id=report.report_id)
        return report
    
    async def create_alert(self, provider: EnrichedProvider, alert_type: str) -> Dict[str, Any]:
        """Create alert for provider requiring attention."""
        return {
            "alert_id": str(uuid.uuid4()),
            "provider_npi": provider.npi,
            "provider_name": f"{provider.first_name} {provider.last_name}",
            "alert_type": alert_type,
            "priority": provider.review_priority,
            "message": f"Provider {provider.npi} requires {alert_type}",
            "created_at": datetime.utcnow().isoformat(),
        }


class DirectoryManagementAgent:
    """Agent for directory management (mostly a wrapper for the service)."""

    def __init__(self, model: GrokModel):
        self.model = model
        self.service = DirectoryManagementService()

    # The orchestrator doesn't currently call methods on this agent in the workflow provided.
    # It just instantiates it. If it were to call something, we'd add it here.
    # For now, we basically provide a placeholder or proxy to the service.

    async def generate_report(self, batch: ValidationBatch, providers: List[EnrichedProvider]) -> ValidationReport:
        return await self.service.generate_report(batch, providers)

    async def create_alert(self, provider: EnrichedProvider, alert_type: str) -> Dict[str, Any]:
        return await self.service.create_alert(provider, alert_type)
