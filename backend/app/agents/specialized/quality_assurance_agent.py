"""Quality Assurance Agent node for discrepancy detection."""
from typing import Dict, Any, List, TYPE_CHECKING
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ...domain.enriched_entities import EnrichedProvider, ValidationStatus
from ...infrastructure.models.grok_model import GrokModel
from ...infrastructure.logging import get_logger
from ..state import AgentState

if TYPE_CHECKING:
    from ..tools.nppes_tool import NppesTool


logger = get_logger(__name__)

async def quality_assurance_node(state: AgentState) -> Dict[str, Any]:
    """
    Node for assessing data quality and detecting discrepancies.
    """
    logger.info("quality_assurance_node_start")
    
    # Local imports
    from ..tools.nppes_tool import NppesTool

    provider_data = state.get("provider_data", {})
    if isinstance(provider_data, dict):
        provider = EnrichedProvider(**provider_data)
    else:
        provider = provider_data

    logger.info("assessing_quality", npi=provider.npi)

    # Tools and Model
    tools = [NppesTool()]
    model = GrokModel()
    
    # Agent
    agent = create_react_agent(model, tools)

    prompt = f"""Assess data quality for provider {provider.first_name} {provider.last_name} (NPI: {provider.npi}).

Provider data:
- Contact: {provider.phone}, {provider.email}
- Address: {provider.address_line1}, {provider.city}, {provider.state}
- Confidence scores: Overall={provider.overall_confidence}, Phone={provider.phone_confidence}, Email={provider.email_confidence}, Address={provider.address_confidence}

Tasks:
1. Use nppes_search to cross-reference with official registry
2. Identify any inconsistencies or discrepancies
3. Flag suspicious or potentially fraudulent information
4. Calculate quality metrics
5. Determine if manual review is required
6. Assign review priority (1-10, higher = more urgent)

Return assessment with:
- Discrepancies found
- Quality score
- Review priority
- Flags for manual review
"""

    # Run agent
    messages = [HumanMessage(content=prompt)]
    result = await agent.ainvoke({"messages": messages})
    
    last_message = result['messages'][-1]
    
    # Post-process logic from original code
    # Existing logic blindly checks confidence scores regardless of agent output.
    # We will preserve that logic as a fallback or enhancement.
    
    validated_discrepancies = []
    
    if provider.overall_confidence < 0.6:
        provider.requires_manual_review = True
        provider.review_priority = 8
        provider.validation_status = ValidationStatus.FLAGGED
        validated_discrepancies.append("Low confidence score detected")
    
    if provider.phone_confidence < 0.5 or provider.email_confidence < 0.5:
        validated_discrepancies.append("Contact information has low confidence")
        provider.requires_manual_review = True
        if provider.review_priority < 7:
            provider.review_priority = 7
    
    # Merge existing discrepancies with new ones
    if validated_discrepancies:
        # Check if attribute exists and is list
        if not hasattr(provider, 'discrepancies') or provider.discrepancies is None:
            provider.discrepancies = []
        provider.discrepancies.extend(validated_discrepancies)
            
    logger.info("quality_assessed", npi=provider.npi, priority=provider.review_priority)
    
    return {
        "provider_data": provider.model_dump(),
        "messages": [last_message]
    }


class QualityAssuranceAgent:
    """Agent for assessing data quality."""

    def __init__(self, model: GrokModel):
        from ..tools.nppes_tool import NppesTool
        
        self.model = model
        self.tools = [NppesTool()]
        self.agent = create_react_agent(self.model, self.tools)

    async def assess_quality(self, provider: EnrichedProvider) -> EnrichedProvider:
        """
        Assess quality of provider data.
        """
        logger.info("assessing_quality", npi=provider.npi)

        prompt = f"""Assess data quality for provider {provider.first_name} {provider.last_name} (NPI: {provider.npi}).

Provider data:
- Contact: {provider.phone}, {provider.email}
- Address: {provider.address_line1}, {provider.city}, {provider.state}
- Confidence scores: Overall={provider.overall_confidence}, Phone={provider.phone_confidence}, Email={provider.email_confidence}, Address={provider.address_confidence}

Tasks:
1. Use nppes_search to cross-reference with official registry
2. Identify any inconsistencies or discrepancies
3. Flag suspicious or potentially fraudulent information
4. Calculate quality metrics
5. Determine if manual review is required
6. Assign review priority (1-10, higher = more urgent)

Return assessment with:
- Discrepancies found
- Quality score
- Review priority
- Flags for manual review
"""

        # Run agent
        messages = [HumanMessage(content=prompt)]
        result = await self.agent.ainvoke({"messages": messages})
        
        last_message = result['messages'][-1]
        
        # Apply the same logic as the node function
        validated_discrepancies = []
        
        if provider.overall_confidence < 0.6:
            provider.requires_manual_review = True
            provider.review_priority = 8
            provider.validation_status = ValidationStatus.FLAGGED
            validated_discrepancies.append("Low confidence score detected")
        
        if provider.phone_confidence < 0.5 or provider.email_confidence < 0.5:
            validated_discrepancies.append("Contact information has low confidence")
            provider.requires_manual_review = True
            if provider.review_priority < 7:
                provider.review_priority = 7
        
        # Merge existing discrepancies with new ones
        if validated_discrepancies:
            if not hasattr(provider, 'discrepancies') or provider.discrepancies is None:
                provider.discrepancies = []
            provider.discrepancies.extend(validated_discrepancies)
            
        logger.info("quality_assessed", npi=provider.npi, priority=provider.review_priority)
        
        return provider
