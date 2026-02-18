import json
import logging
from unittest.mock import AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Test Registry and Skills
print("--- Testing Registry & Skills ---")
from unittest.mock import MagicMock
from enton.skills.skill_registry import SkillRegistry

# Mock dependencies
mock_brain = MagicMock()
mock_brain.register_toolkit = MagicMock()
mock_brain.unregister_toolkit = MagicMock()
mock_bus = MagicMock()
mock_bus.emit = AsyncMock()

# Instantiate registry
registry = SkillRegistry(brain=mock_brain, bus=mock_bus)

print(f"Registry initialized. Loaded skills: {registry.list_skills()}")

# We can't easily assert specific tools without loading files, 
# but we can verify the registry instance is working.
assert isinstance(registry.loaded_skills, dict)
assert isinstance(registry.list_skills(), list)
print("Registry basic checks passed.")

# 2. Test Brain Integration
print("\n--- Testing Brain.think_agent ---")
from enton.cognition.brain import EntonBrain as Brain
from enton.core.config import Provider, Settings


async def test_brain_agent():
    settings = Settings()
    settings.brain_provider = Provider.LOCAL
    
    brain = Brain(settings, toolkits=[])
    # Mock the internal agent
    brain._agent = AsyncMock()
    
    # Mock response object from Agno
    mock_response = AsyncMock()
    mock_response.content = "São 10:00"
    brain._agent.arun.return_value = mock_response

    response = await brain.think_agent("Que horas são?", system="Sys")
    
    print(f"Final Agent Response: {response}")
    assert "São 10:00" in response
    assert brain._agent.arun.call_count == 1
    print("Brain agent test passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_brain_agent())
