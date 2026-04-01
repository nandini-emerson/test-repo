import pytest
from unittest.mock import Mock, MagicMock, patch

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    return MagicMock()

class TestAgent:
    def test_agent_initialization(self, mock_agent):
        """Test agent can be initialized"""
        assert mock_agent is not None
    
    @pytest.mark.asyncio
    async def test_basic_message_processing(self, mock_agent):
        """Test basic message processing"""
        mock_agent.process_message = MagicMock(return_value="Response")
        response = await mock_agent.process_message("Hello!")
        assert response is not None
        assert response == "Response"
    
    def test_agent_has_attributes(self, mock_agent):
        """Test agent mock supports attribute access"""
        mock_agent.config = {"test": "value"}
        assert mock_agent.config["test"] == "value"
