import pytest

from api.internal.grf import GrfAgent
from tests.internal import simulate_agent_step


@pytest.fixture(scope='class')
def agent() -> GrfAgent:
    agent = GrfAgent()
    simulate_agent_step(agent)
    return agent


class TestGrfAgent:
    def test_get_action_probs(self, agent: GrfAgent):
        action_probs = agent.get_action_probs(to_name=False)
        assert len(action_probs) == 19
        assert sum(action_probs.values()) == pytest.approx(1.0)

        action_probs_list = agent.get_action_probs(to_name=True, to_list=True)
        assert len(action_probs) == 19
        assert sum([e[1] for e in action_probs_list]) == pytest.approx(1.0)
        assert action_probs_list[0][0].lower() == 'idle'
