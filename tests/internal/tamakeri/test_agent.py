import pytest

from api.internal.tamakeri.agent import to_action_name, TamakeriAgent
from tests.internal import simulate_agent_step


def test_to_action_name():
    assert to_action_name(0) == 'idle'
    assert to_action_name(1) == 'left'
    assert to_action_name(19) == 'idle'
    assert to_action_name(20) == 'long_pass_left'
    assert to_action_name(21) == 'long_pass_top_left'
    assert to_action_name(51) == 'shot_bottom_left'


@pytest.fixture(scope='class')
def agent() -> TamakeriAgent:
    agent = TamakeriAgent()
    simulate_agent_step(agent)
    return agent


class TestTamakeriAgent:
    def test_get_action_probs(self, agent: TamakeriAgent):
        action_probs = agent.get_action_probs(to_name=False, to_base=False)
        assert len(action_probs) == 52
        assert sum(action_probs.values()) == pytest.approx(1.0)

        action_probs_base = agent.get_action_probs(to_name=False, to_base=True)
        assert len(action_probs_base) == 19
        assert sum(action_probs_base.values()) == pytest.approx(1.0)

        action_probs_list = agent.get_action_probs(to_name=True, to_base=True, to_list=True)
        assert len(action_probs_list) == 19
        assert sum([e[1] for e in action_probs_list]) == pytest.approx(1.0)
        assert action_probs_list[0][0].lower() == 'idle'
