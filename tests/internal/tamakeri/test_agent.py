from api.internal.tamakeri.agent import to_action_name, TamakeriAgent
from tests.internal import simulate_agent_step


def test_to_action_name():
    assert 'Idle' == to_action_name(0)
    assert 'Left' == to_action_name(1)
    assert 'None' == to_action_name(19)
    assert 'LongPassLeft' == to_action_name(20)
    assert 'LongPassTopLeft' == to_action_name(21)
    assert 'ShotBottomLeft' == to_action_name(51)


def test_step():
    agent = TamakeriAgent()
    simulate_agent_step(agent)
