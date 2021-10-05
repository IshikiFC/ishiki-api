from api.internal.grf import GrfAgent
from tests.internal import simulate_agent_step


def test_step():
    agent = GrfAgent()
    simulate_agent_step(agent)
