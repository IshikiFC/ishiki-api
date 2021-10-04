import gfootball.env as football_env
from api.internal.grf import GrfAgent


def test_step():
    agent = GrfAgent()
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    for i in range(10):
        obs, _, _, _ = env.step([agent.get_action()])
        agent.step(obs)
