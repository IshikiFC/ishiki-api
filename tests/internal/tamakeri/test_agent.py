import gfootball.env as football_env
from api.internal.tamakeri.agent import to_action_name, TamakeriAgent


def test_to_action_name():
    assert 'Idle' == to_action_name(0)
    assert 'Left' == to_action_name(1)
    assert 'None' == to_action_name(19)
    assert 'LongPassLeft' == to_action_name(20)
    assert 'LongPassTopLeft' == to_action_name(21)
    assert 'ShotBottomLeft' == to_action_name(51)


def test_step():
    agent = TamakeriAgent()
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    for i in range(10):
        obs, _, _, _ = env.step([agent.get_action()])
        agent.step(obs)
