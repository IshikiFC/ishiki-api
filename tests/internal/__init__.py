import gfootball.env


def simulate_agent_step(agent):
    env = gfootball.env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    for i in range(10):
        action = agent.get_action()
        obs, _, _, _ = env.step([action])
        agent.step(obs)