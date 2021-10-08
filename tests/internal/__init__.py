import gfootball.env


def simulate_agent_step(agent, num_steps=10):
    env = gfootball.env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    for i in range(num_steps):
        obs, _, _, _ = env.step([agent.action])
        agent.step(obs)