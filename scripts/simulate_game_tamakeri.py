from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import gfootball.env as football_env
from api.internal.tamakeri import TamakeriAgent

FLAGS = flags.FLAGS
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')


def main(_):
    env = football_env.create_environment(
        env_name='11_vs_11_hard_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        logdir='/app/logs',
        write_full_episode_dumps=True,
        write_video=True,
        render=True
    )
    agent = TamakeriAgent()

    while True:
        obs, reward, done, info = env.step([agent.get_action()])
        if done:
            break
        agent.step(obs)


if __name__ == '__main__':
    app.run(main)
