from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import gfootball.env
from api.internal.grf import GrfAgent
from api.internal.tamakeri import TamakeriAgent

FLAGS = flags.FLAGS
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')
flags.DEFINE_enum('agent', 'grf', ['grf', 'tamakeri'], 'Agent to play')
flags.DEFINE_enum('level', 'easy', ['easy', 'hard'], 'Opponent level')


def main(_):
    logging.info(f'simulate match: agent={FLAGS.agent}, level={FLAGS.level}')
    env = gfootball.env.create_environment(
        env_name=f'11_vs_11_{FLAGS.level}_stochastic',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        logdir='/app/logs/tmp',
        write_full_episode_dumps=True,
        write_video=True,
        render=True
    )
    if FLAGS.agent == 'grf':
        agent = GrfAgent()
    elif FLAGS.agent == 'tamakeri':
        agent = TamakeriAgent()
    else:
        raise ValueError(f'invalid agent: {FLAGS.agent}')

    while True:
        obs, _, done, _ = env.step([agent.get_action()])
        if done:
            break
        agent.step(obs)


if __name__ == '__main__':
    app.run(main)
