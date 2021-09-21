from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import gfootball.env as football_env

FLAGS = flags.FLAGS
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')


def main(_):
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        number_of_left_players_agent_controls=0,
        number_of_right_players_agent_controls=0,
        extra_players=[
            'ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/app/api/models/11_vs_11_easy_stochastic_v2',
        ],
        logdir='/app/logs',
        write_full_episode_dumps=True,
        write_video=True,
        render=True
    )

    while True:
        _, _, done, _ = env.step([])
        if done:
            break


if __name__ == '__main__':
    app.run(main)
