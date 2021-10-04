from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import gfootball.env as football_env
from api.internal.tamakeri import Environment as TamakeriEnv, load_model
from api.internal.tamakeri.env import to_tamakeri_obs

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

    tamakeri_env = TamakeriEnv()
    model_path = '/app/api/resources/models/1679.pth'
    model = load_model(tamakeri_env.net()(tamakeri_env), model_path)
    model.eval()

    prev_action = 0
    reserved_action = None
    while True:
        obs, reward, done, info = env.step([prev_action])
        if done:
            break
        print(obs[0]['steps_left'])

        info = [{'observation': to_tamakeri_obs(obs), 'action': [prev_action]}, None]
        tamakeri_env.play_info(info)
        x = tamakeri_env.observation(0)

        p, v, r, _ = model.inference(x, None)
        actions = tamakeri_env.legal_actions(0)

        ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
        action = ap_list[0][0]

        if reserved_action is not None:
            prev_action = reserved_action
            reserved_action = None
        else:
            # split action
            prev_action, reserved_action = tamakeri_env.special_to_actions(action)


if __name__ == '__main__':
    app.run(main)
