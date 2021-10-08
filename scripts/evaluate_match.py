from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

from api.internal.grf import GrfAgent
from api.internal.tamakeri import TamakeriAgent
from api.internal.utils.dump import convert_observation, read_dump, write_dump

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'Dump file to read')
flags.DEFINE_string('output', None, 'Dump file to write')


def main(_):
    logging.info(f'evaluate {FLAGS.input}')
    dump = read_dump(FLAGS.input)
    logging.info(f'load {len(dump)} frames')

    grf_agent = GrfAgent()
    tamakeri_agent = TamakeriAgent()

    for frame in tqdm(dump):
        obs = convert_observation(frame['observation'])
        grf_agent.step(obs)
        tamakeri_agent.step(obs)

        evaluation = dict()
        evaluation['grf'] = {
            'action': grf_agent.get_action_probs(to_name=True, to_list=True),
            'value': grf_agent.get_value()
        }
        evaluation['tamakeri'] = {
            'action': tamakeri_agent.get_action_probs(to_name=True, to_base=True, to_list=True),
            'value': tamakeri_agent.get_value()
        }

        frame['debug']['evaluation'] = evaluation

    write_dump(dump, FLAGS.output)
    logging.info(f'wrote evaluation to {FLAGS.output}')


if __name__ == '__main__':
    app.run(main)
