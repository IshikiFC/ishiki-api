from logging import getLogger

import pkg_resources
from pathlib import Path

from api.internal.utils.dump import read_dump

LOGGER = getLogger(__name__)


def _read_match(name):
    """
    read evaluation file and extract necessary information for API
    """

    fp = Path(pkg_resources.resource_filename('api', f'resources/matches/{name}.dump'))
    if not fp.exists():
        raise ValueError(f'match does not exists: {name}')
    LOGGER.info(f'read match from file: {fp}')

    try:
        dump = read_dump(fp)
        prev_frame = 0
        match = []
        for frame in dump:
            observation = dict()
            for obs_key in ['ball', 'left_team', 'right_team']:
                observation[obs_key] = frame['observation'][obs_key].tolist()
            observation['action'] = str(frame['debug']['action'][0])
            evaluation = frame['debug']['evaluation']

            cur_frame = frame['debug']['frame']
            for _ in range(cur_frame - prev_frame):  # duplicate frames to match with movie
                match.append({
                    'observation': observation,
                    'evaluation': evaluation
                })
            prev_frame = cur_frame

    except KeyError as e:
        raise ValueError('match file does not have necessary fields') from e

    return match


def _get_match(name, cache=True):
    global _matches
    if not cache or name not in _matches:
        match = _read_match(name)
        _matches[name] = match
    return _matches[name]


_matches = dict()  # cache matches


def get_match(name=None, num_steps=-1, cache=True):
    name = name or 'tamakeri_hard'
    match = _get_match(name=name, cache=cache)
    return match[:num_steps] if num_steps > 0 else match
