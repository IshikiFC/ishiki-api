import pkg_resources

from api.evaluate.grf import evaluate_internal
from gfootball.env import script_helpers, observation_preprocessing
from gfootball.env.players.ppo2_cnn import ObservationStacker


def _read_match(fp):
    dump_raw = script_helpers.ScriptHelpers().load_dump(fp)
    observation_stacker = ObservationStacker(4)

    match = []
    for frame_raw in dump_raw[:300]:
        observation = _convert_observation(frame_raw['observation'])
        smm = observation_preprocessing.generate_smm([observation])
        smm_stacked = observation_stacker.get(smm)

        frame = dict()
        frame.update(observation)
        frame.update(evaluate_internal(smm_stacked))
        match.append(frame)

    return match


def _convert_observation(observation_raw):
    return {
        'ball': observation_raw['ball'].tolist(),
        'left_team': observation_raw['left_team'].tolist(),
        'right_team': observation_raw['right_team'].tolist(),
        'active': int(observation_raw['right_team_designated_player'])
    }


match = _read_match(
    pkg_resources.resource_filename('api', 'resources/matches/episode_done_20210918-064519149416.dump')
)


def get_match(num_steps=-1):
    return match[:num_steps] if num_steps > 0 else match
