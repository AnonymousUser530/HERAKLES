import copy

from crafter_text_gc.gc_envs.env_trs_uni import Env, ENC_ORDER, DUMMY_BITS
from crafter_text_gc.dicts_action_str_to_int import DICTIONARY_STR_TO_ID
from crafter_text_gc.crafter import constants
import numpy as np
import itertools

import gym
DiscreteSpace = gym.spaces.Discrete
BoxSpace = gym.spaces.Box
DictSpace = gym.spaces.Dict
TextSpace = gym.spaces.Text
BaseClass = gym.Env

id_to_item = [0]*20
dummyenv = Env()
for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
    name = str(name)[str(name).find('objects.')+len('objects.'):-2].lower() if 'objects.' in str(name) else str(name)
    id_to_item[ind] = name
player_idx = id_to_item.index('player')
del dummyenv

#vitals = ["health","food","drink","energy",]
vitals = []

rot = np.array([[0,-1],[1,0]])
directions = ['front', 'right', 'back', 'left']

REF = np.array([0, 1])

def rotation_matrix(v1, v2):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    rotation_matrix = np.array([[dot, -cross],[cross, dot]])
    return rotation_matrix

def describe_loc(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    return "-".join(desc)


def describe_env(info):
    assert(info['semantic'][info['player_pos'][0],info['player_pos'][1]] == player_idx)
    semantic = info['semantic'][info['player_pos'][0]-info['view'][0]//2:info['player_pos'][0]+info['view'][0]//2+1, info['player_pos'][1]-info['view'][1]//2+1:info['player_pos'][1]+info['view'][1]//2]
    center = np.array([info['view'][0]//2,info['view'][1]//2-1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x,y)
    loc = np.stack((y1, x1),axis=-1)
    dist = np.absolute(center-loc).sum(axis=-1)
    obj_info_list = []
    
    facing = info['player_facing']
    target = (center[0] + facing[0], center[1] + facing[1])
    
    if semantic.shape[0] > target[0] and semantic.shape[1] > target[1]:
        target = id_to_item[semantic[target]]
    else:
        target = "void"
    obs = "You face {} at your front.".format(target)
    
    for idx in np.unique(semantic):
        if idx==player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic==idx, dist, np.inf)), semantic.shape)
        obj_info_list.append((id_to_item[idx], dist[smallest], describe_loc(np.array([0,0]), smallest-center)))

    if len(obj_info_list)>0:
        status_str = "You see:"
        for name, dist, loc in obj_info_list:
            if dist>1:
                status_str += "\n- {} {} steps to your {}".format(name, dist, loc)
            else:
                status_str += "\n- {} {} step to your {}".format(name, dist, loc)
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()
    
    return result.strip()

def describe_inventory(info, long_description=False):
    result = ""

    #status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info['inventory'][v]) for v in vitals]))
    #result += status_str + "\n\n"

    inventory_str = "\n".join(["- {}: {}".format(i.replace("_", " "), num) for i,num in info['inventory'].items() if i not in vitals and num!=0])
    inventory_str = "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str #+ "\n\n"

    if long_description and len(info["objects_placed_by_player"])>0:
        result += "\n\n"
        for k, v in info["objects_placed_by_player"].items():
            for pos_v in v:
                result += "You placed {} at ({},{})\n".format(k, pos_v[0], pos_v[1])
        result = result[:-1]
    return result.strip()


def describe_act(info, long_description=False):

    if long_description:
        result = "Elementary actions you can take:\n"
    else:
        result = "Actions you can take:\n"
    for a in info['possible_actions']:
        if long_description:
            if a in info['possible_elementary_actions']:
                if "place" in a or "build" in a or "put" in a:
                    if "place" in a:
                        placed_object = a.split("place ")[1]
                    elif "build" in a:
                        placed_object = a.split("build ")[1]
                    elif "put" in a:
                        placed_object = a.split("put ")[1]
                    if placed_object == "crafting table":
                        placed_object = "table"
                    requirement = ""
                    for k,v in constants.place[placed_object]["uses"].items():
                        if v>1:
                            requirement += "{} {}s, ".format(v,k)
                        else:
                            requirement += "{} {}, ".format(v,k)
                    result += "- {} (require {} in your inventory)\n".format(a, requirement[:-2])
                elif "make" in a or "craft" in a:
                    if "make" in a:
                        made_object = a.split("make ")[1].replace(" ","_")
                    elif "craft" in a:
                        made_object = a.split("craft ")[1].replace(" ","_")
                    requirement = ""
                    for k,v in constants.make[made_object]["uses"].items():
                        if v>1:
                            requirement += "{} {}s, ".format(v,k)
                        else:
                            requirement += "{} {}, ".format(v,k)
                    result += "- {} (require {} in your inventory while facing a {})\n".format(a, requirement[:-2], constants.make[made_object]["nearby"][0])
                elif ("chop" in a) or ("mine" in a) or ("extract" in a):
                    collected_object = a.split(" ")[1]
                    requirement = ""
                    if len(constants.collect[collected_object]["require"])==0:
                        result += "- {} (require facing {})\n".format(a, collected_object)
                    else:
                        for k,v in constants.collect[collected_object]["require"].items():
                            if v>1:
                                requirement += "{} {}s, ".format(v,k.replace("_", " "))
                            else:
                                requirement += "{} {}, ".format(v,k.replace("_", " "))
                        result += "- {} (require {} in your inventory while facing {})\n".format(a, requirement[:-2], collected_object)
                else:
                    result += "- {}\n".format(a)
        else:
            result += "- {}\n".format(a)
    if long_description and len(info['possible_actions']) > len(info['possible_elementary_actions']):
        # there is at least one low level policy that can be called by HL policy
        if (len(info['possible_actions']) -len(info['possible_elementary_actions']))==1:
            result += "\nLow-level policy you can call:\n"
        else:
            result += "\nLow-level policies you can call:\n"
        for a in info['possible_actions']:
            if a not in info['possible_elementary_actions']:
                result += "- {}\n".format(a)
    if 'last_action' in info.keys():
        result += "\nThe last action you took: {}".format(info['last_action'])
    
    return result.strip()

    
def describe_task(info, long_description=False):
    if long_description:
        task_description = "You are playing a Minecraft like game. You can use elementary actions or, if available, more efficient low-level policies.\n"
    else:
        task_description = ""
    task_description += "Your task: {}".format(info['goal'])
    if "hl_traj_len_max" in info.keys() and info['hl_traj_len_max'] is not None:
        task_description += " in {} steps".format(info['hl_traj_len_max'])
    if "nbr_steps" in info.keys() and info['nbr_steps'] is not None:
        if info['nbr_steps']>1:
            task_description += "\n\nYou have already done {} steps.".format(int(info['nbr_steps']))
        else:
            task_description += "\n\nYou have already done {} step.".format(int(info['nbr_steps']))
    return task_description

def describe_task_proba_subgoal_estimator(long_description=False):
    if long_description:
        task_description = "You are playing a Minecraft like game.\n"
    else:
        task_description = ""
    task_description += "Your task is to evaluate your success rate for the goal: {}"

    return task_description

def describe_task_proba_goal_sampler(long_description=False):
    if long_description:
        task_description = "You are playing a Minecraft like game.\n"
    else:
        task_description = ""
    # task_description += "Your have to decide what goal you should achieve taking into account the environment and your inventory."
    task_description += "Your task is to evaluate your success rate for the goal: {}"

    return task_description

def describe_coordinates(info, long_description=False):
    return "Your coordinates: ({},{})".format(info['player_pos'][0], info['player_pos'][1])

    
def describe_frame(info, long_description=False):
    #try:
    result = describe_task(info, long_description)
    result += "\n\n"
    result += describe_coordinates(info)
    result += "\n\n"
    result += describe_env(info)
    result += "\n\n"
    result += describe_inventory(info, long_description)
    result += "\n\n"
    result += describe_act(info, long_description)
    result += "\n\nYour action: "
    
    return result

def describe_frame_proba_subgoal_estimator(info, long_description=False):
    result = describe_task_proba_subgoal_estimator(long_description)
    result += "\n\n"
    result += describe_coordinates(info)
    result += "\n\n"
    result += describe_env(info)
    result += "\n\n"
    result += describe_inventory(info, long_description)
    result += "\n\nYour success rate is: "

    return result

def describe_frame_for_goal_sampler(info, long_description=False):
    result = describe_task_proba_goal_sampler(long_description)
    result += "\n\n"
    result += describe_coordinates(info)
    result += "\n\n"
    result += describe_env(info)
    result += "\n\n"
    result += describe_inventory(info, long_description)
    result += "\n\nYour success rate is: "

    return result

class CrafterTextGC(Env):

    default_iter = 10
    default_steps = 10000

    def __init__(self, area=(64, 64), view=(9, 9), size=(64, 64), reward=True, length=100, train=True, seed=None,
                 hl_traj_len_max=None, long_description=False, reset_word_only_at_episode_termination=False):

        super().__init__(area, view, size, reward, length, seed, train=train,
                         reset_word_only_at_episode_termination=reset_word_only_at_episode_termination)
        self.hl_traj_len_max=hl_traj_len_max
        self.nbr_steps_hl = 0
        self.long_description = long_description

    @property
    def observation_space(self):
        img_shape = (self._size[1], self._size[0], 3)
        return DictSpace({
          'image': BoxSpace(0, 255, img_shape, np.uint8),
          'task_enc': BoxSpace(0, 1, (len(ENC_ORDER) + DUMMY_BITS, ), np.uint8),
          'textual_obs': TextSpace(max_length=10, charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")})


    def add_possible_actions(self, possible_actions:list):
        self.possible_actions = copy.deepcopy(possible_actions)

    def add_possible_elementary_actions(self, possible_elementary_actions:list):
        self.possible_elementary_actions = copy.deepcopy(possible_elementary_actions)

    def update_action_space(self, proba_subgoal):
        action_space =copy.deepcopy(self.possible_elementary_actions)
        for sg in proba_subgoal.keys():
            if proba_subgoal[sg] >= np.random.rand():
                parts = sg.split('_')
                name_sg = parts[0]
                for p in parts[1:]:
                    name_sg += ' '+p
                action_space.append(name_sg)
        self.add_possible_actions(action_space)

    def reset(self, goal=None):
        obs, env_state = super().reset(goal)

        env_state.update({
            'inventory': self._player.inventory.copy(),
            'possible_actions': self.possible_actions,
            'possible_elementary_actions': self.possible_elementary_actions,
            'semantic': self._sem_view(),
            'achievements':self._player.achievements,
            'hl_traj_len_max':self.hl_traj_len_max,
            'nbr_steps':self.nbr_steps_hl,
            'player_pos': self._player.pos,
            'objects_placed_by_player': self._player.placed_objects,
            'player_facing': self._player.facing,
            'view': self._view,
        })

        description_subgaol_estimator = describe_frame_proba_subgoal_estimator(env_state, self.long_description)
        env_state.update({'description_proba_subgoal_estimator': description_subgaol_estimator})
        description_goal_sampler = describe_frame_for_goal_sampler(env_state, self.long_description)
        env_state.update({'description_proba_goal_estimator': description_goal_sampler})

        return  {"image": obs["image"], "task_enc": obs["task_enc"],
                "textual_obs": describe_frame(env_state, self.long_description)}, env_state
    
    def step(self, action):
        a = DICTIONARY_STR_TO_ID[action]
        obs, reward, done, _, env_state = super().step(a)
        
        env_state.update({
            'goal': env_state['goal'],
            'possible_actions': self.possible_actions,
            'possible_elementary_actions': self.possible_elementary_actions,
            'objects_placed_by_player': self._player.placed_objects,
            'last_action': action,
            'hl_traj_len_max':self.hl_traj_len_max,
            'nbr_steps':self.nbr_steps_hl
        })

        description_subgaol_estimator = describe_frame_proba_subgoal_estimator(env_state, self.long_description)
        env_state.update({'description_proba_subgoal_estimator': description_subgaol_estimator})
        description_goal_sampler = describe_frame_for_goal_sampler(env_state, self.long_description)
        env_state.update({'description_proba_goal_estimator': description_goal_sampler})

        return {"image": obs["image"], "task_enc": obs["task_enc"],
                "textual_obs": describe_frame(env_state, self.long_description)}, reward, done, False, env_state