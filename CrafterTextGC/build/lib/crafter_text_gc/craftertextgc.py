from crafter_text_gc.gc_envs.env_trs_uni import Env
import numpy as np
import itertools

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

def describe_inventory(info):
    result = ""
    
    #status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info['inventory'][v]) for v in vitals]))
    #result += status_str + "\n\n"
    
    inventory_str = "\n".join(["- {}: {}".format(i, num) for i,num in info['inventory'].items() if i not in vitals and num!=0])
    inventory_str = "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str #+ "\n\n"
    
    return result.strip()


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
        status_str = "You see:\n{}".format("\n".join(["- {} {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()
    
    return result.strip()


def describe_act(info):

    result = "Actions you can take:\n"
    for a in info['possible_actions']:
        result += "- {}\n".format(a)
    
    result += "\nThe last action you took: {}".format(info['last_action'])
    
    return result.strip()

    
def describe_task(info):
    return "Your task: {}".format(info['goal'])

def describe_coordinates(info):
    return "Your coordinates: ({},{})".format(info['player_pos'][0], info['player_pos'][1])

    
def describe_frame(info):
    #try:
    result = describe_task(info)
    result += "\n\n"
    result += describe_coordinates(info)
    result += "\n\n"
    result += describe_env(info)
    result += "\n\n"
    result += describe_inventory(info)
    result += "\n\n"
    result += describe_act(info)
    result += "\n\nYour action: "
    
    return result

class CrafterTextGC(Env):

    default_iter = 10
    default_steps = 10000

    def __init__(self, area=(64, 64), view=(9, 9), size=(64, 64), reward=True, length=1500, train=True, seed=None):
        self.sentence_to_action = {
            'Noop': 0,
            'Move west': 1,
            'Move east': 2,
            'Move north': 3,
            'Move south': 4,
            'Do': 5,
            'Place stone': 6,
            'Place table': 7,
            'Place furnace': 8,
            'Make wood pickaxe': 9,
            'Make stone pickaxe': 10,
            'Make iron pickaxe': 11,
            'Make wood sword': 12,
            'Make stone sword': 13,
            'Make iron sword': 14
        }
        super().__init__(area, view, size, reward, length, seed, train=train)

    def reset(self, goal=None):
        super().reset(goal)
        _, _, _, _, env_state = self.step('Noop')
        info = {
            'possible_actions': list(self.sentence_to_action.keys()),
            'goal': goal,
        }
        return describe_frame(env_state), info
    
    def step(self, action):
        a = self.sentence_to_action[action]
        obs, reward, done, _, env_state = super().step(a)
        
        env_state.update({
            'goal': env_state['goal'],
            'possible_actions': list(self.sentence_to_action.keys()),
            'last_action': action
        })
                
        
        return describe_frame(env_state), reward, done, False, env_state