import gym
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from crafter_text_gc.crafter import constants
from crafter_text_gc.crafter import objects
from crafter_text_gc.crafter import worldgen
from crafter_text_gc.crafter import env
from crafter_text_gc.gc_envs.env_utils import get_synonym_tasks, SYNONYMS, get_repeat_tasks


DiscreteSpace = gym.spaces.Discrete
BoxSpace = gym.spaces.Box
DictSpace = gym.spaces.Dict
BaseClass = gym.Env

OBJS = ['table', 'furnace']
MATS = ['wood', 'stone', 'coal', 'iron', 'diamond', 'drink']
TOOLS = ['sword', 'pickaxe']
INSTRS = ['collect', 'make', 'place', *np.unique(np.array(list(SYNONYMS.values())).flatten())]
COUNT = [str(i) for i in range(2, 11)]
ENC_ORDER = INSTRS + OBJS + MATS + TOOLS + COUNT
DUMMY_BITS = 10  # for 2^N dummy tasks, min=1
DUMMY_TASKS = np.power(2, DUMMY_BITS) - 1


class Env(env.Env):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=1500, seed=None, train=True, **kwargs):
    super().__init__(area, view, size, reward, length, seed, **kwargs)
    # tasks contain both synonyms and repetitions
    target_achievements, target_achievements_base = get_synonym_tasks(constants.achievements)
    counts = [10 if 'collect' in ach else 5 for ach in target_achievements_base]
    self.target_achievements, self.isreptask = get_repeat_tasks(target_achievements, counts=counts)
    self.target_achievements_base, _ = get_repeat_tasks(target_achievements_base, counts=counts)
    self.task_progress = 0
    # task condition attributes
    self.task_idx = 0
    self.task_enc = np.zeros(len(ENC_ORDER) + DUMMY_BITS)
    self.task_steps = 0
    self.past_achievements = None
    self.follow_achievements = None
    self.given_achievements = None
    self.eval_tsr = np.zeros(len(self.target_achievements))  # evaluated task success rates
    self.id2task, self.task2id = self.generate_mapping()
    self.all_tasks = list(self.id2task.values())
    self.reset_world = True
    self.train = train

  @property
  def observation_space(self):
    img_shape = (self._size[1], self._size[0], 3)
    return DictSpace({
      'image': BoxSpace(0, 255, img_shape, np.uint8),
      'task_enc': BoxSpace(0, 1, (len(ENC_ORDER) + DUMMY_BITS, ), np.uint8),
    })

  def reset(self, goal=None):
    self._episode += 1
    self._step = 0
    self.task_progress = 0
    self.task_done = False
    self.task_steps = 0
    if self.reset_world or not self.train:
      self.reset_world = False
      inventory = None
      # inherit inventory 50% of the time
      if self._player and np.random.rand() < 0.5:
        inventory = self._player.inventory.copy()

      center = (self._world.area[0] // 2, self._world.area[1] // 2)
      self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
      self._update_time()
      self._player = objects.Player(self._world, center)
      self._world.add(self._player)
      self._unlocked = set()
      worldgen.generate_world_og(self._world, self._player)

      if inventory:
        self._player.inventory = inventory
      self.past_achievements = self._player.achievements.copy()
      self.follow_achievements = {k: 0 for k in self.target_achievements}
      self.given_achievements = {k: 0 for k in self.target_achievements}
      self.given_achievements.update({f'dummy{i}': 0 for i in range(DUMMY_TASKS)})
    self.task_idx = self.task2id[goal] if goal else np.random.randint(len(self.target_achievements) + DUMMY_TASKS)
    self.update_given_ach()
    self.task_progress = 0
    self.task_str = self.id2task[self.task_idx]
    return self._obs(), {'goal': self.task_str}

  def update_given_ach(self):
    if self.task_idx < len(self.target_achievements):
      self.given_achievements[self.target_achievements[self.task_idx]] += 1
    else:
      i = self.task_idx - len(self.target_achievements)
      self.given_achievements[f'dummy{i}'] += 1

  def step(self, action):
    obs, reward, done, other_done, info = super().step(action)
    info['given_achs'] = self.given_achievements.copy()
    info['follow_achs'] = self.follow_achievements.copy()
    info['goal'] = self.task_str
    self.reset_world = done
    done = done or self.task_done
    return obs, reward, done, other_done, info

  def _encode_task(self, task_idx):
    encoding = np.zeros(len(ENC_ORDER) + DUMMY_BITS)
    if self.task_idx < len(self.target_achievements):
      task = self.target_achievements[self.task_idx]
      task_words = task.split('_')
      # bag of words encoding
      for i, word in enumerate(ENC_ORDER):
        if word in task_words:
          encoding[i] = 1
    else:
      dummy_enc = np.random.choice([0, 1], size=DUMMY_BITS)
      encoding[-DUMMY_BITS:] = dummy_enc
      # ensure that there is at least one bit flipped in dummy bits
      rdn_idx = np.random.randint(DUMMY_BITS, size=1)
      encoding[-rdn_idx-1] = 1
    return encoding

  def _decode_task(self, task_enc):
    if (task_enc[-DUMMY_BITS:] > 0).any():
      return 'dummy task'
    else:
      return ' '.join([ENC_ORDER[int(i)] for i, c in enumerate(task_enc) if c])

  def render(self, size=None, semantic=False, add_desc=False):
    canvas = super().render(size=size, semantic=semantic)
    if not semantic and add_desc:
      img = Image.fromarray(canvas)
      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 25)
      draw.text((0, 0), self._decode_task(self.task_enc), (255,255,255), font=font)
      draw.text((0, 20), self._player.action,(255,255,255), font=font)
      canvas = np.asarray(img)
    return canvas

  def _get_reward(self):
    reward = 0

    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    self._unlocked |= unlocked

    task_failed = True
    if self.task_idx < len(self.target_achievements):
      task_desc = self.target_achievements[self.task_idx]
      btask_desc = self.target_achievements_base[self.task_idx]
      task_words = task_desc.split('_')
      btask_words = btask_desc.split('_')
      if self.isreptask[self.task_idx]:
        # repeat task
        sub_btaskdesc = '_'.join(btask_words[:-1])
        if self._player.achievements[sub_btaskdesc] - self.past_achievements[sub_btaskdesc] > 0:
          self.task_progress += 1.0 / float(task_words[-1])
        if self.task_progress >= 1.0:
          reward += 1.0
          self.follow_achievements[task_desc] += 1
          self.task_done = True
          task_failed = False
      elif self._player.achievements[btask_desc] - self.past_achievements[btask_desc] > 0:
        # agent successfully completed given task
        reward += 1.0
        self.follow_achievements[task_desc] += 1
        self.task_done = True
        task_failed = False

    if task_failed:
      # increase task step, check if agent is taking too long to complete given task
      self.task_steps += 1
      if self.task_steps > 300:
        self.task_done = True

    self.past_achievements = self._player.achievements.copy()
    return reward

  def _obs(self):
    return {
      'image': self.render(semantic=True),
      'task_enc': self.task_enc,
    }

  def push_info(self, info):
    self.eval_tsr = info['ema_tsr']
    
  def generate_mapping(self):
    plural_exceptions = {
      'coal': 'coals',
      'diamond': 'diamonds',
      'drink': 'drinks',
      'iron': 'irons',
      'stone': 'stones',
      'wood': 'woods',
      'iron pickaxe': 'iron pickaxes',
      'iron sword': 'iron swords',
      'stone pickaxe': 'stone pickaxes',
      'stone sword': 'stone swords',
      'wood pickaxe': 'wood pickaxes',
      'wood sword': 'wood swords',
      'furnace': 'furnaces',
      'table': 'tables',
    }

    id2task = {}
    task2id = {}

    original_list = self.target_achievements + [f'Dummy {i}' for i in range(DUMMY_TASKS)]

    for i, s in enumerate(original_list):
      if s.startswith('Dummy'):
        new_s = s
      else:
        parts = s.split('_')
        if parts[-1].isdigit():
            count = int(parts[-1])
            action = parts[0].capitalize()
            item_parts = parts[1:-1]
        else:
            count = 1
            action = parts[0].capitalize()
            item_parts = parts[1:]
        item_str = ' '.join(item_parts)
        # Handle pluralization
        if count > 1:
            # Get plural form
            plural_item = plural_exceptions.get(item_str, item_str + 's')
        else:
            plural_item = item_str
        # Construct human-readable string
        if count == 1:
            new_s = f"{action} {plural_item}"
        else:
            new_s = f"{action} {count} {plural_item}"
            
      # Add to mapping
      id2task[i] = new_s
      task2id[new_s] = i

    return id2task, task2id
