"""def generate_prompt(obs, infos):
    prompt = "{}\n".format(infos["goal"])
    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt"""

dictionary_english_french_name = {"key": "clef",
                                  "door": "porte",
                                  "ball": "balle",
                                  "box": "boîte"}

dictionary_english_french_adj = {"blue": "bleue",
                                 "green": "verte",
                                 "red": "rouge",
                                 "yellow": "jaune",
                                 "purple": "violette",
                                 "grey": "grise"}

dictionary_english_french_prep = {" a ": "une",
                                  " the ": "la"}


def generate_prompt(past_transitions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["low_level"]))
    prompt += "{}\n".format(info["goal"])
    for transition in past_transitions:
        prompt += "Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Action:{}\n".format(transition["act"])

    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt


class PromptGenerator:

    def __init__(self, memory_lenght=3):
        self.memory_lenght = memory_lenght

    def generate_prompt(self, low_level_actions, obs, acts, infos, nbr_step_max=None, hl_step=0):
        prompt = ''

        head_prompt = "Possible actions of the agent:"
        for ll in low_level_actions:
            head_prompt += " {},".format(ll)
        head_prompt = head_prompt[:-1] + "\n"

        prompt += head_prompt

        if nbr_step_max is not None:
            if nbr_step_max > 1:
                prompt += "You have a maximum of {} steps to complete the task.".format(int(nbr_step_max)) + "\n"
            else:
                prompt += "You have a single step to complete the task." + "\n"

        prompt += "{}".format(infos["goal"])

        for idx, o in enumerate(obs):
            prompt += "\nObservation {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), o[0])
            for k in o[1:]:
                prompt += ", " + k
            prompt += "."
            if idx < len(acts):
                prompt += "\nAction {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), acts[idx])
            else:
                prompt += "\nAction {}: ".format(max(idx, int(hl_step - self.memory_lenght + idx)))
        return prompt

    def generate_prompt_french(self, low_level_actions, obs, acts, infos, nbr_step_max=None, hl_step=0):
        prompt = ''

        head_prompt = "Actions possibles pour l'agent:"
        for ll in low_level_actions:
            head_prompt += " {},".format(ll)
        head_prompt = head_prompt[:-1] + "\n"

        prompt += head_prompt

        if nbr_step_max is not None:
            if nbr_step_max > 1:
                prompt += "Tu as un maximum de {} pas pour compléter la tâche.".format(int(nbr_step_max)) + "\n"
            else:
                prompt += "Tu as un seul pas pour compléter la tâche." + "\n"

        goal = infos["goal"].split("Goal of the agent:")[1]
        adj = ''
        for k, _ in dictionary_english_french_adj.items():
            if k in goal:
                adj = dictionary_english_french_adj[k]
        noun = ''
        for k, _ in dictionary_english_french_name.items():
            if k in goal:
                noun = dictionary_english_french_name[k]
        prep = ''
        for k, _ in dictionary_english_french_prep.items():
            if k in goal:
                prep = dictionary_english_french_prep[k]
        prompt += "Objectif de l'agent: aller à {} {} {}".format(prep, noun, adj)

        for idx, o in enumerate(obs):
            prompt += "\nObservation {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), o[0])
            for k in o[1:]:
                prompt += ", " + k
            prompt += "."
            if idx < len(acts):
                prompt += "\nAction {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), acts[idx])
            else:
                prompt += "\nAction {}:".format(max(idx, int(hl_step - self.memory_lenght + idx)))
        return prompt

    def generate_prompt_generic(self, low_level_actions, obs, acts, infos, nbr_step_max=None, hl_step=0, random_seed=None):
        prompt = ''

        head_prompt = "Possible actions of the agent:"
        len_ll = len(low_level_actions)
        for ll_indx, ll in enumerate(low_level_actions):
            # action + capital alphabetical letter
            head_prompt += " {} = action {},".format(ll, chr(65+(random_seed + ll_indx) % len_ll))
        head_prompt = head_prompt[:-1] + "\n"

        prompt += head_prompt

        if nbr_step_max is not None:
            if nbr_step_max > 1:
                prompt += "You have a maximum of {} steps to complete the task.".format(int(nbr_step_max)) + "\n"
            else:
                prompt += "You have a single step to complete the task." + "\n"

        prompt += "{}".format(infos["goal"])

        for idx, o in enumerate(obs):
            prompt += "\nObservation {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), o[0])
            for k in o[1:]:
                prompt += ", " + k
            prompt += "."
            if idx < len(acts):
                prompt += "\nDo: {}".format(acts[idx])
            else:
                prompt += "\nDo:"
        return prompt

    def generate_prompt_w_act_space(self, action_space, obs, acts, infos, nbr_step_max, hl_step):
        prompt = ''

        head_prompt = "Possible actions of the agent:"
        for act in action_space:
            head_prompt += " {},".format(act)
        head_prompt = head_prompt[:-1] + "\n"

        prompt += head_prompt

        if nbr_step_max > 1:
            prompt += "You have a maximum of {} steps to complete the task.".format(int(nbr_step_max)) + "\n"
        else:
            prompt += "You have a single step to complete the task." + "\n"

        prompt += "{}".format(infos["goal"])

        for idx, o in enumerate(obs):
            prompt += "\nObservation {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), o[0])
            for k in o[1:]:
                prompt += ", " + k
            prompt += "."
            if idx < len(acts):
                prompt += "\nAction {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), acts[idx])
            else:
                prompt += "\nAction {}:".format(max(idx, int(hl_step - self.memory_lenght + idx)))
        return prompt

    def generate_prompt_w_act_space_w_warning(self, action_space, obs, acts, infos, warnings, nbr_step_max, hl_step):
        prompt = ''

        head_prompt = "Possible actions of the agent:"
        for act in action_space:
            head_prompt += " {},".format(act)
        head_prompt = head_prompt[:-1] + "\n"

        prompt += head_prompt

        if nbr_step_max > 1:
            prompt += "You have a maximum of {} steps to complete the task.".format(int(nbr_step_max)) + "\n"
        else:
            prompt += "You have a single step to complete the task." + "\n"

        prompt += "{}".format(infos["goal"])

        for idx, o in enumerate(obs):
            prompt += "\nObservation {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), o[0])
            for k in o[1:]:
                prompt += ", " + k
            prompt += "."
            if idx < len(acts):
                prompt += "\nAction {}: {}".format(max(idx, int(hl_step - self.memory_lenght + idx)), acts[idx])
                if warnings[idx] != "":
                    prompt += "\nWarning: {}".format(warnings[idx])
            else:
                prompt += "\nAction {}:".format(max(idx, int(hl_step - self.memory_lenght + idx)))
        return prompt

    def generate_promp_goal_generator_lp_informed(self, epoch, dict_goal_lp):

        prompt = ("You are in a Minecraft like environment. You are an expert in curriculum learning and reinforcement "
                  "learning. Your goal is to help an agent master a diverse set of interesting tasks. You will be "
                  "provided with the list of goals and their current learning progress. You will have to select the goal "
                  "for the agent to achieve. The goals should be diverse and allow the agent to maximise their learning progess.\n")
        prompt += "Here is the list of goals and their learning progress:\n"
        for goal, lp in dict_goal_lp.items():
            if lp is not None:
                prompt += "{}: {}\n".format(goal, lp)
            else:
                prompt += "{}: learning progress unknown\n".format(goal)
        prompt += "Taking into account the previous commands the next goal of the agent is:"
        return prompt