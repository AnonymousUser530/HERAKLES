def generate_prompt(past_transitions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "{}\n".format(info["goal"])
    for transition in past_transitions:
        prompt += "Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Action:{}\n".format(transition["act"])

    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt