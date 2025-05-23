from collections import defaultdict


class TokenLattice:
    def __init__(self, valid_actions, tokenizer):
        self.tokenizer = tokenizer
        self.lattice = self.build_lattice(valid_actions)
        self.token_lattice = self.tokenize_lattice()
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])[0]

    def build_lattice(self, valid_actions):
        # templates = env.getValidActionObjectCombinations()
        action_strings = valid_actions
        # action_strings = [x['action'] for x in templates]
        # np.random.shuffle(action_strings)
        lookup_by_prefix = defaultdict(set)
        bos_token = self.tokenizer.bos_token
        # pad_token = self.tokenizer.pad_token
        if bos_token is None:
            bos_token = "<pad>"  # "<extra_id_0>"
        # lookup_by_prefix[tuple()] = set([bos_token])

        for action in action_strings:
            tokenized_action = self.tokenizer.tokenize(action)
            tokenized_action.append(self.tokenizer.eos_token)
            action_zero = tokenized_action[0]
            lookup_by_prefix[tuple()] |= set([action_zero])

            for i in range(-1, len(tokenized_action) - 1, 1):
                prefix = tokenized_action[0:i + 1]
                next_token = tokenized_action[i + 1]
                # lookup_by_prefix[tuple(prefix)] |= set([next_token])
                lookup_by_prefix[tuple(prefix)] |= set([next_token])
        return lookup_by_prefix


    def tokenize_lattice(self):
        token_lattice = {}
        for key, tok_list in self.lattice.items():
            # TODO: what about when the tokenizer splits into multiple tokens?
            tokenized_key = self.tokenizer.convert_tokens_to_ids(key)
            if type(tokenized_key) == int:
                tokenized_key = [tokenized_key]
            tokenized_list = [self.tokenizer.convert_tokens_to_ids(x) for x in tok_list]
            token_lattice[tuple(tokenized_key)] = set(tokenized_list)
        return token_lattice

    def next_token_ids(self, id_prefix):
        # pdb.set_trace()
        try:
            return self.token_lattice[tuple(id_prefix)]
        except KeyError:
            return self.eos_token_id


def prefix_fn_global(token_lattice, tokenizer, tokenized_prompts, batch_id, prefix):
    # if len(prefix) == 1:
    #     if prefix[0] == 0:
    #         prefix[0] = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    # prefix[0] = 1
    prefix = prefix.tolist()
    # prefix = list(filter(lambda a: (a != 32000 and a!=32001), prefix))
    # print("===prefix_fn_global===")
    # print("prefix: \n{}".format(prefix))
    len_tkp = None
    for tkp in tokenized_prompts:
        # print("tkp: \n{}".format(tkp))
        # print("prefix[:len(tkp)]: {}".format(prefix[:len(tkp)]))
        if prefix[:len(tkp)] == tkp:
            len_tkp = len(tkp)
            break
    if len_tkp is None:
        # print("len_tkp is None")
        prefix = tuple()
    else:
        # print("len_tkp: {}".format(len_tkp))
        prefix = tuple(prefix[len_tkp:])
    # prefix = tuple(prefix.tolist())
    if type(prefix) == int:
        prefix = [prefix]
    # print("prefix: \n{}".format(prefix))
    next_toks = token_lattice.next_token_ids(prefix)
    if type(next_toks) == int:
        next_toks = [next_toks]

    return list(next_toks)

def prefix_fn_global_2(token_lattice_list, tokenizer, tokenized_prompts, batch_id, prefix):
    """
    token_lattice_list: list of TokenLattice objects
    tokenizer: tokenizer object
    tokenized_prompts: list of tokenized prompts
    batch_id: batch id
    prefix: prefix tensor pass by transformers

    Because the code is parallelized between different LLM, if nbr_LLM>1 the batch_id is not correct (i.e transformers will a batch_id=0 for each LLM)
    We search which tokenized prompt match the prefix to determine the true batch_id and thus call the correct TokenLattice object
    """
    prefix = prefix.tolist()
    len_tkp = None
    idx_lattice = 0
    for idx_tkp, tkp in enumerate(tokenized_prompts):
        if prefix[:len(tkp)] == tkp:
            len_tkp = len(tkp)
            idx_lattice = idx_tkp
            break
    if len_tkp is None:
        prefix = tuple()
    else:
        prefix = tuple(prefix[len_tkp:])
    if type(prefix) == int:
        prefix = [prefix]
    next_toks = token_lattice_list[idx_lattice].next_token_ids(prefix)
    if type(next_toks) == int:
        next_toks = [next_toks]

    return list(next_toks)
