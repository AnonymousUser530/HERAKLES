import os
import torch
import shutil

def LL_loader_separated_actor_critic(Actor_model, Critic_model,
                                     obs_space, task_space, ll_action_space_id_to_act,
                                     memory_size_low_level, saving_path_low_level, config_args,
                                     loading_path_epoch_k_low_level=None):
    if saving_path_low_level is None and loading_path_epoch_k_low_level is None:
        raise ValueError("Either saving_path_low_level or loading_path_epoch_k_low_level must be provided")

    low_level_agents = dict()
    for tsk in task_space:
        if saving_path_low_level is not None:
            path_low_level = os.path.join(os.path.join(saving_path_low_level, tsk), 'last')
        elif loading_path_epoch_k_low_level is not None:
            path_low_level = loading_path_epoch_k_low_level.format(tsk)
        else:
            raise ValueError("Either saving_path_low_level or loading_path_epoch_k_low_level must be provided")
        if os.path.exists(path_low_level + "/actor/model.checkpoint"):
            try:
                print("Loading actor low level")
                actor_model = Actor_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
                actor_model.load_state_dict(torch.load(path_low_level + "/actor/model.checkpoint"))
            except:
                if saving_path_low_level is None:
                    print(path_low_level + "/actor/model.checkpoint")
                    raise ValueError("actor model not found for loading at epoch k")
                print("Loading model low level from backup")
                actor_model = Actor_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
                actor_model.load_state_dict(torch.load(path_low_level.replace("last", "backup") + "/actor/model.checkpoint"))
                dest = path_low_level + "actor"
                src = path_low_level.replace("last", "backup") + "/actor".format(tsk)
                shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            if saving_path_low_level is not None:
                print("Creating actor low level")
            else:
                raise ValueError("can create actor low level only if saving_path_low_level is provided")
            try:
                os.makedirs(os.path.join(os.path.join(os.path.join(saving_path_low_level, tsk), 'last'), 'actor'))
                os.makedirs(os.path.join(os.path.join(os.path.join(saving_path_low_level, tsk), 'backup'), 'actor'))
            except FileExistsError:
                pass
            actor_model = Actor_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
            torch.save(actor_model.state_dict(), saving_path_low_level + "/{}/backup/actor/model.checkpoint".format(tsk))

        if os.path.exists(path_low_level + "/critic/model.checkpoint".format(tsk)):
            try:
                print("Loading critic low level")
                critic_model = Critic_model(obs_space,
                                            action_space_size=int(len(ll_action_space_id_to_act)),
                                            hidsize=1024,
                                            memory=config_args.rl_script_args.memory_ll ,
                                            memory_size=memory_size_low_level)
                critic_model.load_state_dict(torch.load(path_low_level + "/critic/model.checkpoint".format(tsk)))
            except:
                if saving_path_low_level is None:
                    print(path_low_level + "/critic/model.checkpoint")
                    raise ValueError("critic model not found for loading at epoch k")
                print("Loading model low level from backup")
                critic_model = Critic_model(obs_space,
                                            action_space_size=int(len(ll_action_space_id_to_act)),
                                            hidsize=1024,
                                            memory=config_args.rl_script_args.memory_ll ,
                                            memory_size=memory_size_low_level)
                critic_model.load_state_dict(torch.load(path_low_level.replace("last", "backup") + "/critic/model.checkpoint".format(tsk)))
                dest = path_low_level + "/critic".format(tsk)
                src = path_low_level.replace("last", "backup") + "/critic".format(tsk)
                shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            if saving_path_low_level is not None:
                print("Creating critic low level")
            else:
                raise ValueError("can create critic low level only if saving_path_low_level is provided")
            try:
                os.makedirs(os.path.join(os.path.join(os.path.join(saving_path_low_level, tsk), 'last'), 'critic'))
                os.makedirs(os.path.join(os.path.join(os.path.join(saving_path_low_level, tsk), 'backup'), 'critic'))
            except FileExistsError:
                pass
            critic_model = Critic_model(obs_space,
                                        action_space_size=int(len(ll_action_space_id_to_act)),
                                        hidsize=1024,
                                        memory=config_args.rl_script_args.memory_ll ,
                                        memory_size=memory_size_low_level)
            torch.save(critic_model.state_dict(), saving_path_low_level + "/{}/backup/critic/model.checkpoint".format(tsk))

        actor_model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout, batchnorm, ...)
        for n, _ in actor_model.named_children():
            if "rnn" in n:
                actor_model._modules[n].train()
        if torch.cuda.is_available():
            actor_model.cuda()

        critic_model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout, batchnorm, ...)
        for n, _ in critic_model.named_children():
            if "rnn" in n:
                critic_model._modules[n].train()
        if torch.cuda.is_available():
            critic_model.cuda()

        low_level_agents[tsk] = {"actor": actor_model, "critic": critic_model}

    return low_level_agents

def LL_loader_shared_actor_critic(AC_model,
                                         obs_space, task_space, ll_action_space_id_to_act,
                                         memory_size_low_level, saving_path_low_level, config_args,
                                  loading_path_epoch_k_low_level=None):

    if saving_path_low_level is None and loading_path_epoch_k_low_level is None:
        raise ValueError("Either saving_path_low_level or loading_path_epoch_k_low_level must be provided")

    low_level_agents = dict()
    for tsk in task_space:
        if saving_path_low_level is not None:
            path_low_level = os.path.join(os.path.join(saving_path_low_level, tsk), 'last')
        elif loading_path_epoch_k_low_level is not None:
            path_low_level = loading_path_epoch_k_low_level.format(tsk)
        else:
            raise ValueError("Either saving_path_low_level or loading_path_epoch_k_low_level must be provided")
        if os.path.exists(path_low_level + "/model.checkpoint".format(tsk)):
            try:
                print("Loading actor low level")
                actor_critic_model = AC_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
                actor_critic_model.load_state_dict(torch.load(path_low_level + "/model.checkpoint".format(tsk)))
            except:
                if saving_path_low_level is None:
                    print(path_low_level + "/model.checkpoint")
                    raise ValueError("model not found for loading at epoch k")
                print("Loading model low level from backup")
                actor_critic_model = AC_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
                actor_critic_model.load_state_dict(torch.load(path_low_level.replace("last", "backup") + "/model.checkpoint".format(tsk)))
                dest = path_low_level
                src = path_low_level.replace("last", "backup")
                shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            if saving_path_low_level is not None:
                print("Creating actor low level")
            else:
                raise ValueError("can create actor low level only if saving_path_low_level is provided")
            try:
                os.makedirs(os.path.join(os.path.join(saving_path_low_level, tsk), 'last'))
                os.makedirs(os.path.join(os.path.join(saving_path_low_level, tsk), 'backup'))
            except FileExistsError:
                pass
            actor_critic_model = AC_model(obs_space,
                                          action_space_size=int(len(ll_action_space_id_to_act)),
                                          hidsize=1024,
                                          memory=config_args.rl_script_args.memory_ll ,
                                          memory_size=memory_size_low_level)
            torch.save(actor_critic_model.state_dict(), saving_path_low_level + "/{}/backup/model.checkpoint".format(tsk))


        actor_critic_model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout, batchnorm, ...)
        for n, _ in actor_critic_model.named_children():
            if "rnn" in n:
                actor_critic_model._modules[n].train()
        if torch.cuda.is_available():
            actor_critic_model.cuda()

        low_level_agents[tsk] = actor_critic_model

    return low_level_agents