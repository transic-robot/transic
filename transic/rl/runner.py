from rl_games.torch_runner import Runner as _Runner, _override_sigma

from transic.rl.agent import PPOAgent
from transic.rl.player import MyPPOPlayerContinuous as PPOPlayer


def _restore(agent, args, is_train_restore: bool):
    if (
        "checkpoint" in args
        and args["checkpoint"] is not None
        and args["checkpoint"] != ""
    ):
        set_epoch = args.get("from_ckpt_epoch", True)
        if is_train_restore:
            agent.restore(args["checkpoint"], set_epoch)
        else:
            agent.restore(args["checkpoint"])


class Runner(_Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        self.algo_factory.register_builder("ppo", lambda **kwargs: PPOAgent(**kwargs))
        self.player_factory.register_builder(
            "ppo", lambda **kwargs: PPOPlayer(**kwargs)
        )

    def run_train(self, args):
        print("Started to train")
        agent = self.algo_factory.create(
            self.algo_name, base_name="run", params=self.params
        )
        _restore(agent, args, is_train_restore=True)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args):
        print("Started to play")
        save_rollouts_cfg = args.get("save_rollouts", {})
        player = self.create_player(save_rollouts_cfg)
        _restore(player, args, is_train_restore=False)
        _override_sigma(player, args)
        player.run()

    def create_player(self, save_rollouts_cfg):
        return self.player_factory.create(
            self.algo_name, params=self.params, **save_rollouts_cfg
        )
