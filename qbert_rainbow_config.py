from ding.model import RainbowDQN
from easydict import EasyDict
import torch
import os

os.chdir(os.path.dirname(__file__))

qbert_rainbow_config = dict(
    exp_name='qbert_rainbow_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        #'ALE/Qbert-v5' is available. But special setting is needed after gym make.
        frame_stack=4
    ),
    policy=dict(
        cuda=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            iqn=False,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000,)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.05,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=50000, ),
        ),
    ),
)
main_config = EasyDict(qbert_rainbow_config)

qbert_rainbow_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
create_config = EasyDict(qbert_rainbow_create_config)

if __name__ == '__main__':
    # or you can enter ding -m serial -c qbert_rainbow_config.py -s 0
    model = RainbowDQN(**main_config.policy.model)
    state_dict = torch.load(r"./ckpt_best.pth.tar", map_location='cpu')
    model.load_state_dict(state_dict['model'])
    import cereal_entry
    cereal_entry.serial_pipeline((main_config, create_config), seed=0,model=model)
    
    #from ding.entry import serial_pipeline
    #serial_pipeline((main_config, create_config), seed=0,model=model)
