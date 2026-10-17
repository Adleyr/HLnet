import argparse
import os

from trainer import Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/HLnet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)


    trainer=Trainer(config)

    trainer.test()

###################################
if __name__ == '__main__':
    main()
