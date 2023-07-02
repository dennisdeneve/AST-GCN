# -*- coding: utf-8 -*-
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import argparse
import Train.train as train
import time
import yaml

def main():
    time_start = time.time()
    print("Timer started : " ,time_start,'s')

    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to YAML config file')
    args = parser.parse_args()

    print("---- Loading the config.yaml file ----")
    # Load the YAML config file which contains all the required settings for platform
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("---- Finished loading the config.yaml file ----")


    print("************* Starting training the AST-GCN Model ************* ")
    train.train(config)
    print("*************  Finished training the AST-GCN Model ************* ")

    time_end = time.time()
    print("Timer ended : " ,time_end-time_start,'s')

if __name__ == '__main__':
    main()