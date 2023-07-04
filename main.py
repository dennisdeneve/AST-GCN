# -*- coding: utf-8 -*-
import time
import yaml
import argparse
import Train.train_AST_GCN as train_ast_gcn
import Train.train_TGCN as train_tgcn



def main():
    time_start = time.time()
    print("Timer started : " ,time_start,'s')

    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to YAML config file')
    args = parser.parse_args()

    # Load the YAML config file which contains all the required settings for platform
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config['train_ast_gcn']['default']:
        print("************* Starting training process for the AST-GCN Model ************* ")
        train_ast_gcn.train(config)
        print("*************  Finished training process for the AST-GCN Model ************* ")
    
    if config['train_t_gcn']['default']:
        print("************* Starting training process for the T-GCN Model ************* ")
        train_tgcn.train(config)
        print("*************  Finished training process for the T-GCN Model ************* ")
        

    time_end = time.time()
    print("Time taken for the experimental pipeline : " ,time_end-time_start,'s')

if __name__ == '__main__':
    main()