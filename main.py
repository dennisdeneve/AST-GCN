import time
import yaml
import argparse
from Train.stgcn_Train import trainSTGCN as trainSTGCN
from Train.astgcn_Train import trainASTGCN as trainASTGCN
from Evaluation.STGCN_eval import evalSTGCN as evalSTGCN
from Evaluation.ASTGCN_eval import evalASTGCN as evalASTGCN

def main():
    time_start = time.time()
    print("Timer started for experimentation! :p ")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to YAML config file')
    args = parser.parse_args()
    # Load the YAML config file which contains all the required settings for platform
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ##############################  Training  ##################################
    if config['train_ast_gcn']['default']:
        print("************* Starting training process for the AST-GCN Model ************* ")
        trainASTGCN(config)
        print("*************  Finished training process for the AST-GCN Model ************* ")
    
    if config['train_st_gcn']['default']:
        print("************* Starting training process for the ST-GCN Model ************* ")
        trainSTGCN(config)
        print("*************  Finished training process for the ST-GCN Model ************* ")
    
    ##############################  Evaluation  ##################################
    if config['eval_st_gcn']['default']:
        print("************* Starting eval process for the ST-GCN Model ************* ")
        evalSTGCN(config)
        print("*************  Finished eval process for the ST-GCN Model ************* ")
        
    if config['eval_ast_gcn']['default']:
        print("************* Starting eval process for the AST-GCN Model ************* ")
        evalASTGCN(config)
        print("*************  Finished eval process for the AST-GCN Model ************* ")
        
    time_end = time.time()
    print("Time taken for the experimental pipeline :@ " ,time_end-time_start,'s')
if __name__ == '__main__':
    main()