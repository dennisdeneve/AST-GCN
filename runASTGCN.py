import time
import yaml
import argparse
from Execute.astgcnExecute import astgcnExecute
from HPO.astgcnHPO import astgcnHPO as astgcnHPO
from Logs.Evaluation import evalASTGCN as evalASTGCN
#import Visualisations.visualise as visualise
#import Plots.plotter as plotter

def main():
    time_start = time.time()
    print("Timer started for experimentation! :p ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='path to YAML config file')
    args = parser.parse_args()
    # Load the YAML config file which contains all the required settings for platform
    with open('configurations/astgcn_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    with open('configurations/sharedConfig.yaml', 'r') as file:
        sharedConfig = yaml.safe_load(file)
    models_list = ['ASTGCN'] # list of models for plotter

    ##############################  Training  ##################################
    if config['train_ast_gcn']['default']:
        print("************* Starting training process for the AST-GCN Model ************* ")
        trainer = astgcnExecute(sharedConfig,config)
        trainer.train()
        print("*************  Finished training process for the AST-GCN Model ************* ")
    
    ##############################  Evaluation  ##################################     
    if config['eval_ast_gcn']['default']:
        print("************* Starting eval process for the AST-GCN Model ************* ")
        evalASTGCN(config, sharedConfig)
        print("************* Plotting ************* ")
        #plotter.create('ASTGCN',config)
        # plotter.create_boxplots_for_models(models_list, config)
        print("*************  Finished eval process for the AST-GCN Model ************* ")
        
    # ############ Visualisations #############
    #if config['vis']['default'] :
     #   visualise.plot(config)
        
    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken for the experimental pipeline :@ {elapsed_minutes:.2f} minutes")
if __name__ == '__main__':
    main()
