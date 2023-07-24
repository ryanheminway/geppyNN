# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:50:45 2023

GEPNN Post Processing scripts

@author: Ryan Heminway
"""
from geppy.core.entity import *
from geppy.core.symbol import *
from deap import creator, base, tools
from geppy.support.visualization import *
from gepnnFunctions import *

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def read_from_pickle(path):
    """
    Load an object stored in a Pickle file at a given file path. 
    
    Class stored in file must be known at runtime. 
    """
    with open(path, 'rb') as file:
        try:
            return pickle.load(file)
        except EOFError:
            pass

def plot_series(series, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(series)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.show()
    # plt.legend(loc='best', fancybox=True, shadow=True)

def analyze_single_classifier(exp_path, file_name, exp_name):
    """
    Analyze results from a single run of EANT. Given file path should lead to
    a stored Pickle file, storing the deap Logbook object containing stats and 
    Genome objects from the run training a Classifier. Runs solving for a 
    Classifier model are expected to have column names in the Logbook
    object: ['min', 'max', 'avg', 'std', 'best_ind', 'train_acc', 'test_acc'].

    Parameters
    ----------
    exp_path: String
        Absolute path to folder where the Pickle file resides, and analysis 
        results can be stored.
    file_name : String
        Name of pickle file to analyze.
    exp_name : String
        Identifier to use when storing results associated with this experiment.
    """
    file_path = exp_path + file_name
    run_obj = read_from_pickle(file_path)
    
    final_train_acc = run_obj.select('train_acc')[-1]
    final_test_acc = run_obj.select('test_acc')[-1]
    
    print("Got final train / test acc: {} / {}".format(final_train_acc, final_test_acc))
    
    # Plot min / max / avg fitness across generations
    max_list = run_obj.select('max')
    min_list = run_obj.select('min')
    avg_list = run_obj.select('avg')
    med_list = run_obj.select('med')
    max_list = [x for x in max_list if not x == None]
    min_list = [x for x in min_list if not x == None]
    avg_list = [x for x in avg_list if not x == None]
    med_list = [x for x in med_list if not x == None]
    
    fig, ax = plt.subplots()
    ax.plot(min_list, label='min')
    ax.plot(med_list, label="med")
    ax.plot(avg_list, label="avg")
    ax.plot(max_list, label="max")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Trends')
    ax.legend()
    plt.show()
    
    plot_series(min_list, "Generation", "Min Fitness", "Min In Population Trend")
    plot_series(max_list, "Generation", "Max Fitness", "Max In Population Trend")
    plot_series(avg_list, "Generation", "Avg Fitness", "Avg Individual Trend")
    plot_series(med_list, "Generation", "Median Fitness", "Median in Population Trend")
    
    # Store best individual as a graph image
    best_inds = run_obj.select('best_ind')
    best_inds = [x for x in best_inds if not x == None]
    
    if len(best_inds) > 0:
        graph_file = exp_path + "analyzed_best.png"
        rename_labels = {'add': '+', 'sub': '-', 'mult': '*', 'protected_div': '/', "T_link_softmax" : "Softmax"}  
        export_expression_tree_nn(best_inds[-1], rename_labels, graph_file)
        
        graph_prog_folder = "graph_story/"
        
        graph_folder = exp_path + graph_prog_folder
        Path(graph_folder).mkdir(parents=True, exist_ok=True)
            
        # Graph individual progression
        tot = len(best_inds)
        for i in range(tot):
            graph_file = graph_folder + "gen_{}.png".format(i)
            if i < 50:
                export_expression_tree_nn(best_inds[i], rename_labels, graph_file)
            elif i < 100:
                if i % 10 == 0:
                    export_expression_tree_nn(best_inds[i], rename_labels, graph_file)
            else:
                if i % 50 == 0:
                    export_expression_tree_nn(best_inds[i], rename_labels, graph_file)
        
        # Clean out files we dont want
        labeled_files = [f for f in os.listdir(graph_folder) if f.endswith('labeled.png')]
        
        for f in labeled_files:
            del_file = graph_folder + f
            os.remove(del_file)
            
            
def analyze_all(exp_path):
    # pickle_files = [f for f in os.listdir(exp_path) if f.endswith('.pickle')]
    # print("got files: ", pickle_files)
    # for f in pickle_files:
    #     number_pickle = f.split("_iter_", 1)[1]
    #     number = number_pickle.split(".pickl", 1)[0]
    #     new_number = int(number) + 40
    # #    os.rename()
    #     os.rename(exp_path + f, exp_path + "stats_iter_{}.pickle".format(new_number))
        
    # exit(1)
    
    # Make pandas dataframe from aggregated data
    aggr_df = pd.DataFrame(make_experiment_dict(exp_path))
    
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_df, x='Generation', y='Avg', err_style="band") # y='Avg',
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title("Average of 'Avg Fitness' for all Iterations")
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_df, x='Generation', y='Med', err_style="band")
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Median Fitness')
    plt.title("Average of 'Median Fitness' for all Iterations")
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_df, x='Generation', y='Min', err_style="band")
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Min Fitness')
    plt.title("Average of 'Min Fitness' for all Iterations")
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_df, x='Generation', y='Max', err_style="band")
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.title("Average of 'Max Fitness' for all Iterations")
    # Show the plot
    plt.show()
        
    
def make_experiment_dict(exp_path, eant=False, guided=False):
    """
    Create and return a Dictionary containing columns for each statistic
    recorded in the experiment. Aggregates data across ALL pickle objects
    (individual runs) in the given folder.
    
    Returns:
        (dataframe, avg_train_acc, avg_test_acc, file_with_top_acc)
    """
    pickle_files = [f for f in os.listdir(exp_path) if f.endswith('.pickle')]
    
    type_string = "eant" if eant else "gepnn"
    type_string = type_string + " (guided)" if guided else type_string
    
    # Aggregating dictionary
    aggr_dict = {'Generation': [],
                 'Avg': [],
                 'Min': [],
                 'Max': [],
                 'Med': [],
                 'Variation': [],
                 'RunIdx': []}
    
    avg_train_acc = 0
    avg_test_acc = 0
    
    best_file = "N/A"
    best_acc = 0
    
    avg_max_fit = 0 # for regression
    max_fit = 0
    
    for i, file_name in enumerate(pickle_files):
        # Read in Pickle object for a single iteration
        file_path = exp_path + file_name
        run_obj = read_from_pickle(file_path)
        
        final_train_acc = run_obj.select('train_acc')[-1]
        final_test_acc = run_obj.select('test_acc')[-1]
        
        if final_test_acc != None:
            if final_test_acc > best_acc:
                best_acc = final_test_acc
                best_file = file_name
        
            avg_train_acc += final_train_acc
            avg_test_acc += final_test_acc
        
        max_list = run_obj.select('max')
        min_list = run_obj.select('min')
        avg_list = run_obj.select('avg')
        med_list = run_obj.select('med')
        max_list = [x for x in max_list if not x == None]
        min_list = [x for x in min_list if not x == None]
        avg_list = [x for x in avg_list if not x == None]
        med_list = [x for x in med_list if not x == None]
        
        # LIMIT TO 1000 for regression
        max_list = max_list[:1000]
        min_list = min_list[:1000]
        avg_list = avg_list[:1000]
        med_list = med_list[:1000]
        
        avg_max_fit += max_list[-1]
        if max_list[-1] > max_fit:
            max_fit = max_list[-1]
            best_file = file_name # Overloading best_file for regr / classification
        gens = list(range(len(max_list)))
        idxs = [i for j in range(len(max_list))]
        
        aggr_dict['Generation'] = aggr_dict['Generation'] + gens
        aggr_dict['Avg'] = aggr_dict['Avg'] + avg_list
        aggr_dict['Min'] = aggr_dict['Min'] + min_list
        aggr_dict['Max'] = aggr_dict['Max'] + max_list
        aggr_dict['Med'] = aggr_dict['Med'] + med_list
        aggr_dict['RunIdx'] = aggr_dict['RunIdx'] + idxs
   
    avg_test_acc = avg_test_acc / float(len(pickle_files))
    avg_train_acc = avg_train_acc / float(len(pickle_files))
    avg_max_fit = avg_max_fit / float(len(pickle_files))
    aggr_dict['Variation'] = [type_string for i in range(len(aggr_dict['Avg']))] 
    print("Analyzing results from path: ", exp_path)
    print("Got averaged test_accuracy: ", avg_test_acc)
    print("Got averaged train_accuracy: ", avg_train_acc)
    print("Iteration with best test accuracy: ", best_file)
    print("That best accuracy: ", best_acc)
    print("For regression, avg max fitness: ", avg_max_fit)
    print("Iteration with best max fitness: ", best_file)
    
    return aggr_dict
    
def compare_experiments(title, experiments):
    aggr_dict = {'Generation': [],
                 'Avg': [],
                 'Min': [],
                 'Max': [],
                 'Med': [],
                 'Variation': [],
                 'RunIdx': []}
    
    # Each entry in experiments should be (path, eant?, guided?)
    for (path, eant, guided) in experiments:
        path_dict = make_experiment_dict(path, eant, guided)
        
        # Aggregate results into one dict
        aggr_dict['Generation'] = aggr_dict['Generation'] + path_dict['Generation']
        aggr_dict['Avg'] = aggr_dict['Avg'] + path_dict['Avg']
        aggr_dict['Min'] = aggr_dict['Min'] + path_dict['Min']
        aggr_dict['Max'] = aggr_dict['Max'] + path_dict['Max']
        aggr_dict['Med'] = aggr_dict['Med'] + path_dict['Med']
        aggr_dict['Variation'] = aggr_dict['Variation'] + path_dict['Variation']
        aggr_dict['RunIdx'] = aggr_dict['RunIdx'] + path_dict['RunIdx']
        
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_dict, x='Generation', y='Med', hue='Variation', style='Variation', err_style="band")
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Median Fitness')
    plt.title("Median Fitness for Regression") # title)
    # Move the legend to a draggable position
    ax.get_legend().set_draggable(True)
    # Show the plot
    plt.show()        
    
    # # Create the plot using seaborn
    # ax = sns.lineplot(data=aggr_dict, x='Generation', y='Min', hue='Variation', style='Variation', err_style="band")
    # # Set labels and title for the plot
    # plt.xlabel('Generation')
    # plt.ylabel('Min Fitness (Summed Cross Entropy Loss)')
    # plt.title("Minimum Fitness for Iris Classification") # title)
    # # Move the legend to a draggable position
    # ax.get_legend().set_draggable(True)
    # # Show the plot
    # plt.show()      
    
    # Create the plot using seaborn
    ax = sns.lineplot(data=aggr_dict, x='Generation', y='Max', hue='Variation', style='Variation', err_style="band")
    # Set labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.title('Maximum Fitness for Regression') # title)
    # Move the legend to a draggable position
    ax.get_legend().set_draggable(True)
    # Show the plot
    plt.show()      
    
    # # Create the plot using seaborn
    # ax = sns.lineplot(data=aggr_dict, x='Generation', y='Avg', hue='Variation', err_style="band")
    # # Set labels and title for the plot
    # plt.xlabel('Generation')
    # plt.ylabel('Avg Fitness')
    # plt.title(title)
    # # Move the legend to a draggable position
    # ax.get_legend().set_draggable(True)
    # # Show the plot
    # plt.show()      
        

if __name__ == '__main__':
    # Required for any un-pickling to work, when the pickled objects contain
    # Individuals 
    # creator.create("FitnessMin", base.Fitness, weights=(-1,)) # min fitness
    # creator.create("Individual", Chromosome, fitness=creator.FitnessMin)
    
    creator.create("FitnessMax", base.Fitness, weights=(1,)) # maximize fitness
    creator.create("Individual", Chromosome, fitness=creator.FitnessMax)
    
    model_path = str(Path.cwd()) + "/../../experiments/20230707_eant_vs_gepnn/gepnn/20230706_regr_guided/"
    analyze_single_classifier(model_path, "stats_iter_53.pickle", "Testing plotting")
    
    #analyze_all(model_path)
    
    
    # Each experiment is a (path_to_data, eant_experiment?, guided_experiment?)
    experiments = [(str(Path.cwd()) + "/../../experiments/20230707_eant_vs_gepnn/gepnn/202307x_glass_guided_combined/", False, True),
                   (str(Path.cwd()) + "/../../experiments/20230707_eant_vs_gepnn/gepnn/202307x_glass_combined/", False, False)]
    
    #compare_experiments("20230706 Iris GEPNN", experiments)
    
