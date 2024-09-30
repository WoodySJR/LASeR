import os
import numpy as np
import shutil
import random
import math

import torch
import sys
from openai import OpenAI
from parser_llm import parser
from vec2morph import morph_to_vec, vec_to_morph, operator

from ppo.run import run_ppo
from evogym import sample_robot, hashable
from evogym.utils import is_connected, has_actuator, get_full_connectivity
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
sys.path.insert(2, curr_dir)

os.environ['OPENAI_API_KEY'] = "enter your API key here"
os.environ['OPENAI_BASE_URL'] = "enter the base URL here"
client = OpenAI()

operator_2 = operator(5)

system_prompt = "Now you will serve as an intelligent search operator in an Evolutionary Algorithm. In each generation you are given \
a number of evaluated solutions in the format of numpy array, together with their fitness scores. \
Each solution and its fitness score are separated by a comma. \
Different solutions are separated by semicolons. \
The solutions are sorted according to their fitness scores in ascending order. Higher fitness scores are better. \
Your job is to output a new solution that meets a desired fitness. \
Please try your best to logically analyze the relationship between the evaluated solutions and their fitness scores, \
and adhere to this information while proposing the new solution. \
A solution is a 5 times 5 matrix, where each entry is an integer between zero and four. \
Please begin the new solution with <begin> and end it with <end>. The new solution should be formatted in numpy array fashion. \
The new solution must be distinct from the evaluated solutions. Only generate the new solution. No explanation. "

vsr_description = "Hello! We are going to design the structure of a two-dimensional voxel-based soft robot (VSR) in a simulation environment. \
VSRs are composed of square-shaped voxels of different types, aligned into a 5 * 5 matrix. \
Adjacent voxels (that is, in either the same row or the same column) are connected together; \
Voxels located in diagonal positions are not connected together. The robot is subject to gravity, \
and the bottom row touches the ground. There are 5 types of voxels available, including soft voxels \
(for which elastic deformation is possible), rigid voxels (which can not deform), horizontal and vertical actuators \
(which can change their sizes horizontally or vertically), and empty voxels (which basically mean that the corresponding position \
is empty). \
Empty voxels, rigid voxels, soft voxels, horizontal actuators and vertical actuators are represented as 0, 1, 2, 3 and 4, respectively. "

env_description = "The simulation represents objects and their environment as a 2D mass-spring system in a grid-like layout, \
where objects are initialized as a set of non-overlapping, connected voxels. The simulation converts all objects into a set of \
point masses and springs by turning each voxel into a cross-braced square, which may undergo deformation as the simulation progresses. \
The springs obey Hooke's law. Note that adjacent voxels share point masses and springs on their common edge. \
All point masses in the simulation have the same mass and the equilibrium lengths of axis-aligned and diagonal springs \
are constants for simplicity. However, the spring constants assigned vary based on voxel material-type, with ties broken \
in favor of the more-rigid spring. The actuators undergo gradual expansion/contraction either horizontally or vertically \
according to action signals, by changing the lengths of the corresponding springs. "

# carrier
task_description = "Your job is to propose robot designs suitable for completing the following task. \
A three-voxel wide box is initialized right above the robot, and the robot is required to keep the box on top of its head \
stably without letting it slip off, while locomoting rightwards as quickly as possible. "


'''
# walker
task_description = "Your job is to propose high-performing robot designs that can locomote rightwards as quickly as possible. "
'''

'''
# pusher
task_description = "Your job is to propose high-performing robot designs suitable for completing the following task. A three-voxel wide box is initialized on the right of the robot, and the robot is required to push the box rightwards as quickly as possible, while keeping close to it."
'''

constraint_description = "There are two constraints to VSR designs: 1. all voxels must form an entirety and should not fall apart; \
That is, the four voxels, if any, above, below, to the left and to the right of a non-empty voxel mustn't be empty at the same time. \
An example that violates such a constraint is [[2,2,2,2,2],[1,0,1,0,1],[0,4,3,4,0],[1,0,1,0,1],[0,0,0,0,0]], \
in which the voxel '1' in the fourth row and fifth column would fall off; \
(2) there must be at least one actuator (that is, either 3 or 4), so that the robot could interact with the environment. "

# carrier
additional = "Please carefully analyze the relationship between evaluated solutions and their fitness scores, and make use \
of this information to propose the new solution. \
Please make use of empty voxels cleverly so that complex functional substructures could be produced to fulfill the purposes \
of both carrying and locomoting. But do NOT use more than 10 empty voxels. \
Meanwhile, note that a high-performing robot design is not necessarily symmetric. "


'''
# walker
additional = "Please carefully analyze the relationship between evaluated solutions and their fitness scores, and make use \
of this information to propose the new solution. \
Please make use of empty voxels cleverly so that complex functional substructures (such as legs) could be produced to fulfill fast locomotion. \
But do NOT use more than 10 empty voxels. \
Meanwhile, note that a high-performing solution is not necessarily symmetric. "
'''

'''
# pusher
additional = "Please carefully analyze the relationship between evaluated solutions and their fitness scores, and make use \
of this information to propose the new solution. \
Please make use of empty voxels cleverly so that complex functional substructures (such as legs) could be produced to fulfill both pushing and locomoting. \
But do NOT use more than 10 empty voxels. \
Meanwhile, note that a high-performing robot design is not necessarily symmetric. "
'''


def run(args, experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, env_name):
    print()
    all_structures = torch.zeros(1,125)
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)
    
    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    
    #generate a population
    if not is_continuing: 
        for i in range (pop_size):
            
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i, 0))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1
            all_structures = torch.cat((all_structures, morph_to_vec(temp_structure[0].reshape(1,5,5))), 0)

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(*structure_data, i, 0))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen
    
    
    save_flag = True
    
    while True:
        
        # record the feedback trials
        trials = 0
        success_trials = 0
        
        ### UPDATE NUM SURVIORS ###			
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))
        

        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        valid_num_path = os.path.join(root_dir, "saved_data", experiment_name, "valid_rate.txt")
        save_path_before_modify = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "before_modify")
        save_path_after_modify = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "after_modify")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass
        
        try:
            os.makedirs(save_path_before_modify)
            os.makedirs(save_path_after_modify)
        except:
            pass
        
        
        
        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()   
        
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:        
                ppo_args = (0, args, env_name, structure, tc, (save_path_controller, structure.label))
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
      
        group.run_jobs(num_cores)
     
        
        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()
        
        # save evaluated structures into the pool
        for structure in structures:
            if not structure.is_survivor:
                if save_flag:
                    all_structures = morph_to_vec(structure.body.reshape(1,5,5)).float()
                    all_fitnesses = [structure.fitness]
                    save_flag = False
                else:
                    all_structures = torch.cat((all_structures, morph_to_vec(structure.body.reshape(1,5,5)).float()), 0)
                    all_fitnesses.append(structure.fitness)
                    
        
        
        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\t\t" + str(structure.prev_gen_label) + "\n"
        f.write(out)
        f.close()

         ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        
        if generation < 4: # GA for the first 5 generations (warm starting) 
        
            num_children = 0
            while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

                parent_index = random.sample(range(num_survivors), 1)
                child, _ = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)
                prev = survivors[parent_index[0]].prev_gen_label

                if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                    structures[num_survivors + num_children] = Structure(*child, str(num_survivors + num_children)+"-GA", 0)
                    structures[num_survivors + num_children].prev_gen_label = prev
                    population_structure_hashes[hashable(child[0])] = True
                    num_children += 1
                    num_evaluations += 1
                    
        if generation >= 4: # generate the new population with LLM 
            
            prob = 0.4
            
            structures = survivors
            need = 25-len(structures)
            llm_attempts = 0
            llm_robots = 0
            while len(structures) < pop_size and num_evaluations < max_evaluations and llm_attempts<200: 
                try:
                    examples = "The following are a number of evaluated solutions, each followed by its fitness score. \
                    Each solution and its fitness score are separated with a comma. Different solutions are separated with semicolons. \
                    These solutions are sorted according to their fitness scores in ascending order. Higher fitness scores are better. "
                    
                    shots = []
                    fitnesses = []
                    
                    indices = np.argsort(all_fitnesses)
                    best_indices = indices[-num_survivors:]
            
                    for i in best_indices:
                        shots.append(vec_to_morph(all_structures[i].reshape(1,125), operator_2, 5)[0])
                        fitnesses.append(all_fitnesses[i])
                    
                    for i in len(shots):
                        examples = examples + str(np.array(shots[i]).astype("int")) + ", " + str(round(fitnesses[i],3)) + "; "
                    examples = examples[0:-2] + ". "
                    
                    print(examples)
                    
                    target = np.round(np.max(all_fitnesses)*1.2,3)
                    
                    ending = "Now please generate a new solution that has a fitness of " + str(target) + ". "
                    ending += "The new solution must be distinct from the above solutions. Begin it with <begin> and end it with <end>. No explanation. "
            
           
                    user_prompt = vsr_description + env_description + task_description + constraint_description + examples + additional + ending
                    
                    
                    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}],
                    stream=False, max_tokens=1024, temperature=1.5, presence_penalty=1.1, top_p=0.8)
                    
                    llm_attempts += 1
                    content = response.choices[0].message.content
                    print("LLM answers: ", content)
                    
                    robot = parser(content)
                    
                    if hashable(robot) in population_structure_hashes:
                        print("Duplication. ")
                    
                    u = random.uniform(0,1)
                    
                    if hashable(robot) not in population_structure_hashes and is_connected(robot) and has_actuator(robot) and robot.shape==(5,5) and u<1-prob:
                        structures.append(Structure(*(robot, get_full_connectivity(robot)), str(len(structures))+"-LLM", 0))
                        population_structure_hashes[hashable(robot)] = True
                        num_evaluations += 1
                        llm_robots += 1
                    print("Valid robot number: ", len(structures))
                    
                    
                    if hashable(robot) not in population_structure_hashes and is_connected(robot) and has_actuator(robot) and robot.shape==(5,5) and u>=1-prob:
                        print("Start similarity check. ")
                        sims = torch.mm(all_structures.float(), morph_to_vec(robot).reshape(125,1).float()).reshape(-1)
                        if (sims<=20).all():
                            print("Passed the test. ")
                            structures.append(Structure(*(robot, get_full_connectivity(robot)), str(len(structures))+"-LLM", 0))
                            population_structure_hashes[hashable(robot)] = True
                            num_evaluations += 1
                            llm_robots += 1
                        else:
                            print("Failed the test")
                            # save the robot before modification
                            trials += 1
                            temp_path = os.path.join(save_path_before_modify, str(trials)+".npy")
                            np.save(temp_path, robot)
                            prompt = "The solution that you just generated is too similar with an existing one. It needs further modification to improve diversity. \
                            Please decide which voxels in the solution can be replaced by other types of materials, without harming its fitness score. \
                            Change no more than 3 voxels. \
                            Please base your analysis on the characteristics of the evaluated solutions given to you. \
                            Meanwhile, make sure that the modification does not violate the constraints of VSRs. \
                            Now please tell me which voxels exactly do you think can be alterted, and explain the reason. "
                            
                            response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[{"role":"user", "content":user_prompt}, 
                                        {"role":"assistant", "content":content}, {"role":"user","content":prompt}],
                                        stream=False, max_tokens=1024, temperature=0.7, presence_penalty=1.1, top_p=0.8)
                            feedback = response.choices[0].message.content
                            
                            print(feedback)
                            
                            prompt2 = "Based on your analysis above, please generate the resulting solution. \
                            The number of voxels changed should not exceed three. \
                            Do not provide further texual explanation. \
                            Begin the solution with <begin> and end it with <end>. The solution should be formatted in numpy array fashion."
                            
                            response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[{"role":"user", "content":user_prompt}, 
                                        {"role":"assistant", "content":content}, {"role":"user","content":prompt}, 
                                        {"role":"assistant", "content":feedback}, {"role":"user","content":prompt2}],
                                        stream=False, max_tokens=1024, temperature=0.7, presence_penalty=1.1, top_p=0.8)
                            new_content = response.choices[0].message.content
                            print(new_content)
                            
                            robot = parser(new_content)
                            temp_path = os.path.join(save_path_after_modify, str(trials)+".npy")
                            np.save(temp_path, robot)
                            sims = torch.mm(all_structures.float(), morph_to_vec(robot).reshape(125,1).float()).reshape(-1)
                            if hashable(robot) not in population_structure_hashes and is_connected(robot) and has_actuator(robot) and (sims<=20).all() and robot.shape==(5,5):
                                print("New solution passed the test! ")
                                success_trials += 1
                                structures.append(Structure(*(robot, get_full_connectivity(robot)), str(len(structures))+"-LLM", 0))
                                population_structure_hashes[hashable(robot)] = True
                                num_evaluations += 1
                                llm_robots += 1
                            
                except:
                    print("The answer makes no sense! ")
            
            
            ga_robots = 0
            while len(structures) < pop_size and num_evaluations < max_evaluations: # when LLM fails to produce enough robots given limited attempts
                parent_index = random.sample(range(num_survivors), 1)
                child, _ = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)
                prev = survivors[parent_index[0]].prev_gen_label

                if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                    structures.append(Structure(*child, str(len(structures))+"-GA", 0))
                    population_structure_hashes[hashable(child[0])] = True
                    num_evaluations += 1
                    ga_robots += 1
                    
            temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "attempts.txt")
            f = open(temp_path, "w")
            out = str(llm_attempts) + "\t\t" + str(llm_robots) + "\t\t" + str(ga_robots) + "\n" + str(trials) + "\t\t" + str(success_trials)
            f.write(out)
            f.close()

        generation += 1
