# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:35:07 2020

@author: shaheed
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:42:52 2020

@author: mdshaheednazrul
"""

import time
import random
import PIL

from PIL import Image
import numpy as np



img = Image.open('C:/Users/17742/Desktop/GA_FINAL_PROJECT_MNAZRUL/gastly.png').convert('L')

np_img = np.array(img)
np_img = ~np_img  # invert B&W
np_img[np_img > 0] = 1
ref_shape = np_img.shape[0]
num_zeros = np.count_nonzero(np_img==0)
num_ones = np.count_nonzero(np_img==1)
total=num_zeros+num_ones

initial_population_size = int(1.15*ref_shape)#50
i_p = np.random.choice([0, 1], size=(initial_population_size,ref_shape,ref_shape), p=[(num_zeros/total), (num_ones/total)]) #initial population


ss=i_p.shape[0]
ss1=int(((ss+1)/2))
mutation_probability = 1/(ref_shape * ref_shape)
num_generation = 10000

counter = 0

maxVal = 0
maxChrom = []

#------------------------------------------------------------------------------



def fitness(initial_population):
    global maxChrom
    global maxVal
    max_scores=[]
    for i in initial_population:
        identical = i == np_img
        fitness_scores = identical.sum()
        max_scores.append(fitness_scores)
    
    if maxVal < max(max_scores):
        maxChrom = initial_population[max_scores.index(max(max_scores))]
        maxVal = max(max_scores)
    # print(max(max_scores))
    max_scores = np.asarray(max_scores)
    return max_scores


def first_parent_selection(initial_population, fitness_scores):  #will select half of the population as parents 
                                                                     #based on fitness and weight
       
        weight = []
        first_pool = []
        for scores in fitness_scores:
            w=((1-(scores/sum(fitness_scores)))*100000)-99000
            #w=((scores-300)/(sum(fitness_scores)-300*len(fitness_scores)))
            weight.append(w)
        #print(weight)
        
        
        for i in range(ss1):
            random_selection=random.choices(population=fitness_scores,weights=weight,k=ss1)
             
            selected_index=[]
            for i in random_selection:
                index=fitness_scores.index(i)
                selected_index.append(index)
            #print(selected_index)
            #print(len(selected_index))
                 
            for r in selected_index:
                par=initial_population[r]
                first_pool.append(par)
                
            
        first_pool = np.array(first_pool)
     
        return (first_pool)


def second_parent_selection(first_selection):
       
      
       second_batch_parents=[]
       fighters = []
    
       for i in range(ss1):#len(first_selection)):
           fighter_fitness = []
           fighters = []
           #inx1 = random.randint(0, len(first_selection)-1)
           #inx2 = random.randint(0, len(first_selection)-1)
          
           fighter_1 = random.choice(first_selection)
           fighter_2 = random.choice(first_selection)
           #fighter_3 = random.choice(first_selection)
           #fighter_4 = random.choice(first_selection)
      
           fighters.append(fighter_1)
           fighters.append(fighter_2)
           #fighters.append(fighter_3)
           #fighters.append(fighter_4)
           fighter_fitness = fitness(fighters)

           if fighter_fitness[0] > fighter_fitness[1]:
               winner = fighter_1
               #print("Index 1: " + str(inx1))
           else:
               winner = fighter_2
            
           second_batch_parents.append(winner)
       
       
       return second_batch_parents


 
def coitus_for_children(final_parents):
    
    children_gen=[]
    for i in range(int(len(final_parents))):
        f_parent = random.choice(final_parents)
        s_parent = random.choice(final_parents)
        #print('1st parent',f_parent)
        #print()
        #print('2nd parent',s_parent)
        chromosome_length = len(f_parent)
        crossover_point = random.randint(1,chromosome_length-1)
        #crossover_point = int((chromosome_length-1)/2)
        child = np.concatenate((f_parent[0:crossover_point],
                            s_parent[crossover_point:]))
        children_gen.append(child)
    #print(fitness(children_gen))
    return children_gen

def second_mistakes_akachildren(final_parents):
        
        second_child_selection = []
        
        for i in range(int(ss)):
            f_parent = random.choice(final_parents)
            s_parent = random.choice(final_parents)
            #print('1st parent',f_parent)
            #print()
            #print('2nd parent',s_parent)
            chromosome_length = len(f_parent)
            crossover_point = random.randint(1,chromosome_length-1)
            child = np.concatenate((f_parent[0:crossover_point],
                                s_parent[crossover_point:]))
            ss_child = np.where((child==0)|(child==1), child^1, child)
            
            second_child_selection.append(ss_child)
        return second_child_selection


def mutate_the_mistakes(mistakes, mutation_probability):
        
        random_mutation_array = np.random.random(
            size=(mistakes.shape))
        
        random_mutation_boolean = random_mutation_array <= mutation_probability
        mistakes[random_mutation_boolean] = np.logical_not(mistakes[random_mutation_boolean])
    
        return mistakes


start_time = time.time()

for i in range(num_generation):   

        
    fitness_scores = list(fitness(i_p))

    #first_selection = first_parent_selection(i_p, fitness_scores)   
    #first_selection = first_parent_selection(i_p, fitness_scores)
    #fitness(first_selection)
    #type(first_selection)
        
 
    final_parents = second_parent_selection(i_p)
    
    np.asarray(final_parents)
    
    final_parents_fitness = fitness(final_parents)
    max(final_parents_fitness)
 
    child_batch_uno = coitus_for_children(final_parents)
    child_batch_uno = np.asarray(child_batch_uno)
    #print(child_batch_uno[0])
    #fitness(child_batch_uno)
        
    child_batch_dos = coitus_for_children(final_parents)   
    #print(child_batch_dos)
    
    child_batch_dos = np.asarray(child_batch_dos)
    
    mutation1 = mutate_the_mistakes(child_batch_uno, mutation_probability)
    #print(mutation1[0])
    first_batch = np.concatenate((final_parents, mutation1))
    first_batch_fitness= fitness(mutation1)
    first_batch_idx = np.argmax(first_batch_fitness)
    
    
    mutation2 = mutate_the_mistakes(child_batch_dos, mutation_probability)
    #print(mutation2)
    second_batch = np.concatenate((final_parents, mutation2))
    second_batch_fitness= fitness(second_batch)
    second_batch_idx = np.argmax(second_batch_fitness)
    
        
    first_child_arr = mutation1[first_batch_idx]
    first_child_arr = maxChrom

    # print(len(first_child_arr))
    # print(len(first_child_arr[0]))
    # for arr in range(len(first_child_arr)):
    #     print(first_child_arr[arr])
    second_child_arr = second_batch[second_batch_idx]
    
    first_child_arr = first_child_arr.reshape(ref_shape, ref_shape)

    im1 = Image.fromarray(second_child_arr.astype('uint8')*255)
    im2 = Image.fromarray(second_child_arr.astype('uint8')*255)
    
    
    next_gen = []
    next_gen.append(first_batch)
    next_gen.append(second_batch)
    next_gen = np.array(next_gen)
    
    
    nxgnpop = random.choice(next_gen)
    nxgnpop = np.array(nxgnpop)
    
    counter+=1

    i_p = []

    i_p = nxgnpop

    i_p[np.argmin(nxgnpop)] = maxChrom
    # i_p[0] = maxChrom


    #print(len(i_p))
    #print(mutation_probability)
    
    #break

    if counter%50==0:
        print("---" + str(counter))
        print(maxVal)
    if counter%1000 ==0:
        im1.show()

    if maxVal == ref_shape*ref_shape:
        im1.show()
        print(counter)
        break
    # elif counter%250==0:
    #     im2.show()
        
        
    
end_time = time.time()

print('time elapsed' ,end_time - start_time)    
    
    
    









