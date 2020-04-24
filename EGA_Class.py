import numpy as np
# import Fitness_TSP
import matplotlib.pyplot as plt
import functools


def genInitPop(individuals, var, n_class):
    # This function create the first population
    tmp_pop = np.ndarray(shape=(individuals, 1), dtype=('U', var))
    for ind in range(individuals):
        gen = ""
        for v in range(var):
            gen += str(np.random.randint(0, n_class))
        tmp_pop[ind] = gen
    return tmp_pop


def annularCross(I_double, n, n_vars, Pc):
    # This function apply Deterministic Annular Crossover to the population
    I_tmp = np.copy(I_double)
    chromosome = n_vars
    for count in range(int(n / 2)):
        cross_prob = np.random.random(1)
        if cross_prob <= Pc:
            cut1 = np.random.randint(0, chromosome - 1)
            len_cut = np.random.randint(0, int(chromosome / 2))
            cut2 = (cut1 + len_cut) % chromosome
            tmp_str1 = I_double[count][0]
            tmp_str2 = I_double[n - count - 1][0]
            if (cut2 < cut1):
                cut2, cut1 = cut1, cut2
            tmp_str1, tmp_str2 = tmp_str1[0:cut1] + tmp_str2[cut1:cut2] + tmp_str1[cut2:], tmp_str2[0:cut1] + \
                                 tmp_str1[cut1:cut2] + tmp_str2[cut2:]
            I_tmp[count][0] = str(tmp_str1)
            I_tmp[n - count - 1][0] = str(tmp_str2)
    return I_tmp



def mutation(I_double, n, n_var, n_class,B2M):
    I_tmp = np.copy(I_double)
    for count in range(B2M):
        string = ''
        p1 = np.random.randint(0, n)
        p2 = np.random.randint(0, n_var)
        print("mutación: ({}, {})".format(p1, p2))
        tmp = list(I_tmp[p1][0])
        mut_comp = tmp[p2]
        tmp[p2] = np.random.randint(0, n_class)
        while mut_comp == tmp[p2]:
            tmp[p2] = np.random.randint(0, n_class)
        for i in tmp:
            print(type(string))

        #I_tmp[p1][0] = np.array(''.join(str(tmp)))

    return I_tmp



I = genInitPop(4, 10, 3)
I_double = np.concatenate((I, I), axis=0)
print("Antes: ")
print(I_double)
I_double = mutation(I_double, 4, 10, 3, 4)
print("Después: ")
print(I_double)

#
# ### Variables for flexibility of the algorithm
# # number of variables for one solution
# # in this case, we have the number of rows in the Dataset
# n_vars = 50
# # Number of classes in the Dataset
# n_class = 3
# ## Variables what EGA needs
# # Number of generations
# G = 2000
# # Number of individuals
# n = 100
# # Length of chromosome
# L = n_vars
# # Population
# I = np.ndarray(shape=(n, n_vars), dtype=np.int16)
# # Crossing probability
# Pc = 0.9
# # Mutation probability
# Pm = 0.05
# # list of fitness
# fitness = np.ndarray(shape=(2, n), dtype=float)
# # Expected number of mutations for each generation
# B2M = int(n * L * Pm)
#
# ### Auxiliary variables
# # Double of the population
# I_double = np.ndarray(shape=(2 * n, n_vars), dtype=np.int16)
# # double of the list of fitness
# fitness_double = np.ndarray(shape=(2, 2 * n), dtype=float)
#
# # Initial population
# I = genInitPop(n, n_vars)
#
# # Save a best of the all generation
# champ = np.ndarray(shape=(n, n_vars), dtype= np.int16)
# champ = champ[0]
# champ = list(range(n_vars))
# fit_champ = Fitness_TSP.fitness(champ)
#
# for gen in range(G):
#     # Double of length of the population
#     I_double = np.concatenate((I, I), axis=0)
#
#     # Apply Annular Crossover
#     I_double = crossover(I_double, n, n_vars, Pc)
#
#     # Apply Mutation
#     I_double = mutation(I_double, len(I_double), n_vars, B2M)
#
#     # Apply fitness
#     count = 0
#     for i in range(fitness_double.shape[1]):
#         fitness_double[0][i] = count
#         fitness_double[1][i] = Fitness_TSP.fitness(I_double[i])
#         count += 1
#     # Order by fitness
#     fitness_double = fitness_double[:, fitness_double[1].argsort()]
#
#
#     # Apply Elitism
#     ind_eli = fitness_double[0][0:n]
#     ind_eli = np.array(ind_eli, dtype=np.int16)
#     count = 0
#     for i in ind_eli:
#         I[count] = I_double[i]
#         count += 1
#     print("best way {}, Best Travel Cost= {}".format(list(I[0]), fitness_double[1][0]))
#
#     # we compare the best of this generation with the champ
#     if fitness_double[1][0] < fit_champ:
#         champ = I[0]
#         fit_champ = fitness_double[1][0]
#         print("nuevo champ")
#
# way = list(champ)
# way.append(way[0])
# print("Aproaches: ")
# print(list(way), fit_champ)
# print()
#
# points = Fitness_TSP.return_points(way)
#
# plt.plot(points[0], points[1])
# plt.title("Traveling Salesman Problem")
# plt.scatter(points[0], points[1], c='red')
# for inx, poi  in enumerate(I[0]):
#     plt.annotate(poi, (points[0][inx]+0.3, points[1][inx]+0.3))
# plt.show()
