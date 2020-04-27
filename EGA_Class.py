import numpy as np
import Fitness_VNND as vnnd

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
        tmp = list(I_tmp[p1][0])
        mut_comp = tmp[p2]
        tmp[p2] = np.random.randint(0, n_class)
        while str(mut_comp) == str(tmp[p2]):
            tmp[p2] = np.random.randint(0, n_class)
        for k in range(len(tmp)):
            string += str(tmp[k])
        I_tmp[p1][0] = string
    return I_tmp






### Variables for flexibility of the algorithm
# number of variables for one solution
# in this case, we have the number of rows in the Dataset
n_vars = 150
# Number of classes in the Dataset
n_class = 3
## Variables what EGA needs
# Number of generations
G = 1300
# Number of individuals
n = 100
# Length of chromosome
L = n_vars
# Population
I = np.ndarray(shape=(n, n_vars), dtype=np.int16)
# Crossing probability
Pc = 0.9
# Mutation probability
Pm = 0.05
# list of fitness
fitness = np.ndarray(shape=(2, n), dtype=float)
# Expected number of mutations for each generation
B2M = int(n * L * Pm)

### Auxiliary variables
# Double of the population
I_double = np.ndarray(shape=(2 * n, n_vars), dtype=np.int16)
# double of the list of fitness
fitness_double = np.ndarray(shape=(2, 2 * n), dtype=float)

# Initial population
I = genInitPop(n, n_vars, n_class)



for gen in range(G):
    # Double of length of the population
    I_double = np.concatenate((I, I), axis=0)

    # Apply Annular Crossover
    I_double = annularCross(I_double, n, n_vars, Pc)

    # Apply Mutation
    I_double = mutation(I_double, n, n_vars, n_class, B2M)

    # Apply fitness
    count = 0
    for i in range(fitness_double.shape[1]):
        fitness_double[0][i] = count
        fitness_double[1][i] = vnnd.fitness(I_double[i][0])
        count += 1
    # Order by fitness
    fitness_double = fitness_double[:, fitness_double[1].argsort()]

    # Apply Elitism
    ind_eli = fitness_double[0][0:n]
    ind_eli = np.array(ind_eli, dtype=np.int16)
    count = 0
    for i in ind_eli:
        I[count] = I_double[i]
        count += 1
    print("VNND= {}".format(fitness_double[1][0]))

print("Aproaches: ")
print(fitness_double[1][0])
vnnd.plot_cluster(I[0][0])

