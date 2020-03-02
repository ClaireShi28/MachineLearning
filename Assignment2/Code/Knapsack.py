import mlrose_hiive as mlrose
from matplotlib import pyplot as plt
import random

# Knapsack problem
def Knapsack():
    rs=2 #random state
    ma=200 #max attempts

    items = 40  # number of items
    random.seed(6)

    weights=[]
    values=[]
    for i in range(0,items):
        weights.append((random.random()+0.1)*30)
        #weights.append(random.randint(1,31))
        #values.append(random.randint(1, 500))
        values.append((random.random()+0.1)*500)

    #weights=[9,13,153,50,15,68,27,39,23,52,11,32,24,48,73,42,43,22,7,18,4,30,153,50,15,68,68,27,27,39]
    #values=[150,35,200,60,60,45,60,40,30,10,70,30,15,10,40,70,75,80,20,12,50,10,200,60,60,45,45,60,60,40]
    #print(len(weights))
    #print(weights)

    max_weight_pct = 0.6
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    problem_fit = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True)

    # Fitness curve
    alltime = []
    import time
    start = time.time()
    best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, pop_size=300, mutation_prob=0.7, curve=True,
                                                                   max_attempts=ma,
                                                                   random_state=rs)
    end = time.time()
    alltime.append((end - start))

    start=time.time()
    best_state, best_fitness, rhcfitness_curve = mlrose.random_hill_climb(problem_fit, curve=True,max_attempts=ma,
                                                                          random_state=rs)
    end=time.time()
    alltime.append((end-start))

    start = time.time()
    SA_schedule = mlrose.GeomDecay(init_temp=100000, decay=0.95, min_temp=1)
    best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit,
                                                                           schedule=SA_schedule, curve=True,max_attempts=ma,
                                                                           random_state=rs)
    end = time.time()
    alltime.append((end - start))

    start = time.time()
    best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit, curve=True,max_attempts=ma,
                                                                pop_size=400, keep_pct=0.3,
                                                                random_state=rs)
    end = time.time()
    alltime.append((end - start))

    # Plot time comparison
    plt.figure()
    algorithms = ['GA', 'RHC', 'SA', 'MIMIC']
    plt.bar(algorithms, alltime)
    plt.title("Running time for Knapsack problem (seconds)")
    plt.ylabel('Time (s)')
    plt.xlabel('Random search algorithms')
    plt.tight_layout()
    i = 0
    for a in algorithms:
        plt.text(a, alltime[i] + 0.05, '%.2f' % alltime[i], ha='center', va='bottom', fontsize=11)
        i += 1
    plt.savefig("Running time for Knapsack problem")
    plt.show()

    plt.title("Knapsack problem fitness vs iterations")
    plt.plot(gafitness_curve, label='GA', color='r')
    plt.plot(rhcfitness_curve, label='RHC', color='b')
    plt.plot(safitness_curve, label='SA', color='orange')
    plt.plot(mimicfitness_curve, label='MIMIC', color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack fitness curve")
    plt.show()


    # MIMIC Fitness vs Iterations as cpt changes
    CPT = [0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure()
    for c in CPT:
        best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit, keep_pct=c, curve=True,max_attempts=ma,
                                                                    random_state=rs)
        plt.plot(mimicfitness_curve, label='pct = ' + str(c))

    plt.title("Knapsack problem using MIMIC with different values of pct parameter")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack MIMIC parameter")
    plt.show()

    # GA Fitness vs Iterations as mutation prob changes
    Mutate = [0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure()
    for m in Mutate:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, mutation_prob=m, curve=True,
                                                                       max_attempts=ma, random_state=rs)
        plt.plot(gafitness_curve, label='mutation = ' + str(m))

    plt.title("Knapsack problem using GA with  different values of mutation probability")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack GA parameter")
    plt.show()
    
    # SA Fitness vs Iterations as schedule changes
    # schedule = mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1)

    init_temp = 1.0
    decay_r = [0.15, 0.35, 0.55, 0.75, 0.95]
    plt.figure()
    for d in decay_r:
        SAschedule = mlrose.GeomDecay(init_temp=100000, decay=d, min_temp=1)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True,max_attempts=ma,
                                                                               random_state=rs)
        plt.plot(safitness_curve, label='decay rate = ' + str(d))

    plt.title("Knapsack problem using SA with different values of decay rate")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack SA parameter")
    plt.show()
    
    
    init_temp = 1.0
    temps = [100000, 10000, 1000, 100, 10, 5]
    plt.figure()
    for t in temps:
        SAschedule = mlrose.GeomDecay(init_temp=t,  decay=0.95, min_temp=1)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True, max_attempts=ma,
                                                                               random_state=rs)
        plt.plot(safitness_curve, label='Temperature = ' + str(t))

    plt.title("Knapsack problem using SA with different values of temperature")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack SA temp")
    plt.show()
    
    
    Mutate = [0.1, 0.1, 0.1, 0.1, 0.1]
    pop = [50, 100, 200, 300, 400]
    Mutatepop = [(100, 0.2), (100, 0.5), (100, 0.7), (200, 0.2), (200, 0.5),
                 (200, 0.7), (300, 0.2), (300, 0.5), (300, 0.7)]
    plt.figure()
    for m in Mutatepop:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, pop_size=m[0],
                                                                       mutation_prob=m[1], curve=True,
                                                                       max_attempts=ma,
                                                                       random_state=rs)
        plt.plot(gafitness_curve, label='pop size = ' + str(m[0]) + ', mutation = ' + str(m[1]))

    plt.title("Knapsack using GA with  different parameters")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack GA parameter mutate pop")
    plt.show()
    
    temps = [10000000, 1000000, 100000, 10000, 1000, 100, 10, 5]
    plt.figure()
    for t in temps:
        SAschedule = mlrose.GeomDecay(init_temp=t, decay=0.95, min_temp=1)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True, max_attempts=ma,
                                                                               random_state=rs)
        plt.plot(safitness_curve, label='Temperature = ' + str(t))

    plt.title("Knapsack problem using SA with different values of temperature")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack SA temp")
    plt.show()
    
    CPT = [0.1, 0.3, 0.5, 0.9]
    pp=[(100,0.2), (100, 0.5), (100, 0.7), (100, 0.9),
        (200, 0.2), (200, 0.5), (200, 0.7), (200, 0.9),
        (500, 0.2), (500, 0.5), (500, 0.7), (500, 0.9)]
    plt.figure()
    Pop=[100,200,300,400,500]
    for p in Pop:
        best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit,
                                                                    pop_size=p,
                                                                    keep_pct=0.3, curve=True,
                                                                    max_attempts=ma,
                                                                    random_state=rs)
        plt.plot(mimicfitness_curve, label='pop size = ' + str(p))


    plt.title("Knapsack problem using MIMIC with different parameters")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Knapsack MIMIC parameter pop pct")
    plt.show()

Knapsack()