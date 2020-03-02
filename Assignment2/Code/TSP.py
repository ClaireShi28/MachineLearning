import mlrose_hiive as mlrose
from matplotlib import pyplot as plt
import random
import itertools

#TSP problem
def TSP(loop=False):
    rs=1 #set random state
    ma=200 #max_attempts
    # Create list of city coordinates
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

    #Generate 20 cities coordinate randomely (0-99)

    cities=30 #10 cities random seed 60
    random.seed(6)
    #Generate 20 unique cities coordinates randomly
    random_list = list(itertools.product(range(0, 99), range(0, 99)))
    coords_list=random.sample(random_list, cities)
    print (coords_list)


    print (len(coords_list))
    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords = coords_list)

    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness_coords, maximize=True)

    # Solve problem using the genetic algorithm
    best_state, best_fitness = mlrose.genetic_alg(problem=problem_fit, mutation_prob=0.2,max_attempts=200, random_state=rs)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness = mlrose.random_hill_climb(problem_fit, random_state=rs)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness = mlrose.simulated_annealing(problem_fit, random_state=rs)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness = mlrose.mimic(problem_fit, keep_pct=0.2, random_state=rs)

    print(best_state)
    print(best_fitness)

    #iterations=[i*2 for i in range(0,50)]
    iterations=[5,10,30,50,100,250,500,1000,2000,3000]
    GA_fitness=[]
    RHC_fitness=[]
    SA_fitness=[]
    MIMIC_fitness=[]

    GA_time=[]
    RHC_time=[]
    SA_time=[]
    MIMIC_time=[]

    if loop==True:

        for max_it in iterations:
            import time
            start_time=time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem=problem_fit, pop_size=200, mutation_prob=0.2,
                                                          max_iters=1000, random_state=rs)

            end_time = time.time()
            time = end_time - start_time
            GA_time.append(time)
            GA_fitness.append(1/best_fitness)

            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_iters=max_it, random_state=rs, max_attempts=ma)
            end_time = time.time()
            time = end_time - start_time
            RHC_time.append(time)
            RHC_fitness.append(1/best_fitness)

            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem_fit, max_iters=max_it, random_state=rs, max_attempts=ma)
            end_time = time.time()
            time = end_time - start_time
            SA_time.append(time)
            SA_fitness.append(1/best_fitness)

            start_time = time.time()
            best_state, best_fitness = mlrose.mimic(problem_fit, keep_pct=0.1, max_iters=max_it,
                                                    random_state=rs, max_attempts=ma)
            end_time = time.time()
            time = end_time - start_time
            MIMIC_time.append(time)
            MIMIC_fitness.append(1/best_fitness)


        # plt.plot(nn_model.fitness_curve, 'b-')
        # plt.plot(iterations,train_accuracy,'bo-')
        # plt.plot(iterations,test_accuracy,'bo-',label='Accuracy')
        plt.title("GA Fitness loop")
        plt.plot(iterations, GA_fitness, 'r-', label='GA')
        plt.plot(iterations, RHC_fitness, 'b-', label='RHC')
        plt.plot(iterations, SA_fitness, '-', color='orange', label='SA')
        plt.plot(iterations, MIMIC_fitness, 'g-', label='MIMIC')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig("GA Fitness loop")
        plt.show()

        #Time
        plt.title("GA time loop")
        plt.plot(iterations, GA_time, 'r-', label='GA')
        plt.plot(iterations, RHC_time, 'b-', label='RHC')
        plt.plot(iterations, SA_time, '-', color='orange', label='SA')
        plt.plot(iterations, MIMIC_time, 'g-', label='MIMIC')
        plt.xlabel('Iterations')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.savefig("GA time loop")
        plt.show()

    timecompare=[]
    #Fitness curve
    import time
    start_time = time.time()
    best_state, best_fitness, gafitness_curve=mlrose.genetic_alg(problem=problem_fit, pop_size=200, mutation_prob=0.5, curve=True,
                                                                 max_attempts=ma, random_state=rs)
    end_time = time.time()
    timecompare.append((end_time-start_time))

    start_time = time.time()
    best_state, best_fitness, rhcfitness_curve = mlrose.random_hill_climb(problem=problem_fit,  curve=True,
                                                                          max_attempts=ma,
                                                                          random_state=rs)
    end_time = time.time()
    timecompare.append((end_time - start_time))


    start_time = time.time()
    best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem=problem_fit, curve=True,
                                                                           max_attempts=ma,
                                                                           random_state=rs)
    end_time = time.time()
    timecompare.append((end_time - start_time))

    start_time = time.time()
    best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem=problem_fit,pop_size=200, keep_pct=0.5, curve=True,
                                                                max_attempts=ma, random_state=rs)
    end_time = time.time()
    timecompare.append((end_time - start_time))

    plt.figure()
    plt.title("TSP fitness vs iterations using 4 random algorithm")
    plt.plot(gafitness_curve, label='GA',color='r')
    plt.plot(rhcfitness_curve, label='RHC',color='b')
    plt.plot(safitness_curve, label='SA',color='orange')
    plt.plot(mimicfitness_curve, label='MIMIC',color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("TSP fitness curve")
    plt.show()

    # Plot time comparison
    plt.figure()
    algorithms=['GA', 'RHC', 'SA', 'MIMIC']
    plt.bar(algorithms, timecompare)
    plt.title("Running time for TSP (seconds)")
    plt.ylabel('Time (s)')
    plt.xlabel('Random search algorithms')
    # plt.ylim(bottom=0,top=1.1)
    # plt.plot(train_size, score, 'o-', label='score')
    plt.tight_layout()
    i=0
    for a in algorithms:
        plt.text(a, timecompare[i]+0.05, '%.2f' % timecompare[i], ha='center', va= 'bottom',fontsize=11)
        i+=1
    plt.savefig("Running time for TSP")
    plt.show()
    
    #MIMIC Fitness vs Iterations as cpt changes
    CPT=[0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure()
    for c in CPT:
        best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit, keep_pct=c, curve=True,
                                                                    max_attempts=ma,
                                                                    random_state=rs)
        plt.plot(mimicfitness_curve, label='pct = '+ str(c))

    plt.title("TSP using MIMIC with different values of pct parameter")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("MIMIC parameter")
    plt.show()

    #GA Fitness vs Iterations as mutation prob changes
    Mutate=[0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure()
    for m in Mutate:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, mutation_prob=m, curve=True,
                                                                       max_attempts=ma,
                                                                       random_state=rs)
        plt.plot(gafitness_curve, label='mutation = ' + str(m))

    plt.title("TSP using GA with  different values of mutation probability")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("GA parameter")
    plt.show()

    #SA Fitness vs Iterations as schedule changes
    #schedule = mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1)

    init_temp=1.0
    decay_r=[0.15, 0.35, 0.55, 0.75, 0.95]
    plt.figure()
    for d in decay_r:
        SAschedule = mlrose.GeomDecay(init_temp=10, decay=d, min_temp=1)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               max_attempts=ma,
                                                                               curve=True, random_state=rs)
        plt.plot(safitness_curve, label='decay rate = ' + str(d))

    plt.title("TSP using SA with different values of decay rate")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("SA parameter")
    plt.show()

    temps = [100000, 10000, 1000, 100, 10, 5]
    plt.figure()
    for t in temps:
        SAschedule = mlrose.GeomDecay(init_temp=t, decay=0.55, min_temp=1)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True, max_attempts=ma,
                                                                               random_state=rs)
        plt.plot(safitness_curve, label='Temperature = ' + str(t))

    plt.title("TSP using SA with different values of temperature")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("TSP SA temp")
    plt.show()

    Mutate = [0.1, 0.1, 0.1, 0.1, 0.1]
    pop=[50,100,200,300,400]
    Mutatepop=[(100,0.2),(100,0.5), (100,0.7), (200,0.2), (200, 0.5),
               (200,0.7), (300, 0.2),(300,0.5), (300,0.7)]
    plt.figure()
    for m in Mutatepop:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit,pop_size=m[0],
                                                                       mutation_prob=m[1], curve=True,
                                                                       max_attempts=ma,
                                                                       random_state=rs)
        plt.plot(gafitness_curve, label='pop size = '+ str(m[0])+', mutation = ' + str(m[1]))

    plt.title("TSP using GA with  different parameters")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("GA parameter mutate pop")
    plt.show()

TSP(loop=False)