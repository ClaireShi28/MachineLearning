from sklearn.model_selection import train_test_split, GridSearchCV
import LoadData
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import time

def NeuralNetworkWeightOptimization():
    rs=2 #random_state
    cv=5

    lr = 0.015# 0.064
    nodes = [10,10,10]
    act = 'relu'
    maxiter = 500
    X, y = LoadData.load_wine_quality_data()
    #X, y = LoadData.load_ozone_data()
    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, test_size=0.3)

    #Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.fit_transform(X_test)

    #One hot encode y values
    one_hot=OneHotEncoder()

    y_train_hot=one_hot.fit_transform(y_train.reshape(-1,1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    #Initialize neural network object and fit object
    #fit_curves=[]
    #algorithms=['GC','GA', 'RHC','SA']



    #GD
    # GridSearchCV for NN
    params = {
        'hidden_layer_sizes': [(5, 2), (50,), (100,)],
        'activation': ['tanh', 'relu'],
        'learning_rate': [0.001, 0.005, 0.0001, 0.003, 0.01],
    }

    nn_model_gd = mlrose.NeuralNetwork(hidden_nodes=[10,10,10],activation='relu',algorithm='gradient_descent',
                                       max_iters=maxiter,bias=True, is_classifier=True,learning_rate=0.015,
                                       early_stopping=True, clip_max=5, max_attempts=100,
                                       random_state=rs,curve=True)
    nn_model_gd.fit(X_train_scaled, y_train_hot)

    plt.figure()
    plt.title("NN - GD fitness curve")
    plt.plot(nn_model_gd.fitness_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig("NN GD fitness curve")
    plt.show()

    #GA
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, algorithm='genetic_alg',
                                       max_iters=maxiter, bias=True, is_classifier=True, learning_rate=lr,
                                       early_stopping=True, clip_max=5, max_attempts=100,
                                       random_state=rs, curve=True)
    nn_model_ga.fit(X_train_scaled, y_train_hot)
    plt.figure()
    plt.title("NN - GA fitness curve")
    plt.plot(nn_model_ga.fitness_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig("NN GA fitness curve")
    plt.show()

    #RHC
    nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act,algorithm='random_hill_climb',
                                        max_iters=maxiter,bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        curve=True, random_state=rs)
    nn_model_rhc.fit(X_train_scaled, y_train_hot)
    plt.figure()
    plt.title("NN - RHC fitness curve")
    plt.plot(nn_model_rhc.fitness_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig("NN RHC fitness curve")
    plt.show()

    #SA
    nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=nodes,activation=act,algorithm='simulated_annealing',
                                       max_iters=maxiter,bias=True, is_classifier=True,learning_rate=lr,
                                       early_stopping=True, clip_max=5, max_attempts=100,
                                       curve=True,random_state=rs)
    nn_model_sa.fit(X_train_scaled, y_train_hot)
    plt.figure()
    plt.title("NN - SA fitness curve")
    plt.plot(nn_model_sa.fitness_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig("NN SA fitness curve")
    plt.show()

    plt.figure()
    plt.title("NN fitness vs iterations of 4 algorithms")
    plt.plot(nn_model_gd.fitness_curve, label='Gradient descent')
    plt.plot(nn_model_ga.fitness_curve, label="GA")
    plt.plot(nn_model_rhc.fitness_curve, label='RHC')
    plt.plot(nn_model_sa.fitness_curve, label='SA')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig("NN fitness vs iterations of 4 algorithms")
    plt.show()
    
    #Accuracy score
    nn_model=mlrose.NeuralNetwork(hidden_nodes=nodes,activation=act,
                                  algorithm='random_hill_climb', max_iters=maxiter,
                                  bias=True,is_classifier=True, learning_rate=lr,
                                  early_stopping=False,curve=True,
                                  random_state=rs)
    nn_model.fit(X_train_scaled,y_train_hot)

    #Once model fitted, can use it to predict the labels (y values) of training and test datasets and
    #use predictions to asses training and test accuracy
    y_train_pred=nn_model.predict(X_train_scaled)
    y_train_accuracy=accuracy_score(y_train_hot, y_train_pred)
    print ("Train Accuracy: ", y_train_accuracy)

    y_test_pred=nn_model.predict(X_test_scaled)
    y_test_accuracy=accuracy_score(y_test_hot, y_test_pred)
    print("Test Accuracy: ", y_test_accuracy)


    traintime=[]
    trainlabels=['RHC','Gradient descent','SA','GA']

    iterations = [10,50,100,250, 500, 1000, 2000, 3000, 4000, 5000]#, 500, 1000, 2500, 5000]
    iterations=[10,50,100,200,300,400,500,1000]
    iterations=[1000]
    #iterations = [i*10 for i in range(10000)]
    test_accuracy = []
    rhc_train_accuracy=[]
    rhc_cross_val=[]
    f1_train=[]
    f1_cv=[]
    rhc_test_accuracy=[]
    start=time.time()
    for max_i in iterations:
        nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act,max_iters=max_i,
                                        algorithm='random_hill_climb',
                                        bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=rs,curve=False)
        nn_model.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        rhc_test_accuracy.append(y_test_accuracy)
        score=cross_val_score(nn_model, X_test_scaled, y_test_hot, cv=5)


        y_train_pred = nn_model.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        rhc_train_accuracy.append(y_train_accuracy)

        #print(f1_score(y_train_hot, y_train_pred))
        #print(f1_score(y_test_hot,score))
        rhc_cross_val.append(score.mean())
    end = time.time()
    traintime.append((end - start))
    #np.savetxt('RHC',nn_model.fitness_curve)

    #Plot test accuracy vs iterations
    #ax=plt.plot()
    title = "NN - RHC score vs iterations"
    plt.figure()
    plt.title(title)
    #plt.plot(nn_model.fitness_curve, 'b-')
    #plt.plot(iterations,f1_train,'bo-', label='Training score')
    plt.plot(iterations,rhc_train_accuracy,'bo-',label='Accuracy score')
    plt.plot(iterations,rhc_cross_val,'ro-',label='Cross validation score')
    #plt.plot(iterations, f1_cv, 'go-', label='CV F1 score')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(title)
    plt.show()


    #GD GD GD

    gd_train_accuracy = []
    gd_cross_val = []
    gd_test_accuracy=[]
    start=time.time()
    for max_i in iterations:
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[50], activation='tanh', max_iters=max_i,
                                        algorithm='gradient_descent',
                                        bias=True, is_classifier=True, learning_rate=0.001,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=rs, curve=False)
        nn_model.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        gd_test_accuracy.append(y_test_accuracy)
        score = cross_val_score(nn_model, X_test_scaled, y_test_hot, cv=5)

        y_train_pred = nn_model.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        print("max_iters = ", max_i, ", Y Train Accuracy ", y_train_accuracy)
        gd_train_accuracy.append(y_train_accuracy)

        print("max_iters = ", max_i, ", CV Score ", score.mean())
        gd_cross_val.append(score.mean())

    # np.savetxt('RHC',nn_model.fitness_curve)
    print (gd_train_accuracy)
    print(gd_cross_val)
    end = time.time()
    traintime.append((end - start))
    # Plot test accuracy vs iterations
    # ax=plt.plot()
    title = "NN - Gradient descent score vs iterations"
    plt.figure()
    plt.title(title)
    # plt.plot(nn_model.fitness_curve, 'b-')
    #plt.plot(iterations, f1_train, 'bo-', label='Training score')
    plt.plot(iterations,gd_train_accuracy,'bo-',label='Accuracy score')
    plt.plot(iterations, gd_cross_val, 'ro-', label='Cross validation score')
    #plt.plot(iterations, f1_cv, 'go-', label='CV F1 score')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(title)
    plt.show()


    #SA SA SA
    sa_train_accuracy = []
    sa_cross_val = []
    start=time.time()
    sa_test_accuracy=[]
    for max_i in iterations:
        nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=max_i,
                                        algorithm='simulated_annealing',
                                        bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=rs, curve=False)
        nn_model.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        sa_test_accuracy.append(y_test_accuracy)
        score = cross_val_score(nn_model, X_test_scaled, y_test_hot, cv=5)

        y_train_pred = nn_model.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        sa_train_accuracy.append(y_train_accuracy)

        #print(f1_score(y_train_hot, y_train_pred))
        #print(f1_score(y_test_hot, score))
        sa_cross_val.append(score.mean())
    end = time.time()
    traintime.append((end - start))
    # np.savetxt('RHC',nn_model.fitness_curve)

    # Plot test accuracy vs iterations
    # ax=plt.plot()
    title = "NN - SA score vs iterations"
    plt.figure()
    plt.title(title)
    # plt.plot(nn_model.fitness_curve, 'b-')
    #plt.plot(iterations, f1_train, 'bo-', label='Training score')
    plt.plot(iterations,sa_train_accuracy,'bo-',label='Accuracy score')
    plt.plot(iterations, sa_cross_val, 'ro-', label='Cross validation score')
    #plt.plot(iterations, f1_cv, 'go-', label='CV F1 score')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(title)
    plt.show()


    #GA GA GA

    ga_train_accuracy = []
    ga_cross_val = []
    start=time.time()
    ga_test_accuracy=[]
    for max_i in iterations:
        nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=max_i,
                                        algorithm='genetic_alg',
                                        bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=rs, curve=False)
        nn_model.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        ga_test_accuracy.append(y_test_accuracy)
        score = cross_val_score(nn_model, X_test_scaled, y_test_hot, cv=5)

        y_train_pred = nn_model.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        ga_train_accuracy.append(y_train_accuracy)

        ga_cross_val.append(score.mean())
    end=time.time()
    traintime.append((end-start))
    # np.savetxt('RHC',nn_model.fitness_curve)

    # Plot test accuracy vs iterations
    # ax=plt.plot()
    title = "NN - GA score vs iterations"
    plt.figure()
    plt.title(title)
    # plt.plot(nn_model.fitness_curve, 'b-')
    #plt.plot(iterations, f1_train, 'bo-', label='Training score')
    plt.plot(iterations,ga_train_accuracy,'bo-',label='Accuracy')
    plt.plot(iterations, ga_cross_val, 'ro-', label='Cross validation score')
    #plt.plot(iterations, f1_cv, 'go-', label='CV F1 score')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(title)
    plt.show()

    #gd_cross_val=[0.5583333333333333, 0.7166666666666666, 0.7208333333333333, 0.7291666666666666, 0.7270833333333334, 0.7312500000000001, 0.73125, 0.7229166666666667, 0.7145833333333333, 0.7125]
    plt.figure()
    plt.title("NN - score vs iterations")
    plt.plot(iterations, ga_cross_val, 'o-', label='GA')
    plt.plot(iterations, gd_cross_val, 'o-', label='Gradient descent')
    plt.plot(iterations, rhc_cross_val, 'o-',label='RHC')
    plt.plot(iterations, sa_cross_val, 'o-',label='SA')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig("NN - score vs iterations")
    plt.show()

    CV=[]
    CV.append(ga_cross_val[0])
    CV.append(gd_cross_val[0])
    CV.append(sa_cross_val[0])
    CV.append(rhc_cross_val[0])

    Acc=[]
    Acc.append(ga_test_accuracy[0])
    Acc.append(gd_test_accuracy[0])
    Acc.append(sa_test_accuracy[0])
    Acc.append(rhc_test_accuracy[0])
    ##Accuracy comparison
    plt.figure()
    plt.title("NN training accuracy score comparison")
    plt.bar(trainlabels, CV, color='green')
    plt.ylabel('Accuracy score')
    plt.tight_layout()
    i = 0
    for a in trainlabels:
        plt.text(a, CV[i], '%.2f' % CV[i], ha='center', va='bottom', fontsize=11)
        i += 1
    plt.savefig("NN train score comparison")
    plt.show()

    # Plot Acc
    plt.figure()
    plt.title("NN testing accuracy score comparison")
    plt.bar(trainlabels, Acc, color='orange')
    plt.ylabel('Accuracy score')
    plt.tight_layout()
    i = 0
    for a in trainlabels:
        plt.text(a, Acc[i], '%.2f' % Acc[i], ha='center', va='bottom', fontsize=11)
        i += 1
    plt.savefig("NN test score comparison")
    plt.show()

    plt.figure()
    plt.title("NN running time comparison")
    plt.bar(trainlabels, traintime)
    plt.ylabel('Time (s)')
    plt.tight_layout()
    i = 0
    for a in trainlabels:
        plt.text(a, traintime[i] + 0.05, '%.2f' % traintime[i], ha='center', va='bottom', fontsize=11)
        i += 1
    plt.savefig("NN running time comparison")
    plt.show()


NeuralNetworkWeightOptimization()
#model = mlrose.NeuralNetwork(hidden_nodes = , activation ='tanh', etc.,)
#learning_curve(estimator=model, X, y, cv, train_sizes, return_times=True, scoring='accuracy')
