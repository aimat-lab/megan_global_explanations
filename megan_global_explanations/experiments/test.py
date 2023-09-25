import numpy as np
import matplotlib.pyplot as plt


N_POINTS = 100
MU = 1.5
STD = .25

def prob(x, mu=0., std=1.):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-.5*((x-mu)/std)**2)


def entropy(samples, mu=0., std=1.):
    h = 0
    for x in samples:
        p = prob(x, mu, std)
        h += -p*np.log10(p)
    return h


def study_entropy(population):
    cluster_low = np.array([sample for sample in population if sample<=0])
    cluster_high = np.array([sample for sample in population if sample>0])
    entropy_one_cluster = entropy(population)
    entropy_two_clusters = entropy(cluster_low, mu=-MU, std=STD) + entropy(cluster_high, mu=MU, std=STD)
    print(f'1 cluster: {entropy_one_cluster}, 2 clusters: {entropy_two_clusters}')
    print(f'info gain: {entropy_one_cluster-entropy_two_clusters}')


def plot(population):
    plt.scatter(population, np.zeros_like(population))
    plt.show()
    plt.close()


print('ground truth = 1 cluster')
population = np.random.randn(N_POINTS)
plot(population)
study_entropy(population)


print('ground truth = 2 clusters')
pop1 = MU + np.random.randn(N_POINTS//2)*STD
pop2 = -MU + np.random.randn(N_POINTS//2)*STD
population = np.array([sample for sample in pop1]+[sample for sample in pop2])
plot(population)
study_entropy(population)
