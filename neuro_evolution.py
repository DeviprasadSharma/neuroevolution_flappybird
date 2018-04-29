import math
import numpy as np
from copy import deepcopy

POPULATION = 50
SIZE = [5,25,2]

def sigmoid(z):
	f = lambda x: 0. if x < -500 else 1.0 if x >500 else 1.0/(1.0 + math.exp(-x))
	if isinstance(z,(float,int)):
		return f(z)
	else:
		return np.array(list(map(f,z)))

class NeuroNetwork():
	SIZE = [5,25,2]
	def __init__(self,size = SIZE):
		self.num_layers = len(size)
		self.size = size
		# without input layer
		#self.biases = [np.random.randn(y) for y in size[1:]]
		self.weights = [np.random.randn(y,x)
						for x,y in zip(size[:-1],size[1:])]

	def feedforward(self, a):
		''' a is the input, could be the position of pipes'''
		a = a.copy()
		for w in self.weights:
			a = sigmoid(np.dot(w,a))
		return a

class individual():
	''' an individual is a neruo network'''
	def __init__(self,score,netweights):
		self.score = score
		self.netweights = netweights

class Generation():
	def __init__(self):
		self.individuals = []
		self.mutation_rate = 0.05
		self.elitism = 0.2
		self.population = POPULATION
		self.random_behavior = 0.1

	def add(self,new):
		'''new is a individual, add it into list with order'''
		index = 0
		for i,nn in enumerate(self.individuals):
			# sort from high to low
			if new.score > nn.score:
				index = i
				break
		self.individuals.insert(index,new)

	def breed(self,ind1,ind2):
		childweights = deepcopy(ind1.netweights)
		weight_n = [] # e.g [6,6] for a [2,3,2] network
		for w in ind2.netweights:
			weight_n.append(w.size)
		for i,n in zip(range(len(ind2.netweights)),weight_n):
			for j in range(n):
				if np.random.random()<0.5:
					childweights[i].flat[j] = ind2.netweights[i].flat[j]
				# mutation
				if np.random.random()<self.mutation_rate:
					childweights[i].flat[j] += np.random.random() * 4 - 2
		return childweights

	def generate_next(self):
		nextgen = [] # a list of weights
		for i in range(round(self.elitism*self.population)):
			if len(nextgen) < self.population:
				nextgen.append(self.individuals[i].netweights)
		# random
		for i in range(round(self.random_behavior*self.population)):
			if len(nextgen) < self.population:
				nextgen.append([np.random.randn(y,x)
						for x,y in zip(SIZE[:-1],SIZE[1:])])
		# breed
		p = np.array([x.score for x in self.individuals])
		p = p - np.min(p)
		p = p/sum(p)
		while len(nextgen) < self.population:
			i,j = np.random.choice(range(len(self.individuals)),2,replace = False,p=p)
			child = self.breed(self.individuals[i],self.individuals[j])
			nextgen.append(child)
		return nextgen

class Generations():
	def __init__(self):
		self.generations = []

	def first_generation(self):
		outweight = []
		out = []
		for i in range(POPULATION):
			nn = NeuroNetwork()
			out.append(nn)
			outweight.append(nn.weights)
		self.generations.append(Generation())
		return out,outweight

	def next_generation(self):
		if len(self.generations) == 0:
			return False
		gen = self.generations[-1].generate_next()
		self.generations.append(Generation())
		return gen

	def add_indi(self,indi):
		if len(self.generations) == 0:
			return False
		return self.generations[-1].add(indi)



class NeuroEvolution():
	def __init__(self):
		self.gene = Generations()

	def restart(self):
		self.gene = Generations()

	def next_generation(self):
		networks = []
		if len(self.gene.generations) == 0:
			nn,networks = self.gene.first_generation()
		else:
			networks = self.gene.next_generation()
			nn = []
			for item in networks:
				n = NeuroNetwork()
				n.weights = item
				nn.append(n)
		historic = 1 # number of generations keeps in gene
		if historic != 0:
			L = len(self.gene.generations)
			if L > historic:
				self.gene.generations = self.gene.generations[-historic:]
		return nn

	def network_score(self,score,nn):
		self.gene.add_indi(individual(score,nn.weights))











