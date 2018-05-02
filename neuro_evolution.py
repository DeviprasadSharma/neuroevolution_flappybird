import math
import numpy as np
from copy import deepcopy
import re

POPULATION = 50
SIZE = [3,9,1]

def sigmoid(z):
	f = lambda x: 0. if x < -500 else 1.0 if x >500 else 1.0/(1.0 + math.exp(-x))
	if isinstance(z,(float,int)):
		return f(z)
	else:
		return np.array(list(map(f,z)))

class NeuroNetwork():
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
		self.mutation_rate = 0.1
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
					childweights[i].flat[j] += np.random.random() - 0.5
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
		#p = np.array([x.score for x in self.individuals])
		#p = p / sum(p) if sum(p)!= 0 else None
		#while len(nextgen) < self.population:
		#	i,j = np.random.choice(range(len(self.individuals)),2,replace = False,p=p)
		#	child = self.breed(self.individuals[i],self.individuals[j])
		#	nextgen.append(child)
		max_n = 0
		while True:
			for i in range(max_n):
				child = self.breed(self.individuals[i],self.individuals[max_n])
				nextgen.append(child)
				if len(nextgen) == self.population:
					return nextgen
			max_n += 1
			if max_n > len(self.individuals):
				max_n = 0

	def output(self,path='gene.txt'):
		with open(path,'a') as f:
			for x in self.individuals:
				f.write(str(x.score)+',')

class Generations():
	def __init__(self):
		self.generations = []

	@staticmethod
	def readlog(path):
		data = []
		try:
			with open(path,'r') as f:
				w = f.read()
			p1 = r'\[array\(\[([^\)]*)\]\)'
			a = re.findall(p1,w)
			p2 = r', array\(\[([^\)]*)\]\)'
			b = re.findall(p2,w)
			p3 = r'\[([^\]]*)\]'
			tofloat = lambda x: list(map(float,x.split(',')))
			pipeline = lambda x: list(map(tofloat,re.findall(p3,x)))
			a = list(map(pipeline,a))
			b = list(map(pipeline,b))
			for i,j in zip(a,b):
				data.append([np.array(i),np.array(j)])
			return data
		except IOError:
			return data

	def first_generation(self):
		outweight = Generations.readlog('lastgeneration.txt')
		out = []
		for w in outweight:
			nn = NeuroNetwork()
			nn.weights = w
			out.append(nn)
		while len(out) < POPULATION:
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

	def output(self):
		'''wirte the last generation into file'''
		self.generations[-1].output()




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

	def output(self):
		self.gene.output()











