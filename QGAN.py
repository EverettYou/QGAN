import itertools
import numpy
import scipy.sparse, scipy.sparse.linalg
import torch
# ================= MATH ===================
def tSVD(A, max_dim = None, tol = 1.E-3):
	# truncated SVD that split matrix A = U.V to U and V
	# given the max bond dimension max_dim and the relative tolerance tol.
	if max_dim is None: # if max_dim not specified
		max_dim = min(A.shape) # set it to the compact dimension
	# compute compact SVD
	U, s, V = numpy.linalg.svd(A, full_matrices = False)
	# determine truncation position
	if len(s) == 0: # if singular value list empty
		cut = 0 # cut at 0
	else: # normal case
		eps = s[0] * tol # estimate absolute tolerance based on s[0]
		cut = min(len(s), max_dim) # set initial cut position
		while abs(s[cut-1]) < eps: # if last element still in the zero zone
			cut -= 1 # decrease the cut by one
		# until last element out of the zero zone
	# make truncation according to the determined cut
	Sq = numpy.diag(numpy.sqrt(s[:cut]))
	U = numpy.dot(U[:, :cut], Sq)
	V = numpy.dot(Sq, V[:cut, :])
	return U, V, cut # return U, V and cut
def tenprod(vs):
	# normalized tensor product of a list of vectors
	# vs = [v1, v2, ...] is a list of vectors
	# return v1 x v2 x ... tensor
	if len(vs) == 0: # no vector input
		return numpy.array([]) # return empty array
	t = vs[0] # take the first vector
	for v in vs[1:]: # for rest of the vectors
		t = numpy.tensordot(t, v, axes = 0) # tensor to t one by one
	return t
def randvec(size):
	# sample a random unit vector
	v = numpy.random.normal(size = size)
	return v / numpy.linalg.norm(v)
def GS(H, method = 'dense'):
	# find a ground state of H, assuming H Hermitian
	if method == 'dense':
		if scipy.sparse.issparse(H):
			H = H.toarray() # convert to dense
		w, U = numpy.linalg.eigh(H)
		return U[:, numpy.argmin(w)]
	elif method == 'sparse':
		w, U = scipy.sparse.linalg.eigsh(H, k = 1, which = 'SA')
		return U[:, 0]
	else:
		raise ValueError('method must be "dense" or "sparse".')
# ================ PHYSICS =================
# Pauli operator class ---------------------
class Pauli:
	def __init__(self, Xs, Zs, L):
		self.Xs = frozenset(x%L for x in Xs) # a set of positions of X
		self.Zs = frozenset(z%L for z in Zs) # a set of positions of Z
		self.L = L # total length of Pauli string
		self.ipw = len(self.Xs & self.Zs) # number of Y's determines the power of i
		self.dim = 2**self.L # Hilbert space dimension
		self._mat = None # to hold matrix representation
	def __repr__(self):
		return 'σ' + repr(self.indices())
	def __eq__(self, other):
		return (self.Xs == other.Xs and self.Zs == other.Zs and self.L == other.L)
	def __gt__(self, other):
		return self.indices() > other.indices()
	def __lt__(self, other):
		return self.indices() < other.indices()
	def __hash__(self):
		return hash((self.Xs, self.Zs, self.L))
	def indices(self):
		Xs, Zs = self.Xs, self.Zs
		Ys = Xs & Zs
		ids = [0] * self.L
		for X in Xs - Ys:
			ids[X] = 1
		for Y in Ys:
			ids[Y] = 2
		for Z in Zs - Ys:
			ids[Z] = 3
		return ids
	def shift(self, d):
		Xs = frozenset((x + d) for x in self.Xs)
		Zs = frozenset((z + d) for z in self.Zs)
		return Pauli(Xs, Zs, self.L)
	def mat(self):
		# matrix representation of Pauli operator
		if self._mat is None:
			# encode the sets Xs and Zs into two binary integers x and z for fast inference of matrix indexes by bitwise operation
			x = numpy.sum(2**(self.L - 1 - numpy.array(list(self.Xs), dtype = int)))
			z = numpy.sum(2**(self.L - 1 - numpy.array(list(self.Zs), dtype = int)))
			# prepare holders for data, row indexes and column indexes
			data = []
			rows = []
			cols = []
			# collect array rules
			for i in range(self.dim):
				j = numpy.bitwise_xor(i, x)
				n = bin(numpy.bitwise_and(j, z)).count('1')
				data.append(1j**(self.ipw + 2 * n))
				rows.append(i)
				cols.append(j)
			# construct CSR matrix
			self._mat = scipy.sparse.csr_matrix((data, (rows, cols)), shape = (self.dim, self.dim))
		return self._mat
# Model class ------------------------------
# 1D chain of qubits
class Model:
	def __init__(self, L, loc = 2):
		self.L = L # length of the qubit chain
		self.loc = loc # locality of local operators
		self.ops = self.build_ops()
		self.opdim = len(self.ops) # operator space dimension
		self.dim = 2**self.L # Hilbert space dimension
	def __repr__(self):
		return '[Model %d sites, %d local operators]'%(self.L, self.dim)
	def build_ops(self):
		# build the set of local operators
		ks = list(range(min(self.loc, self.L)))
		ops = []
		for xs, zs in itertools.product(itertools.chain.from_iterable(itertools.combinations(ks, n) for n in range(self.loc + 1)), repeat = 2):
			Xs = frozenset(xs)
			Zs = frozenset(zs)
			if 0 in Xs|Zs:
				ops.append(Pauli(Xs, Zs, self.L))
		ops = sorted({op.shift(d) for d in range(self.L) for op in ops}) # local operators, use set construction to remove duplicated operators
		return ops
	def M(self, h):
		# construct the Hamiltonian
		H = scipy.sparse.csr_matrix(([],([],[])), shape = (self.dim, self.dim)) # create an empty Hamiltonian
		for hi, op in zip(h, self.ops):
			H += hi * op.mat() # add local operator to Hamiltonian with coefficient from vector h
		# find the ground state of H
		if H.shape[0] > 32: # for large matrix, use sparse method
			g = GS(H, method = 'sparse')
		else: # for small matrix, use dense method
			g = GS(H, method = 'dense')
		# disconnected correlation
		A = numpy.array(list(op.mat().dot(g) for op in self.ops))
		C = numpy.real(A.dot(numpy.conj(A.transpose())))
		# operator expectation value
		O = numpy.real(numpy.array(list(op.mat().dot(g).dot(numpy.conj(g)) for op in self.ops)))
		# return connected correlation
		return C - numpy.tensordot(O, O, axes = 0)
# ========== MPS related classes ===========
# MPS unit class ---------------------------
# tensor legs arrangement: (left, right, phys)
class MPSunit:
	def __init__(self, tensor):
		self.tensor = tensor
		self.left = None
		self.right = None
		self.left_dim, self.right_dim, self.phys_dim = tensor.shape
	def __repr__(self):
		return '[MPS unit %d, %d, %d]'%(self.left_dim, self.right_dim, self.phys_dim)
	def pin(self, phys_env):
		# pin the physical leg to phys_env
		self.phys_env = phys_env # keep the physical environment
		self.matrix = self.tensor.dot(phys_env)
		return self.matrix
	def fromright(self, right_env):
		self.right_env = right_env # keep input in right environment
		return self.matrix.dot(right_env)
	def fromleft(self, left_env):
		self.left_env = left_env # keep input in left environment
		return self.matrix.transpose().dot(left_env)
# MPS class --------------------------------
# specialized to the double ended MPS that looks like TTTTT
class MPS:
	def __init__(self, length, dim, max_dim = 8, lr = 0.01):
		self.length = length # MPS chain length
		# dim must match model's operator space dimension
		self.left_dim = dim
		self.right_dim = dim
		self.phys_dim = dim + 1
		self.max_dim = max_dim # max bond dimension
		self.leftend = None
		self.rightend = None
		self.assemble()
		self.P = torch.autograd.Variable(torch.zeros(dim, dim), requires_grad = True)
		self.optimizer = MPSAdam([self.P], lr = lr) # optimizer
	def __repr__(self):
		return '[MPS %d, %d, (%d)^%d]'%(self.left_dim, self.right_dim, self.phys_dim, self.length)
	def assemble(self):
		# set an default auxiliary dimension
		dim = min(max(self.left_dim, self.right_dim), self.max_dim)
		for i in range(self.length):
			if i == 0: # left end unit
				unit = MPSunit(numpy.empty((self.left_dim, dim, self.phys_dim)))
				self.leftend = unit
			elif i == self.length - 1: # right end unit
				unit.right = MPSunit(numpy.empty((dim, self.right_dim, self.phys_dim)))
				unit.right.left = unit
				unit = unit.right
			else: # bulk unit
				unit.right = MPSunit(numpy.empty((dim, dim, self.phys_dim)))
				unit.right.left = unit
				unit = unit.right
			if i == self.length - 1:
				self.rightend = unit
	def initialize(self, val):
		# initialize MPS such that when all physical legs are pinned to [1,0,0,...], the MPS is simply a two-way multiplier with multiplication factor = val
		unit = self.leftend # start from left
		while unit is not None:
			if unit.phys_dim != 0:
				unit.tensor[:,:,0] = val**(1/self.length) * numpy.eye(unit.left_dim, unit.right_dim) # set eye to the first physical slice
				unit.tensor[:,:,1:] = 0. # the rest physical slices are set to zero
			unit = unit.right # move to right
	def pin(self, phys_env):
		# broadcast phys_env to every tensor and pin their physical legs
		unit = self.leftend # start from left
		while unit is not None:
			unit.pin(phys_env) # pin physical environment
			unit = unit.right # move to right
	def fromright(self, right_env):
		# must pin the physical legs before calling me
		unit = self.rightend # start from right
		while unit is not None:
			right_env = unit.fromright(right_env) # get new right_env
			unit = unit.left # move to left
		return right_env # return the last right_env
	def fromleft(self, left_env):
		# must pin the physical legs before calling me
		unit = self.leftend  # start from left
		while unit is not None:
			left_env = unit.fromleft(left_env) # get new left_env
			unit = unit.right # move to right
		return left_env # return the last left_env
	def evaluate(self):
		# contracting the right leg of MPS to form a double MPS. With the physical legs pinned, returns the matrix representation supported on the left double legs
		unit = self.rightend # take the right most unit
		while unit is not None:
			if unit.right is None: # right most unit
				mat = unit.matrix # create a matrix holder
			else: # for the remaining units to the left
				mat = unit.matrix.dot(mat)
			if unit.left is not None:
				unit = unit.left # move to left
			else: # has reached the left most unit
				# construct density matrix and return
				return mat.dot(mat.transpose())
	def update(self, grad, k = 1):
		# consider contracting the right leg of MPS to form a double MPS. Let grad be the gradient supported on the doubled left legs. Update the MPS tensors by ascending this gradient, such that: double MPS += grad (approximately). This is done by probing the gradient matrix with k rounds of random signals and each signal leads to a small update of the order 1/k. By default, k = 1.
		for i in range(k):
			# sample a random vector with norm = sqrt(right_dim), s.t. when z is averaged over, the density matrix is an identity matrix
			z = numpy.sqrt(self.right_dim) * randvec(self.right_dim)
			x = self.fromright(z) # push z to the left
			y = grad.dot(x) # take gradient signal
			# now environments are prepared
			unit = self.leftend
			while unit is not None:
				if unit.left is None: # left most unit
					unit.left_env = y # set y to the left environment
					if unit.right is None: # the unit is both left and right end when the MPS length = 1
						self.siteupdate(unit, 2./k) # site update by 2/k
					else:
						self.siteupdate(unit, 1./k) # site update by 1/k
				elif unit.right is None: # right most unit
					self.bondupdate(unit.left, unit, 1./k)
					self.siteupdate(unit, 1./k)
				else: # bulk unit
					self.bondupdate(unit.left, unit, 1./k)
				unit = unit.right # move to right
	def siteupdate(self, unit, rate = 1.):
		# single site update
		unit.tensor += rate * tenprod([unit.left_env, unit.right_env, unit.phys_env]) # gradient descend with rate
		if unit.right is None: # for the right end unit
			self.relatent() # rearrange latent space
		else: # for bulk units
			unit.pin(unit.phys_env) # pin the physical leg again since the tensor has changed
			unit.right.left_env = unit.fromleft(unit.left_env) # update the left environment of its right unit
	def bondupdate(self, left_unit, right_unit, rate = 1.):
		# forming bond tensor
		bond_tensor = numpy.tensordot(left_unit.tensor, right_unit.tensor, axes = ((1),(0)))
		# gradient descend of bond tensor
		bond_tensor += rate * tenprod([left_unit.left_env, left_unit.phys_env, right_unit.right_env, right_unit.phys_env])
		# reshape bond tensor to matrix for SVD
		left_dim = left_unit.left_dim * left_unit.phys_dim
		right_dim = right_unit.right_dim * right_unit.phys_dim
		bond_matrix = numpy.reshape(bond_tensor, (left_dim, right_dim))
		# perform truncated SVD, return U, V and actual bond dimension
		U, V, dim = tSVD(bond_matrix, max_dim = self.max_dim)
		# set the bond dimension to neighboring units
		left_unit.right_dim = dim
		right_unit.left_dim = dim
		# update tensors
		shape = (left_unit.left_dim, left_unit.phys_dim, left_unit.right_dim)
		left_unit.tensor = numpy.swapaxes(numpy.reshape(U, shape), 1, 2)
		shape = (right_unit.left_dim, right_unit.right_dim, right_unit.phys_dim)
		right_unit.tensor = numpy.reshape(V, shape)
		# pins physical leg of the left unit
		left_unit.pin(left_unit.phys_env)
		# update left environment of right unit
		right_unit.left_env = left_unit.fromleft(left_unit.left_env)
	def relatent(self):
		# rearrange the latent space basis
		unit = self.rightend # take the right end unit
		# self contraction by right leg
		block_tensor = numpy.tensordot(unit.tensor, unit.tensor, axes = ((1),(1)))
		dim = unit.left_dim * unit.phys_dim # calculate total dimension
		block_matrix = numpy.reshape(block_tensor, (dim, dim)) # reshape to matrix for diagonalization
		w, U = numpy.linalg.eigh(block_matrix)
		# w will follow the ascending order
		# truncate to the right dimension (removing zero modes only)
		w = w[-unit.right_dim:]
		U = U[:, -unit.right_dim:]
		U = U.dot(numpy.diag(numpy.sqrt(w))) # rescaling
		# update the tensor
		shape = (unit.left_dim, unit.phys_dim, unit.right_dim)
		unit.tensor = numpy.swapaxes(numpy.reshape(U, shape), 1, 2)
		unit.pin(unit.phys_env) # pin the physical leg again as the tensor had changed
	def getP(self, h):
		# return density matrix P as torch Variable
		v = numpy.concatenate((numpy.ones(1), h)) # vector concatenation
		self.pin(v) # pin to physical legs
		out = self.evaluate() # evaluate density matrix
		self.P.data = torch.Tensor(out) # update to P
		return self.P
	def optimize(self, V):
		# one step of optimization of the value function V
		self.optimizer.zero_grad() # clear gradients
		V.backward() # back propagate gradient from value function
		self.optimizer.stepby() # one step optimization, only gradient updated
		grad = self.P.grad.data.numpy() # this is the amount of MPS update that the optimizer wants to make, we should use this signal to guide the MPS update
		self.update(grad) # update MPS tensors
# Adam optimizer for MPS -------------------
class MPSAdam(torch.optim.Adam):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	def stepby(self, closure = None):
		# Performs a single optimization step.
		# only updates gradient signal of the parameters 
		loss = None
		if closure is not None:
			loss  = closure()
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				state = self.state[p]
				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(p.data)
				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']
				state['step'] += 1
				if group['weight_decay'] != 0:
					grad = grad.add(group['weight_decay'], p.data)
				# Decay the first and second moment running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				denom = exp_avg_sq.sqrt().add_(group['eps'])
				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']
				step_size = group['lr'] * numpy.sqrt(bias_correction2) / bias_correction1
				p.grad.data = -step_size * exp_avg / denom
		return loss
# ========= ML related classes =============
# Generative Model -------------------------
class GM:
	def __init__(self):
		pass























