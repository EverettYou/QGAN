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
	# compute SVD
	U, s, V = torch.svd(A)
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
	s = s[:cut]
	U = U[:, :cut]
	V = V[:, :cut]
	return U, s, V
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
def rexp(M):
	w, U = numpy.linalg.eigh(M)
	w = numpy.exp(- w)
	w /= numpy.sum(w)
	return U.dot(numpy.diag(w)).dot(U.transpose())
def reln(P):
	w, U = numpy.linalg.eigh(P)
	w = - numpy.log(w)
	w -= numpy.min(w)
	return U.dot(numpy.diag(w)).dot(U.transpose())
def f(x):
	# f(x) = arctanh(x)/x
	if abs(x) < 0.01:
		x2 = x**2
		return (4*x2-15)/(9*x2-15)
	else:
		return numpy.arctanh(x)/x
f = numpy.vectorize(f)
def B(p):
	l = len(p)
	pL = numpy.tile(p,(l,1))
	pR = pL.transpose()
	pM = (pL + pR)/2
	eps = (pL - pR)/pM/2
	return f(eps)/pM
class Xent(torch.autograd.Function):
	# Xent(P0, P1) = - Tr P0.ln(P1)
	def forward(self, P0, P1):
		self.p1, self.V = torch.symeig(P1, eigenvectors = True)
		self.lnp1 = torch.log(self.p1)
		xent = - torch.sum(torch.sum(self.V * P0.mm(self.V), 0)*self.lnp1, 0)
		self.save_for_backward(P0, P1)
		return xent
	def backward(self, grad):
		P0, P1 = self.saved_tensors
		dP0 = self.V.mm(torch.diag(self.lnp1)).mm(self.V.t())
		dP1 = self.V.t().mm(P0).mm(self.V)
		dP1.mul_(torch.Tensor(B(self.p1.numpy())))
		dP1 = self.V.mm(dP1).mm(self.V.t())
		return (-grad * dP0, -grad * dP1)
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
		return '[Model %d sites, %d local operators]'%(self.L, self.opdim)
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
	def sample(self):
		# draw a random sample
		h = randvec(self.opdim)
		M = self.M(h)
		return (h, M)
# ========== MPS related classes ===========
# MPS unit class ---------------------------
class MPSunit:
	def __init__(self, dim):
		self.dim = dim # dim of operator space
		sqrdim = numpy.sqrt(dim) # square root dimension
		# T tensor - auxiliary space operator
		# s vector - singular values
		# E matrix - physical space encoder
		self.T = torch.eye(dim).div_(sqrdim).expand(1, -1, -1).permute(1,2,0)
		self.s = torch.ones(1).mul_(sqrdim)
		self.E = torch.zeros(dim + 1, 1)
		self.E[0,0] = 1.
		# create a Variable for optimization
		self.mat = torch.autograd.Variable(torch.eye(dim), requires_grad = True)
		self.push(torch.zeros(dim)) # make an initial push
	def __repr__(self):
		return '[MPS unit internal dim %d]'%len(self.s)
	def push(self, h):
		# vector concatenation
		self.v = torch.cat((torch.ones(1), h))
		# push v through, resulting in mat0
		self.mat0 = self.T.matmul(self.s * self.E.t().mv(self.v))
		# clone mat0 to mat
		self.mat.data = self.mat0.clone()
		return self.mat
	def update(self):
		dmat = self.mat.data - self.mat0 # get difference of mat
		dvec = dmat.view(self.dim**2) # vector form of dmat
		TMat = self.T.view(self.dim**2, -1) # reshape T tensor
		TExt = torch.cat((TMat, dvec.expand(1, -1).t()), dim = 1) # + dvec
		QT, RT = torch.qr(TExt) # QR for T
		vn = self.v/torch.sum(self.v**2)
		EExt = torch.cat((self.E, vn.expand(1, -1).t()), dim = 1) # E extend
		QE, RE = torch.qr(EExt) # QR for E
		S = torch.diag(torch.cat((self.s, torch.ones(1)))) # S mat
		U, self.s, V = tSVD(RE.mm(S).mm(RT.t()))
		self.T = QT.mm(V).view(self.dim, self.dim, -1)
		self.E = QE.mm(U)
# MPS class --------------------------------
class MPS:
	def __init__(self, depth, dim):
		self.depth = depth # MPS chain depth (length)
		self.dim = dim # dim must match operator space dimension
		self.units = [MPSunit(dim) for i in range(depth)]
		self.params = [unit.mat for unit in self.units]
	def __repr__(self):
		return '[MPS depth %d, dim %d]'%(self.depth, self.dim)
	def mul(self, val):
		# multiply P matrix by val
		for unit in self.units:
			unit.s.mul_(val**(0.5/self.depth))
	def P(self, h):
		# prepare an identity matrix A
		A = torch.autograd.Variable(torch.eye(self.dim))
		for unit in self.units:
			A = A.mm(unit.push(h)) # right multiply by MPS matrices
		P = A.mm(A.t()) # construct P matrix
		return P
	def M(self, h):
		P = self.P(h).data.numpy()
		return reln(P)
	def update(self):
		# update every MPS unit
		for unit in self.units:
			unit.update()
# ========= ML related classes =============
# Generative Model -------------------------
class GM:
	def __init__(self, L, depth, loss_fn = 'MSE', lr = 0.002):
		self.model = Model(L)
		self.dim = self.model.opdim
		self.generator = MPS(depth, self.dim)
		self.generator.mul(numpy.sqrt(1./self.dim))
		self.optimizer = torch.optim.SGD(self.generator.params, lr)
		self.loss_fn = loss_fn
	def train(self, k = 8):
		h, M = self.model.sample() # draw a (h, M) pair
		h = torch.Tensor(h)
		PM = torch.autograd.Variable(torch.Tensor(rexp(M))) # data P
		for i in range(k): # k = number of gradient descents
			PG = self.generator.P(h) # model P
			loss = self.loss(PM, PG) # get loss function
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		self.generator.update()
		return numpy.asscalar(loss.data.numpy())
	def test(self, size = 8):
		res = {'h fidelity': 0., 'M deviation': 0.}
		for i in range(size):
			h, M = self.model.sample() # draw a (h, M) pair
			h = torch.Tensor(h)
			MG = self.generator.M(h)
			res['h fidelity'] += abs(GS(MG).dot(h))
			res['M deviation'] += numpy.sum((M - MG)**2)/self.dim**2
		return {name: val/size for name, val in res.items()}
	def loss(self, PM, PG):
		if self.loss_fn == 'MSE':
			return torch.dist(PM, PG)
		elif self.loss_fn == 'KLD':
			# create instances of cross entropy function
			xent = Xent()
			# for KLD, regularization is needed to ensure Tr PG = 1
			return xent(PM, PG) + (torch.trace(PG)-1)**2
		elif self.loss_fn == 'JSD':
			# create instances of cross entropy function
			xent1 = Xent()
			xent2 = Xent()
			xent3 = Xent()
			PA = (PM + PG)/2
			return xent1(PA, PA) - (xent2(PM, PM) + xent3(PG, PG))/2
		
























