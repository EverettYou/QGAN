import numpy
import torch
# ================= math ===================
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
# ========== MPS related classes ===========
# MPS unit -------------------------------
# tensor legs arranged as (left, right, phys)
class MPSunit:
	def __init__(self, tensor):
		self.tensor = tensor
		self.left_unit = None
		self.right_unit = None
		(self.left_dim, self.right_dim, self.phys_dim) = tensor.shape
	def __repr__(self):
		return '[MPS unit %d]'%self.index
	def pin_phys(self, phys_env):
		# pin the physical leg to phys_env environment vector
		self.phys_env = phys_env # keep the physical environment
		self.matrix = self.tensor.dot(phys_env)
	def forward(self, input):
		self.right_env = input # keep input in right environment
		if self.left_unit is not None:
			# matrix product input and push result to the left unit
			return self.left_unit.forward(self.matrix.dot(input))
		else: # left-most unit, return the result
			return self.matrix.dot(input)
	def backward(self, grad):
		if self.left_unit is None: # left-most unit
			if self.right_unit is None: # if also right-most unit
				self.site_update(grad)
				self.pin_phys(self.phys_env) # pin physical legs again because tensor has been updated
			else: # if right unit exist
				self.right_unit.backward(self.site_update(grad/2))
		elif self.right_unit is None: # right-most unit
			self.site_update(self.block_update(grad))
			self.pin_phys(self.phys_env) # pin physical legs again because tensor has been updated
		else: # bulk tensors
			self.right_unit.backward(self.block_update(grad))
	def site_update(self, grad):
		# gradient descend
		self.tensor += tenprod([grad, self.right_env, self.phys_env])
		return grad # propagate gradient backward to the right
	def block_update(self, grad):
		# forming two-site block tensor
		block_tensor = numpy.tensordot(self.left_unit.tensor, self.tensor, axes = ((1),(0)))
		# gradient descent of the block tensor
		block_tensor += tenprod([grad, self.left_unit.phys_env, self.right_env, self.phys_env])
		# reshape block tensor to matrix for SVD
		left_dim = self.left_unit.left_dim * self.left_unit.phys_dim
		right_dim = self.right_dim * self.phys_dim
		block_matrix = numpy.reshape(block_tensor, (left_dim, right_dim))
		# perform truncated SVD, return U.V and the actual bond dimension
		U, V, dim = tSVD(block_matrix, max_dim = self.max_dim)
		# set the bond dimension to the neighboring units
		self.left_unit.right_dim = dim
		self.left_dim = dim
		# update the tensor of the left unit
		shape = (self.left_unit.left_dim, self.left_unit.phys_dim, self.left_unit.right_dim)
		self.left_unit.tensor = numpy.swapaxes(numpy.reshape(U, shape), 1, 2)
		# update the tensor of the current unit
		shape = (self.left_dim, self.right_dim, self.phys_dim)
		self.tensor = numpy.reshape(V, shape)
		# pin physical leg again to calculate the new grad
		self.left_unit.pin_phys(self.left_unit.phys_env)
		# push gradient to the left leg of the current unit
		new_grad = grad.dot(self.left_unit.matrix)
		return new_grad # propagate gradient backward to the right
# MPS --------------------------------------
# specialized to the double-headed MPS that looks like: TTTTT
class MPS:
	def __init__(self, length = 1, left_dim = 2, right_dim = 2,  phys_dim = 2, max_dim = 64):
		self.length = length # length of MPS chain
		self.left_dim = left_dim # dimension of left boundary leg
		self.right_dim = right_dim # dimension of right boundary leg
		self.phys_dim = phys_dim # dimension of physical legs
		self.max_dim = max_dim # max_dim: max auxiliary bond dimension
		self.assemble() # assemble MPS chain
	def __repr__(self):
		return '[MPS %d * (%d)^%d * %d]'%(self.left_dim, self.phys_dim, self.length, self.right_dim)
	def assemble(self):
		self.units = [] # prepare a container for MPS units
		for i in range(self.length):
			# specify leg dimensions
			if i == 0:
				left_dim = self.left_dim
			else:
				left_dim = 2
			if i == self.length - 1:
				right_dim = self.right_dim
			else:
				right_dim = 2
			phys_dim = self.phys_dim
			# create a MPS unit (random initialized)
			new_unit = MPSunit(numpy.random.normal(size = (left_dim, right_dim, phys_dim)))
			# index the unit for representation purpose 
			new_unit.index  = self.length - i
			# broadcast max_dim to each unit
			new_unit.max_dim = self.max_dim
			self.units.append(new_unit) # add unit to units
		# chain up the units
		for i in range(self.length - 1):
			self.units[i].right_unit = self.units[i + 1]
		for i in range(1, self.length):
			self.units[i].left_unit = self.units[i - 1]
	def relatent(self):
		# rearrange the latent space basis
		unit = self.units[-1] # take the last unit
		# self contraction by right leg (indexed 1)
		block_tensor = numpy.tensordot(unit.tensor, unit.tensor, axes = ((1), (1)))
		dim = unit.left_dim * unit.phys_dim # left and physical dimensions
		# reshape to matrix for diagonalize
		block_matrix = numpy.reshape(block_tensor, (dim, dim))
		w, U = numpy.linalg.eigh(block_matrix)
		# w will follow the ascending order
		# truncate to right dimension (removing zero modes only)
		w = w[-unit.right_dim:]
		U = U[:,-unit.right_dim:]
		U = U.dot(numpy.diag(numpy.sqrt(w)))
		# reconstruct the tensor
		shape = (unit.left_dim, unit.phys_dim, unit.right_dim)
		# update tensor
		unit.tensor = numpy.swapaxes(numpy.reshape(U, shape), 1, 2)
	def pin_phys(self, phys_env):
		# broadcast phys_env to every tensor and pin their physical legs
		for unit in self.units:
			unit.pin_phys(phys_env)
	def forward(self, input):
		# must pin the physical legs before calling me
		# take input from right and get output from left
		return self.units[-1].forward(input)
	def backward(self, grad):
		# must pin the physical legs before calling me
		# take grad from left and update the MPS to the right
		# in the meanwhile all MPS tensors are updated
		self.units[0].backward(grad)








