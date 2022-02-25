import tensorflow as tf
import tensorflow.keras as tfk


'''
Based on https://github.com/dawenl/vae_cf
'''
def MLP(dims, dropout_rate = 0.0):
	"""
	Input:
	dims (list): The architecture of MLP, e.g., [100, 50, 10].

	Example:
	mlp = MLP([50, 10, 25, 10], 0.5)
	x = tf.random.uniform([32, 50])
	mlp(x, training=True)
	"""
	if len(dims) == 2:
		return tfk.layers.Dense(dims[1], input_shape = (dims[0],))

	#assert len(dims) >= 3, "The number of layer should at least be 3"

	model = tfk.Sequential()
	model.add(tfk.layers.Dense(dims[1], input_shape = (dims[0],), activation = 'tanh'))
	if dropout_rate > 0: model.add(tfk.layers.Dropout(dropout_rate))
	for dim in dims[2:-1]:
		model.add(tfk.layers.Dense(dim, activation = 'tanh'))
		if dropout_rate > 0: model.add(tfk.layers.Dropout(dropout_rate))
	model.add(tfk.layers.Dense(dims[-1]))
	return model


def Linear(dims, activation = False):
	if activation:
		return tfk.layers.Dense(dims[1], input_shape = (dims[0],), activation = 'tanh')
	else:
		return tfk.layers.Dense(dims[1], input_shape = (dims[0],))



class MultiVAE(tfk.Model):
	"""
	vae = MultiVAE([200, 600, 2000])
	x = tf.random.uniform([32, 2000])
	logits, KLD = vae(x, training=True)
	"""
	def __init__(self, p_dims, dropout_rate = 0.5):

		super(MultiVAE, self).__init__()

		self.p_dims = p_dims
		self.q_dims = p_dims[1:][::-1] + [2 * p_dims[0]]
		self.dropout_rate = dropout_rate

		self.q = MLP(self.q_dims)
		self.p = MLP(self.p_dims)


	def call(self, x, training = False):

		h = tf.nn.l2_normalize(x, axis = 1)
		if training:
			h = tf.nn.dropout(h, rate = self.dropout_rate)

		## inference
		o = self.q(h, training = training)
		μ, logσ2 = tf.split(o, 2, axis = 1)

		## KL Divergence
		σ = tf.exp(0.5 * logσ2)
		KLD = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logσ2 - σ**2 - μ**2, axis = 1))
		z = self.reparameterize(μ, logσ2, training)

		## recognition
		logits = self.p(z, training = training)

		return logits, KLD


	def reparameterize(self, μ, logσ2, training):
		if training:
			σ = tf.exp(0.5 * logσ2)
			ϵ = tf.random.normal(σ.shape, dtype = σ.dtype)
			return μ + ϵ * σ
		else:
			return μ


