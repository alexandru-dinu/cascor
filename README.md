# Cascor

	start with simple FC (784 + 1 x 10) network

	repeat
		train until no improvement can be made
		save the state of weights up to this point
		generate a set of hidden units candidates (current: 1)

		for each candidate c
			maximize the correlation between c's input weights and network error

		insert the hidden unit with the maximum correlation
    	freeze its input weights
    	add a column of size (1, 10) to the output weights matrix
	until there is no significant improvement
