include("misc.jl")

function leastSquaresRBFL2(X,y, sigma, lambda)

	# Add bias column
	n = size(X,1)
	Z = rbf(X, X, sigma)

	# Find regression weights minimizing squared error
	w = (Z'*Z+lambda*eye(n))\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = rbf(Xtilde, X, sigma)*w

	# Return model
	return LinearModel(predict,w)
end

function rbf(Xtilde, X, sigma)

	dist = distancesSquared(Xtilde, X)

	return sqrt(1/(2*pi*sigma^2))*exp.(-dist/2/sigma^2)

end
