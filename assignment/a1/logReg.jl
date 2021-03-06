include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 + exp.(-yXw)))
	g = -X'*(y./(1+exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] = -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end

function softmaxclassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	print(size(W))

	W[:] = findMin(funObj,W[:],maxIter = 500, derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end


function softmaxObj(w,X,y,k)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W # dim n * k

	Z = sum(exp.(XW),2) # dim n

	f = 0
	g = zeros(d,k)

	for i in 1:n

		f += -XW[i, y[i]]+log(Z[i])

		p = exp.(XW[i,:])/Z[i]

		for c in 1:k

			g[:,c]+= X[i,:]*(p[c]-(y[i]==c))

		end


	end

	G = reshape(g,d*k,1)

	return (f,G)
end
