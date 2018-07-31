include("misc.jl")
using MathProgBase, GLPKMathProgInterface

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function lesatAbsolutes(X, y)

	(n, d) = size(X)
	Z = [ones(n,1) X]

    c = zeros(d+1+n,1)
	c[d+1:end] = 1

	A = [-Z eye(n);Z eye(n)]
	d = [-y[:];y[:]]
	b = fill(Inf, d+1+n)

    lb = fill(-Inf, d+1+n)
	ub = fill(Inf, d+1+n)

	solution = linprog(c,A,d,b,lb,ub,GLPKSolverLP())
	w = solution.sol[1:d+1]

	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

    return LinearModel(predict,w)

end
