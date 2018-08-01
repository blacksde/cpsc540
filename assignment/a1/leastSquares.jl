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

    c = zeros(d+1+n)
	c[d+2:end] = 1.0

	A = [-Z eye(n);Z eye(n)]
	da = [-y[:];y[:]]
	b = fill(Inf, 2*n)

    lb = fill(-Inf, d+1+n)
	ub = fill(Inf, d+1+n)

	solution = linprog(c,A, da,b,lb,ub,GLPKSolverLP())
	w = solution.sol[1:d+1]

	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

    return LinearModel(predict,w)

end


function lesatmax(X, y)

	(n, d) = size(X)
	Z = [ones(n,1) X]

    c = zeros(d+1+1)
	c[d+2:end] = 1.0

	A = [-Z ones(n,1);Z ones(n,1)]
	da = [-y[:];y[:]]
	b = fill(Inf, 2*n)

    lb = fill(-Inf, d+1+1)
	ub = fill(Inf, d+1+1)

	solution = linprog(c,A, da,b,lb,ub,GLPKSolverLP())
	w = solution.sol[1:d+1]

	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

    return LinearModel(predict,w)

end
