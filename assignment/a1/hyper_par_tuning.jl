# Load X and y variable
using JLD
data = load("/Users/jianguo/Documents/cpsc540/assignment/a1/nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d)    = size(X)
splitval = Int64(n/2)

Xtrain = X[1:splitval,:]
ytrain = y[1:splitval]
Xval   = X[(splitval+1):end,:]
yval   = y[(splitval+1):end]

# Fit least squares model
include("/Users/jianguo/Documents/cpsc540/assignment/a1/leastSquaresRBFL2.jl")

max_err    = Inf
bestSigma  = []
bestLambda = []

for lambda in 2.0.^(-5:5)
    for sigma in 2.0.^(-5:5)
        model = leastSquaresRBFL2(Xtrain, ytrain, sigma, lambda)

        # Report the error on the test set
        t = size(Xval,1)
        yhat = model.predict(Xval)
        valError = sum((yhat - yval).^2)/t
        @printf("with sigma = %.3f and lambda = %.3f valError = %.2f\n",sigma, lambda, valError)

        if valError < max_err
            max_err   = valError
            bestSigma = sigma
            bestLambda= lambda
        end
    end
end

model = leastSquaresRBFL2(X, y, bestSigma, bestLambda)


# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
