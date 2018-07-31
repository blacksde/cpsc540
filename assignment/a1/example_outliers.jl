# Load X and y variable
using JLD
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least absolutes model
include("leastSquares.jl")
model = lesatAbsolutes(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(abs(yhat - y))
@printf("Squared train Error with least absolutes: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(abs(yhat - ytest))
@printf("Squared test Error with least absolutes: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
