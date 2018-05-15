# Load X and y variable
using JLD
data = load("logisticData.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
X = [ones(n,1) X]
d += 1
w = zeros(d,1)
lambda = 1
maxPasses = 500
progTol = 1e-4
verbose = true

## Run and time coordinate descent to minimize L2-regularization logistic loss

# Start timer
tic()

# Compute Lipschitz constant of 'f'
(eigenvalues,~) = eig(X'*X)
L = .25*maximum(eigenvalues) + lambda;

# Start running coordinate descent
w_old = copy(w);
for k in 1:maxPasses*d

    # Choose variable to update 'j'
    j = rand(1:d)

    # Compute partial derivative 'g_j'
    yXw = y.*(X*w);
    sigmoid = 1./(1+exp.(-yXw));
    g = -X'*(y.*(1-sigmoid)) + lambda*w;
    g_j = g[j];

    # Update variable
    w[j] -= (1/L)*g_j;

    # Check for lack of progress after each "pass"
    if mod(k,d) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 + exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        w_old = copy(w);
    end

end

# End timer
toc()
