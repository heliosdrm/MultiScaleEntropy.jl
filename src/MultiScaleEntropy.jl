module MultiScaleEntropy

using RecurrenceAnalysis

function approximateentropy(x, m, r)
    xe_plus = embed(x, m+1, 1)
    xe = [xe_plus[:,1:end-1]; xe_plus[end,2:end]]
    n = size(xe)
    rmat = recurrencematrix(xe, r)
    c = sum(rmat, 1)/n
    apen = sum(log(c))/n
    rmat = recurrencematrix(xe_plus, r)
    c = sum(rmat, 1)/(n-1)
    apen_plus = sum(log(c))/(n-1)
    apen - apen_plus
end

function sampleentropy(x, m, r)
    xe_plus = embed(x, m+1, 1)
    xe = [xe_plus[:,1:end-1]; xe_plus[end,2:end]]
    n = size(xe)
    rmat = recurrencematrix(xe, r)
    b = recurrencerate(rmat, theiler=1)*n*(n-1)
    rmat = recurrencematrix(xe_plus, r)
    a = recurrencerate(rmat)*(n-1)*(n-2)
    -log(a/b)
end

# MSE with coarse-grained series
# CompMSE with moving average

# Loop over different scales, and calculate slope or area