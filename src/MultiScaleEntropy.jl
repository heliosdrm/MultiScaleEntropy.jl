module MultiScaleEntropy

using RecurrenceAnalysis

function approximateentropy(x, m, r)
    xe_plus = embed(x, m+1, 1)
    xe = [xe_plus[:,1:end-1]; xe_plus[end,2:end]]
    n = size(xe)[1]
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
    n = size(xe)[1]
    rmat = recurrencematrix(xe, r)
    b = recurrencerate(rmat, theiler=1)*n*(n-1)
    rmat = recurrencematrix(xe_plus, r)
    a = recurrencerate(rmat, theiler=1)*(n-1)*(n-2)
    -log(a/b)
end

# MSE with coarse-grained series
# CompMSE with moving average

moving_average(x, ns) = filt(ones(ns)/ns, 1, x)[ns:end]
grained(x, ns) = moving_average(x, ns)[1:ns:end]

function mse(x, m, r, bounds; ent_type="ApEn", scale="grained")
    ent_fun = Dict("ApEn"=>approximateentropy, "SampEn"=>sampleentropy)
    scale_fun = Dict("grained"=>grained, "moving"=>moving_average)
    ent = zeros(bounds[2]-bounds[1]+1)
    for s in range(bounds...)
        xs = scale_fun(x, s)
        ent[s] = scale_fun(xs, m, r)
    end
    ent
end

# Loop over different scales, and calculate slope or area