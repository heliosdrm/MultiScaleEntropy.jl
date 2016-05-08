module MultiScaleEntropy

using RecurrenceAnalysis

local_rr(x::AbstractMatrix{Bool}) = [countnz(x[:,t]) for t in (1:size(x)[2])]

# Note: in this function x is assumed to be previously normalized and embedded, radius is absolute
function sampled_local_rr(x, radius, scale=1; kwargs...)
    n = div(size(x)[2], scale)
    recurrences = zeros(scale, n)
    # normalize = false by default
    !haskey(kwargs, :normalize) && kwargs[:normalize] = false
    for s = 1:scale
        xs = x[s:scale:end,:]
        rmat = recurrencematrix(xs, radius; kwargs...)
        recurrences[s,:] = local_rr(rmat)
    end
    recurrences
end

apen_factors(recurrences) = prod(recurrences)
sampen_factors(recurrences) = sum(recurrences)
comp_sampen_factors(recurrences) = prod(sum(recurrences,2))


function entropyfactors(x, m, delay)
    xe = embed(x/std(x), m, delay)
    xe_plus = [xe[1:end-delay,:] xe[1+delay:end,end]]
end

function approximateentropy(x, m, r)
    xe_plus = embed(x, m+1, 1)
    xe = [xe_plus[:,1:end-1]; xe_plus[end,2:end]]
    n = size(xe)[1]
    rmat = recurrencematrix(xe, std(xe)*r)
    c = sum(rmat, 1)/n
    apen = sum(log(c))/n
    rmat = recurrencematrix(xe_plus, std(xe_plus)*r)
    c = sum(rmat, 1)/(n-1)
    apen_plus = sum(log(c))/(n-1)
    apen - apen_plus
end

function sampleentropy_nd(x, m, r, delay=1)
    xe_plus = embed(x, m+1, delay)
    xe = [xe_plus[:,1:end-1]; xe_plus[end-delay+1:end,2:end]]
    n = size(xe)[1]
    rmat = recurrencematrix(xe, std(xe)*r)
    b = recurrencerate(rmat, theiler=1)*n*(n-1)
    rmat = recurrencematrix(xe_plus, std(xe_plus)*r)
    a = recurrencerate(rmat, theiler=1)*(n-delay)*(n-delay-1)
    (a, b)
end

function sampleentropy(args...)
    a, b = sampleentropy_nd(args...)
    -log(a/b)
end



# MSE with coarse-grained series
# MMSE with moving average

moving_average(x, ns) = filt(ones(ns)/ns, 1, x)[ns:end]
grained(x, ns) = moving_average(x, ns)[1:ns:end]

function mse(x, m, r, bounds; ent_type="SampEn")
    ent_fun = Dict("ApEn"=>approximateentropy, "SampEn"=>sampleentropy)[ent_type]
    ent = zeros(bounds[2]-bounds[1]+1)
    for s in range(bounds...)
        xs = grained(x, s)
        ent[s] = ent_fun(xs, m, r)
    end
    ent
end

function mmse(x, m, r, bounds)
    ent = zeros(bounds[2]-bounds[1]+1)
    for s in range(bounds...)
        xs = moving_average(x, s)
        ent[s] = sampleentropy(xs, m, r, s)
    end
    ent
end

function cmse(x, m, r, bounds)
    ent = zeros(bounds[2]-bounds[1]+1)
    for s in range(bounds...)
        for k = 1:s
            xs = grained(x[k:end], s)
            ent[s] += sampleentropy(xs, m, r) / s
        end
    end
    ent
end

function rcmse(x, m, r, bounds)
    ent = zeros(bounds[2]-bounds[1]+1)
    for s in range(bounds...)
        a = b = 0.
        for k = 1:s
            xs = grained(x[k:end], s)
            ak, bk = sampleentropy_nd(xs, m, r)
            a += ak
            b += bk
        end
        ent[s] = -log(a/b)
    end
    ent
end

# Loop over different scales, and calculate slope or area
