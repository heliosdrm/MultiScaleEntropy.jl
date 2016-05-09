module MultiScaleEntropy

using RecurrenceAnalysis

# Local recurrences (excluding self-recurrence)
local_rr(x::AbstractMatrix{Bool}) = [countnz(x[:,t])-1 for t in (1:size(x)[2])]

# Note: in this function x is assumed to be previously normalized and embedded, radius is absolute
function sampled_local_rr(x, r, scale=1, delayed=false)
    n = div(size(x)[1], scale)
    !delayed && rem(size(x)[1], scale) > 0 && (n += 1)
    recurrences = zeros(n, delayed?scale:1)
    for s = 1:size(recurrences)[2]
        xs = x[s:scale:scale*n,:]
        rmat = recurrencematrix(xs, r, normalize=false)
        recurrences[:,s] = local_rr(rmat)
    end
    recurrences
end

function full_local_rr(x, radius, scale)
    rmat = recurrencematrix(x, radius, normalize=false)
    recurrences = hcat(local_rr(rmat))
end

# Auxiliary functions
moving_average(x, ns) = filt(ones(ns)/ns, 1, x)[ns:end]
arithmean(x) = mean(x)
geommean(x) = prod(x)^(1/length(x))
mixmean(x) = geommean(mean(x, 1))

function multiscaleentropy(x, m, r, bounds; filtf=moving_average, normbyscale=false, rcount="simple", avg=arithmean)
    # Select methods
    countrecurrences = Dict(
        "simple"    => sampled_local_rr,
        "full"      => full_local_rr,
        "composite" => (x,r,s)->sampled_local_rr(x,r,s,true)
    )[rcount]
    ent = zeros(bounds[2]-bounds[1]+1)
    # Normalize signal
    x ./= std(x)
    for s in range(bounds...)
        # 1. Filter signal by scale
        xf = filtf(x, s)
        normbyscale && (xf ./=std(xf))
        # 2. Embed (and trim) for m and m+1
        xe = embed(xf, m, s)
        xe_plus = [xe[1:end-s,:] xe[1+s:end,end]]
        xe = xe[1:end-s,:]
        # 3. Count recurrences
        rc = countrecurrences(xe, r, s)
        rc_plus = countrecurrences(xe_plus, r, s)
        # 4. Calculate factors of entropy
        # (add self-recurrence if using geometric mean)
        (avg == geommean) && (rc += 1; rc_plus += 1)
        a = avg(rc_plus)
        b = avg(rc)
        ent[s] = -log(a/b)
    end
    ent
end


# To delete ...

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
    xe = xe[1:end-delay,:]
    n = size(xe)[1]
    rmat = recurrencematrix(xe, r, normalize=false)
    b = recurrencerate(rmat, theiler=1)*(n-delay)*(n-delay-1)
    rmat = recurrencematrix(xe_plus, r, normalize=false)
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
