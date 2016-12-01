module MultiScaleEntropy

using RecurrenceAnalysis

export multiscaleentropy,
       moving_average,
       arithmean, geommean, mixmean

# Local recurrences (excluding self-recurrence)
local_rr(x::AbstractMatrix{Bool}, exclude=1) = [countnz(x[1:t-exclude,t+exclude:end,t]) for t in (1:size(x)[2])]

# Note: in this function x is assumed to be previously normalized and embedded, radius is absolute
function sampled_local_rr(x, r, scale=1, delayed=false)
    n = div(size(x)[1], scale)
    if !delayed && rem(size(x)[1], scale) > 0
        n += 1
    end
    recurrences = zeros(n, delayed?scale:1)
    for s = 1:size(recurrences)[2]
        xs = x[s:scale:scale*n,:]
        rmat = recurrencematrix(xs, r, scale=1)
        recurrences[:,s] = local_rr(rmat)
    end
    recurrences
end

function full_local_rr(x, radius, scale)
    rmat = recurrencematrix(x, radius, scale=1)
    recurrences = hcat(local_rr(rmat,1))
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

end
