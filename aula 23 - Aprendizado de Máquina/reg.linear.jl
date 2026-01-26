using DataFrames
using Statistics
using StatsBase

f_objetivo(x::Number) = (2 + 20 * x - 25 * x^2 + 0.25 * x^3)/datasize

datasize = 100
data = DataFrame(:x => rand(datasize)*datasize)

data[:, :y] = [(f_objetivo(x) + randn()*25) for x in data[!, :x]]

data[:, :x]

sort!(data, :x)

function encontrarparâmetrosestimador(data::AbstractDataFrame, grau=1)
    X = ones((size(data, 1), grau + 1))

    if (grau > 0)
        for i in 1:grau
            X[:, i+1] = data[:, :x] .^ i
        end
    end
    Y = data[:, :y]

    Θ_hat = inv(X' * X) * X' * Y
end

function estimar(x::Number, parâmetros)
    y = 0
    for i in eachindex(parâmetros)
        y += parâmetros[i] * x^(i - 1)
    end
    y
end

function treinar(data, tam_treino, dim_max)
    @assert tam_treino < size(data, 1)

    rand_data = data[sample(axes(data, 1), datasize), :]

    treino = rand_data[1:tam_treino, :]
    teste = rand_data[tam_treino + 1:end, :]

    dim = 0
    result = []
    while dim <= dim_max
        parâmetros = encontrarparâmetrosestimador(treino, dim)
        estimativas = [estimar(x, parâmetros) for x in teste[:, :x]]

        erro_médio = mean(abs.(estimativas .- teste[:, :y]))
        push!(result, (dim, erro_médio, parâmetros))

        dim += 1
    end

    result

end

resultado = treinar(data, 90, 6)

using GLMakie

fig = Figure(size = (600, 600));
ax = Axis(fig[1, 1], title = "Regressão Linear", limits = (0, datasize, min(min(data.y...), 0), max(max(data.y...), datasize))
)

scatter!(ax, data.x, data.y, color = :limegreen, marker = :cross, label = "Dados")
lines!(ax, 0:datasize, f_objetivo.(0:datasize), color = (:tomato, 0.25), label = L"f(̇.)")

for i in eachindex(resultado)
    xs = 0:datasize
    ys = [estimar(x, resultado[i][3]) for x in xs]
    lines!(ax, xs, ys, label = "Estimador grau $(i - 1)", linestyle = :dash, color = i, colorrange = (1, length(resultado)))
end

axislegend(position = :rb)

display(fig)