using CSV, DataFrames, Statistics, StatsBase

irisfile = CSV.File("/home/gfgmzk/Downloads/Iris.csv")
irisDF = DataFrame(irisfile)
display(irisDF)

onlyparameters = DataFrames.view(irisDF, :, 2:5)
onlyparameters[1, 1]
spearman = zeros(4, 4)
for i in 1:4, j in 1:4
    print(i, j)
    spearman[i, j] = cov(competerank(onlyparameters[!, i]), competerank((onlyparameters[!, j]))) /
                     (std(competerank(onlyparameters[!, i])) * std(competerank(onlyparameters[!, j])))
end

spearman

using GLMakie

names = DataFrames.names(onlyparameters)
figure = Figure(size=(600, 600));
axis = figure[1, 1] = Axis(figure, title="Spearman Matrix", xticks=(1:4, names),
    yticks=(1:4, names))
heatmap!(axis, spearman, colorrange=(-1, 1))

for i in 1:4, j in 1:4
    text!(axis, j, i,
        text=string(round(spearman[i, j], digits=2)),
        align=(:center, :center),
        color=:black
    )
end
axis.yreversed = true

include("reg.linear.jl")
using .RegressãoLinear: treinar, estimar

data = DataFrames.view(onlyparameters, :, [1, 3])

resultado = treinar(data, 100, 5)
datasize = 150
axis2 = Axis(figure[2, 1], title="Regressão Linear", limits=(0, 10, min(min(data[!, 2]...), 0), max(max(data[!, 2]...), 10)
    ), xlabel=names[1], ylabel=names[3]
)

scatter!(axis2, data[!, 1], data[!, 2], color=:limegreen, marker=:cross, label="Dados")

for i in eachindex(resultado)
    xs = 0:20
    ys = [estimar(x, resultado[i][3]) for x in xs]
    lines!(axis2, xs, ys, label="Estimador grau $(i - 1)", linestyle=:dash, color=i, colorrange=(1, length(resultado)))
end

axislegend(axis2, position=:rb)

display(figure)