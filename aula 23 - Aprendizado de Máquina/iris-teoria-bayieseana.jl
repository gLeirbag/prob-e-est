using CSV, DataFrames, Statistics, StatsBase, Random, Distributions

irisfile = CSV.File("/home/gfgmzk/Downloads/Iris.csv")
irisDF = DataFrame(irisfile)

# Embaralho as linhas porque os dados vieram ordenados por espécies.
Random.shuffle!(irisDF)

numbertraining = 20

# Uma parte dos dados vai ser usado para inferir as probabilidades.
trainingDF = DataFrames.view(irisDF, 1:numbertraining, 2:lastindex(irisDF, 2))
# Outra parte para testar as probabilidades calculadas
testDF = DataFrames.view(irisDF, numbertraining+1:lastindex(irisDF, 1), 2:lastindex(irisDF, 2))

# Variável contendo a proporção de espécies no conjunto de treinamente, somente para ilustrar, não é usado para nada.
speciesproportion = StatsBase.countmap(trainingDF.Species) |> function (speciesCount)
    Dict(species => count / sum(values(speciesCount)) for (species, count) in speciesCount)
end

"""
Computa a função de densidade de probabilidade de um atributo a partir de uma amostra, assumindo que o atributo venha de uma distribuição normal.
"""
function getnaiveattributedist(sample)
    # Assumindo que o atributo possui uma distribuição normal no universo. Usando uma média aleatória a partir de um intervalo de confiança de 99%.
    samplemean = Statistics.mean(sample)
    n = length(sample)
    confidence = 0.99
    deviation = std(sample)
    t = Distributions.TDist(n - 1)
    quantil = Distributions.quantile(t, 1 - confidence / 2)
    confidenceinterval = range(samplemean - (quantil * deviation / sqrt(n)), samplemean + (quantil * deviation / sqrt(n)), step=0.01)

    Distributions.Normal(rand(confidenceinterval), deviation)
end

"""
Calcula a probabilidade dos atributos dentro da amostra. Em outras palavras estamos calculando o prior.
"""
function computeattributesprobabilitymatrix(sample::Matrix, attributes::Vector)
    p_vec = zeros(length(attributes))
    for i in 1:length(attributes)
        p_vec[i] = pdf(getnaiveattributedist(sample[:, i]), attributes[i])
    end
    p_vec
end

function computeclassscore(sample::Matrix, class::String15, attributes::Vector; classposition=:end)
    # Estamos calculando algo que chamei arbitrariamente de score por que não estamos calculando a probabilidade da evidência para chamarmos de probabilidade.

    # P(X_vec | C) é a probabilidade de toda a intersecção de X_vec dado C.
    # Ou, em outras palavras, é a probabilidade de todos os elementos de X_vec acontecerem ao mesmo tempo.
    # Assumindo a independência dos elementos internos de X_vec, então basta multiplicar a probabilidade de cada atributo.


    # classsample = filter(eachrow(sample)) do row
    #     row[classposition == :end ? end : 1] == class
    # end

    classsample = sample[ sample[:, classposition == :end ? end : 1].== class ,:]

    classprobability = size(classsample, 1) / size(sample, 1)
    prior = classposition == :end ? computeattributesprobabilitymatrix(classsample[:, 1:end-1], attributes) : computeattributesprobabilitymatrix(classsample[:, 2:end], attributes)
    
    prod(prior) * classprobability
end

function computescores(sample::Matrix, attributes::Vector; classposition=:end)
    classes = classposition == :end ? unique(sample[:, end]) : unique(sample[:, 1])
    scores = Float64[]

    for class in classes
        push!(scores, computeclassscore(sample, class, attributes; classposition))
    end

    scores = scores ./ sum(scores)

    (classes, scores)
end

function test(data, numbertraining = 20)
    trainingDF = DataFrames.view(data, 1:numbertraining, 2:lastindex(data, 2))
    testDF = DataFrames.view(data, numbertraining+1:lastindex(data, 1), 2:lastindex(data, 2))

    numbercorrects = 0
    for row in eachrow(testDF)
        (classes, scores) = computescores(Matrix(trainingDF), collect(row)[1:end-1])
        predictedclass = classes[argmax(scores)]
        if predictedclass == row[end]
            numbercorrects += 1
        end
    end

    return numbercorrects / size(testDF, 1)
end

test(irisDF, 15)