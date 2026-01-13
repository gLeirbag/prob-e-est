# Em uma maternidade, são medidos os pesos de 10 recém-nascidos, obtendo-se os valores (em gramas):
amostras_pesos = (3200, 3050, 3400, 3180, 3300, 3120, 3280, 3350, 3250, 3100)

# a) Determine o intervalo de 90% para a média dos pesos.

# Traduzindo o enunciado, com base na amostra, eu quero _estimar_ o peso média de recém-nascidos através da amostra. Para isso, vamos utilizar o fato que a distribuição da média das amostras
# Se aproxima de uma dist. normal.

# No caso, como o número de amostras é pequenos, vamos considerar que vai se aproximar de uma distribuição T-Student.
using Statistics
média = Statistics.mean(amostras_pesos)

# Calculamos o desvio padrão.
d_p = sqrt(sum( (amostra) -> (amostra - média)^2, amostras_pesos )/ (length(amostras_pesos) - 1))

# Quero calcular P(x_low < média < x_high) = 90%
# Assim quero calcular P(média < x_low) = 5% e P(média < x_high) = 95%

using Distributions
tstudent = TDist(length(amostras_pesos) - 1)

x_low_tstudent = quantile(tstudent, 0.05)
x_high_tstudent = quantile(tstudent, 0.95) # Poderiamos assumir, como a dist é simétrica, que x_high_tstudent = -x_low_tstudent

# Certo, sabemos que x_low_tstudent = est.Média - média_amostral/ d_p/sqrt(v)
# Então, est.Média = d_p/sqrt(v)*_low_tstudent + média_amostral

mean_low =  d_p/sqrt(length(amostras_pesos))*x_low_tstudent + média
mean_high = d_p/sqrt(length(amostras_pesos))*x_high_tstudent + média

println("Intervalo de 90% para a média dos pesos: [$(mean_low), $(mean_high)]")

# Através dos cálculos: Intervalo de 90% para a média dos pesos: [3157.356808596692, 3288.643191403308]

# b) Agora, resolva o mesmo problema, mas usando bootstrap.
# Vamos tratar nossa amostra como se fosse a população real.

qtd_amostras = 1000
qtd_por_amostra = 3

médias = []

for i in 1:1000
    média = [rand(amostras_pesos), rand(amostras_pesos), rand(amostras_pesos)]
    push!(médias, média...)
end
mean(médias)

