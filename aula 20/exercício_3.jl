# Assume-se que a pressão arterial humana tem média igual a μ = 120 mm Hg. Em um estudo com um
# grupo de n = 10 pacientes idosos, observou-se uma pressão média igual a 130,1 mmHg e desvio padrão 21,21
# mmHg. Esse grupo tem pressão arterial mais alta do que a população em geral?
μ = 120
n = 10
média_amostral = 130.1
desvio_amostral = 21.21

# Vamos fazer um teste de hipótese.
# Hipótese nula, H0: μ = 120
# Hipótese alternativa, H1: μ > 120

# Não nos foi dado um coeficiente de sensibilidade α, então vamos simplesmente fazer o cálculo da probabilidade de rejeitar H0 e ver se o resultado é alto.
# Queremos calcular P(Rejeitar H0 | H0 é verdadeiro) => P( estimador > μ | μ = 120)
# Vamos considerar que a média das amostras segue uma distribuição t student.
#
# t = (estimador - μ) / (dp/sqrt(n-1))
# P( estimador > μ | μ = 120)  => P( (Est. - μ) / (s/ sqrt(n)) < 130,1 - μ / (s / sqrt(n)) )
t = (média_amostral - μ)/ (desvio_amostral/ sqrt(10) )

using Distributions
tstudent = TDist(n - 1)
p_da_amostra = 1 - cdf(tstudent, t)