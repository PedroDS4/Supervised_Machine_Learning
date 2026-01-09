# **Aprendizado Supervisionado**
Este repositório contempla projetos de aprendizado supervisionado para praticar o aprendizado de redes neurais.

#**Aprendizado Supervisionado por Gradiente Descendente**
Um clássico problema na literatura matemática é o problema de um ajuste de curva a partir de pontos experimentais de um certo experimento físico, onde é requerido um modelo capaz de melhor se ajustar ao formato dos pontos.
Considere agora o seguinte problema onde temos os pontos de entrada de um sistema, e também os pontos de saída, então.

$$
(x_i, y_i) ∀  i  ɛ [1,n]
$$

e queremos obter um modelo que melhor mapeie $x$ em $y$, então

$$
f(x) = y
$$

Para fazer isso, definimos uma função que meça o erro entre o modelo e os dados experimentais, uma função de custo bastante comum é o erro médio quadrático entre as variáveis, dado por

$$
MSE = \frac{ \sum_{i = 0}^{N}|y_i - f(x_i)|^2}{N}
$$

Essa função é chamada de função de custo, e podemos minimizá-la em função dos parâmetros do modelo $$ f(x_i,z_1,z_2,z_3,...,z_l) $$
Então a estratégia utilizada é calcular o gradiente da função de custo

$$
J = \frac{ \sum_{i = 0}^{N}|y_i - f(x_i,z_1,z_2,z_3,...,z_l)|^2}{N}
$$

para obter a minimização do erro e ajustar os parâmetros do modelo, ao mesmo tempo, então temos

$$
∇J(z_1,z_2,z_3,...,z_l) = ∇\frac{ \sum_{i = 0}^{N}|y_i - f(x_i,z_1,z_2,z_3,...,z_l)|^2}{N}
$$

e os parâmetros podem ser estimados iterativamente, como segue pela equação abaixo

$$
z_i[k+1] = z_i[k] - μ∇J_{z_i}
$$

O sinal de $-$ significa um problema de minimização de uma função de custo, e essa equação pode ser enxergada como um aprendizado supervisionado de um parâmetro de um certo modelo, de acordo com a $k$-ésima época(ou geração) de evolução do algorítmo.

##**Problema de Regressão Linear**
Para um problema de regressão linear comum, precisamos encontrar a melhor reta de equação $$y_i = ax_i + b$$
e precisamos dos melhores parâmetros a e b que faça a reta ficar mais ajustável aos dados.
Usando agora a função custo aplicada a esse modelo, temos

$$
J = \frac{ \sum_{i = 0}^{N}(y_i - f(x_i,a,b))^2}{N}
$$

Calculando o gradiente

$$
\nabla J_{a} = \frac{\partial }{\partial a}\frac{ \sum_{i = 0}^{N}(y_i - f(x_i,a,b))^2}{N}
$$

Utilizando que a derivada da soma é a soma das derivadas, temos

$$
∇J_{a} =\frac{1}{N} \cdot \sum_{i = 0}^{N} \frac{\partial }{\partial a}(y_i - f(x_i,a,b))^2
$$

E agora utilizando a regra da cadeia

$$
∇J_{a} =\frac{1}{N} \cdot \sum_{i = 0}^{N} 2\cdot (y_i - f(x_i,a,b))\cdot (-\frac{\partial }{\partial a}f(x_i,a,b))
$$

Então agora calculando a derivada da equação da reta, temos

$$
\nabla J_{a} = -\frac{2}{N}\sum_{i = 0}^{N} x_i (y_i - f(x_i,a,b))
$$

Fazendo o mesmo procedimento para a derivada em relação a $b$

$$
∇J_{b} = \frac{\partial }{\partial b}\frac{ \sum_{i = 0}^{N}(y_i - f(x_i,a,b))^2}{N}
$$

temos

$$
∇J_{b} =\frac{1}{N} \cdot \sum_{i = 0}^{N} \frac{\partial }{\partial b}(y_i - f(x_i,a,b))^2
$$

$$
\frac{1}{N} \cdot \sum_{i = 0}^{N} 2\cdot (y_i - f(x_i,a,b))\cdot (-\frac{\partial }{\partial b}f(x_i,a,b))
$$

e finalmente temos

$$
-\frac{2}{N} \cdot \sum_{i = 0}^{N}(y_i - f(x_i,a,b))
$$

Igualando a zero


$$
\begin{cases}
\nabla J_{a} = -\frac{2}{N}\sum_{i = 0}^{N} x_i (y_i - f(x_i,a,b)) = 0 \\
\nabla J_{b} = -\frac{2}{N} \cdot \sum_{i = 0}^{N}(y_i - f(x_i,a,b)) = 0
\end{cases}
$$

Agora montando o algorítmo da rede neural, ficamos com
$$
\begin{cases}
\ a_{n+1} = a_{n} - \mu \nabla J_{a} \\
\ b_{n+1} = b_{n} - \mu \nabla J_{b}
\end{cases}
$$



#**Regressão linear múltiplas**
A extensão do modelo linear para multiplas dimensões permite mapear pontos experimentais de um certo experimento físico, através de um modelo, nesse caso linear.
Considere agora o seguinte problema onde temos várias entradas $x_{ij}$  em um sistema, e também os pontos de saída $y_i$, então

$$
(x_{11},x_{12},x_{13},x_{14},...,x_{1M}) \rightarrow y_1
$$

$$
y_i = \sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ
$$

onde $w_j$ são os pesos de cada variável e o termo $\theta$ é conhecido como bias.

e queremos obter um modelo linear para $y_i$ em função das M variáveis do problema

$$
f(x_{i1},x_{i2},...,x_{iM}) = y_i
$$

Utilizando como função de custo o erro médio quadrático mais uma vez, temos

$$
MSE = \frac{ \sum_{i = 0}^{N}(y_i - f(x_{i1},x_{i2},...,x_{iM}))^2}{N}
$$

porém o modelo depende mesmo é dos pesos e do bias, sendo assim, uma vez que os dados são fixos, podemos alterar a dependência da função para os parâmetros de peso e bias

$$
J(w_1,w_2,w_3,...,w_M) = \frac{ \sum_{i = 0}^{N}(y_i - f(\theta,w_1,w_2,w_3,...,w_M))^2}{N}
$$

para obter a minimização do erro e ajustar os parâmetros do modelo, ao mesmo tempo, achamos o seu gradiente em relação aos parâmetros do modelo

$$
∇J(w_1,w_2,w_3,...,w_M) = ∇\frac{ \sum_{i = 0}^{N}(y_i - f(\theta,w_1,w_2,w_3,...,w_M))^2}{N}
$$

e os parâmetros podem ser estimados iterativamente, como segue pela equação abaixo

$$
\begin{cases}
  w_i^{k+1} = w_i^{k} - μ∇J_{w_i} \\
  \theta^{k+1} = \theta^{k} - μ∇J_{\theta}
\end{cases}
$$



Onde k é a $k$-ésima iteração e i é o $i$-ésimo peso.

O sinal de $-$ significa um problema de minimização de uma função de custo, e essa equação pode ser enxergada como um aprendizado supervisionado de um parâmetro de um certo modelo, de acordo com a $k$-ésima época(ou geração) de evolução do algorítmo.


##**Problema de Regressão Linear com M variáveis**
Para um problema de regressão linear multivariável, precisamos do modelo que melhor ajuste a equação do $ℜ^n$ dada por $$\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ$$ , e precisamos dos melhores parâmetros $\vec{w}$ e $\theta$ que faça o ajuste.
Para o caso multidimensional, a maximização é feita calculando o gradiente em relação a cada um dos pesos, e ao bias, como segue

$$
J = \frac{ \sum_{i = 0}^{N}(y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ) )^2}{N}
$$

Calculando o gradiente em relação a um k-ésimo coeficiente

$$
∇J_{w_k} = \frac{\partial }{\partial w_k}\frac{ \sum_{i = 0}^{N}(y_i - (\sum_{j = 1}^{M} w_j \cdot x_{ij}  + θ))^2}{N}
$$

Utilizando que a derivada da soma é a soma das derivadas, temos

$$
∇J_{w_k} = \frac{1}{N} \cdot \sum_{i = 0}^{N} \frac{\partial }{\partial w_k}(y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ))^2
$$

E agora utilizando a regra da cadeia

$$
∇J_{w_k} = \frac{1}{N} \cdot \sum_{i = 0}^{N} 2\cdot (y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ)) \cdot (-\frac{\partial }{\partial w_k} (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ))
$$

Então agora calculando a derivada do modelo com respeito à um $k$-ésimo coeficiente, temos

$$
\nabla J_{w_k} = -\frac{1}{N} \cdot \sum_{i = 0}^{N} 2\cdot (y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ)) \cdot \frac{\partial }{\partial w_k} (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ)
$$

e temos que

$$
\frac{\partial }{\partial w_k} (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ) = x_{ik}
$$

então finalmente obtemos

$$
\nabla J_{w_k} = -\frac{2}{N} \cdot \sum_{i = 0}^{N} x_{ik}(y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ))
$$

e em relação ao bias temos algo similar

$$
∇J_{\theta} = \frac{\partial }{\partial \theta}\frac{ \sum_{i = 0}^{N}(y_i - (\sum_{j = 1}^{M} w_j \cdot x_{ij}  + θ))^2}{N}
$$

desenvolvendo, temos

$$
\nabla J_{\theta} = -\frac{2}{N} \cdot \sum_{i = 0}^{N} (y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ)) \cdot \frac{\partial }{\partial \theta} (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ) = -\frac{2}{N} \cdot \sum_{i = 0}^{N} (y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ))
$$


finalmente temos as equações do gradiente

$$
\begin{cases}
\nabla J_{w_k} =-\frac{2}{N} \cdot \sum_{i = 0}^{N} x_{ik}(y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ)) \\
\nabla J_{\theta} = -\frac{2}{N} \cdot \sum_{i = 0}^{N} (y_i - (\sum_{j = 1}^{M}w_j \cdot x_{ij}  + θ))
\end{cases}
$$

Agora montando o algorítmo da rede neural, ficamos com
$$
\begin{cases}
\ w_k^{n+1} = w_k^{n} - μ\nabla J_{w_k} \\
\ \theta^{n+1} = \theta^{n} - μ\nabla J_{\theta}
\end{cases}
$$






