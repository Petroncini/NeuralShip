# Foguetes Genéticos

Este projeto visa utilizar um algoritmo genético para otimizar o controlador de um foguete simulado e fazê-lo aprender a pousar seguramente em uma plataforma após reentrada. Inspirado pela aula de Sistemas Bioinspirados do Professor Eduardo do Vale Simões, o projeto foi implementado em Python (mil desculpas) e utiliza a biblioteca Pygame para visualização.

## Controlador Neural

O foguete é controlado por uma rede neural com duas camadas ocultas de 32 neurônios, utilizando a função de ativação tangente hiperbólica (tanh).

### Inputs da Rede Neural
A rede recebe 7 inputs normalizados:
- Posição X do foguete (normalizada pela largura)
- Posição Y do foguete (normalizada pela altura)
- Velocidade no eixo X
- Velocidade no eixo Y
- Ângulo do foguete
- Distância do foguete à plataforma de pouso no eixo X
- Fração de combustível remanescente

### Outputs da Rede
Três neurônios de saída representam as decisões de controle:
- Aplicar impulso (ligar o motor)
- Rotacionar para a esquerda
- Rotacionar para a direita

## Dinâmica de Simulação

Em cada frame, o estado atual do foguete é alimentado na rede neural, e sua saída determina as ações do agente. Uma rodada de teste termina quando todos os foguetes:
- Colidem com as bordas
- Pousam com sucesso

![Foguetes pousando, já vi melhores](https://github.com/user-attachments/assets/c22ba342-e943-4ece-9a6a-60a7ecc8bb3f)

## Função de Fitness

A avaliação dos indivíduos considera múltiplos parâmetros:
- Distância final à plataforma de pouso (desencorajado)
- Velocidade vertical final (fortemente desencorajada)
- Velocidade horizontal final (desencorajada)
- Diferença do ângulo em relação a 90 graus (desencorajada)
- Combustível consumido (incentivado)
- Sucesso de pouso
- Penalidade adicional por colisão lateral

## Estratégia de Treinamento

### Rodadas Múltiplas
Cada foguete é avaliado em várias rodadas, com o número de testes aumentando progressivamente. Isso favorece indivíduos mais robustos, capazes de lidar com diferentes condições iniciais.

### Progressão da Dificuldade
O programa implementa uma política de treinamento que gradualmente aumenta a complexidade:
- Inicialmente, a rede aprende a reduzir a velocidade de queda
- Depois, aprende a endireitar a trajetória
- Finalmente, adapta-se a condições iniciais cada vez mais desafiadoras

Os parâmetros iniciais (posição, ângulo, velocidade) têm componentes aleatórios, com seus intervalos expandindo-se a partir da geração 150.

## Reprodução e Evolução

### Seleção
- Os melhores indivíduos são diretamente copiados para a próxima geração
- Indivíduos são selecionados para reprodução com probabilidade proporcional ao seu fitness

### Cruzamento
- Pesos das redes são trocados aleatoriamente entre pais
- Filhos recebem mutações com taxa variável
  - Maior exploração no início
  - Mais refinamento nas gerações finais
 
## Observações

O objetivo final do treinamento era obter um algoritmo capaz de fazer o controle dinâmico do foguete, realizando um pouso controlado e robusto. Essa robustez foi o principal desafio no desenvolvimento da política de treinamento. 

Inicialmente, observamos que ao treinar os foguetes com condições iniciais fixas ou com pouca variação, os indivíduos ficavam extremamente bons em realizar o pouso, mas qualquer variação fora dos parâmetros do treinamento resultava em falha total do algoritmo. Eles estavam efetivamente seguindo um plano de voo, não se adaptando às condições do ambiente.

O treinamento com grande variação de condições iniciais apresentava o problema de que as redes eram incapazes de fazer o menor progresso. Não havia nenhum indivíduo que aprendesse a direcionar seu voo. No máximo, eles aprendiam a desacelerar sua descida. O problema foi muito complexo para o algoritmo resolver de uma vez. Talvez com mais gerações e mais indivíduos ainda seria possível resolver.

O meio termo que apresentou o resultado ótimo foi uma escala gradual da dificuldade do problema. Assim, nas primeiras gerações, eram avaliadas em um ambiente com pouca variação das condições iniciais. Nessas primeiras gerações, as redes aprenderam a reduzir sua velocidade e a direcionar sua descida para o centro da plataforma. Nas gerações mais avançadas, as redes eram avaliadas com condições iniciais mais extremas e com mais ciclos de avaliação por indivíduo. Nessa etapa, as redes refinaram sua capacidade de controle adquirido na etapa inicial de treinamento. As redes então se tornaram capazes de realizar manobras extremas de correção de trajetória.

O teste final desse procedimento de treinamento foi a adição de uma nova variável ao ambiente: a posição da plataforma de pouso. Uma rede que saiu de um processo evolucionário com mais de 1000 gerações foi avaliada nesse novo ambiente. Se o algoritmo de treino é realmente robusto, ele conseguiria se adaptar a essa mudança. 

Os resultados foram inicialmente decepcionantes, com uma taxa de sucesso de 52%. Claro, considerando que a plataforma de pouso tem 100 pixels e a tela 600, 52% é muito melhor que chance aleatória de 16% (desconsiderando a capacidade de desacelerar). No entanto, uma nave que só pousa 50% das vezes não é desejável. 

Mas uma segunda análise mostra que a distância final média do foguete à plataforma é de apenas 13 pixels. O que acontece é que o foguete sempre erra a plataforma por muito pouco, sempre do lado da plataforma mais próximo do centro da tela, mostrando que a rede neural desenvolvida pelo algoritmo genético depende principalmente da entrada de distância do foguete à plataforma, mas também da sua posição x absoluta, que é otimizada para chegar em LARGURA / 2.

## Tecnologias
- Python (desculpa Simões!)
- Pygame
- NumPy
- Algoritmos Genéticos
- Redes Neurais

## 📝 Licença
Projeto de código aberto - Tá liberado explorar e sugerir melhorias!
