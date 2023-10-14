# Datasets
7 bases (3 classificação; 4 regressão)
## Classificação
- [Weareable](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) ([Paper])(https://ieeexplore.ieee.org/document/7350781)
- [Facies classification](https://github.com/arturjordao/TowardsAutomaticAccurateCore-logProcessing) ([Paper](https://www.sciencedirect.com/science/article/pii/S092698512300068X?via%3Dihub))
- [Biomedical Images](https://medmnist.com/)
## Regressão
- [AirBnBNY](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/data) (Kaggle)
- [NASA](https://www.kaggle.com/code/wassimderbel/nasa-predictive-maintenance-rul) (Kaggle)
- SSH
- [Predição de Óleo](https://github.com/2M-kotb/DLSTM2/tree/master) ([Paper](https://www.sciencedirect.com/science/article/pii/S0925231218311639))

- Sobre o dataset do petróleo:
- Dataset de petróleo que você forneceu é da produção acumulada de mês em mês durante 63 meses. A janela de lookback é um parâmetro ajustável que o autor original muda depende do modelo usado. Lembro de termos conversado sobre salvar o dataset no formato X (cada linha sendo n amostras seguidas) e y (a amostra n+1 a ser prevista pelo modelo), mas não seria possível que o autoML testasse e otimizasse esse parâmetro em vez de eu já salvar com um n fixo? Se não for possível, qual n devo escolher (o autor varia de 1 a 6).
- Eu deixei a divisão treino/teste=70/30 (o paper original usa 80/20).
- Eu fui atrás do paper e descobri que foram usados 2 dataset na pesquisa: 1 do petróleo chinês em uma unidade de tempo não especificada pelo paper e 1 do petróleo indiano de mês em mês (esse foi o que você me passou, ele está com o nome errado como se fosse o chinês). Devo mexer nesse outro dataset chinês ou ficar apenas com o indiano o qual você me passou o arquivo?

- Sobre o dataset do ssh:
- Pode me passar o link do paper por favor? Não encontrei ele para ver aquela diferença nos intervalos de tempo entre as leituras que havíamos conversado.

- Sobre o dataset do airbnb:
- Preciso fazer o pré-processamento dos dados categóricos e scaling por agora ou posso só dividir em treino/teste? Fiquei na dúvida pois se diferentes tipos de pré-processamento de dados são necessários dependendo do dataset, então não seria possível apenas trocar o nome do arquivo .npz e rodar o autoML se não fizermos esses pré-processamentos antes de salvar o .npz.