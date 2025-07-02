# rede-neural

# 🧠 Rede Neural Multicamadas em PHP com ZTensor

  Uma implementação simples e eficiente de uma **rede neural multicamada (MLP)** em PHP, utilizando **tensores otimizados via ZTensor** e suporte a **ativações, otimizadores, perda BCE e softmax+CCE**. Ideal para aprendizado de máquina com dados vetoriais e classificação binária/multiclasse.

  ---

  ## 🚀 Demonstração: Previsão de Vendas de Carros

  Treinamento de rede para prever se um carro será vendido com base em quilometragem, ano e preço.

  ### 📂 Estrutura dos dados (`car-prices.csv`)

  | Mileage (km) | Year | Price (R$) | Sold |
  |--------------|------|------------|------|
  | 41235        | 2018 | 55000      | yes  |
  | 75210        | 2015 | 38000      | no   |
  | ...          | ...  | ...        | ...  |

  ---

  ## 🧪 Exemplo de Uso

  ```php
  use Omgaalfa\Ztensor\rede\neural\Activation\ReLUActivation;
  use Omgaalfa\Ztensor\rede\neural\Activation\SigmoidActivation;
  use Omgaalfa\Ztensor\rede\neural\Activation\TanhActivation;
  use Omgaalfa\Ztensor\rede\neural\NeuralNetwork;
  use Omgaalfa\Ztensor\rede\neural\utils\ModelManager;
  use ZMatrix\ZTensor;

  $nn = new NeuralNetwork(0.1);
  $nn->addLayer(new TanhActivation(3, 12));
  $nn->addLayer(new ReLUActivation(12, 8));
  $nn->addLayer(new SigmoidActivation(8, 1));

  $nn->train($X_train, $y_train, epochs: 1000, lossFunction: 'bce', verbose: true);

  ModelManager::save($nn, 'car_sales_model.json');

  $nn_loaded = new NeuralNetwork(0.1);
  // ... Adicionar camadas na mesma ordem ...
  ModelManager::load($nn_loaded, 'car_sales_model.json');
  $pred = $nn_loaded->predict($X_test);
```

```text

🚗📊 Iniciando modelo de previsão de vendas de carros...
📦 Modelo carregado de car_sales_model.json

🧪 Avaliando o modelo carregado...
📊 Acurácia de teste: 76.56%

✅ Processo completo finalizado com sucesso.

```

# Instalation

```bash

git clone https://github.com/webtec3/rede-neural.git
cd rede-neural
composer install

```
🧬 Funcionalidades


✅ Rede Neural Multicamadas com suporte a múltiplas ativações (ReLU, Tanh, Sigmoid)

🧠 Treinamento com otimização Adam

🧪 Suporte a funções de perda bce e cce

🧮 Tensores otimizados com ZTensor

📦 Serialização de modelos via ModelManager

📊 Acurácia automática nos testes
