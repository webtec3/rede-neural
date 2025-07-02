<?php

declare(strict_types=1);

use Omgaalfa\Ztensor\rede\neural\Activation\ReLUActivation;
use Omgaalfa\Ztensor\rede\neural\Activation\SigmoidActivation;
use Omgaalfa\Ztensor\rede\neural\Activation\TanhActivation;
use Omgaalfa\Ztensor\rede\neural\NeuralNetwork;
use Omgaalfa\Ztensor\rede\neural\utils\Metric;
use Omgaalfa\Ztensor\rede\neural\utils\ModelManager;
use ZMatrix\Ztensor;


require_once __DIR__ . "/../vendor/autoload.php";

// --- Problema AND (Omgaalfa) ---
echo "\n========== Problema AND - Rede Omgaalfa ==========\n";


$samples = ZTensor::arr([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]);

$labels = ZTensor::arr([
    0.0, // 0 OR 0 = 0
    1.0, // 0 OR 1 = 1
    1.0, // 1 OR 0 = 1
    1.0, // 1 OR 1 = 1
]);

// ═══════════════════════════════════════════════════════════════════════════════
// 2) Montagem da rede: 2 → 8 (ReLU) → 1 (Sigmoid), learning rate 0.1
// ═══════════════════════════════════════════════════════════════════════════════
$nn = new NeuralNetwork(0.1);
$nn->addLayer(new ReLUActivation(2, 8));
$nn->addLayer(new SigmoidActivation(8, 1));


function treinaSalva($nn, $samples, $labels, $modelFile = 'or_model.json')
{
// ═══════════════════════════════════════════════════════════════════════════════
// 3) Treinamento
// ═══════════════════════════════════════════════════════════════════════════════
    echo "🛠️ Iniciando treinamento da rede (OR gate)...\n";
    $nn->train($samples, $labels, 1000, 'bce', true);
    echo "✅ Treinamento concluído.\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 4) Salvando o modelo
// ═══════════════════════════════════════════════════════════════════════════════

    ModelManager::save($nn, $modelFile);
    echo "💾 Modelo salvo em $modelFile\n";
}

$modelFile = 'or_model.json';
treinaSalva($nn, $samples, $labels, $modelFile);

// ═══════════════════════════════════════════════════════════════════════════════
// 5) Carregando o modelo
// ═══════════════════════════════════════════════════════════════════════════════
$nn_loaded = new NeuralNetwork(0.1);
$nn_loaded->addLayer(new ReLUActivation(2, 8));
$nn_loaded->addLayer(new SigmoidActivation(8, 1));


ModelManager::load($nn_loaded, $modelFile);
echo "📦 Modelo carregado de $modelFile\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 6) Predição usando o modelo carregado
// ═══════════════════════════════════════════════════════════════════════════════
echo "\n🧪 Testando a rede carregada:\n";
$predictions = $nn_loaded->predict($samples);

for ($i = 0; $i < $samples->shape()[0]; $i++) {
    $input = [$samples->key([$i, 0]), $samples->key([$i, 1])];
    $expected = $labels->key([$i]);
    $predicted = $predictions->key([$i, 0]);

    echo sprintf(
        "Input: [%d, %d] | Expected: %.1f | Predicted: %.4f | %s\n",
        $input[0], $input[1], $expected, $predicted,
        (abs($expected - $predicted) < 0.5) ? "✅" : "❌"
    );
}

echo "\n✅ Teste concluído.\n";


// ═══════════════════════════════════════════════════════════════════════════════
// 1) Dados do Iris dataset
// ═══════════════════════════════════════════════════════════════════════════════

// 🔢 Features: Sepal length, Sepal width, Petal length, Petal width
$samples = [
    [5.1, 3.5, 1.4, 0.2],  // setosa
    [4.9, 3.0, 1.4, 0.2],  // setosa
    [7.0, 3.2, 4.7, 1.4],  // versicolor
    [6.4, 3.2, 4.5, 1.5],  // versicolor
    [6.3, 3.3, 6.0, 2.5],  // virginica
    [5.8, 2.7, 5.1, 1.9],  // virginica
];

// 2) Rótulos em one‑hot (3 classes)
$labels = [
    [1, 0, 0],  // setosa
    [1, 0, 0],
    [0, 1, 0],  // versicolor
    [0, 1, 0],
    [0, 0, 1],  // virginica
    [0, 0, 1],
];

// Converte para ZTensor
$X = ZTensor::arr($samples);   // shape [6×4]
$Y = ZTensor::arr($labels);    // shape [6×3]

// 3) Define a rede: 4 → 8 (ReLU) → 3 (Softmax)
$nn = new NeuralNetwork(0.001);
$nn->addLayer(new TanhActivation(4, 16));
$nn->addLayer(new ReLUActivation(16, 8));
$nn->addLayer(new SigmoidActivation(8, 3));

// 4) Treina a rede
echo "🛠️ Iniciando treinamento (Iris toy)...\n";
$nn->train($X, $Y, 2000, 'cce', true);
echo "✅ Treinamento concluído.\n\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 4) Salvando o modelo
// ═══════════════════════════════════════════════════════════════════════════════
$modelFile = 'iris_model.json';
ModelManager::save($nn, $modelFile);
echo "💾 Modelo salvo em $modelFile\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 5) Carregando o modelo
// ═══════════════════════════════════════════════════════════════════════════════
$nn = new NeuralNetwork(0.001);
$nn->addLayer(new TanhActivation(4, 16));
$nn->addLayer(new ReLUActivation(16, 8));
$nn->addLayer(new SigmoidActivation(8, 3));

ModelManager::load($nn, $modelFile);
echo "📦 Modelo carregado de $modelFile\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 6) Predição usando o modelo carregado
// ═══════════════════════════════════════════════════════════════════════════════
echo "\n🧪 Testando a rede carregada:\n";
// 5) Testa predições
echo "🧪 Testando a rede:\n";
$preds = $nn->predict($X)->toArray();  // [6×3] array de probabilidades

foreach ($preds as $i => $prob) {
    // Classe prevista = índice do maior valor em $prob
    $predClass = array_search(max($prob), $prob, true);
    echo sprintf(
        "Amostra %d → Probabilidades: [%.2f, %.2f, %.2f] | Previsto: %d | Esperado: %d\n",
        $i,
        $prob[0], $prob[1], $prob[2],
        $predClass,
        array_search(1, $labels[$i], true)
    );
}

echo "\n✅ Teste concluído.\n";

exit();
// Dados do problema AND
$X = ZTensor::arr([[0,0], [0,1], [1,0], [1,1]]);
$y = ZTensor::arr([0, 0, 0, 1]);

// Cria a rede
$nn = new NeuralNetwork(0.1);
$nn->addLayer(new ReLUActivation(2, 4));
$nn->addLayer(new SigmoidActivation(4, 1));

// Treina
$nn->train($X, $y, 1000, 'bce', true);

$metrics = [
    'accuracy'  => [Metric::class, 'accuracy'],
    'precision' => [Metric::class, 'precision'],
    'recall'    => [Metric::class, 'recall'],
    'f1_score'  => [Metric::class, 'f1'],
    'mse'       => [Metric::class, 'mse'],
];

$results = ModelManager::evaluate($nn, $X, $y, $metrics);
print_r($results);

// Salva modelo
ModelManager::save($nn, 'xor_model.json');

// Carrega modelo
// 🔥 Carrega os pesos salvos do modelo treinado
ModelManager::load($nn, 'xor_model.json');

echo "✅ Modelo carregado com sucesso.\n";

// 🔢 Dados de entrada
$test = ZTensor::arr([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]);

// 🧠 Faz as predições
$output = $nn->predict($test);

// 📊 Mostra o resultado formatado
echo "\n🧪 Testando a rede carregada:\n";

for ($i = 0; $i < $test->shape()[0]; $i++) {
    $input = [$test->key([$i, 0]), $test->key([$i, 1])];
    $predicted = $output->key([$i, 0]);

    echo sprintf(
        "Input: [%d, %d] | Predicted: %.4f\n",
        $input[0], $input[1], $predicted
    );
}

echo "\n✅ Teste concluído.\n";