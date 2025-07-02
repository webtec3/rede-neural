<?php

declare(strict_types=1);

use Omgaalfa\Ztensor\rede\neural\Activation\ReLUActivation;
use Omgaalfa\Ztensor\rede\neural\Activation\SigmoidActivation;
use Omgaalfa\Ztensor\rede\neural\Activation\TanhActivation;
use Omgaalfa\Ztensor\rede\neural\NeuralNetwork;
use Omgaalfa\Ztensor\rede\neural\utils\ModelManager;
use ZMatrix\ZTensor;

require_once __DIR__ . "/../vendor/autoload.php";

echo "\n🚗📊 Iniciando modelo de previsão de vendas de carros...\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 1) Carregamento e pré-processamento dos dados
// ═══════════════════════════════════════════════════════════════════════════════

$arquivo =  __DIR__ . '/car-prices.csv';
$features = [];
$labels = [];

if (($fp = fopen($arquivo, 'r')) !== false) {
    fgetcsv($fp); // ignora cabeçalho
    while (($row = fgetcsv($fp)) !== false) {
        $mileage = (float) $row[1];
        $year    = (float) $row[2];
        $price   = (float) $row[3];
        $soldStr = strtolower(trim($row[4]));

        $label = match ($soldStr) {
            'yes' => 1.0,
            'no'  => 0.0,
            default => null,
        };

        if (!is_null($label)) {
            $features[] = [$mileage, $year, $price];
            $labels[] = $label;
        }
    }
    fclose($fp);
}

// Normaliza os dados
function normalizeFeatures(array $X): array {
    $numCols = count($X[0]);
    $mins = array_fill(0, $numCols, INF);
    $maxs = array_fill(0, $numCols, -INF);

    foreach ($X as $row) {
        for ($j = 0; $j < $numCols; $j++) {
            $mins[$j] = min($mins[$j], $row[$j]);
            $maxs[$j] = max($maxs[$j], $row[$j]);
        }
    }

    $X_norm = [];
    foreach ($X as $row) {
        $normRow = [];
        for ($j = 0; $j < $numCols; $j++) {
            $range = $maxs[$j] - $mins[$j];
            $normRow[] = ($range > 0) ? ($row[$j] - $mins[$j]) / $range : 0.0;
        }
        $X_norm[] = $normRow;
    }

    return $X_norm;
}

$features = normalizeFeatures($features);

// Split treino/teste
function splitTrainTest(array $X, array $y, float $ratio = 0.75): array {
    $n = count($X);
    $idx = range(0, $n - 1);
    shuffle($idx);
    $cut = (int) round($n * $ratio);

    $trainX = []; $trainY = [];
    $testX = []; $testY = [];

    foreach ($idx as $i => $orig) {
        if ($i < $cut) {
            $trainX[] = $X[$orig];
            $trainY[] = $y[$orig];
        } else {
            $testX[] = $X[$orig];
            $testY[] = $y[$orig];
        }
    }

    return [$trainX, $trainY, $testX, $testY];
}

[$trainX, $trainY, $testX, $testY] = splitTrainTest($features, $labels);

// ═══════════════════════════════════════════════════════════════════════════════
// 2) Preparação dos tensores
// ═══════════════════════════════════════════════════════════════════════════════

$X_train = ZTensor::arr($trainX);
$y_train = ZTensor::arr($trainY)->reshape([count($trainY), 1]);

$X_test = ZTensor::arr($testX);
$y_test = ZTensor::arr($testY)->reshape([count($testY), 1]);

// ═══════════════════════════════════════════════════════════════════════════════
// 3) Construção e treinamento da rede
// ═══════════════════════════════════════════════════════════════════════════════

$nn = new NeuralNetwork(0.1);
$nn->addLayer(new TanhActivation(3, 12));
$nn->addLayer(new ReLUActivation(12, 8));
$nn->addLayer(new SigmoidActivation(8, 1));

echo "\n🛠️ Treinando a rede...\n";
$nn->train($X_train, $y_train, epochs: 1000, lossFunction: 'bce', verbose: true);
echo "✅ Treinamento concluído.\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 4) Salvando o modelo
// ═══════════════════════════════════════════════════════════════════════════════

$modelFile = 'car_sales_model.json';
ModelManager::save($nn, $modelFile);
echo "💾 Modelo salvo em $modelFile\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 5) Carregando o modelo e testando
// ═══════════════════════════════════════════════════════════════════════════════

$nn_loaded = new NeuralNetwork(0.1);
$nn_loaded->addLayer(new TanhActivation(3, 12));
$nn_loaded->addLayer(new ReLUActivation(12, 8));
$nn_loaded->addLayer(new SigmoidActivation(8, 1));

ModelManager::load($nn_loaded, $modelFile);
echo "📦 Modelo carregado de $modelFile\n";

// ═══════════════════════════════════════════════════════════════════════════════
// 6) Avaliação no conjunto de teste
// ═══════════════════════════════════════════════════════════════════════════════

echo "\n🧪 Avaliando o modelo carregado...\n";
$pred = $nn_loaded->predict($X_test);
$predArr = $pred->toArray();
$yArr = $y_test->toArray();

$correct = 0;
foreach ($predArr as $i => [$p]) {
    $predLabel = ($p >= 0.5 ? 1.0 : 0.0);
    if ($predLabel === $yArr[$i][0]) {
        $correct++;
    }
}
$acc = $correct / count($predArr);
echo "📊 Acurácia de teste: ", round($acc * 100, 2), "%\n";
echo "\n✅ Processo completo finalizado com sucesso.\n";
