<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural;

use Exception;
use Omgaalfa\Ztensor\rede\neural\Activation\AbstractActivation;
use Omgaalfa\Ztensor\rede\neural\Interfaces\ActivationInterface;
use RuntimeException;
use ZMatrix\ZTensor;

class NeuralNetwork
{
    /** @var ActivationInterface[] */
    private array $layers = [];
    /**
     * @var AdamOptimizer
     */
    private AdamOptimizer $optimizer;

    /**
     * @var int
     */
    private int $patience;

    /**
     * @param float $learningRate
     * @param int $patience
     */
    public function __construct(
        float $learningRate = 0.01,
        int   $patience = 500
    )
    {
        $this->optimizer = new AdamOptimizer($learningRate);
        $this->patience = $patience;
    }

    /**
     * @param ActivationInterface $activation
     * @return void
     */
    public function addLayer(ActivationInterface $activation): void
    {
        $this->layers[] = $activation;
    }

    /**
     * @param Ztensor $X
     * @param Ztensor $y
     * @param int $epochs
     * @param string $lossFunction
     * @param bool $verbose
     * @return void
     * @throws Exception
     */
    public function train(Ztensor $X, Ztensor $y, int $epochs = 1000, string $lossFunction = 'bce', bool $verbose = false): void
    {
        $best = PHP_FLOAT_MAX;
        $noImp = 0;

        for ($e = 0; $e < $epochs; $e++) {
            // forward
            $out = $X;
            foreach ($this->layers as $layer) {
                $out = $layer->forward($out);
            }

            // loss
            $loss = $this->calculateLoss($y, $out, $lossFunction);

            if ($loss < $best - 1e-6) {
                $best = $loss;
                $noImp = 0;
                if ($verbose && $e % 100 === 0) {
                    echo "🚀 Epoch $e, Loss: $best\n";
                }
            } else if (++$noImp >= $this->patience) {
                echo "🛑 Early stop at epoch $e, Loss: $loss\n";
                break;
            }

            // backward
            $grad = $this->calculateGradient($y, $out);
            for ($i = count($this->layers) - 1; $i >= 0; $i--) {
                $grad = $this->layers[$i]->backward($grad);
                if($this->layers[$i] instanceof AbstractActivation) {
                    $this->optimizer->updateLayer($this->layers[$i]);
                }
            }
        }
    }

    /**
     * Gera predições (forward completo).
     *
     * @param Ztensor $X Entradas [N x features]
     * @return Ztensor   Saídas [N x outputs]
     */
    public function predict(Ztensor $X): Ztensor
    {
        $out = $X;
        foreach ($this->layers as $layer) {
            $out = $layer->forward($out);
        }
        return $out;
    }

    /**
     * @param ZTensor $trueLabels
     * @param ZTensor $predictions
     * @return ZTensor[]
     */
    private function reshapeForLoss(Ztensor $trueLabels, Ztensor $predictions): array
    {
        // 2) Obtém formas
        $shapeT = $trueLabels->shape();   // ex.: [N] ou [N×1] ou [N×M]
        $shapeP = $predictions->shape();

        // 3) Verifica e reshapa para 2D [N×1]
        if (count($shapeT) === 1) {
            // se veio [N], passa para [N×1]
            $N = $shapeT[0];
            $true2d = $trueLabels->reshape([$N, 1]);
        } elseif (count($shapeT) === 2) {
            // já é 2D: mantém
            $true2d = $trueLabels;
        } else {
            throw new RuntimeException("reshapeForLoss: rótulos têm dimensão inválida (" . count($shapeT) . "D). Esperado 1D ou 2D.");
        }

        if (count($shapeP) === 1) {
            $Np = $shapeP[0];
            $pred2d = $predictions->reshape([$Np, 1]);
        } elseif (count($shapeP) === 2) {
            $pred2d = $predictions;
        } else {
            throw new RuntimeException("reshapeForLoss: previsões têm dimensão inválida (" . count($shapeP) . "D). Esperado 1D ou 2D.");
        }

        return [$true2d, $pred2d];
    }

    /**
     * @param ZTensor $trueLabels
     * @param ZTensor $predictions
     * @return float
     */
    public function calculateMseLoss(Ztensor $trueLabels, Ztensor $predictions): float
    {
        [$trueLabels, $predictions] = $this->reshapeForLoss($trueLabels, $predictions);
        $difference = $trueLabels->copy()->sub($predictions);
        return $difference->pow(2)->mean();
    }

    /**
     * @param ZTensor $trueLabels
     * @param ZTensor $predictions
     * @return float
     */
    public function calculateBinaryCrossEntropyLoss(Ztensor $trueLabels, Ztensor $predictions): float
    {
        $epsilon = 1e-12;
        [$trueLabels, $predictions] = $this->reshapeForLoss($trueLabels, $predictions);

        $predClipped = Ztensor::clip($predictions->copy(), $epsilon, 1.0 - $epsilon);

        $logP = $predClipped->copy()->log(); // log(p)

        $oneMinusP = Ztensor::ones($predClipped->shape())->sub($predClipped->copy());
        $oneMinusP = Ztensor::clip($oneMinusP, $epsilon, 1.0); // protege contra log(0)
        $logOneMinusP = $oneMinusP->log(); // log(1 - p)

        $term1 = $trueLabels->copy()->mul($logP);
        $term2 = Ztensor::ones($trueLabels->shape())
            ->sub($trueLabels->copy())->mul($logOneMinusP);

        return $term1->add($term2)->scalarMultiply(-1.0)->mean();
    }

    /**
     * @param ZTensor $trueLabels
     * @param ZTensor $predictions
     * @return float|int
     */
    public function calculateCategoricalCrossEntropyLoss(Ztensor $trueLabels, Ztensor $predictions): float|int
    {
        $epsilon = 1e-12;
        [$yTrue, $yPred] = $this->reshapeForLoss($trueLabels, $predictions);

        $yPredClipped = Ztensor::clip($yPred, $epsilon, 1.0 - $epsilon);

        $logP = $yPredClipped->copy()->log();  // Evita alterar yPred
        $crossEntropy = $yTrue->copy()->mul($logP);  // Evita alterar yTrue

        $batchSize = $crossEntropy->shape()[0];
        $sampleLosses = Ztensor::zeros([$batchSize]);

        $crossEntropy->sum($sampleLosses, 1); // assume que sum é in-place no destino
        $sampleLosses->scalarMultiply(-1.0);

        return $sampleLosses->mean();
    }

    /**
     * @param ZTensor $trueLabels
     * @param ZTensor $predictions
     * @param string $lossFunction
     * @return float
     */
    private function calculateLoss(Ztensor $trueLabels, Ztensor $predictions, string $lossFunction): float
    {
        return match ($lossFunction) {
            'mse' => $this->calculateMseLoss($trueLabels, $predictions),
            'bce' => $this->calculateBinaryCrossEntropyLoss($trueLabels, $predictions),
            'cce' => $this->calculateCategoricalCrossEntropyLoss($trueLabels, $predictions),
            default => throw new \InvalidArgumentException("Função de perda inválida: $lossFunction. Use 'mse', 'bce' ou 'cce'.")
        };
    }

    /**
     * @param ZTensor $labels
     * @param ZTensor $predictions
     * @return float|int|ZTensor
     */
    private function calculateGradient(Ztensor $labels, Ztensor $predictions): float|int|Ztensor
    {
        $lastLayerIndex = count($this->layers) - 1;
        $outputActivation = $this->layers[$lastLayerIndex]->getActivation();

        if ($outputActivation === 'sigmoid') {
            return $predictions->copy()->sub($labels->reshape($predictions->shape()));
        }


        if ($outputActivation === 'softmax') {
            return $predictions->copy()->sub($labels);  // Assumimos que a perda é categorical cross-entropy (CCE)
        }

        throw new RuntimeException("Gradiente inicial não definido para a ativação de saída: $outputActivation");
    }


    /**
     * @return ActivationInterface[]
     */
    public function getLayers(): array
    {
        return $this->layers;
    }
}
