<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural;

use Omgaalfa\PhpMl\rede\dev\neuron\DenseLayer;
use Omgaalfa\Ztensor\rede\neural\Activation\AbstractActivation;
use ZMatrix\ZTensor;

class AdamOptimizer
{
    private int $t = 0;

    /** @var array<int, ZTensor> */
    private array $mW = [], $vW = [];
    /** @var array<int, ZTensor> */
    private array $mB = [], $vB = [];

    public function __construct(
        protected float $learningRate = 0.001,
        protected float $beta1 = 0.9,
        protected float $beta2 = 0.999,
        protected float $epsilon = 1e-8
    )
    {
    }

    /**
     * Atualiza parâmetros (pesos e bias) de uma camada densa usando Adam.
     */
    public function updateLayer(DenseLayer|AbstractActivation $layer): void
    {
        $this->t++;
        $key = spl_object_id($layer);

        // Obtém gradientes e parâmetros atuais
        [$dW, $dB] = array_values($layer->getGradients());
        [$W, $B] = $layer->getParams();

        // Inicializa momentos se necessário
        if (!isset($this->mW[$key])) {
            $this->mW[$key] = ZTensor::zeros($dW->shape());
            $this->vW[$key] = ZTensor::zeros($dW->shape());
            $this->mB[$key] = ZTensor::zeros($dB->shape());
            $this->vB[$key] = ZTensor::zeros($dB->shape());
        }

        // Moments for weights
        $mW = $this->mW[$key];
        $vW = $this->vW[$key];
        $mW->scalarMultiply($this->beta1);
        $tmp1 = ZTensor::arr($dW)->scalarMultiply(1 - $this->beta1);
        $mW->add($tmp1);
        $vW->scalarMultiply($this->beta2);
        $tmp2 = $dW->mul($dW)->scalarMultiply(1 - $this->beta2);
        $vW->add($tmp2);
        $mW_hat = ZTensor::arr($mW)->scalarMultiply(1 / (1 - ($this->beta1 ** $this->t)));
        $vW_hat = ZTensor::arr($vW)->scalarMultiply(1 / (1 - ($this->beta2 ** $this->t)));
        $denW = ZTensor::arr($vW_hat);
        $denW->sqrt()->add(ZTensor::full($denW->shape(), $this->epsilon));
        $updW = ZTensor::arr($mW_hat)->divide($denW)->scalarMultiply($this->getLearningRate());
        $W_new = ZTensor::arr($W)->sub($updW);

        // Moments for bias
        $mB = $this->mB[$key];
        $vB = $this->vB[$key];
        $mB->scalarMultiply($this->beta1);
        $tmpB1 = ZTensor::arr($dB)->scalarMultiply(1 - $this->beta1);
        $mB->add($tmpB1);
        $vB->scalarMultiply($this->beta2);
        $tmpB2 = ZTensor::arr($dB)->mul($dB)->scalarMultiply(1 - $this->beta2);
        $vB->add($tmpB2);
        $mB_hat = $mB->scalarMultiply(1 / (1 - ($this->beta1 ** $this->t)));
        $vB_hat = $vB->scalarMultiply(1 / (1 - ($this->beta2 ** $this->t)));

        // CORREÇÃO: Calcula o denominador para o bias de forma vetorizada (elemento a elemento)
        $denB = ZTensor::arr($vB_hat);
        $denB->sqrt()->add(ZTensor::full($denB->shape(), $this->epsilon));

        // Calcula a atualização do bias usando o denominador correto
        $updB = ZTensor::arr($mB_hat)->divide($denB)->scalarMultiply($this->getLearningRate());
        $B_new = ZTensor::arr($B)->sub($updB);

        // Aplica novos parâmetros na camada
        $layer->setParams($W_new, $B_new);

        // Armazena momentos atualizados
        $this->mW[$key] = $mW;
        $this->vW[$key] = $vW;
        $this->mB[$key] = $mB;
        $this->vB[$key] = $vB;
    }

    public function getLearningRate(): float
    {
        return $this->learningRate;
    }
}
