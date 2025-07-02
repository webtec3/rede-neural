<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural\Activation;

use Exception;
use Omgaalfa\Ztensor\rede\neural\Activation;
use Omgaalfa\Ztensor\rede\neural\Interfaces\ActivationInterface;
use ZMatrix\ZTensor;

abstract class AbstractActivation implements ActivationInterface
{
    /**
     * @var float
     */
    protected float $scale;

    /**
     * @var Ztensor
     */
    protected Ztensor $weights;
    /**
     * @var Ztensor
     */
    protected Ztensor $bias;
    /**
     * @var string
     */
    protected string $activation;

    /**
     * @var Ztensor
     */
    protected Ztensor $lastInput;
    /**
     * @var Ztensor
     */
    protected Ztensor $lastZ;
    /**
     * @var Ztensor
     */
    protected Ztensor $lastOutput;
    /**
     * @var Ztensor
     */
    protected Ztensor $cached_dW;
    /**
     * @var Ztensor
     */
    protected Ztensor $cached_dB;


    /**
     * @param Ztensor $input
     * @return Ztensor
     * @throws Exception
     */
    public function forward(Ztensor $input): Ztensor
    {
        $shapeIn = $input->shape();
        if ($input->ndim() === 1) {
            $input = $input->reshape([1, $shapeIn[0]]);
        }
        $this->lastInput = $input;

        $Z = $input->matmul($this->weights);
        $this->lastZ = $Z->add($this->bias);
        $this->lastOutput = Activation::apply($this->lastZ, $this->activation);

        return $this->lastOutput;
    }

    /**
     * @param Ztensor $gradOutput
     * @return Ztensor
     */
    public function backward(Ztensor $gradOutput): Ztensor
    {
        // Verifica se é softmax na saída
        if ($this->getActivation() === 'softmax') {
            // Não aplica derivada — o gradOutput já é (ŷ - y) para CCE
            $dZ = $gradOutput;
        } else {
            // Para outras ativações, aplica a derivada normalmente
            $dAct = Activation::derivative($this->lastZ, $this->activation);
            $dZ = $gradOutput->mul($dAct);
        }

        $batchSize = $this->lastInput->shape()[0];
        $outputSize = $this->outputSize;

        // Gradiente dos pesos
        $dW = $this->lastInput
            ->transpose()
            ->matmul($dZ)
            ->scalarMultiply(1.0 / $batchSize);

        // Gradiente do bias
        $dB = Ztensor::zeros([$outputSize]);
        $dZ->sum($dB, 0); // Soma eixo 0 → shape [outputSize]
        $dB = $dB->scalarMultiply(1.0 / $batchSize);

        // Guarda os gradientes para o otimizador (Adam)
        $this->cached_dW = $dW;
        $this->cached_dB = $dB;

        // Gradiente para a camada anterior
        return $dZ->matmul($this->weights->transpose());
    }

    /**
     * @return array
     */
    public function getParams(): array
    {
        return [$this->weights, $this->bias];
    }

    /**
     * @param Ztensor $W
     * @param Ztensor $B
     * @return void
     */
    public function setParams(Ztensor $W, Ztensor $B): void
    {
        $this->weights = $W;
        $this->bias = $B;
    }

    public function getActivation(): string
    {
        return $this->activation;
    }

    /**
     * @return array
     */
    public function getGradients(): array
    {
        return [
            'dW' => $this->cached_dW,
            'dB' => $this->cached_dB,
        ];
    }
}