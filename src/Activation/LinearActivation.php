<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural\Activation;

use ZMatrix\ZTensor;

class LinearActivation extends AbstractActivation
{

    public function __construct(public readonly int $inputSize, public readonly int $outputSize)
    {
        $this->setActivation('linear');
        $scale = sqrt(2.0 / ($inputSize + $outputSize));
        $this->weights = Ztensor::random([$inputSize, $outputSize], -1.0, 1.0)
            ->scalarMultiply($scale);
        $this->bias = Ztensor::random([$outputSize], -0.1, 0.1);
    }


    public function setActivation(string $activation): void
    {
        $this->activation = $activation;
    }
}