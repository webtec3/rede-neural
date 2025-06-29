<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural\Interfaces;

use ZMatrix\ZTensor;

interface ActivationInterface
{

    /**
     * @param Ztensor $input
     * @return Ztensor
     */
    public function forward(Ztensor $input): Ztensor;

    /**
     * @param Ztensor $gradOutput
     * @return Ztensor
     */
    public function backward(Ztensor $gradOutput): Ztensor;
}