<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural;

use ZMatrix\ZTensor;


class Activation
{

    /**
     * @param ZTensor $x
     * @param string $activation
     * @return ZTensor
     */
    public static function apply(Ztensor $x, string $activation): Ztensor
    {
        return match ($activation) {
            'relu' => self::reluActive($x),
            'leaky_relu' => self::leakyReluActive($x),
            'tanh' => self::tanhActive($x),
            'sigmoid' => self::sigmoidActive($x),
            'softmax' => self::softmaxActive($x),
            'linear' => self::linearActive($x),
            default => throw new \InvalidArgumentException("Ativação desconhecida: $activation")
        };
    }

    /**
     * @param ZTensor $x
     * @param string $activation
     * @return ZTensor
     */
    public static function derivative(ZTensor $x, string $activation): Ztensor
    {
        return match ($activation) {
            'relu' => self::reluDerivative($x),
            'leaky_relu' => self::leakyReluDerivative($x),
            'tanh' => self::tanhDerivative($x),
            'sigmoid' => self::sigmoidDerivative($x),
            'softmax' => self::softmaxDerivative($x),
            'linear' => self::linearDerivative($x),
            default => throw new \InvalidArgumentException("Derivada desconhecida para ativação: $activation")
        };
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function leakyReluActive(Ztensor $x): Ztensor
    {
        return $x->leakyRelu();
    }

    /**
     * @param ZTensor $x
     * @return float|int|ZTensor
     */
    public static function leakyReluDerivative(Ztensor $x): float|int|Ztensor
    {
        return $x->leakyReluDerivative();
    }


    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function tanhActive(Ztensor $x): Ztensor
    {
        return $x->tanh();
    }


    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function tanhDerivative(Ztensor $x): Ztensor
    {
        return $x->tanhDerivative();
    }


    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function reluActive(Ztensor $x): Ztensor
    {
        return $x->relu();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function reluDerivative(Ztensor $x): Ztensor
    {
        return $x->reluDerivative();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function sigmoidActive(Ztensor $x): Ztensor
    {
        // 1 / (1 + e^(-x))
        return $x->sigmoid();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function sigmoidDerivative(Ztensor $x): Ztensor
    {
        return $x->sigmoidDerivative();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function softmaxActive(Ztensor $x): Ztensor
    {
        return $x->softmax();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function softmaxDerivative(Ztensor $x): Ztensor
    {
        return $x->softmaxDerivative();
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function linearActive(Ztensor $x): Ztensor
    {
        // Ativação linear: f(x) = x
        return $x;
    }

    /**
     * @param ZTensor $x
     * @return ZTensor
     */
    public static function linearDerivative(Ztensor $x): Ztensor
    {
        // Derivada da função linear: f'(x) = 1
        return Ztensor::ones($x->shape());
    }
}