<?php

namespace Omgaalfa\Ztensor\rede\neural\utils;

use ZMatrix\ZTensor;

class Metric
{
    /**
     * Acurácia para classificação binária ou multiclasse.
     */
    public static function accuracy(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = $yTrue->reshape($yPred->shape());
        $pred = $yPred->copy();

        if ($pred->shape()[1] === 1) {
            // Classificação binária
            $predBin = $pred->greater([0.5]);
            $correct = $predBin->sub($yTrue)->abs()->greater([1e-8])->sub([1])->abs();
            return $correct->mean();
        } else {
            // Multiclasse (argmax)
            $pred = self::argmax($pred);
            $yTrue = self::argmax($yTrue);
            $correct = $pred->sub($yTrue)->abs()->greater([1e-8])->sub([1])->abs();
            return $correct->mean();
        }
    }

    /**
     * Precisão (Precision) — Classificação binária
     */
    public static function precision(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = $yTrue->reshape($yPred->shape());
        $predBin = $yPred->greater([0.5]);

        $tp = $predBin->mul($yTrue)->sumtotal();
        $fp = $predBin->sumtotal() - $tp;

        return ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
    }

    /**
     * Recall — Classificação binária
     */
    public static function recall(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = $yTrue->reshape($yPred->shape());
        $predBin = $yPred->greater([0.5]);

        $tp = $predBin->mul($yTrue)->sumtotal();
        $fn = $yTrue->sumtotal() - $tp;

        return ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
    }

    /**
     * F1-Score — Classificação binária
     */
    public static function f1(ZTensor $yTrue, ZTensor $yPred): float
    {
        $precision = self::precision($yTrue, $yPred);
        $recall    = self::recall($yTrue, $yPred);

        return ($precision + $recall) > 0 ? 2 * ($precision * $recall) / ($precision + $recall) : 0.0;
    }

    /**
     * MSE — Erro quadrático médio (Regressão)
     */
    public static function mse(ZTensor $yTrue, ZTensor $yPred): float
    {
        $diff = $yTrue->reshape($yPred->shape())->sub($yPred);
        return $diff->pow(2)->mean();
    }

    /**
     * MAE — Erro absoluto médio (Regressão)
     */
    public static function mae(ZTensor $yTrue, ZTensor $yPred): float
    {
        $diff = $yTrue->reshape($yPred->shape())->sub($yPred);
        return $diff->abs()->mean();
    }

    /**
     * Argmax manual até que haja suporte nativo no ZTensor.
     */
    public static function argmax(ZTensor $tensor): ZTensor
    {
        $arr = $tensor->toArray();
        $result = [];

        foreach ($arr as $row) {
            $maxIdx = array_keys($row, max($row))[0];
            $vec = array_fill(0, count($row), 0.0);
            $vec[$maxIdx] = 1.0;
            $result[] = $vec;
        }

        return ZTensor::arr($result);
    }
}
