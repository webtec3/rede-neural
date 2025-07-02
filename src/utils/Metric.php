<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural\utils;

use ZMatrix\ZTensor;

class Metric
{
    /**
     * Acurácia para classificação binária ou multiclasse.
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function accuracy(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = self::alignShapes($yTrue, $yPred);
        $pred = $yPred->copy();

        if ($pred->shape()[1] === 1) {
            // Classificação binária
            $predBin = $pred->greater([0.5]);
            $correct = $predBin->sub($yTrue)->abs()->greater([1e-8])->sub([1])->abs();
            return $correct->mean();
        }

        $pred = self::argmax($pred);
        $yTrue = self::argmax($yTrue);
        $correct = $pred->sub($yTrue)->abs()->greater([1e-8])->sub([1])->abs();

        return $correct->mean();
    }

    /**
     * Precisão (Precision) — Classificação binária
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function precision(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = self::alignShapes($yTrue, $yPred);
        $predBin = $yPred->greater([0.5]);

        $tp = $predBin->mul($yTrue)->sumtotal();
        $fp = $predBin->sumtotal() - $tp;

        return ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
    }

    /**
     * Recall — Classificação binária
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function recall(ZTensor $yTrue, ZTensor $yPred): float
    {
        $yTrue = self::alignShapes($yTrue, $yPred);
        $predBin = $yPred->greater([0.5]);
        $tp = $predBin->mul($yTrue)->sumtotal();
        $fn = $yTrue->sumtotal() - $tp;

        return ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
    }

    /**
     * Garante que yTrue tenha o mesmo shape de yPred (reshape se necessário).
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return ZTensor
     */
    public static function alignShapes(ZTensor $yTrue, ZTensor $yPred): ZTensor
    {
        $shapeTrue = $yTrue->shape();
        $shapePred = $yPred->shape();

        if ($shapeTrue === $shapePred) {
            return $yTrue;
        }

        if (count($shapeTrue) === 1 && count($shapePred) === 2 && $shapePred[1] === 1) {
            return $yTrue->reshape([$shapePred[0], 1]);
        }

        if (count($shapeTrue) === 2 && $shapeTrue[0] === $shapePred[0] && $shapeTrue[1] === 1 && $shapePred[1] === 1) {
            return $yTrue;
        }

        throw new \RuntimeException("Não foi possível alinhar shapes: " . json_encode($shapeTrue) . " -> " . json_encode($shapePred));
    }

    /**
     * F1-Score — Classificação binária
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function f1(ZTensor $yTrue, ZTensor $yPred): float
    {
        $precision = self::precision($yTrue, $yPred);
        $recall = self::recall($yTrue, $yPred);

        return ($precision + $recall) > 0 ? 2 * ($precision * $recall) / ($precision + $recall) : 0.0;
    }

    /**
     * MSE — Erro quadrático médio (Regressão)
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function mse(ZTensor $yTrue, ZTensor $yPred): float
    {
        $diff = $yTrue->reshape($yPred->shape())->sub($yPred);
        return $diff->pow(2)->mean();
    }

    /**
     * MAE — Erro absoluto médio (Regressão)
     *
     * @param ZTensor $yTrue
     * @param ZTensor $yPred
     * @return float
     */
    public static function mae(ZTensor $yTrue, ZTensor $yPred): float
    {
        $diff = $yTrue->reshape($yPred->shape())->sub($yPred);
        return $diff->abs()->mean();
    }

    /**
     * Argmax manual até que haja suporte nativo no ZTensor.
     *
     * @param ZTensor $tensor
     * @return ZTensor
     */
    public static function argmax(ZTensor $tensor): ZTensor
    {
        $arr = $tensor->toArray();
        $result = [];

        foreach ($arr as $row) {
            if (is_array($row)) {
                $maxIdx = array_keys($row, max($row))[0];
                $vec = array_fill(0, count($row), 0.0);
                $vec[$maxIdx] = 1.0;
                $result[] = $vec;
            }
        }

        return ZTensor::arr($result);
    }
}
