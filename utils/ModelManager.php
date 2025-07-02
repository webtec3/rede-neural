<?php

declare(strict_types=1);

namespace Omgaalfa\Ztensor\rede\neural\utils;

use Omgaalfa\Ztensor\rede\neural\NeuralNetwork;
use RuntimeException;
use ZMatrix\ZTensor;

class ModelManager
{
    /**
     * Avalia o modelo usando métricas nomeadas.
     * @param NeuralNetwork $model
     * @param ZTensor $X entradas
     * @param ZTensor $y verdadeiros
     * @param array<string, callable> $metrics callbacks(fn(yTrue, yPred): float)
     * @return array<string, float>
     */
    public static function evaluate(NeuralNetwork $model, ZTensor $X, ZTensor $y, array $metrics = []): array
    {
        $pred = $model->predict($X);
        $results = array_map(static function ($fn) use ($pred, $y) {
            return $fn($y, $pred);
        }, $metrics);
        return $results;
    }

    /**
     * Salva pesos e bias em arquivo JSON ou binário.
     * @param NeuralNetwork $model
     * @param string $filename
     * @param bool $binary
     */
    public static function save(NeuralNetwork $model, string $filename, bool $binary = false): void
    {
        $data = [];
        $filename = dirname(__DIR__) . "/models/{$filename}";
        foreach ($model->getLayers() as $layer) {
            [$W, $B] = $layer->getParams();
            $data[] = [
                'weights' => $W->toArray(),
                'bias' => $B->toArray(),
                'activation' => $layer->getActivation()
            ];
        }
        $content = $binary ? serialize($data) : json_encode($data);
        file_put_contents($filename, $content);
    }

    /**
     * Carrega parâmetros de arquivo JSON ou binário.
     * @param NeuralNetwork $model
     * @param string $filename
     * @param bool $binary
     * @throws RuntimeException
     */
    public static function load(NeuralNetwork $model, string $filename, bool $binary = false): void
    {
        $filename = dirname(__DIR__) . "/models/{$filename}";
        $content = file_get_contents($filename);
        $data = $binary ? unserialize($content) : json_decode($content, true);
        if (!is_array($data)) {
            throw new RuntimeException("Formato de modelo inválido");
        }
        $layers = $model->getLayers();
        foreach ($data as $i => $params) {
            if (!isset($layers[$i])) {
                throw new RuntimeException("Parâmetros faltando para layer $i");
            }
            $W = ZTensor::arr($params['weights']);
            $B = ZTensor::arr($params['bias']);
            $layers[$i]->setParams($W, $B);
        }
    }
}