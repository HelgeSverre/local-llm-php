<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

final class Statistics
{
    /**
     * @param list<float|int> $values
     * @return array{count:int,min:float,max:float,mean:float,median:float,stddev:float}
     */
    public static function summarize(array $values): array
    {
        if ($values === []) {
            return [
                'count' => 0,
                'min' => 0.0,
                'max' => 0.0,
                'mean' => 0.0,
                'median' => 0.0,
                'stddev' => 0.0,
            ];
        }

        $numbers = array_map(static fn(float|int $value): float => (float) $value, $values);
        sort($numbers);

        $count = count($numbers);
        $sum = array_sum($numbers);
        $mean = $sum / $count;
        $middle = intdiv($count, 2);
        $median = $count % 2 === 0
            ? ($numbers[$middle - 1] + $numbers[$middle]) / 2
            : $numbers[$middle];

        $variance = 0.0;
        foreach ($numbers as $number) {
            $variance += ($number - $mean) ** 2;
        }

        $variance /= $count;

        return [
            'count' => $count,
            'min' => $numbers[0],
            'max' => $numbers[$count - 1],
            'mean' => $mean,
            'median' => $median,
            'stddev' => sqrt($variance),
        ];
    }
}
