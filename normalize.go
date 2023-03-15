package simpleneural

import "math"

func DecimalScaling(data []float64) []float64 {
	max := math.Max(math.Abs(data[0]), math.Abs(data[len(data)-1]))
	var j float64
	if max > 1 {
		for max >= 1 {
			max = max / 10
			j++
		}
	} else {
		for max < 0.1 {
			max = max * 10
			j--
		}
	}
	for i := 0; i < len(data); i++ {
		data[i] = data[i] / math.Pow(10, j)
	}
	return data
}

func ZScoreNormalization(data []float64) (float64, float64) {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	mean := sum / float64(len(data))

	var variance float64
	for _, val := range data {
		variance += math.Pow(val-mean, 2)
	}
	variance /= float64(len(data))

	stdDev := math.Sqrt(variance)

	for i, val := range data {
		data[i] = (val - mean) / stdDev
	}

	return mean, stdDev
}

func InverseZScoreNormalization(data []float64, mean float64, stdDev float64) {
	for i, val := range data {
		data[i] = val*stdDev + mean
	}

}
