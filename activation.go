package simpleneural

import (
	"math"
	"math/rand"
)

const (
	SoftmaxA = "softmax"
	TanhA    = "tanh"
	SigmoidA = "sigmoid"
	ReluA    = "relu"
)

func (nn *NeuralNetwork) OutActivation(x float64) float64 {
	if nn.OActivation == SigmoidA {
		return sigmoid(x)
	} else if nn.OActivation == TanhA {
		return math.Tanh(x)
	} else if nn.OActivation == ReluA {
		return relu(x)
	} else {
		return sigmoid(x)
	}

}

func (nn *NeuralNetwork) HiddenActivation(x float64) float64 {
	if nn.HActivation == "sigmoid" {
		return sigmoid(x)
	} else if nn.HActivation == "tanh" {
		return math.Tanh(x)
	} else if nn.HActivation == "relu" {
		return relu(x)
	} else {
		return sigmoid(x)
	}

}

func CostDerivative(output []float64, target []float64) []float64 {
	delta := make([]float64, len(output))
	for i := range delta {
		delta[i] = output[i] - target[i]
	}
	return delta
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime
func SigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func randomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = rand.Float64()*2 - 1
		}
	}
	return matrix
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func Softmax(x []float64) []float64 {
	exp := make([]float64, len(x))
	sumExp := 0.0

	for i := 0; i < len(x); i++ {
		exp[i] = math.Exp(x[i])
		sumExp += exp[i]
	}

	result := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		result[i] = exp[i] / sumExp
	}

	return result
}
