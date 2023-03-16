package simpleneural

/*
#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Tanh(double x) {
    return sinh(x) / cosh(x);
}

double Exp(double x) {
	return exp(x);
}

*/
import "C"
import (
	"math"
	"math/rand"

	"github.com/attarrumi/simpleneural/assembly"
)

const (
	SoftmaxA = "softmax"
	TanhA    = "tanh"
	SigmoidA = "sigmoid"
	ReluA    = "relu"
)

func (nn *NeuralNetwork) OutActivation(x float64) float64 {
	if nn.OActivation == SigmoidA {
		return SigmoidC(x)
	} else if nn.OActivation == TanhA {
		return math.Tanh(x)
	} else if nn.OActivation == ReluA {
		return Relu(x)
	} else {
		return Sigmoid(x)
	}

}

func (nn *NeuralNetwork) HiddenActivation(x float64) float64 {
	if nn.HActivation == "sigmoid" {
		return SigmoidC(x)
	} else if nn.HActivation == "tanh" {
		return math.Tanh(x)
	} else if nn.HActivation == "relu" {
		return Relu(x)
	} else {
		return Sigmoid(x)
	}

}

func CostDerivative(output []float64, target []float64) []float64 {
	delta := make([]float64, len(output))
	for i := range delta {
		delta[i] = output[i] - target[i]
	}
	return delta
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidC(x float64) float64 {
	return 1.0 / (1.0 + assembly.ClangOne(C.Exp, -x))
}

// sigmoidPrime
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
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

func Relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func Softmax(x []float64) []float64 {

	max := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	exps := make([]float64, len(x))
	sum := 0.0
	for i := 0; i < len(x); i++ {
		exps[i] = assembly.ClangOne(C.Exp, x[i]-max)

		sum += exps[i]
	}
	result := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		result[i] = exps[i] / sum
	}
	return result
}
