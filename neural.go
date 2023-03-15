package simpleneural

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/panjf2000/ants/v2"
)

type NeuralNetwork struct {
	inputNodes  int
	hiddenNodes int
	outputNodes int
	weightsIH   [][]float64
	weightsHO   [][]float64
	biasH       []float64
	biasO       []float64
	// hidden      []float64
	Config
}

func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes int, config *Config) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputNodes:  inputNodes,
		hiddenNodes: hiddenNodes,
		outputNodes: outputNodes,
		weightsIH:   randomMatrix(hiddenNodes, inputNodes),
		weightsHO:   randomMatrix(outputNodes, hiddenNodes),
		biasH:       make([]float64, hiddenNodes),
		biasO:       make([]float64, outputNodes),
	}
	nn.Config = *configInit(config)
	for i := 0; i < hiddenNodes; i++ {
		nn.biasH[i] = rand.Float64()*2 - 1
	}
	for i := 0; i < outputNodes; i++ {
		nn.biasO[i] = rand.Float64()*2 - 1
	}
	return nn
}

func (nn *NeuralNetwork) FeedForward(input []float64) []float64 {
	// Calculate activations for the hidden layer
	hiddenActivations := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		weightedSum := nn.biasH[i]
		for j := 0; j < nn.inputNodes; j++ {
			weightedSum += nn.weightsIH[i][j] * input[j]
		}
		if nn.HActivation != SoftmaxA {
			hiddenActivations[i] = nn.HiddenActivation(weightedSum)

		}
	}

	if nn.HActivation == SoftmaxA {
		hiddenActivations = Softmax(hiddenActivations)
	}

	// Calculate activations for the output layer
	outputActivations := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		weightedSum := nn.biasO[i]
		for j := 0; j < nn.hiddenNodes; j++ {
			weightedSum += nn.weightsHO[i][j] * hiddenActivations[j]
		}
		if nn.OActivation != SoftmaxA {
			outputActivations[i] = nn.OutActivation(weightedSum)
		}
	}
	if nn.OActivation == SoftmaxA {
		outputActivations = Softmax(outputActivations)
	}
	return outputActivations
}

var sum int32
var start = time.Now()
var totalLoss = 0.0

func (nn *NeuralNetwork) Runn(i interface{}, learningRate float64, num, numInput int) {
	atomic.AddInt32(&sum, 1)
	rn := i.([][]float64)
	input := rn[0]
	target := rn[1]

	hiddenActivations := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		weightedSum := nn.biasH[i]
		for j := 0; j < nn.inputNodes; j++ {
			weightedSum += nn.weightsIH[i][j] * input[j]
		}
		weightedSum += nn.biasH[i]

		if nn.HActivation != SoftmaxA {
			hiddenActivations[i] = nn.HiddenActivation(weightedSum)

		}
	}

	if nn.HActivation == SoftmaxA {
		hiddenActivations = Softmax(hiddenActivations)
	}

	if nn.Dropout > 0.0 {
		hiddenActivations, _ = Dropout(hiddenActivations, nn.Dropout)
	}

	outputActivations := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		weightedSum := nn.biasO[i]
		for j := 0; j < nn.hiddenNodes; j++ {
			weightedSum += nn.weightsHO[i][j] * hiddenActivations[j]
		}
		weightedSum += nn.biasO[i]

		if nn.OActivation != SoftmaxA {
			outputActivations[i] = nn.OutActivation(weightedSum)
		}
	}

	if nn.OActivation == SoftmaxA {
		outputActivations = Softmax(outputActivations)
	}

	loss := 0.0

	outputErrors := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		outputErrors[i] = target[i] - outputActivations[i]
		loss += 0.5 * outputErrors[i] * outputErrors[i]

	}

	totalLoss += loss

	// Calculate hidden layer errors
	hiddenErrors := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		errorSum := 0.0
		for j := 0; j < nn.outputNodes; j++ {
			errorSum += nn.weightsHO[j][i] * outputErrors[j]
		}
		hiddenErrors[i] = errorSum * (1 - hiddenActivations[i])
	}

	// Update output layer weights and biases
	for i := 0; i < nn.outputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			nn.weightsHO[i][j] += learningRate * outputErrors[i] * hiddenActivations[j]
		}
		nn.biasO[i] += learningRate * outputErrors[i]
	}

	// Update hidden layer weights and biases
	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.inputNodes; j++ {
			nn.weightsIH[i][j] += learningRate * hiddenErrors[i] * input[j]
		}
		nn.biasH[i] += learningRate * hiddenErrors[i]
	}

	if (int(sum)+1)%num == 0 {
		last := time.Since(start)
		avgLoss := totalLoss / float64(numInput)
		loss2 := MeanSquaredError(target, outputActivations)

		fmt.Printf("Time %s Epoch %d: Loss = %.4f TotalLost: %.4f\n", last, sum+1, loss2, avgLoss)
	}
}

func (nn *NeuralNetwork) Train(x, y [][]float64) {
	epoch := nn.Epoch
	batch := nn.Batch
	var wg sync.WaitGroup
	p, _ := ants.NewPoolWithFunc(nn.Gorontine, func(i interface{}) {
		nn.Runn(i, nn.Learning, nn.NumShow, len(x)*nn.Epoch)
		wg.Done()
	})

	defer p.Release()
	for j := 0; j < epoch; j++ {
		for i := 0; i < len(x); i += batch {

			end := i + batch
			if end > len(x) {
				end = len(x)
			}
			batchX := x[i:end]
			batchY := y[i:end]

			for o, input := range batchX {
				wg.Add(1)
				_ = p.Invoke([][]float64{input, batchY[o]})
			}

		}
	}

	last := time.Since(start)

	fmt.Printf("Time %s Gorontine: %d \n\n", last, p.Running())
	p.Free()

}
func CategoricalCrossEntropyLoss(yTrue, yPred [][]float64) float64 {
	var loss float64

	for i := 0; i < len(yTrue); i++ {
		for j := 0; j < len(yTrue[i]); j++ {
			loss -= yTrue[i][j] * math.Log(yPred[i][j])
		}
	}

	loss /= float64(len(yTrue))

	return loss
}

func MeanSquaredError(yTrue, yPred []float64) float64 {
	var mse float64

	for i := 0; i < len(yTrue); i++ {
		mse += math.Pow(yTrue[i]-yPred[i], 2)
	}

	mse /= float64(len(yTrue))

	return mse
}

func BinaryCrossEntropy(yTrue []float64, yPred []float64) float64 {
	var loss float64
	for i := 0; i < len(yTrue); i++ {
		loss += -(yTrue[i]*math.Log(yPred[i]) + (1-yTrue[i])*math.Log(1-yPred[i]))
	}
	return loss / float64(len(yTrue))
}

func Dropout(input []float64, dropoutProb float64) ([]float64, []bool) {
	output := make([]float64, len(input))
	mask := make([]bool, len(input))
	for i, x := range input {
		if rand.Float64() < dropoutProb {
			output[i] = 0.0
			mask[i] = true
		} else {
			output[i] = x / (1.0 - dropoutProb)
			mask[i] = false
		}
	}
	return output, mask
}
