package simpleneural

import "math/rand"

// generateData generates random data for binary classification
func GenerateData(n int) ([][]float64, [][]float64) {

	// Inisialisasi slice untuk menyimpan data XOR
	xorData := make([][3]float64, n)

	// Loop untuk membuat data XOR secara acak
	for i := 0; i < n; i++ {
		x1 := rand.Float64()
		x2 := rand.Float64()
		x3 := rand.Float64()

		// XOR
		if (x1 < 0.5 && x2 >= 0.5) || (x1 >= 0.5 && x2 < 0.5) {
			xorData[i][0] = x1
			xorData[i][1] = x2
			xorData[i][2] = x3
		} else {
			xorData[i][0] = x2
			xorData[i][1] = x1
			xorData[i][2] = x3
		}
	}

	data := make([][]float64, len(xorData))
	out := make([][]float64, len(xorData))
	for i := 0; i < len(xorData); i++ {
		var output float64
		if xorData[i][0] <= xorData[i][1] {
			output = 0.0
		} else {
			output = 1.0

		}
		data[i] = []float64{xorData[i][0], xorData[i][1], xorData[i][2]}
		out[i] = []float64{output, output}
	}
	return data, out
}
