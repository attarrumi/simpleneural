
build for linux:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 CC="zig cc -target x86_64-linux-musl" CXX="zig c++ -target x86_64-linux-musl"  go build


	rand.Seed(time.Now().UnixNano())
	x, y := GenerateData(1000)
	
	epoch := 1000
	batch := 10
	// create neural network
	n := NewNeuralNetwork(len(x[0]), 3, len(y[0]), &Config{
		Epoch:       epoch,
		Batch:       batch,
		NumShow:     100000,
		Gorontine:   10,
		Learning:    0.01,
		Dropout:     0.1,
		HActivation: SigmoidA,
		OActivation: TanhA,
	})

	n.Train(x, y)

	// test neural network
	for i, input := range x {
		if i == 10 {
			break
		}
		p := n.FeedForward(input)
		fmt.Printf("Input: %v, Target: %v, Output: %v\n", input, y[i], p)
	}

