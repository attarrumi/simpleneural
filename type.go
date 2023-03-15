package simpleneural

type Config struct {
	Epoch       int
	Batch       int
	Learning    float64
	NumShow     int
	Gorontine   int
	Dropout     float64
	OActivation string
	HActivation string
}

func configInit(c *Config) *Config {
	if c.Batch == 0 {
		c.Batch = 1
	}
	if c.Epoch == 0 {
		c.Epoch = 1
	}
	if c.Gorontine == 0 {
		c.Gorontine = 1
	}
	if c.Learning == 0 {
		c.Learning = 0.1
	}
	return c
}
