package rbm

import "math"

type Classifier struct {
	rbm

	b      []float64
	input  int
	output int
}

func NewClassifier(input, output, hidden int) *Classifier {
	c := &Classifier{
		rbm:    *newRBM(input+output, hidden),
		input:  input,
		output: output,
		b:      make([]float64, input+output),
	}
	for i := 0; i < c.Visible(); i++ {
		if i < c.Input() {
			c.vt[i] = binaryUnit
		} else {
			c.vt[i] = softmaxUnit
		}
	}
	return c
}

func (c *Classifier) Input() int {
	return c.input
}

func (c *Classifier) Output() int {
	return c.output
}

func (c *Classifier) Train(input [][]float64, output []int, opt *Option) {
	for r := 0; r < opt.Iteration; r++ {
		for b := 0; b < len(input); b += opt.BatchSize {
			size := opt.BatchSize
			if size > len(input)-b {
				size = len(input) - b
			}
			c.resetDelta()
			for i := b; i < b+size; i++ {
				c.cd(opt.GibbsStep, c.vis(input[i], output[i]))
			}

			// To avoid having to change the learning rate when the size of a mini-batch is changed, it is helpful
			// to divide the total gradient computed on a mini-batch by the size of the mini-batch, so when talking
			// about learning rates we will assume that they multiply the average, per-case gradient computed on
			// a mini-batch, not the total gradient for the mini-batch.
			c.update(LearningRate / float64(size))
		}
	}
}

func softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func (c *Classifier) freeEnergy(v []float64) float64 {
	var e float64
	for i := c.Input(); i < c.Input()+c.Output(); i++ {
		e -= c.bv[i] * v[i]
	}
	for i := 0; i < c.Hidden(); i++ {
		e -= softplus(c.eh(i, v))
	}
	return e
}

func (c *Classifier) vis(input []float64, n int) []float64 {
	copy(c.b, input)
	for i := 0; i < c.Output(); i++ {
		if i == n {
			c.b[i+c.Input()] = 1
		} else {
			c.b[i+c.Input()] = 0
		}
	}
	return c.b
}

func (c *Classifier) Classify(input []float64) int {
	idx := -1
	min := 0.0
	for i := 0; i < c.Output(); i++ {
		e := c.freeEnergy(c.vis(input, i))
		if idx == -1 || e < min {
			idx = i
			min = e
		}
	}
	return idx
}
