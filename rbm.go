package rbm

import (
	"fmt"
	"io"
	"math"
	"math/rand"
)

const (
	learningRate = 0.1
	// Sensible values for the weight-cost coefficient for L2 weight-decay typically range from
	// 0.01 to 0.00001.
	weightDecay = 0.001
	// Use small random values for the weights chosen from a zero-mean Gaussian with a standard deviation
	// of 0.01.
	weightStdDev = 0.01
)

type unitType uint8

const (
	binaryUnit unitType = iota
	gaussianUnit
	softmaxUnit
)

type rbm struct {
	w  [][]float64 // weight v * h
	dw [][]float64 // delta weight

	v   []float64  // visible
	bv  []float64  // bias visible
	dbv []float64  // delta of bias visible
	vt  []unitType // visible type

	h   []float64 // hidden
	rh  []float64 // hidden (added for contrastive divergence)
	bh  []float64 // bias hidden
	dbh []float64 // delta of bias hidden
}

func newRBM(visible, hidden int) *rbm {
	w := make([][]float64, visible)
	dw := make([][]float64, visible)
	for i := 0; i < visible; i++ {
		w[i] = make([]float64, hidden)
		dw[i] = make([]float64, hidden)
	}

	m := &rbm{
		w:  w,
		dw: dw,

		v:   make([]float64, visible),
		bv:  make([]float64, visible),
		dbv: make([]float64, visible),
		vt:  make([]unitType, visible),

		h:   make([]float64, hidden),
		rh:  make([]float64, hidden),
		bh:  make([]float64, hidden),
		dbh: make([]float64, hidden),
	}
	m.Reset()
	return m
}

func (m *rbm) resetDelta() {
	for i := 0; i < m.Visible(); i++ {
		for j := 0; j < m.Hidden(); j++ {
			m.dw[i][j] = 0
		}
	}

	for i := 0; i < m.Visible(); i++ {
		m.dbv[i] = 0
	}

	for i := 0; i < m.Hidden(); i++ {
		m.dbh[i] = 0
	}
}

func (m *rbm) Reset() {
	for i := 0; i < m.Visible(); i++ {
		for j := 0; j < m.Hidden(); j++ {
			m.w[i][j] = weightStdDev * rand.NormFloat64()
		}
	}

	// Set the hidden biases to 0.
	for i := 0; i < m.Visible(); i++ {
		m.v[i] = 0
		m.bv[i] = 0
	}

	// TODO: Set the visible biases to log[pi/(1−pi)] where pi
	// is the proportion of training vectors in which unit i is on.
	for i := 0; i < m.Hidden(); i++ {
		m.h[i] = 0
		m.rh[i] = 0
		m.bh[i] = 0
	}
}

func writeSlice(w io.Writer, v []float64) (err error) {
	for i := 0; i < len(v); i++ {
		if i > 0 {
			_, err = fmt.Fprint(w, " ")
			if err != nil {
				return
			}
		}
		_, err = fmt.Fprint(w, v[i])
		if err != nil {
			return
		}
	}
	_, err = fmt.Fprint(w, "\n")
	return
}

// line 1: visible and hidden unit count
//
// line 2: visible bias separated by space
//
// line 3: hidden bias separated by space
//
// line N: weight separated by space
func (m *rbm) WriteTo(w io.Writer) (err error) {
	_, err = fmt.Fprintf(w, "%d %d\n", m.Visible(), m.Hidden())
	if err != nil {
		return
	}

	err = writeSlice(w, m.bv)
	if err != nil {
		return
	}

	err = writeSlice(w, m.bh)
	if err != nil {
		return
	}

	for i := 0; i < m.Visible(); i++ {
		err = writeSlice(w, m.w[i])
		if err != nil {
			return
		}
	}
	return
}

func (m *rbm) ReadFrom(r io.Reader) (err error) {
	var visible, hidden int
	_, err = fmt.Fscan(r, &visible, &hidden)
	if err != nil {
		return
	}

	for i := 0; i < m.Visible(); i++ {
		_, err = fmt.Fscan(r, &m.bv[i])
		if err != nil {
			return
		}
	}

	for i := 0; i < m.Hidden(); i++ {
		_, err = fmt.Fscan(r, &m.bh[i])
		if err != nil {
			return
		}
	}

	for i := 0; i < m.Visible(); i++ {
		for j := 0; j < m.Hidden(); j++ {
			_, err = fmt.Fscan(r, &m.w[i][j])
			if err != nil {
				return
			}
		}
	}
	return
}

func (m *rbm) Visible() int {
	return len(m.v)
}

func (m *rbm) Hidden() int {
	return len(m.h)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sample(x float64) float64 {
	if x > rand.Float64() {
		return 1
	}
	return 0
}

func softmax(x []float64, mask []unitType) {
	var max float64
	for i := 0; i < len(x); i++ {
		if mask[i] != softmaxUnit {
			continue
		}
		if x[i] > max {
			max = x[i]
		}
	}
	var sum float64
	for i := 0; i < len(x); i++ {
		if mask[i] != softmaxUnit {
			continue
		}
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}
	for i := 0; i < len(x); i++ {
		if mask[i] != softmaxUnit {
			continue
		}
		x[i] /= sum
	}
}

// visible unit activation energy
func (m *rbm) ev(v int, h []float64) float64 {
	e := m.bv[v]
	for i := 0; i < m.Hidden(); i++ {
		e += m.w[v][i] * h[i]
	}
	return e
}

// hidden unit activation energy
func (m *rbm) eh(h int, v []float64) float64 {
	e := m.bh[h]
	for i := 0; i < m.Visible(); i++ {
		e += m.w[i][h] * v[i]
	}
	return e
}

func (m *rbm) ph(v []float64) []float64 {
	for i := 0; i < m.Hidden(); i++ {
		m.h[i] = sigmoid(m.eh(i, v))
	}
	return m.h
}

// Reconstruct returns reconstructed visible units and hidden units with gibbs sampling
func (m *rbm) Reconstruct(v []float64, step int) ([]float64, []float64) {
	copy(m.v, v)
	for s := 0; s < step; s++ {
		for i := 0; i < m.Hidden(); i++ {
			// It is very important to make these hidden states binary, rather than using the probabilities
			// themselves. If the probabilities are used, each hidden unit can communicate a real-value to the
			// visible units during the reconstruction. This seriously violates the information bottleneck created by
			// the fact that a hidden unit can convey at most one bit (on average). This information bottleneck
			// acts as a strong regularizer.
			m.h[i] = sample(sigmoid(m.eh(i, m.v)))
		}
		for i := 0; i < m.Visible(); i++ {
			e := m.ev(i, m.h)
			switch m.vt[i] {
			case binaryUnit:
				// Assuming that the visible units are binary, the correct way to update the visible states when generating
				// a reconstruction is to stochastically pick a 1 or 0 with a probability determined by the total top-down
				// input.
				m.v[i] = sample(sigmoid(e))
			case gaussianUnit:
				m.v[i] = rand.NormFloat64() + e
			case softmaxUnit:
				m.v[i] = e
			}
		}
		// softmax
		softmax(m.v, m.vt)
		for i := 0; i < m.Visible(); i++ {
			switch m.vt[i] {
			case softmaxUnit:
				m.v[i] = sample(m.v[i])
			}
		}
	}
	return m.v, m.h
}

func (m *rbm) updateDelta(v, rv, h, rh []float64) {
	// w
	for i := 0; i < m.Visible(); i++ {
		for j := 0; j < m.Hidden(); j++ {
			m.dw[i][j] += h[j]*v[i] - rh[j]*rv[i]
		}
	}
	// bv
	for i := 0; i < m.Visible(); i++ {
		m.dbv[i] += v[i] - rv[i]
	}
	// bh
	for i := 0; i < m.Hidden(); i++ {
		m.dbh[i] += h[i] - rh[i]
	}
}

// contrastive divergence for weight updates
func (m *rbm) cd(step int, v []float64) {
	rv, _ := m.Reconstruct(v, step)
	for i := 0; i < m.Hidden(); i++ {
		// pj is a probability and hj is a binary state that takes value 1 with probability pj.
		// Using hj is closer to the mathematical model of an rbm, but using pj usually has less sampling noise which
		// allows slightly faster learning.
		m.h[i] = sigmoid(m.eh(i, v))
		// For the last update of the hidden units, it is silly to use stochastic binary states because nothing
		// depends on which state is chosen. So use the probability itself to avoid unnecessary sampling noise.
		m.rh[i] = sigmoid(m.eh(i, rv))
	}
	m.updateDelta(v, rv, m.h, m.rh)
	return
}

type Option struct {
	// It is possible to update the weights after estimating the gradient on a single training case, but it is
	// often more efficient to divide the training set into small “mini-batches” of 10 to 100 cases.
	// For datasets that contain a small number of equiprobable classes, the ideal mini-batch size is often
	// equal to the number of classes and each mini-batch should contain one example of each class to reduce
	// the sampling error when estimating the gradient for the whole training set from a single mini-batch.
	// For other datasets, first randomize the order of the training examples then use minibatches of size
	// about 10.
	BatchSize int
	Iteration int
	// rbms typically learn better models if more steps of alternating Gibbs sampling are used before
	// collecting the statistics for the second term in the learning rule, which will be called the negative
	// statistics. CDn will be used to denote learning using n full steps of alternating Gibbs sampling.
	GibbsStep int
}

func (m *rbm) update(rate float64) {
	// w
	for i := 0; i < m.Visible(); i++ {
		for j := 0; j < m.Hidden(); j++ {
			// It is important to multiply the derivative of the penalty term by the learning rate. Otherwise,
			// changes in the learning rate change the function that is being optimized rather than just changing
			// the optimization procedure.
			m.w[i][j] += rate * (m.dw[i][j] - weightDecay*m.w[i][j])
		}
	}
	// bv
	for i := 0; i < m.Visible(); i++ {
		m.bv[i] += rate * (m.dbv[i] - weightDecay*m.bv[i])
	}
	// bh
	for i := 0; i < m.Hidden(); i++ {
		m.bh[i] += rate * (m.dbh[i] - weightDecay*m.bh[i])
	}
}

func (m *rbm) Train(data [][]float64, opt *Option) {
	for r := 0; r < opt.Iteration; r++ {
		for b := 0; b < len(data); b += opt.BatchSize {
			size := opt.BatchSize
			if size > len(data)-b {
				size = len(data) - b
			}
			m.resetDelta()
			for i := b; i < b+size; i++ {
				m.cd(opt.GibbsStep, data[i])
			}

			// To avoid having to change the learning rate when the size of a mini-batch is changed, it is helpful
			// to divide the total gradient computed on a mini-batch by the size of the mini-batch, so when talking
			// about learning rates we will assume that they multiply the average, per-case gradient computed on
			// a mini-batch, not the total gradient for the mini-batch.
			m.update(learningRate / float64(size))
		}
	}
}
