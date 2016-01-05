package rbm

type Gaussian struct {
	rbm
}

func NewGaussian(visible, hidden int) *Gaussian {
	m := &Gaussian{rbm: *newRBM(visible, hidden)}
	for i := 0; i < m.Visible(); i++ {
		m.vt[i] = gaussianUnit
	}
	return m
}
