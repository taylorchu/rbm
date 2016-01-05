package rbm

type Binary struct {
	rbm
}

func New(visible, hidden int) *Binary {
	return &Binary{rbm: *newRBM(visible, hidden)}
}
