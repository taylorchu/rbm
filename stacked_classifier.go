package rbm

import (
	"errors"
	"io"
)

type StackedClassifier struct {
	gaussian   *Gaussian
	binary     []*Binary
	classifier *Classifier
}

var (
	ErrInvalidLayer = errors.New("not enough layer specified for stacked classifier")
)

func NewStackedClassifier(withGaussian bool, units ...int) (*StackedClassifier, error) {
	// top layer should be a classifier, followed by
	// some binary layers and optionally one gaussian layer.
	s := new(StackedClassifier)

	// 3 for classifier + 1 for each additional layer
	if len(units) < 3 {
		return nil, ErrInvalidLayer
	}
	s.classifier = NewClassifier(units[len(units)-3], units[len(units)-2], units[len(units)-1])
	for i := 0; i < len(units)-3; i++ {
		if withGaussian && i == 0 {
			s.gaussian = NewGaussian(units[i], units[i+1])
		} else {
			s.binary = append(s.binary, New(units[i], units[i+1]))
		}
	}

	return s, nil
}

func (s *StackedClassifier) ReadFrom(r io.Reader) (err error) {
	if s.gaussian != nil {
		err = s.gaussian.ReadFrom(r)
		if err != nil {
			return
		}
	}
	for _, b := range s.binary {
		err = b.ReadFrom(r)
		if err != nil {
			return
		}
	}
	err = s.classifier.ReadFrom(r)
	if err != nil {
		return
	}
	return
}

func (s *StackedClassifier) WriteTo(w io.Writer) (err error) {
	if s.gaussian != nil {
		err = s.gaussian.WriteTo(w)
		if err != nil {
			return
		}
	}
	for _, b := range s.binary {
		err = b.WriteTo(w)
		if err != nil {
			return
		}
	}
	err = s.classifier.WriteTo(w)
	if err != nil {
		return
	}
	return
}

func (s *StackedClassifier) Train(input [][]float64, output []int, opt *Option) {
	next := func(r *rbm) {
		r.Train(input, opt)

		input2 := make([][]float64, len(input))
		for i, v := range input {
			rh := r.ph(v)
			copied := make([]float64, len(rh))
			copy(copied, rh)
			input2[i] = copied
		}
		input = input2
	}
	if s.gaussian != nil {
		next(s.gaussian.rbm)
	}
	for _, b := range s.binary {
		next(b.rbm)
	}
	s.classifier.Train(input, output, opt)
}

func (s *StackedClassifier) Classify(input []float64) int {
	next := func(r *rbm) {
		rh := r.ph(input)
		copied := make([]float64, len(rh))
		copy(copied, rh)
		input = copied
	}
	if s.gaussian != nil {
		next(s.gaussian.rbm)
	}
	for _, b := range s.binary {
		next(b.rbm)
	}
	return s.classifier.Classify(input)
}
