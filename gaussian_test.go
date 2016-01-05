package rbm

import (
	"bytes"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestGaussianTrain(t *testing.T) {
	shift := 3.0
	m := NewGaussian(2, 11)
	var data [][]float64
	for i := 0; i < 100; i++ {
		data = append(data, []float64{rand.NormFloat64(), rand.NormFloat64() + shift})
	}

	m.Train(data, &Option{
		BatchSize: 10,
		Iteration: 20,
		GibbsStep: 10,
	})

	total := 1000
	for _, test := range []struct {
		in []float64
	}{
		{
			in: []float64{0, shift},
		},
		{
			in: []float64{0.1, shift - 0.1},
		},
		{
			in: []float64{0, shift - 0.1},
		},
	} {
		var count int
		for i := 0; i < total; i++ {
			got, _ := m.Reconstruct(test.in, 2)
			diff := math.Abs(got[1] - got[0] - shift)
			if diff > 3 {
				count++
			}
		}
		errRate := float64(count) / float64(total)
		if errRate > 0.05 {
			t.Fatalf("reconstruct error rate %f", errRate)
		}
	}
}

func TestGaussianMarshal(t *testing.T) {
	m := NewGaussian(3, 2)
	buf := new(bytes.Buffer)
	err := m.WriteTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	m2 := NewGaussian(3, 2)
	err = m2.ReadFrom(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m, m2) {
		t.Fatalf("not equal")
	}
}
